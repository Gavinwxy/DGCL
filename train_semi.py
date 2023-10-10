import argparse
import copy
import logging
import os
import os.path as osp
import pprint
import random
import time
from datetime import datetime

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from tensorboardX import SummaryWriter

from dgcl.dataset.augmentation import generate_unsup_data
from dgcl.dataset.builder import get_loader
from dgcl.models.model_helper import ModelBuilder
from dgcl.utils.dist_helper import setup_distributed
from dgcl.utils.loss_helper import (
    get_criterion,
)
from dgcl.utils.lr_helper import get_optimizer, get_scheduler
from dgcl.utils.utils import (
    AverageMeter,
    init_log,
    intersectionAndUnion,
    load_state,
    set_random_seed,
)

from torch.cuda.amp import GradScaler, autocast


from dgcl.dgcl import (
    compute_dgcl_loss,
    FeatureMemory, 
    )


parser = argparse.ArgumentParser(description="Semi-Supervised Semantic Segmentation")
parser.add_argument("--config", type=str, default="config.yaml")
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--port", default=None, type=int)




def compute_unsupervised_loss(predict, target, ignore_mask):
    batch_size, num_class, h, w = predict.shape

    with torch.no_grad():
        target[ignore_mask==255] = 255
        weight = batch_size * h * w / torch.sum(target != 255)

    loss = weight * F.cross_entropy(predict, target, ignore_index=255)  

    return loss


def main():
    global args, cfg
    args = parser.parse_args()
    seed = args.seed
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    scaler = GradScaler()

    logger = init_log("global", logging.INFO)
    logger.propagate = 0

    cfg["exp_path"] = os.path.dirname(args.config)
    cfg["save_path"] = os.path.join(cfg["exp_path"], cfg["saver"]["snapshot_dir"])

    cudnn.enabled = True
    cudnn.benchmark = True

    rank, word_size = setup_distributed(port=args.port)

    if rank == 0:
        logger.info("{}".format(pprint.pformat(cfg)))
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        tb_logger = SummaryWriter(
            osp.join(cfg["exp_path"], "log/events_seg/" + current_time)
        )
    else:
        tb_logger = None

    if args.seed is not None:
        print("set random seed to", args.seed)
        set_random_seed(args.seed)

    if not osp.exists(cfg["saver"]["snapshot_dir"]) and rank == 0:
        os.makedirs(cfg["saver"]["snapshot_dir"])

    # Create network
    model = ModelBuilder(cfg["net"])
    modules_back = [model.encoder]
    if cfg["net"].get("aux_loss", False):
        modules_head = [model.auxor, model.decoder]
    else:
        modules_head = [model.decoder]

    if cfg["net"].get("sync_bn", True):
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model.cuda()

    sup_loss_fn = get_criterion(cfg)

    train_loader_sup, train_loader_unsup, val_loader = get_loader(cfg, seed=seed)

    # Optimizer and lr decay scheduler
    cfg_trainer = cfg["trainer"]
    cfg_optim = cfg_trainer["optimizer"]
    times = 10 if "pascal" in cfg["dataset"]["type"] else 1

    params_list = []
    for module in modules_back:
        params_list.append(
            dict(params=module.parameters(), lr=cfg_optim["kwargs"]["lr"])
        )
    for module in modules_head:
        params_list.append(
            dict(params=module.parameters(), lr=cfg_optim["kwargs"]["lr"] * times)
        )

    optimizer = get_optimizer(params_list, cfg_optim)

    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
    )

    # Teacher model
    model_teacher = ModelBuilder(cfg["net"])
    model_teacher = model_teacher.cuda()
    model_teacher = torch.nn.parallel.DistributedDataParallel(
        model_teacher,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
    )

    for p in model_teacher.parameters():
        p.requires_grad = False

    best_prec = 0
    last_epoch = 0 #############################

    # auto_resume > pretrain
    if cfg["saver"].get("auto_resume", False):
        lastest_model = os.path.join(cfg["save_path"], "ckpt_ori.pth")
        if not os.path.exists(lastest_model):
            "No checkpoint found in '{}'".format(lastest_model)
        else:
            print(f"Resume model from: '{lastest_model}'")
            best_prec, last_epoch = load_state(
                lastest_model, model, optimizer=optimizer, key="model_state"
            )
            _, _ = load_state(
                lastest_model, model_teacher, optimizer=optimizer, key="teacher_state"
            )

    elif cfg["saver"].get("pretrain", False):
        load_state(cfg["saver"]["pretrain"], model, key="model_state")
        load_state(cfg["saver"]["pretrain"], model_teacher, key="teacher_state")

    optimizer_start = get_optimizer(params_list, cfg_optim)
    lr_scheduler = get_scheduler(
        cfg_trainer, len(train_loader_sup), optimizer_start, start_epoch=last_epoch
    )

    # build class-wise memory bank
    memobank = FeatureMemory(memory_per_class=10000, n_classes=cfg["net"]["num_classes"], k=[8,16,32])

    # Start to train model
    for epoch in range(last_epoch, cfg_trainer["epochs"]):
        # Training
        train(
            model,
            model_teacher,
            optimizer,
            lr_scheduler,
            sup_loss_fn,
            train_loader_sup,
            train_loader_unsup,
            epoch,
            tb_logger,
            logger,
            memobank,
            scaler,
        )

        # Validation
        if cfg_trainer["eval_on"]:
            if rank == 0:
                logger.info("start evaluation")

            if epoch < cfg["trainer"].get("sup_only_epoch", 1):
                prec = validate(model, val_loader, epoch, logger)
                # prec = validate_slide(model_teacher, val_loader, epoch, logger)
            else:
                prec_s = validate(model, val_loader, epoch, logger)
                prec_t = validate(model_teacher, val_loader, epoch, logger)
                prec = max(prec_s, prec_t)

            if rank == 0:
                state = {
                    "epoch": epoch + 1,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "teacher_state": model_teacher.state_dict(),
                    "best_miou": best_prec,
                }
                if prec > best_prec:
                    best_prec = prec
                    torch.save(
                        state, osp.join(cfg["saver"]["snapshot_dir"], "ckpt_best.pth")
                    )

                torch.save(state, osp.join(cfg["saver"]["snapshot_dir"], "ckpt.pth"))

                logger.info(
                    "\033[31m * Currently, the best val result is: {:.2f}\033[0m".format(
                        best_prec * 100
                    )
                )
                
                if epoch < cfg["trainer"].get("sup_only_epoch", 1):
                    tb_logger.add_scalar("mIoU val", prec, epoch)
                else:
                    tb_logger.add_scalar("mIoU val", prec_t, epoch)
                    tb_logger.add_scalar("mIoU val student", prec_s, epoch)
                    


def train(
    model,
    model_teacher,
    optimizer,
    lr_scheduler,
    sup_loss_fn,
    loader_l,
    loader_u,
    epoch,
    tb_logger,
    logger,
    memobank,
    scaler,
):
    ema_decay_origin = cfg["net"]["ema_decay"]

    model.train()

    loader_l.sampler.set_epoch(epoch)
    loader_u.sampler.set_epoch(epoch)
    loader_l_iter = iter(loader_l)
    loader_u_iter = iter(loader_u)
    assert len(loader_l) == len(
        loader_u
    ), f"labeled data {len(loader_l)} unlabeled data {len(loader_u)}, imbalance!"

    rank, world_size = dist.get_rank(), dist.get_world_size()

    sup_losses = AverageMeter(10)
    uns_losses = AverageMeter(10)
    con_losses = AverageMeter(10)
    data_times = AverageMeter(10)
    batch_times = AverageMeter(10)
    learning_rates = AverageMeter(10)

    batch_end = time.time()
    for step in range(len(loader_l)):
        batch_start = time.time()
        data_times.update(batch_start - batch_end)

        i_iter = epoch * len(loader_l) + step
        lr = lr_scheduler.get_lr()
        learning_rates.update(lr[0])
        lr_scheduler.step()

        image_l, label_l = loader_l_iter.next()
        _, h, w = label_l.size()
        image_l, label_l = image_l.cuda(), label_l.cuda()

        image_u, ignore_mask = loader_u_iter.next()
        image_u = image_u.cuda()
        ignore_mask = ignore_mask.cuda()


        # unsupervised loss
        pre_epoch = cfg["trainer"].get("sup_only_epoch", 1)
        total_epoch = cfg["trainer"]["epochs"]
        progress = (epoch-pre_epoch) / (total_epoch-pre_epoch)
        

        percent_unreliable = 20 * (1 - progress) # from 20 to 10
        alpha_t = max(80 * (1 - progress), 10) # from 80 to 10 
        
        num_anchor = 200
        contra_weight = 0.5


        if epoch < cfg["trainer"].get("sup_only_epoch", 1):
            with autocast():
                # forward
                outs = model(image_l)
                pred, rep = outs["pred"], outs["rep"]

                pred = F.interpolate(pred, (h, w), mode="bilinear", align_corners=True)

                # supervised loss
                if "aux_loss" in cfg["net"].keys():
                    aux = outs["aux"]
                    aux = F.interpolate(aux, (h, w), mode="bilinear", align_corners=True)
                    sup_loss = sup_loss_fn([pred, aux], label_l)
                else:
                    sup_loss = sup_loss_fn(pred, label_l)

                model_teacher.train()
                _ = model_teacher(image_l)

                unsup_loss = 0 * rep.sum()
                contra_loss = 0 * rep.sum()


            loss = sup_loss + unsup_loss + contra_loss
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        else:
            if epoch == cfg["trainer"].get("sup_only_epoch", 1):
            # copy student parameters to teacher
                with torch.no_grad():
                    for t_params, s_params in zip(
                        model_teacher.parameters(), model.parameters()
                    ):
                        t_params.data = s_params.data


            num_labeled = len(image_l)
            num_unlabeled = len(image_u)
            image_bi = torch.cat((image_l, image_u))
    

            model_teacher.train()
            with torch.no_grad():
                out_bi_t = model_teacher(image_bi)
                pred_bi_t = out_bi_t["pred"].detach()
                
                # Get predictions of original unlabeled data
                pred_u_t = pred_bi_t[num_labeled:]
                
                pred_u_t_large = F.interpolate(pred_u_t, (h, w), mode="bilinear", align_corners=True)
                prob_u_t_large = F.softmax(pred_u_t_large, dim=1)
                logits_u_t, label_u_t = torch.max(prob_u_t_large, dim=1)

                # Create entropy based threshold
                prob = prob_u_t_large.detach()
                entropy = -torch.sum(prob * torch.log(prob + 1e-10), dim=1)

                thresh = torch.quantile(
                    entropy[ignore_mask != 255].detach().flatten(), (100-percent_unreliable)/100
                )
                thresh_mask = entropy.ge(thresh).bool() * (ignore_mask != 255).bool()
                label_u_t[thresh_mask] = 255

   
                if random.uniform(0,1) < 0.5:
                    image_u_aug, label_u_aug, ignore_mask_aug = generate_unsup_data(
                        image_u,
                        label_u_t.clone(),
                        ignore_mask.clone(),
                        mode='cutmix',
                    )
                else:
                    image_u_aug, label_u_aug, ignore_mask_aug = image_u, label_u_t.clone(), ignore_mask.clone()



            image_tri = torch.cat((image_l, image_u, image_u_aug), dim=0)
            with autocast():

                out_tri_s = model(image_tri)
                pred_tri_s, rep_tri_s, fts_tri_s = out_tri_s["pred"], out_tri_s["rep"], out_tri_s['fts']
                
                pred_tri_s_large = F.interpolate(pred_tri_s, size=(h, w), mode="bilinear", align_corners=True)

                pred_l_s_large = pred_tri_s_large[:num_labeled]
                pred_u_aug_s_large = pred_tri_s_large[num_labeled+num_unlabeled:]

                # supervised loss
                if "aux_loss" in cfg["net"].keys():
                    aux = out_tri_s["aux"][:num_labeled]
                    aux = F.interpolate(aux, (h, w), mode="bilinear", align_corners=True)
                    sup_loss = sup_loss_fn([pred_l_s_large, aux], label_l.clone())
                else:
                    sup_loss = sup_loss_fn(pred_l_s_large, label_l.clone())



            # Unspervised consistency training
            with autocast():
                unsup_loss = compute_unsupervised_loss(
                            pred_u_aug_s_large,
                            label_u_aug.clone(),
                            ignore_mask_aug.clone(),
                        )
            

            # contrastive loss using pseudo labels
            with torch.no_grad():
                # pred_bi_t 
                probs_bi_t = torch.softmax(pred_bi_t, dim=1) 
                _, probs_u_t = probs_bi_t[:num_labeled], probs_bi_t[num_labeled:]
                

                label_l_small = F.interpolate(label_l.unsqueeze(1).float(),size=pred_bi_t.shape[2:],mode="nearest").squeeze(1)
                _, label_u_small = torch.max(probs_u_t, dim=1)
                ignore_mask_small = F.interpolate(ignore_mask.unsqueeze(1).float(),size=pred_bi_t.shape[2:],mode="nearest").squeeze(1).long()

            
                entropy_u = -torch.sum(probs_u_t * torch.log(probs_u_t + 1e-10), dim=1)
                high_thresh_u = torch.quantile(entropy_u[ignore_mask_small != 255].detach().flatten(), (100 - alpha_t)/100)
                high_entropy_mask_u = (entropy_u.ge(high_thresh_u).bool() * (ignore_mask_small != 255).bool())
                label_u_small[high_entropy_mask_u] = 255        
                label_u_small[ignore_mask_small == 255] = 255 

                # The important label
                label_contra_memo = torch.cat([label_l_small.long(), label_u_small.long()], dim=0)


            # Extract predictions on labeled and original unlabeled 
            rep_bi_s = rep_tri_s[:num_labeled+num_unlabeled]
            fts_bi_s = fts_tri_s[:num_labeled+num_unlabeled] 

            if memobank.check_if_full():
                with autocast():

                    contra_loss = compute_dgcl_loss(                        
                        rep = rep_bi_s,                    
                        fts = fts_bi_s.float().detach(),
                        memo = memobank,
                        label = label_contra_memo.detach(),
                        k_low_thresh=num_anchor,
                        k_den_cal=[8,16,32])


                if contra_loss is None:
                    contra_loss = 0*rep_tri_s.sum()
                dist.barrier()
                dist.all_reduce(contra_loss)
                contra_loss = (
                    contra_loss
                    / world_size
                )

            else:
                with autocast():
                    contra_loss = 0*rep_tri_s.sum()


            loss = sup_loss + contra_weight*unsup_loss + contra_weight*contra_loss

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()


            # Update memory bank
            memobank.update(fts_bi_s.float().detach(), rep_bi_s.float().detach(), label_contra_memo.detach()) 
            

            with torch.no_grad():
                ema_decay = min(
                    1
                    - 1
                    / (
                        i_iter
                        - len(loader_l) * cfg["trainer"].get("sup_only_epoch", 1)
                        + 1
                    ),
                    ema_decay_origin,
                )

                for t_params, s_params in zip(
                    model_teacher.parameters(), model.parameters()
                ):
                    t_params.data = (
                        ema_decay * t_params.data + (1 - ema_decay) * s_params.data
                    )
                

        # gather all loss from different gpus
        reduced_sup_loss = sup_loss.clone().detach()
        dist.all_reduce(reduced_sup_loss)
        sup_losses.update(reduced_sup_loss.item())

        reduced_uns_loss = unsup_loss.clone().detach()
        dist.all_reduce(reduced_uns_loss)
        uns_losses.update(reduced_uns_loss.item())

        reduced_con_loss = contra_loss.clone().detach()
        con_losses.update(reduced_con_loss.item())

        batch_end = time.time()
        batch_times.update(batch_end - batch_start)

        if i_iter % 10 == 0 and rank == 0:
            logger.info(
                "[{}] "
                "Iter [{}/{}]\t"
                "Time {batch_time.val:.2f} ({batch_time.avg:.2f})\t"
                "Sup {sup_loss.val:.3f} ({sup_loss.avg:.3f})\t"
                "Uns {uns_loss.val:.3f} ({uns_loss.avg:.3f})\t"
                "Con {con_loss.val:.3f} ({con_loss.avg:.3f})\t".format(
                    cfg["dataset"]["n_sup"],
                    i_iter,
                    cfg["trainer"]["epochs"] * len(loader_l),
                    batch_time=batch_times,
                    sup_loss=sup_losses,
                    uns_loss=uns_losses,
                    con_loss=con_losses,
                )
            )

            tb_logger.add_scalar("lr", learning_rates.val, i_iter)
            tb_logger.add_scalar("Sup Loss", sup_losses.val, i_iter)
            tb_logger.add_scalar("Uns Loss", uns_losses.val, i_iter)
            tb_logger.add_scalar("Con Loss", con_losses.val, i_iter)




def validate(
    model,
    data_loader,
    epoch,
    logger,
):
    model.eval()
    data_loader.sampler.set_epoch(epoch)

    num_classes, ignore_label = (
        cfg["net"]["num_classes"],
        cfg["dataset"]["ignore_label"],
    )
    rank, world_size = dist.get_rank(), dist.get_world_size()

    intersection_meter = AverageMeter()
    union_meter = AverageMeter()

    for step, batch in enumerate(data_loader):
        images, labels = batch
        images = images.cuda()
        labels = labels.long().cuda()

        with torch.no_grad():
            outs = model(images)

        # get the output produced by model_teacher
        output = outs["pred"]
        output = F.interpolate(
            output, labels.shape[1:], mode="bilinear", align_corners=True
        )
        output = output.data.max(1)[1].cpu().numpy()
        target_origin = labels.cpu().numpy()

        # start to calculate miou
        intersection, union, target = intersectionAndUnion(
            output, target_origin, num_classes, ignore_label
        )

        # gather all validation information
        reduced_intersection = torch.from_numpy(intersection).cuda()
        reduced_union = torch.from_numpy(union).cuda()
        reduced_target = torch.from_numpy(target).cuda()

        dist.all_reduce(reduced_intersection)
        dist.all_reduce(reduced_union)
        dist.all_reduce(reduced_target)

        intersection_meter.update(reduced_intersection.cpu().numpy())
        union_meter.update(reduced_union.cpu().numpy())

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)

    if rank == 0:
        for i, iou in enumerate(iou_class):
            logger.info(" * class [{}] IoU {:.2f}".format(i, iou * 100))
        logger.info(" * epoch {} mIoU {:.2f}".format(epoch, mIoU * 100))

    return mIoU




if __name__ == "__main__":
    main()
