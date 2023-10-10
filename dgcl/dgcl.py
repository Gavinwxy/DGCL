from random import sample
import torch
import torch.distributed as dist
import numpy as np
from torch.cuda.amp import autocast


@torch.no_grad()
def gather_together(tensor):
    dist.barrier()
    
    world_size = dist.get_world_size()
    tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensor_list, tensor)
    output_tensor = torch.cat(tensor_list, dim=0)
    return output_tensor


def dist_based_density_cal_multi_scale(feat, memo, k=[16,32,64], norm=False, mode='dot'):
    ''' Function to calculate feature to memory density based on feature distances
    Inputs:
        feat: tensor of shape [M C]
        memo: tensor of shape [N C]
        k: number of samples for density evaluation
        norm: bool, if feature and memory bank is l2 normalized
        mode: mode of distance
    Outputs:
        density: tensor of shape [N M] 
    '''
    if norm:
        feat = torch.nn.functional.normalize(feat, p=2, dim=1)
        memo = torch.nn.functional.normalize(memo, p=2, dim=1)

    assert mode in ['l2','dot','cos']
    if mode == 'l2':
        feat_to_memo_sim = -torch.cdist(feat.unsqueeze(0), memo.unsqueeze(0)).squeeze(0) # N M
    if mode == 'dot':
        feat_to_memo_sim = torch.matmul(feat, memo.T)
    if mode == 'cos':
        feat_to_memo_sim = torch.cosine_similarity(feat.unsqueeze(1), memo.unsqueeze(0), dim=2) 

    k_1, k_2, k_3 = k


    v_1, _ = torch.topk(feat_to_memo_sim, k=k_1, dim=1)
    v_2, _ = torch.topk(feat_to_memo_sim, k=k_2, dim=1)
    v_3, _ = torch.topk(feat_to_memo_sim, k=k_3, dim=1)

    den_1 = torch.sum(v_1, dim=1) / k_1
    den_2 = torch.sum(v_2, dim=1) / k_2
    den_3 = torch.sum(v_3, dim=1) / k_3
    
    feat_to_memo_density = (den_1 + den_2 + den_3) / 3

    
    return feat_to_memo_density



class FeatureMemory:
    def __init__(self, memory_per_class=10000, n_classes=21, feat_dim=256, proj_dim=256, k=[8,16,32]):
        self.memory_per_class = memory_per_class
        self.n_classes = n_classes
        self.k = k
        self.fts_memory = []
        self.proj_memory = []
        self.feat_density = []

        for i in range(self.n_classes):
            self.fts_memory.append(torch.zeros(0,feat_dim).cuda()) # Memo for fts
            self.proj_memory.append(torch.zeros(0,proj_dim).cuda()) # Memo for rep
            self.feat_density.append(torch.zeros(1).cuda())


    def check_if_full(self):
        full = True
        for item in self.fts_memory:
            if item.size(0) < self.memory_per_class:
                full = False
        return full


    @torch.no_grad()
    def update(self, fts, rep, labels):
        fts_dim, rep_dim = fts.size(1), rep.size(1)
        
        fts = fts.permute(0,2,3,1) # B H W C
        rep = rep.permute(0,2,3,1) # B H W C
        
        data = torch.cat([fts, rep, labels.unsqueeze(-1)], dim=-1)
        data = gather_together(data)

        fts = data[:,:,:,:fts_dim] # B H W C
        rep = data[:,:,:,fts_dim:fts_dim+rep_dim] # B H W C
        labels = data[:,:,:,-1] # B H W

        
        for c in range(self.n_classes):

            mask_c = labels == c  # get mask for class c
            
            fts_c = fts[mask_c, :].clone()# get fts from class c
            rep_c = rep[mask_c, :].clone()

            c_num = fts_c.size(0) # number of selected fts

            if c_num > 1000:
                idx = np.random.choice(c_num, 1000, False)
                fts_c = fts_c[idx]
                rep_c = rep_c[idx]
            
            if self.fts_memory[c].size(0) > 1000:
                with autocast(enabled=False):
                    density_c = dist_based_density_cal_multi_scale(fts_c.clone(), 
                                        self.fts_memory[c].clone(),
                                        k=self.k,
                                        norm=True,
                                        mode='dot')
            else:
                density_c = torch.zeros(fts_c.size(0)).cuda()

            
            self.fts_memory[c] = torch.cat((fts_c, self.fts_memory[c]), dim=0)[:self.memory_per_class]
            self.proj_memory[c] = torch.cat((rep_c, self.proj_memory[c]), dim=0)[:self.memory_per_class]
            self.feat_density[c] = torch.cat((density_c, self.feat_density[c]), dim=0)[:self.memory_per_class]                
  



def compute_dgcl_loss(
        rep,
        fts,
        memo,
        label,
        temperature=0.5, 
        k_den_cal=[8,16,32],
        k_high_thresh=50,  
        k_low_thresh=256,
        ): 
    

    feat_memory = torch.stack(memo.fts_memory)
    proj_memory = torch.stack(memo.proj_memory)
    # Get in-memory feature densities
    in_memo_density = torch.stack(memo.feat_density)


    # Scan classes in current batches
    batch_cls = torch.unique(label)
    cls_filt = (batch_cls!=255)
    # In pascal, do not perform contrastive loss on background category
    cls_filt = cls_filt*(batch_cls!=0)
    batch_cls = batch_cls[cls_filt]

    fts = fts.permute(0,2,3,1) # B H W C
    rep = rep.permute(0,2,3,1) # B H W C

    # Begin loss calculation
    loss = 0
    cls_cnt = 0
    for cls in batch_cls:
        low_thresh = k_low_thresh
        high_thresh = k_high_thresh

        cls_map = label == cls

        cls_fts_cnt = cls_map.sum() # check number of featurs for current class
        if cls_fts_cnt < k_low_thresh: #500
            low_thresh = cls_fts_cnt  # Select all of them

        if cls_fts_cnt < k_high_thresh:
            high_thresh = cls_fts_cnt

        if cls_fts_cnt == 0: # if class fts are not enough, skip current class
            continue

        feat_cls = fts[cls_map] # N 256
        proj_cls = rep[cls_map] # N 256


        if cls_fts_cnt > 10000: # sample fts if number of feature is too large # 2000
            sample_idx = np.random.choice(feat_cls.size(0), 10000, False)
            feat_cls = feat_cls[sample_idx]
            proj_cls = proj_cls[sample_idx]

        feat_memo_cls = feat_memory[cls] # 5000 256
        if feat_cls.size(0) > max(k_den_cal):
            with autocast(enabled=False):
                # Student to teacher density 
                s_to_b_density = dist_based_density_cal_multi_scale(
                    feat=feat_cls, 
                    memo=feat_memo_cls, #feat_cls, 
                    k=k_den_cal,
                    norm=True,
                    mode='dot'
                    )

        else: 
            s_to_b_density = feat_cls.new_ones(feat_cls.size(0))


        # Select low density anchors
        _, idx = torch.topk(s_to_b_density, k=low_thresh, dim=0, largest=False)
        proj_anchor = proj_cls[idx].clone().cuda()
        
        # Select high density batch samples as positive samples.
        _, idx_pos = torch.topk(s_to_b_density, k=high_thresh)
        proj_pos_b = proj_cls[idx_pos].detach().clone() # No gradients 
        
        # Select high density memory samples 
        memo_cls_density = in_memo_density[cls] # memo_size
        _, idx_pos = torch.topk(memo_cls_density, k=k_high_thresh)

        proj_pos_m = proj_memory[cls][idx_pos] # 50 256
        proj_pos = torch.cat([proj_pos_b, proj_pos_m], dim=0)

        # Randomly sample negative samples
        not_cls_map = (label!=cls)*(label!=255) # Remember to filt out ignored labels
        not_cls_cnt = not_cls_map.sum()
        
        proj_neg = rep[not_cls_map].detach().clone()# (num_cls-1)*50 256
    
        if not_cls_cnt > 512: 
            sample_neg_idx = np.random.choice(int(not_cls_cnt), 512, False)
            proj_neg = proj_neg[sample_neg_idx].detach().clone()


        proto = torch.mean(proj_pos, dim=0, keepdim=True) # 1 256 
        all_proj = torch.cat((proto, proj_neg), dim=0) # M 256

        proj_anchor_norm = torch.nn.functional.normalize(proj_anchor, p=2, dim=1)
        all_proj_norm = torch.nn.functional.normalize(all_proj, p=2, dim=1)

        # Calculating loss
        logits_cls = torch.div(torch.matmul(proj_anchor_norm, all_proj_norm.T), temperature) # K M
            
        logits_max, _ = torch.max(logits_cls, dim=1, keepdim=True)
        logits = logits_cls - logits_max.detach()

        # compute log_prob
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)
        
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = log_prob[:,0]

        # loss
        loss = loss - mean_log_prob_pos.mean()
        
        cls_cnt += 1
    
    if cls_cnt == 0:
        return None

    loss = loss / cls_cnt
    
    return loss


