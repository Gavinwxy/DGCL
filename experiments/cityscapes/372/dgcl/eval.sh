#!/bin/bash
# now=$(date +"%Y%m%d_%H%M%S")
job='eval'
ROOT=../../../..

mkdir -p log
mkdir -p checkpoints/results


now=$(date +"%Y%m%d_%H%M%S")
CUDA_VISIBLE_DEVICES=0 python $ROOT/eval.py \
    --config=config.yaml \
    --base_size 2048 \
    --scales 1.0 \
    --model_path=checkpoints/ckpt_best.pth \
    --save_folder=checkpoints/results \
    2>&1 | tee log/val_last_$now.txt
