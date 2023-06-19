#!/bin/bash

DEVICE=$1
SEED=$2 # 0, 1, 2, 3, 4
CREDENTIALS=$3


# 0.00001, 0.0001,  0.001, 0.01
LR=0.001

#-------------------------------------------------------------------------------#

CUDA_VISIBLE_DEVICES=$DEVICE python3 train.py \
--epochs 100 \
--batch-size 100 --seed $SEED   \
--network-type nn --lr $LR \
--num-workers 0 --p-num 20 --credentials $CREDENTIALS \
--exp-name vqa_c2
