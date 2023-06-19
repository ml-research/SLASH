#!/bin/bash

DEVICE=$1
SEED=$2 # 0, 1, 2, 3, 4
CREDENTIALS=$3



#-------------------------------------------------------------------------------#
# Train on CLEVR_v1 with cnn model

CUDA_VISIBLE_DEVICES=$DEVICE python3 test.py \
--seed $SEED \
--network-type nn --batch-size 100 \
--num-workers 0 --p-num 16 --credentials $CREDENTIALS

