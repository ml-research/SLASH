#!/bin/bash

#MODEL="slot-attention-set-pred-shapeworld4"
#MODEL="slot-attention-set-pred-shapeworld4-cogent"
MODEL="slot-attention-set-pred-clevr"
#MODEL="slot-attention-set-pred-clevr-cogent"


#DATA="../data/shapeworld4"
#DATA="../data/shapeworld_cogent"
DATA="../../../data/CLEVR_v1.0"
#DATA="../../../data/CLEVR_CoGenT_v1.0"


#DATASET=shapeworld4
DATASET=clevr

DEVICE=$1
SEED=$2 # 0, 1, 2, 3, 4
CREDENTIALS=$3
#-------------------------------------------------------------------------------#

CUDA_VISIBLE_DEVICES=$DEVICE python3 train.py \
--data-dir $DATA --dataset $DATASET --epochs 1000 \
--name $MODEL --lr 0.0004 --batch-size 512 --n-slots 10 --n-iters-slot-att 3 --n-attr 15 --ap-log 1 --seed $SEED \
--warmup-epochs 8 --decay-epochs 360 --num-workers 8 --credentials $CREDENTIALS



# # CLEVR
# for S in 0 1 2 3 4
# do
#     CUDA_VISIBLE_DEVICES=$DEVICE python3 train.py \
#     --data-dir $DATA --dataset $DATASET --epochs 1000 \
#     --name $MODEL --lr 0.0004 --batch-size 512 --n-slots 10 --n-iters-slot-att 3 --n-attr 15 --ap-log 1 --seed $S \
#     --warmup-epochs 8 --decay-epochs 360 --num-workers 8 --credentials $CREDENTIALS
# done

# # CLEVR COGENT
# for S in 0 1 2 3 4
# do
#     CUDA_VISIBLE_DEVICES=$DEVICE python3 train.py \
#     --data-dir $DATA --dataset $DATASET --epochs 1000 \
#     --name $MODEL --lr 0.0004 --batch-size 512 --n-slots 10 --n-iters-slot-att 3 --n-attr 15 --ap-log 1 --seed $S \
#     --warmup-epochs 8 --decay-epochs 360 --num-workers 8 --credentials $CREDENTIALS
# done

# # SHAPEWORLD4
# for S in 0 1 2 3 4
# do

#     CUDA_VISIBLE_DEVICES=$DEVICE python3 train.py \
#     --data-dir $DATA --dataset $DATASET --epochs 1000 \
#     --name $MODEL --lr 0.0004 --batch-size 512 --n-slots 4 --n-iters-slot-att 3 --n-attr 15 --ap-log 1 --seed $S \
#     --warmup-epochs 8 --decay-epochs 360 --num-workers 8 --credentials $CREDENTIALS
# done

# SHAPEWORLD4 COGENT
# for S in 0 1 2 3 4
# do

#     CUDA_VISIBLE_DEVICES=$DEVICE python3 train.py \
#     --data-dir $DATA --dataset $DATASET --epochs 1000 \
#     --name $MODEL --lr 0.0004 --batch-size 512 --n-slots 4 --n-iters-slot-att 3 --n-attr 15 --ap-log 1 --seed $S \
#     --warmup-epochs 8 --decay-epochs 360 --num-workers 8 --credentialpns $CREDENTIALS --cogent
# done

