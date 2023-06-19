#!/bin/bash

DEVICE=$1
SEED=$2 # 0, 1, 2, 3, 4
CREDENTIALS=$3

METHOD=same # same, top_k, exact
K=0 #1,3,5,10

#-------------------------------------------------------------------------------#
# Train MNIST with SAME


CUDA_VISIBLE_DEVICES=$DEVICE python3 train.py \
--epochs 30 \
--batch-size 100 --seed 42 --method=$METHOD --images-per-addition=2 --k=$K \
--network-type nn --lr 0.005 \
--num-workers 0 --p-num 8 --credentials $CREDENTIALS



#NN
# for S in 0 1 2 3 4
# do

#     CUDA_VISIBLE_DEVICES=$DEVICE python3 train.py \
#     --epochs 30 \
#     --batch-size 100 --seed $S --method=$METHOD --images-per-addition=2 --k=$K \
#     --network-type nn --lr 0.005 \
#     --num-workers 0 --p-num 8 --credentials $CREDENTIALS


#     CUDA_VISIBLE_DEVICES=$DEVICE python3 train.py \
#     --epochs 30 \
#     --batch-size 100 --seed $S --method=$METHOD --images-per-addition=3 --k=$K \
#     --network-type nn --lr 0.005 \
#     --num-workers 0 --p-num 8 --credentials $CREDENTIALS


#     CUDA_VISIBLE_DEVICES=$DEVICE python3 train.py \
#     --epochs 30 \
#     --batch-size 100 --seed $S --method=$METHOD --images-per-addition=4 --k=$K \
#     --network-type nn --lr 0.005 \
#     --num-workers 0 --p-num 8 --credentials $CREDENTIALS
# done


#PC
# for S in 0 1 2 3 4
# do
#     CUDA_VISIBLE_DEVICES=$DEVICE python3 train.py \
#     --epochs 30 \
#     --batch-size 100 --seed $S --method=$METHOD --images-per-addition=2 --k=$K \
#     --network-type pc --pc-structure poon-domingos --lr 0.01 \
#     --num-workers 0 --p-num 8 --credentials $CREDENTIALS


#     CUDA_VISIBLE_DEVICES=$DEVICE python3 train.py \
#     --epochs 30 \
#     --batch-size 100 --seed $S --method=$METHOD --images-per-addition=3 --k=$K \
#     --network-type pc --pc-structure poon-domingos  --lr 0.01 \
#     --num-workers 0 --p-num 8 --credentials $CREDENTIALS

#     CUDA_VISIBLE_DEVICES=$DEVICE python3 train.py \
#     --epochs 30 \
#     --batch-size 100 --seed $S --method=$METHOD --images-per-addition=4 --k=$K \
#     --network-type pc --pc-structure poon-domingos  --lr 0.01 \
#     --num-workers 0 --p-num 8 --credentials $CREDENTIALS
# done