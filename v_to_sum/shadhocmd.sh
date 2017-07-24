#!/usr/bin/env bash

#$ -q gpu@@csecri-titanxp
#$ -pe smp 6
#$ -N vtosumtest


module load python/2.7.13
module load cuda/8.0
module load cudnn/5.1

gpu=$(nvidia-smi -q -d PIDS | awk 'BEGIN { gpu = 0; n = 0; } /Attached/ { n += $4; } /Processes/ && NF < 3 { gpu++; } END { if (gpu >= n) { print -1; } if (gpu < n) { print gpu; } }')

source /afs/crc.nd.edu/user/k/klannon/local_root_rhel7/root/bin/thisroot.csh

mkdir .theano
wd=$(pwd)
THEANO_FLAGS="base_compiledir=${wd}/.theano:$THEANO_FLAGS"
CUDA_VISIBLE_DEVICES=$gpu

python train.py   -N 25 --train-fraction 0.85

tar czf out.tar.gz performance.json
