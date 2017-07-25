#!/usr/bin/env bash

#$ -q gpu@@csecri-titanxp
#$ -pe smp 6
#$ -N vtosumtest

source /afs/crc.nd.edu/user/j/jkinniso/Public/apps/x86_64/scripts/env.sh
module load python/2.7.11
module load cuda/8.0
module load cudnn/5.1

gpu=$(nvidia-smi -q -d PIDS | awk 'BEGIN { gpu = 0; n = 0; } /Attached/ { n += $4; } /Processes/ && NF < 3 { gpu++; } END { if (gpu >= n) { print -1; } if (gpu < n) { print gpu; } }')

source /afs/crc.nd.edu/user/k/klannon/local_root_rhel7/root/bin/thisroot.sh

mkdir .theano
wd=$(pwd)
THEANO_FLAGS="base_compiledir=${wd}/.theano:$THEANO_FLAGS"
CUDA_VISIBLE_DEVICES=$gpu

python train.py train.npz -N 25 --train-fraction 0.85

tar czf out.tar.gz performance.json