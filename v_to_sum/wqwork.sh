#!/usr/bin/env csh

#$ -q gpu@@csecri-titanxp
#$ -pe smp 6

module load cctools/6.0.10

work_queue_worker -M shadho
