#!/bin/bash

#SBATCH --job-name=exp
#SBATCH --nodes=1
#SBATCH --nodelist=c05
#SBATCH --gres=gpu:2
#SBATCH --time=0-12:00:00  # 12 hours timelimit
#SBATCH --mem=64000MB

source /home/${USER}/.bashrc
conda activate tch

srun python -u /home/gyuseonglee/workspace/2301_OCR/main.py \
--ep 20 \
--bs 128 \
--cnn vgg \
--seq dec \
--tqdm_off \
--trs \
--rot \
--one \
$*  