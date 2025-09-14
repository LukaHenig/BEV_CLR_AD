#!/bin/bash
#SBATCH --partition=gpu8            # oder passender GPU-Partitionname
#SBATCH --gres=gpu:h100:8
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=96                   # 1 Task pro GPU
#SBATCH --mem=1400G
#SBATCH --output=logs_train/DDP_H100/%x_%j.out
#SBATCH --job-name=bev_clr_ad_train_ddp

# Lade Module oder aktiviere Umgebung
#source ~/.bashrc
#conda activate bev_clr_ad


########### Irgendwie läuft es nicht mehr, muss man noch mal analysieren 


# Training starten
export OMP_NUM_THREADS=16
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB__SERVICE_WAIT=300

torchrun --nproc_per_node=8 --nnodes=1 --master_port=12345 \
  train.py --config='configs/train/train_bev_clr_ad.yaml'
