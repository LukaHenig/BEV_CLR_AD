#!/bin/bash
#SBATCH --partition=gpu1            # oder passender GPU-Partitionname
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=48                   # 1 Task pro GPU
#SBATCH --mem=64G
#SBATCH --output=logs_train/train_L40S/%x_%j.out
#SBATCH --job-name=bevcar_train

# Lade Module oder aktiviere Umgebung
#source ~/.bashrc
#conda activate bev_clr_ad

# Training starten
export OMP_NUM_THREADS=16
export CUDA_VISIBLE_DEVICES=0
export WANDB__SERVICE_WAIT=300

torchrun --nproc_per_node=8 --nnodes=1 --master_port=12345 \
  train.py --config='configs/train/train_bevcar.yaml'
