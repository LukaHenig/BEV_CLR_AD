#!/bin/bash
#SBATCH --partition=gpu1            # oder passender GPU-Partitionname
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=48                   # 1 Task pro GPU
#SBATCH --mem=64G
#SBATCH --output=logs_train/train_L40S/%x_%j.out
#SBATCH --job-name=bevcar_train

# Lade Module oder aktiviere Umgebung
#source ~/.bashrc
conda activate bev_clr_ad
#cd BEV_CLR_AD
#test
# Training starten
train.py --config='configs/train/train_bevcar.yaml'
