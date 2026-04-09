#!/bin/bash -l
#SBATCH --partition=gpu1            # oder passender GPU-Partitionname
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1                  # 1 Task pro GPU
#SBATCH --mem=64G
#SBATCH --job-name=bev_clr_ad_eval
#SBATCH --output=logs_eval/%x_%j.out
#SBATCH --chdir=/home/es/es_es/es_luheit04/forschungsprojekt/BEV_CLR_AD


set -e  

mkdir -p logs_eval

module load devel/cuda/12.4

# Conda initialisieren 
if ! command -v conda >/dev/null 2>&1; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || true
  source "$HOME/anaconda3/etc/profile.d/conda.sh" 2>/dev/null || true
fi

# WICHTIG: kein -u aktiv
conda activate bev_clr_ad

# Debug
hostname; pwd; date
which python; python -V
nvidia-smi || true


export CUDA_VISIBLE_DEVICES=0
export WANDB__SERVICE_WAIT=300

python eval.py  --config='configs/eval/eval_bev_clr_ad.yaml'
