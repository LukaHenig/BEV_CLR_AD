#!/bin/bash -l
#SBATCH --partition=gpu1
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=64G
#SBATCH --job-name=bev_clr_ad_L40s_train
#SBATCH --chdir=/home/es/es_es/es_luheit04/forschungsprojekt/BEV_CLR_AD
#SBATCH --output=logs_train/train_L40S/%x_%j.out

set -e  

mkdir -p logs_train/train_L40S

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

srun -u python train.py --config=configs/train/train_bev_clr_ad.yaml

#########################################################################################
#Start: sbatch BEV_CLR_AD/configs/slurm/Train_L40S.slurm
#Kill: scancel "JOBID"
#See: squeue -u $USER or sacct -j "JOBID" --format=JobID,JobName,State,ExitCode,Elapsed
##########################################################################################

