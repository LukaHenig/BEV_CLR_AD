#!/bin/bash -l
#SBATCH --partition=gpu8 
#SBATCH --gres=gpu:h100:8 
#SBATCH --time=48:00:00           
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96 
#SBATCH --mem=80G         
#SBATCH --job-name=bevclr_H100_train              
#SBATCH --output=logs_train/train_H100/%x_%j.out
#SBATCH --chdir=/home/es/es_es/es_luheit04/forschungsprojekt/BEV_CLR_AD


set -e  

mkdir -p logs_train/train_H100

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

# --- quiet runtime env ---
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1

# Silence chatty libs:
export PYTHONWARNINGS="ignore"       # hide UserWarnings (e.g., sklearn/numpy)
export NCCL_DEBUG=ERROR              # hide NCCL INFO/WARN
export TQDM_DISABLE=1                # hide tqdm progress bars (if used)
export WANDB_SILENT=false             # minimize W&B chatter
export WANDB_CONSOLE=auto            # (off) for no console hijack


# --- paths ---
PROJECT_DIR="/home/es/es_es/es_luheit04/forschungsprojekt/BEV_CLR_AD"
CONFIG_PATH="configs/train/train_bev_clr_ad.yaml"

cd "$PROJECT_DIR"

# --- launch ---
# Single-node, 8 GPUs, torchrun standalone rendezvous
torchrun --nproc_per_node=${SLURM_GPUS_ON_NODE:-8} --standalone \
  train_DDP.py --config="$CONFIG_PATH"

###########################################################################################################
# If your already on the GPU just run the sript directly (./configs/slurm/Train_DDP_H100_8nodes.slurm)
# important configs ( batch size = 4, nworkers = 4 , device_ids = [0,1,2,3,4,5,6,7] )
#Start: sbatch BEV_CLR_AD/configs/slurm/Train_DDP_H100_8nodes.slurm
#Kill: scancel "JOBID"
#See: squeue -u $USER or sacct -j "JOBID" --format=JobID,JobName,State,ExitCode,Elapsed
##########################################################################################################
