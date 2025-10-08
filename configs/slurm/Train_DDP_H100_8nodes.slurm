#!/bin/bash
#SBATCH --job-name=bevclr_ddp
#SBATCH --partition=gpu8             
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96          # CPU threads total on the node; adjust to your node
#SBATCH --time=2:00:00              # TODO: adjust
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# SBATCH --account=YOUR_ACCOUNT     # (optional) uncomment & set if your cluster requires it


###########################################################################################################
## If your already on the GPU just run the sript directly (./configs/slurm/Train_DDP_H100_8nodes.slurm)
## important configs ( batch size = 4, nworkers = 4 , device_ids = [0,1,2,3,4,5,6,7] )
##########################################################################################################

# --- env / modules ---
module purge
module load devel/cuda/12.4

# Conda (adapt if your site uses a different init script)
source ~/.bashrc
conda activate bev_clr_ad

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

mkdir -p "$PROJECT_DIR/logs_train/DDP_H100"
cd "$PROJECT_DIR"

# --- launch ---
# Single-node, 8 GPUs, torchrun standalone rendezvous
torchrun --nproc_per_node=${SLURM_GPUS_ON_NODE:-8} --standalone \
  train_DDP.py --config="$CONFIG_PATH"