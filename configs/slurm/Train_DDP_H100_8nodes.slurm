#!/bin/bash
#SBATCH --job-name=bev_ddp
#SBATCH --partition=gpu8
#SBATCH --nodes=1
#SBATCH --ntasks=1                 # ONE task; torchrun will fork 8 workers
#SBATCH --gpus-per-node=8          # or: --gres=gpu:h100:8   (pick one, not both)
#SBATCH --cpus-per-task=16         # CPU cores for the *single* launcher task
#SBATCH --time=48:00:00
#SBATCH --mem=0                    # let Slurm give full node mem; or set e.g. 512G
#SBATCH --output=logs_train/DDP_H100/%x_%j.out

# always start in the submit dir
cd "$SLURM_SUBMIT_DIR"

# env (no manual CUDA_VISIBLE_DEVICES)
export OMP_NUM_THREADS=16
unset CUDA_VISIBLE_DEVICES
export PYTHONUNBUFFERED=1
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1           # single-node only
export WANDB__SERVICE_WAIT=300

# quick sanity (shows up in your log file)
echo "== Host $(hostname) =="
nvidia-smi -L
python - <<'PY'
import os, torch
print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("cuda.is_available    =", torch.cuda.is_available())
print("device_count         =", torch.cuda.device_count())
print("torch                =", torch.__version__)
print("torch.version.cuda   =", torch.version.cuda)
PY

# launch (torchrun creates 8 local worker processes)
srun --ntasks=1 --gpus=8 --cpu-bind=cores \
     bash -lc "torchrun --standalone --nproc_per_node=8 \
       train_ddp.py --config configs/train/train_bev_clr_ad.yaml"