# BEV-CLR-AD
[**arXiv**](https://arxiv.org/abs/2403.11761) | [**IEEE Xplore**](https://ieeexplore.ieee.org/document/10802147) | [**Website**](http://bevcar.cs.uni-freiburg.de/) | [**Video**](https://youtu.be/bB_k_6IvPHQ?feature=shared)

This repository extends the official BEVCar implementation with a configurable **C**amera–**L**iDAR–**R**adar (CLR) stack for bird's-eye-view (BEV) semantic map and vehicle segmentation. In addition to the camera–radar fusion presented in the paper, the code base now ships with:

* LiDAR ingestion through the same voxelized encoder interface that is used for radar (with an optional lightweight occupancy compressor fallback).
* Transformer-based fusion blocks that let you independently enable/disable the camera, radar, and LiDAR branches and control how the modalities initialize the BEV queries.
* Reworked configuration files that capture all data, training, and evaluation toggles—including support for different GPU topologies (L40S or H100) and for resuming experiments from checkpoints.
* Unit tests that validate the most critical data loading utilities, transformer components, and end-to-end training/evaluation helpers.

If you use this project in your research, please cite the BEVCar paper:

```
@inproceedings{schramm2024bevcar,
  author={Schramm, Jonas and Vödisch, Niclas and Petek, Kürsat and Kiran, B Ravi and Yogamani, Senthil and Burgard, Wolfram and Valada, Abhinav},
  booktitle={2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  title={{BEVCar}: Camera-Radar Fusion for BEV Map and Object Segmentation},
  year={2024},
  pages={1435-1442},
}
```

---

## 📦 Repository structure

```
BEV_CLR_AD/
├── configs/                   # Training / evaluation configs for different GPU targets.
├── nets/                      # Network components (transformer fusion, voxel encoders, deformable ops bindings, …).
├── train.py / train_DDP.py    # Entry points for DataParallel and DistributedDataParallel training.
├── eval.py                    # Evaluation + metric aggregation over DAY/RAIN/NIGHT splits.
├── nuscenes_data.py           # nuScenes dataloader with camera, radar, and LiDAR support.
├── custom_nuscenes_splits.py  # Helper for custom DAY/RAIN/NIGHT validation split.
└── saverloader.py, utils/     # Checkpoint and misc utility helpers.
```


## 💾 Data preparation

We evaluate on the nuScenes dataset. Download the dataset from the [nuScenes website](https://www.nuscenes.org/download) and extract it, e.g. to `/datasets/nuscenes`, which should yield:

```
/datasets/nuscenes
    samples/          # Sensor data for keyframes (RGB + RADAR + LiDAR).
    sweeps/           # Intermediate sensor sweeps.
    maps/             # Rasterized PNGs and vector JSONs.
    v1.0-trainval/    # Metadata + annotations for train/val.
    v1.0-test/        # Metadata + annotations for test.
    v1.0-mini/        # Metadata + annotations for the mini split.
```

* Update the `data_dir` entry inside your chosen config file in [`configs/`](./configs) to point at this directory.
* Optionally set `custom_dataroot` and `use_pre_scaled_imgs: true` if you created prescaled images using [`nuscenes_image_converter.py`](./nuscenes_image_converter.py).
* Radar sweeps (`nsweeps`) and LiDAR sweeps (`lidar_nsweeps`) are configurable per run and are passed directly to the dataloader.
* For the custom DAY/RAIN/NIGHT split, replace `create_splits_scenes()[split]` with [`create_drn_eval_split_scenes()[split]`](./custom_nuscenes_splits.py) and set `do_drn_val_split: true` in your config. [`eval.py`](./eval.py) and the provided evaluation configs already do this for you.

> ℹ️ Expected Shapely < 2.0 warnings emitted by the nuScenes map API are filtered automatically inside the training/evaluation entry points.

---

## 🧪 Environment setup

The project targets **Python 3.10** and is tested with **PyTorch 2.5.1 (cu121 wheels)**.
On our cluster, the deformable attention CUDA ops are compiled on a GPU node with the **CUDA 12.4 module** loaded.

A typical conda-based setup looks like:

```
conda env create -f environment.yml
conda activate bev_clr_ad

# Safety pins for this project/toolchain
python -m pip install "setuptools<82" wheel
python -m pip install "numpy<2"

# Install PyTorch separately (must match our working setup)
python -m pip install --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1

# Python dependencies (project requirements, excluding torch/xformers/spconv/cumm)
python -m pip install -r requirements-pip-freeze.txt

# Optional (otherwise only warnings about missing xFormers)
conda install xformers -c xformers -y

# Required for VoxelNeXt sparse backbone
python -m pip install cumm-cu124==0.7.11 spconv-cu124==2.3.8
```

Compile the deformable attention CUDA ops once per environment (on a GPU node with CUDA toolkit available):

```
# Example: start an interactive GPU shell (cluster-specific)
srun --partition=gpu1 --gres=gpu:1 --time=08:00:00 --nodes=1 --ntasks=1 --mem=64G --pty /bin/bash

# Load CUDA toolkit for compilation (cluster-specific module name)
module purge
module load devel/cuda/12.4

conda activate bev_clr_ad
cd nets/ops
rm -rf build *.so **/*.so 2>/dev/null || true
sh make.sh
python test.py  # optional correctness check (should print multiple True checks)
cd ../..
```

Start training (example):

```
CUDA_VISIBLE_DEVICES=0 python train.py --config='configs/train/train_bev_clr_ad_L40S_run1.yaml'
```

### Notes

* Do **not** install `MultiScaleDeformableAttention` via `pip`; it is built locally in `nets/ops`.
* Do **not** add `torch`, `torchvision`, `torchaudio`, `xformers`, `spconv`, or `cumm` to `requirements-pip-freeze.txt` (they are installed separately as shown above).
* If you see a NumPy compatibility issue, re-pin NumPy with `python -m pip install "numpy<2"`.
* If you see `No module named pkg_resources`, re-pin setuptools with `python -m pip install "setuptools<82"`.

---

## 📥 Pre-trained checkpoints

so far none

You can also resume CLR experiments by pointing `init_dir` to an existing folder (see the configs for examples).

---

## 🚀 Training

Two launcher scripts are provided:

* [`train.py`](./train.py) for single-node `torch.nn.DataParallel` (convenient for debugging).
* [`train_DDP.py`](./train_DDP.py) for multi-GPU `torch.distributed` training.

Example (single GPU, DataParallel):

```
CUDA_VISIBLE_DEVICES=0 python train.py --config configs/train/train_bev_clr_ad_L40S.yaml
```

Example (8× GPU, DistributedDataParallel):

```
OMP_NUM_THREADS=16 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
WANDB__SERVICE_WAIT=300 \
torchrun --nproc_per_node=8 --nnodes=1 --master_port=1234 \
    train_DDP.py --config configs/train/train_bev_clr_ad_H100.yaml
```

Key training options are surfaced in the YAML configs: modality toggles (`use_radar`, `use_lidar`, `use_pre_scaled_imgs`), voxel encoder types (`radar_encoder_type`, `lidar_encoder_type`), transformer depth (`num_layers`), and scheduling (`grad_acc`, `use_scheduler`). Adjust `device_ids`, `batch_size`, and gradient accumulation to match your hardware.

Saved checkpoints live under `model_checkpoints/<exp_name>/`, with retention controlled by `keep_latest`.

---

## 📊 Evaluation

[`eval.py`](./eval.py) runs inference, aggregates IoUs (overall + distance buckets), and optionally reports per-condition metrics on the DAY/RAIN/NIGHT split. Use the provided config as a starting point:

```
CUDA_VISIBLE_DEVICES=0 python eval.py --config configs/eval/eval_bev_clr_ad.yaml
```

Important flags:

* `init_dir` and `load_step` select which checkpoint to evaluate.
* `train_task` governs whether to score map-only, object-only, or joint heads.
* `use_radar`, `use_lidar`, and `use_radar_filters` should mirror the settings used during training.
* Metrics are printed in tabulated form at the end of the run; TensorBoard logs are saved under `log_dir`.

Batch size is fixed to 1 for evaluation because the script still relies on deterministic ordering of the nuScenes samples.
