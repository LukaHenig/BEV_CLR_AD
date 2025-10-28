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
├── tests/                     # Pytest suite covering dataloading, evaluation, and transformer utilities.
├── train.py / train_DDP.py    # Entry points for DataParallel and DistributedDataParallel training.
├── eval.py                    # Evaluation + metric aggregation over DAY/RAIN/NIGHT splits.
├── nuscenes_data.py           # nuScenes dataloader with camera, radar, and LiDAR support.
├── custom_nuscenes_splits.py  # Helper for custom DAY/RAIN/NIGHT validation split.
└── saverloader.py, utils/     # Checkpoint and misc utility helpers.
```

---

## 📔 Abstract

Semantic scene segmentation from a bird's-eye-view (BEV) perspective plays a crucial role in facilitating planning and decision-making for mobile robots. Although recent vision-only methods have demonstrated notable advancements in performance, they often struggle under adverse illumination conditions such as rain or nighttime. While active sensors offer a solution to this challenge, the prohibitively high cost of LiDARs remains a limiting factor. Fusing camera data with automotive radars poses a more inexpensive alternative but has received less attention in prior research. In this work, we aim to advance this promising avenue by introducing BEVCar, a novel approach for joint BEV object and map segmentation. The core novelty of our approach lies in first learning a point-based encoding of raw radar data, which is then leveraged to efficiently initialize the lifting of image features into the BEV space. We perform extensive experiments on the nuScenes dataset and demonstrate that BEVCar outperforms the current state of the art. Moreover, we show that incorporating radar information significantly enhances robustness in challenging environmental conditions and improves segmentation performance for distant objects.

---

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

The project targets Python 3.10, PyTorch 2.1.2, and CUDA 11.8. A typical conda-based setup looks like:

```
conda create --name bevcar python=3.10
conda activate bevcar
conda install pytorch=2.1.2 torchvision=0.16.2 torchaudio=2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install pip
conda install xformers -c xformers
pip install -r requirements.txt
```

Compile the deformable attention CUDA ops once per environment:

```
cd nets/ops
sh make.sh
python test.py  # optional correctness check
cd ../..
```

---

## 📥 Pre-trained checkpoints

Pretrained BEVCar checkpoints remain available for camera-only, camera–radar, and ResNet-based variants. Download the desired weights and place them under `model_checkpoints/` following this structure:

```
model_checkpoints/
    BEVCar/model-000050000.pth
    BEVCar_ResNet/model-000050000.pth
    CAM_ONLY/model-000050000.pth
```

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

---

## 🧰 Utilities & tests

* [`nuscenes_data.py`](./nuscenes_data.py) exposes helper functions for camera projection, radar/LiDAR voxelization, BEV occupancy creation, and dataset compilation. These utilities power both training and evaluation pipelines.
* [`saverloader.py`](./saverloader.py) handles checkpoint I/O—including optimizer/scheduler state restoration when `load_optimizer`/`load_scheduler` are enabled.
* The [`tests/`](./tests) folder ships with a Pytest suite. Run it locally via:

  ```
  pytest
  ```

  or use the convenience wrapper:

  ```
  bash scripts/run_tests.sh
  ```

The tests create light-weight stubs for nuScenes to validate geometric transforms, dataloader augmentations, and transformer fusion components without needing the full dataset.

---

## 👩‍⚖️ License

The code is released under the [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license. For commercial inquiries, please contact the authors.

---

## 🙏 Acknowledgment

We thank the authors of [Simple-BEV](https://github.com/aharley/simple_bev) for publicly releasing their [source code](https://github.com/aharley/simple_bev). This work was supported by Qualcomm Technologies Inc., the German Research Foundation (DFG) Emmy Noether Program grant No. 468878300, and an academic grant from NVIDIA.
