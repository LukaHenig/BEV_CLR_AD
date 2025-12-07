"""
Quick utility to grab a single nuScenes sample and render the raw inputs:
- Six camera views
- Combined 5-radar point cloud (3D scatter)
- LiDAR point cloud (3D scatter)

Usage:
    python single_frame_viz.py --config configs/train/train_bev_clr_ad_L40S.yaml \
        --output assets/sample_frame.png

The script reuses the same nuScenes loader settings as training. It assumes the
config contains data_dir/custom_dataroot entries that point to your dataset.
"""
from __future__ import annotations

import argparse
import itertools
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

import nuscenes_data

# Match the defaults used in train.py
SCENE_CENTROID = np.array([[0.0, 1.0, 0.0]])
BOUNDS = (-50, 50, -5, 5, -50, 50)


def _load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_data_aug_conf(cfg: dict) -> dict:
    final_dim = cfg.get("final_dim", [448, 896])
    if cfg.get("rand_crop_and_resize", True):
        resize_lim = [0.8, 1.2]
        crop_offset = int(final_dim[0] * (1 - resize_lim[0]))
    else:
        resize_lim = [1.0, 1.0]
        crop_offset = 0
    return {
        "crop_offset": crop_offset,
        "resize_lim": resize_lim,
        "final_dim": final_dim,
        "H": 900,
        "W": 1600,
        "cams": [
            "CAM_FRONT_LEFT",
            "CAM_FRONT",
            "CAM_FRONT_RIGHT",
            "CAM_BACK_LEFT",
            "CAM_BACK",
            "CAM_BACK_RIGHT",
        ],
        "ncams": cfg.get("ncams", 6),
    }


def _unpack_sample(sample, radar_encoder_type: str):
    if radar_encoder_type == "voxel_net":
        (
            imgs,
            rots,
            trans,
            intrins,
            seg_bev_g,
            valid_bev_g,
            radar_data,
            lidar_data,
            bev_map_mask_g,
            bev_map_g,
            egocar_bev,
            *_,
        ) = sample
    else:
        (
            imgs,
            rots,
            trans,
            intrins,
            seg_bev_g,
            valid_bev_g,
            radar_data,
            lidar_data,
            bev_map_mask_g,
            bev_map_g,
            egocar_bev,
        ) = sample
    return imgs, radar_data, lidar_data


def _prep_camera_axes(fig):
    axes = []
    for row in range(2):
        for col in range(3):
            axes.append(fig.add_subplot(3, 3, row * 3 + col + 1))
    return axes


def _plot_cameras(fig, imgs: torch.Tensor):
    # imgs: (S, C, H, W) with values in [0, 1]
    cam_axes = _prep_camera_axes(fig)
    for ax, (cam_idx, cam) in zip(cam_axes, enumerate(imgs)):
        cam_np = cam.permute(1, 2, 0).clamp(0, 1).cpu().numpy()
        ax.imshow(cam_np)
        ax.axis("off")
        ax.set_title(f"Camera {cam_idx}")


def _plot_pointcloud(ax, points: np.ndarray, title: str, color_dim: int | None = None):
    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        s=1,
        c=points[:, color_dim] if color_dim is not None else "tab:blue",
        cmap="viridis",
    )
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect([1, 1, 0.5])


def main():
    parser = argparse.ArgumentParser(description="Render a single nuScenes sample with raw inputs.")
    parser.add_argument("--config", required=True, help="Path to a training YAML config.")
    parser.add_argument("--output", default="assets/sample_frame.png", help="Where to save the figure.")
    parser.add_argument("--sample-index", type=int, default=0, help="Index of the sample to visualize (within the dataset).")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    data_aug_conf = _build_data_aug_conf(cfg)

    loader, _ = nuscenes_data.compile_data(
        cfg.get("dset", "trainval"),
        cfg["data_dir"],
        data_aug_conf=data_aug_conf,
        centroid=SCENE_CENTROID,
        bounds=BOUNDS,
        res_3d=tuple(cfg.get("grid_dim", [200, 8, 200])),
        bsz=1,
        nworkers=cfg.get("nworkers", 1),
        shuffle=False,
        nsweeps=cfg.get("nsweeps", 1),
        use_radar_filters=cfg.get("use_radar_filters", False),
        do_shuffle_cams=False,
        radar_encoder_type=cfg.get("radar_encoder_type", "voxel_net"),
        use_shallow_metadata=cfg.get("use_shallow_metadata", True),
        use_pre_scaled_imgs=cfg.get("use_pre_scaled_imgs", False),
        custom_dataroot=cfg.get("custom_dataroot", None),
        use_obj_layer_only_on_map=cfg.get("use_obj_layer_only_on_map", True),
        use_radar_occupancy_map=cfg.get("use_radar_occupancy_map", False),
        use_lidar=cfg.get("use_lidar", False),
        lidar_nsweeps=cfg.get("lidar_nsweeps", 1),
    )

    all_samples = itertools.islice(loader, args.sample_index, args.sample_index + 1)
    try:
        sample = next(all_samples)
    except StopIteration:
        raise RuntimeError(f"Dataset is empty or sample_index {args.sample_index} is out of bounds")

    imgs, radar_data, lidar_data = _unpack_sample(sample, cfg.get("radar_encoder_type", "voxel_net"))

    # Strip time dimension and batch dim
    imgs = imgs[0, 0]
    radar_points = radar_data[0, 0].T.cpu().numpy()
    lidar_points = lidar_data[0, 0].T.cpu().numpy() if lidar_data is not None else None

    fig = plt.figure(figsize=(18, 12))
    _plot_cameras(fig, imgs)

    radar_ax = fig.add_subplot(3, 3, 7, projection="3d")
    _plot_pointcloud(radar_ax, radar_points, "Radar point cloud", color_dim=3 if radar_points.shape[1] > 3 else None)

    lidar_ax = fig.add_subplot(3, 3, 8, projection="3d")
    if lidar_points is not None:
        _plot_pointcloud(lidar_ax, lidar_points, "LiDAR point cloud", color_dim=3 if lidar_points.shape[1] > 3 else None)
    else:
        lidar_ax.set_title("LiDAR point cloud (not provided)")
        lidar_ax.axis("off")

    Path(os.path.dirname(args.output) or ".").mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(args.output, dpi=200)
    print(f"Saved frame visualization to {args.output}")


if __name__ == "__main__":
    main()
