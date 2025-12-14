"""Generate a histogram of lidar point counts for all nuScenes LIDAR_TOP frames.

This script iterates over every LIDAR_TOP ``sample_data`` record in a nuScenes
installation, counts the number of lidar points, and plots the distribution.

Example:
    python scripts/nuscenes_lidar_histogram.py \
        --dataroot /../../../../../../beegfs/scratch/workspace/es_luheit04-NuScneDataset_new/es_luheit04-NuSceneDataset-1756172124/nuscenes \
        --version v1.0-trainval \
        --output lidar_hist.png
"""

import argparse
import os
from typing import List

import matplotlib

# Use a non-interactive backend so the script works in headless environments.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataroot",
        type=str,
        required=True,
        help="Path to the nuScenes dataset root (e.g., /datasets/nuscenes)",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v1.0-trainval",
        help="nuScenes version to load (e.g., v1.0-trainval, v1.0-mini)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="nuscenes_lidar_histogram.png",
        help="Where to save the histogram image",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=50,
        help="Number of histogram bins",
    )
    return parser.parse_args()


def collect_lidar_counts(nusc: NuScenes) -> List[int]:
    """Return the number of lidar points for each LIDAR_TOP sample_data frame."""
    counts: List[int] = []
    lidar_sample_data = [sd for sd in nusc.sample_data if sd["channel"] == "LIDAR_TOP"]

    for sd in tqdm(lidar_sample_data, desc="Counting lidar points"):
        lidar_path = os.path.join(nusc.dataroot, sd["filename"])
        lidar_points = LidarPointCloud.from_file(lidar_path)
        counts.append(lidar_points.points.shape[1])

    return counts


def plot_histogram(counts: List[int], bins: int, output_path: str) -> None:
    plt.figure(figsize=(10, 6))
    plt.hist(counts, bins=bins, color="steelblue", edgecolor="black")
    plt.xlabel("Number of lidar points per frame")
    plt.ylabel("Frame count")
    plt.title("nuScenes LIDAR_TOP point count distribution")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path)


def main() -> None:
    args = parse_args()
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True)

    counts = collect_lidar_counts(nusc)
    if not counts:
        raise RuntimeError("No LIDAR_TOP frames found. Check the dataset path and version.")

    plot_histogram(counts, bins=args.bins, output_path=args.output)

    summary = (
        f"Frames: {len(counts)} | "
        f"Min: {min(counts)} | Max: {max(counts)} | "
        f"Mean: {sum(counts) / len(counts):.2f}"
    )
    print(summary)
    print(f"Histogram saved to: {args.output}")


if __name__ == "__main__":
    main()
