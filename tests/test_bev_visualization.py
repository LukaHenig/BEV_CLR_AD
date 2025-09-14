import os
import numpy as np
import matplotlib.pyplot as plt


def visualize_bev(map_mask, radar_pts, lidar_pts, out_path):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(map_mask, extent=[-50, 50, -50, 50], origin='lower', cmap='gray')
    ax.scatter(radar_pts[:, 0], radar_pts[:, 1], c='r', s=8, label='Radar')
    ax.scatter(lidar_pts[:, 0], lidar_pts[:, 1], c='b', s=2, label='LiDAR')
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.legend(loc='upper right')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def test_bev_visualization(tmp_path):
    map_mask = np.zeros((200, 200))
    map_mask[80:120, 80:120] = 1.0
    radar_pts = np.random.uniform(-50, 50, size=(32, 2))
    lidar_pts = np.random.uniform(-50, 50, size=(128, 2))
    out_file = tmp_path / 'bev_overlay.png'
    path = visualize_bev(map_mask, radar_pts, lidar_pts, str(out_file))
    assert os.path.exists(path)
