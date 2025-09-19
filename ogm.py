#!/usr/bin/env python3
# ogm_synced.py — builds & visualizes an occupancy grid from synced_data.npz

import argparse
import numpy as np
import matplotlib.pyplot as plt
from robot import RobotConfig, LidarConfig, MapConfig
from grid_mapping import bresenham2D, mapCorrelation, update_occupancy_grid, logodds_to_prob

def load_synced_data(npz_file):
    data = np.load(npz_file)
    # Convert arrays from npz to a dict for legacy compatibility
    return {key: data[key] for key in data}

def build_occupancy_grid(data, map_cfg, lidar_cfg, debug_file=None):
    grid = np.zeros((map_cfg.sizex, map_cfg.sizey), dtype=np.float32)
    x_im = np.arange(map_cfg.xmin, map_cfg.xmax + map_cfg.res, map_cfg.res)
    y_im = np.arange(map_cfg.ymin, map_cfg.ymax + map_cfg.res, map_cfg.res)
    lidar_angles = getattr(lidar_cfg, "angle_min", -2.356194490192345) + np.arange(data["lidar"].shape[0]) * getattr(lidar_cfg, "angle_increment", 0.00436332)
    lidar_angles = lidar_angles.reshape((-1,))
    if debug_file is not None:
        debug_file.write(f"lidar_angles shape:{lidar_angles.shape}\n")

    for k in range(data["sync_times"].size):
        pose = data["pose"][:, k]
        scan = data["lidar"][:, k]
        update_occupancy_grid(grid, pose, scan, lidar_angles, lidar_cfg, map_cfg, debug_file)
    return grid

def update_occupancy_grid(grid, pose, scan, angles, lidar_cfg, map_cfg, debug_file=None):
    """
    Update the occupancy grid in place based on current robot pose and lidar scan.

    Parameters:
        grid      : np.ndarray, shape (map_cfg.sizex, map_cfg.sizey), log-odds grid (float32)
        pose      : np.ndarray, (3,), [x, y, theta] (robot pose)
        scan      : np.ndarray, (n_beams,), lidar ranges
        angles    : np.ndarray, (n_beams,), corresponding angles
        lidar_cfg : LidarConfig object
        map_cfg   : MapConfig object
        debug_file: optional file handle for logging
    """
    # Parameters for log-odds update
    free_val = 1.0    # Experiment: try larger (e.g., 2.0) for clearer free area
    occ_val = 2.0     # Make occupied stand out

    x0 = int((pose[0] - map_cfg.xmin) / map_cfg.res)
    y0 = int((pose[1] - map_cfg.ymin) / map_cfg.res)

    for i in range(len(scan)):
        r = scan[i]
        if r < lidar_cfg.rmin or r > lidar_cfg.rmax:
            continue
        angle = angles[i]
        end_x = pose[0] + r * np.cos(pose[2] + angle)
        end_y = pose[1] + r * np.sin(pose[2] + angle)
        x1 = int((end_x - map_cfg.xmin) / map_cfg.res)
        y1 = int((end_y - map_cfg.ymin) / map_cfg.res)
        free_cells = bresenham2D(x0, y0, x1, y1)
        free_x = np.clip(np.round(free_cells[0]).astype(int), 0, grid.shape[0] - 1)
        free_y = np.clip(np.round(free_cells[1]).astype(int), 0, grid.shape[1] - 1)
        grid[free_x, free_y] -= free_val          # FILL free space along ray
        x1_clipped = np.clip(x1, 0, grid.shape[0] - 1)
        y1_clipped = np.clip(y1, 0, grid.shape[1] - 1)
        grid[x1_clipped, y1_clipped] += occ_val   # Mark hit as occupied


def visualize_grid(grid, map_cfg):
    prob = logodds_to_prob(grid)
    extent = [map_cfg.ymin, map_cfg.ymax, map_cfg.xmin, map_cfg.xmax]
    plt.figure(figsize=(8,8))
    plt.imshow(prob.T, origin='lower', cmap='gray', extent=extent)
    plt.title("Occupancy Grid Map")
    plt.colorbar(label="Occupancy Probability")
    plt.xlabel("Y [m]")
    plt.ylabel("X [m]")
    plt.tight_layout()
    plt.show(block=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synced_file", type=str, default="synced_data.npz")
    args = parser.parse_args()

    debug_file = open("debug_log.txt", "w")
    data = load_synced_data(args.synced_file)

    # Print shape/type information
    print("--- Synced data keys and shapes ---")
    for key, value in data.items():
        print(f"{key}: shape={value.shape}, dtype={value.dtype}")
    print("-----------------------------------")

    map_cfg = MapConfig()
    lidar_cfg = LidarConfig()
    grid = build_occupancy_grid(data, map_cfg, lidar_cfg, debug_file)
    np.savez("ogm_grid.npz", grid=grid, map_cfg=vars(map_cfg))
    print("OGM grid saved to ogm_grid.npz")

    visualize_grid(grid, map_cfg)
    debug_file.close()


if __name__ == "__main__":
    main()
