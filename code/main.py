from config import MapConfig, LidarConfig, RobotConfig
from dataset_utils import load_dataset, save_synced_dataset, load_synced_data
from occupancy_grid import build_occupancy_grid, update_occupancy_grid_vectorized, ogm_plot_vectorized
from particle_filter_cpu import motion_update, measurement_update_cpu, compute_neff, resample_particles, particle_filter_cpu
from particle_filter_gpu import motion_update, measurement_update_gpu, compute_neff, resample_particles, particle_filter_gpu
from visualize import visualize_ogm
import cupy as cp
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from tqdm import tqdm

def main():
    global x_im, y_im, grid, particles, weights, traj_estimates
    save_synced_dataset()
    data = load_synced_data("synced_data.npz")

    # Print shape/type information
    print("--- Synced data keys and shapes ---")
    for key, value in data.items():
        print(f"{key}: shape={value.shape}, dtype={value.dtype}")
    print("-----------------------------------")

    map_cfg = MapConfig()
    lidar_cfg = LidarConfig()
    grid = build_occupancy_grid(data, map_cfg, lidar_cfg)
    np.savez("ogm_grid.npz", grid=grid, map_cfg=vars(map_cfg))
    print("OGM grid saved to ogm_grid.npz")
    #visualize_ogm(grid, map_cfg)

    NUM_PARTICLES = 1000
    particle_filter_cpu(NUM_PARTICLES)
    # particle_filter_gpu(NUM_PARTICLES) 

if __name__ == "__main__":
    main()