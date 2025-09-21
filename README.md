# particleSLAM
ERL 2025 Summer Project

This project implements simultaneous localization and mapping (SLAM) using encoder and IMU odometry, and 2-D
LiDAR scans from a differential-drive robot.

## Modular Source Structure
```
your_project/
├── config.py # All dataclass configs (robot, map, lidar)
├── dataset_utils.py # Dataset loading, saving, and synchronization
├── odom.py # DifferentialDrive class, odometry utilities
├── occupancy_grid.py # Occupancy grid build/update and grid math
├── particle_filter.py # Particle filter logic (CPU & GPU), measurements
├── visualization.py # All plotting/visualization routines
├── main.py # Entry point: ties entire pipeline together
├── README.md
├── 
└── (other scripts, data, docs)
```
## Features

- Differential-drive odometry integration
- LiDAR scan interpolation and world transformation
- Occupancy grid mapping with Bresenham ray tracing
- Particle Filter SLAM using CPU (soon be updated)
- Particle Filter SLAM using CuPy (GPU)
- Trajectory and particle visualization in world coordinates (meters)

## Requirements

- Python 3.8+
- [NumPy](https://numpy.org/)
- [CuPy](https://cupy.dev/)
- [Matplotlib](https://matplotlib.org/)
- [tqdm](https://tqdm.github.io/)


## Usage

1. **Prepare dataset:** Place encoder, LiDAR, and IMU files (e.g., `Encoders20.npz`) in the `../data/` folder.
2. **Run the main SLAM pipeline:**
python main.py
- This performs data sync, mapping, and SLAM, then generates PNG figures and saves mapping results (e.g., `.npz`, `.csv`).

---

## Output

- `ogm_grid.npz`: Final occupancy grid for analysis
- `pf_grid_cells_####.png`: Snapshots of mapping and trajectory
- Optionally `.csv` or `.npy` for map export

## License

## Contributors
- Hyungjun Doh