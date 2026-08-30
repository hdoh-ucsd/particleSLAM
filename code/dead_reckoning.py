"""Deterministic odometry baseline using encoder speed and filtered IMU yaw."""

from typing import Mapping

import numpy as np
from tqdm import tqdm

from config import LidarConfig, MapConfig, RobotConfig
from occupancy_grid import update_occupancy_grid_vectorized
from particle_filter_cpu import compute_differential_drive_update


def run_dead_reckoning(
    data: Mapping[str, np.ndarray],
    map_cfg: MapConfig | None = None,
    lidar_cfg: LidarConfig | None = None,
    robot_cfg: RobotConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the deterministic trajectory and its occupancy grid."""
    map_cfg = map_cfg or MapConfig()
    lidar_cfg = lidar_cfg or LidarConfig()
    robot_cfg = robot_cfg or RobotConfig()
    grid = np.zeros((map_cfg.sizex, map_cfg.sizey), dtype=np.float32)
    pose = np.zeros(3, dtype=float)
    trajectory: list[np.ndarray] = []
    angles = lidar_cfg.angle_min + np.arange(data["lidar"].shape[0]) * lidar_cfg.angle_increment
    sync_times = data["sync_times"]

    for index in tqdm(range(1, sync_times.size), desc="Dead reckoning"):
        dt = float(sync_times[index] - sync_times[index - 1])
        if dt <= 0:
            continue
        if "body_motion" in data:
            forward, lateral = data["body_motion"][:2, index]
            delta_heading = float(data["imu_angular_velocity"][2, index]) * dt
            midpoint_heading = pose[2] + delta_heading / 2.0
            pose[0] += forward * np.cos(midpoint_heading) - lateral * np.sin(midpoint_heading)
            pose[1] += forward * np.sin(midpoint_heading) + lateral * np.cos(midpoint_heading)
            pose[2] = (pose[2] + delta_heading + np.pi) % (2.0 * np.pi) - np.pi
            trajectory.append(pose.copy())
            update_occupancy_grid_vectorized(
                grid, pose, data["lidar"][:, index], angles, lidar_cfg, map_cfg
            )
            continue
        velocity, yaw_rate = compute_differential_drive_update(
            data["encoder_counts"][:, index],
            data["imu_angular_velocity"][2, index],
            dt,
            robot_cfg,
        )
        delta_heading = yaw_rate * dt
        midpoint_heading = pose[2] + delta_heading / 2.0
        pose[0] += velocity * dt * np.cos(midpoint_heading)
        pose[1] += velocity * dt * np.sin(midpoint_heading)
        pose[2] = (pose[2] + delta_heading + np.pi) % (2.0 * np.pi) - np.pi
        trajectory.append(pose.copy())
        update_occupancy_grid_vectorized(
            grid,
            pose,
            data["lidar"][:, index],
            angles,
            lidar_cfg,
            map_cfg,
        )

    return np.asarray(trajectory), grid
