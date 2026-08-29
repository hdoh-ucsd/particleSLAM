"""CuPy particle-filter backend."""

from pathlib import Path
from typing import Mapping

import cupy as cp
import numpy as np
from tqdm import tqdm

from config import LidarConfig, MapConfig, ParticleFilterConfig, RobotConfig
from occupancy_grid import update_occupancy_grid_vectorized
from particle_filter_cpu import compute_differential_drive_update
from visualize import visualize_particles


def motion_update(
    particles: cp.ndarray,
    encoder_counts: np.ndarray,
    imu_angular_velocity: np.ndarray,
    index: int,
    dt: float,
    robot_cfg: RobotConfig,
    rng: cp.random.RandomState,
    filter_cfg: ParticleFilterConfig,
) -> cp.ndarray:
    velocity, yaw_rate = compute_differential_drive_update(
        encoder_counts[:, index], imu_angular_velocity[2, index], dt, robot_cfg
    )
    sampled_velocity = velocity + rng.normal(
        0.0, filter_cfg.linear_noise_std, particles.shape[0]
    )
    sampled_yaw_rate = yaw_rate + rng.normal(
        0.0, filter_cfg.angular_noise_std, particles.shape[0]
    )
    heading = particles[:, 2]
    particles[:, 0] += sampled_velocity * dt * cp.cos(heading)
    particles[:, 1] += sampled_velocity * dt * cp.sin(heading)
    particles[:, 2] += sampled_yaw_rate * dt
    return particles


def transform_lidar_to_world_gpu(
    scan: cp.ndarray, particles: cp.ndarray, lidar_cfg: LidarConfig
) -> tuple[cp.ndarray, cp.ndarray]:
    angles = lidar_cfg.angle_min + cp.arange(scan.size) * lidar_cfg.angle_increment
    valid = (scan > lidar_cfg.rmin) & (scan < lidar_cfg.rmax)
    local_x = scan[valid] * cp.cos(angles[valid]) + lidar_cfg.x
    local_y = scan[valid] * cp.sin(angles[valid]) + lidar_cfg.y
    headings = particles[:, 2, None]
    world_x = particles[:, 0, None] + local_x * cp.cos(headings) - local_y * cp.sin(headings)
    world_y = particles[:, 1, None] + local_x * cp.sin(headings) + local_y * cp.cos(headings)
    return world_x, world_y


def measurement_update_gpu(
    particles: cp.ndarray,
    scan: np.ndarray,
    _grid: np.ndarray,
    _x_coordinates: np.ndarray,
    _y_coordinates: np.ndarray,
    lidar_cfg: LidarConfig,
    filter_cfg: ParticleFilterConfig | None = None,
    rng: cp.random.RandomState | None = None,
) -> cp.ndarray:
    """GPU scan-to-grid correlation over a local pose neighborhood."""
    del rng
    filter_cfg = filter_cfg or ParticleFilterConfig()
    scan_gpu = cp.asarray(scan)[:: filter_cfg.correlation_beam_stride]
    beam_indices = cp.arange(scan_gpu.size) * filter_cfg.correlation_beam_stride
    angles = lidar_cfg.angle_min + beam_indices * lidar_cfg.angle_increment
    valid = (scan_gpu > lidar_cfg.rmin) & (scan_gpu < lidar_cfg.rmax_used)
    scan_gpu = scan_gpu[valid]
    angles = angles[valid]
    if scan_gpu.size < filter_cfg.min_valid_beams or not np.any(_grid):
        return cp.full(particles.shape[0], 1.0 / particles.shape[0])

    map_gpu = cp.asarray(_grid)
    local_x = scan_gpu * cp.cos(angles) + lidar_cfg.x
    local_y = scan_gpu * cp.sin(angles) + lidar_cfg.y
    xy_offsets = np.arange(
        -filter_cfg.correlation_xy_window,
        filter_cfg.correlation_xy_window + filter_cfg.correlation_xy_step / 2.0,
        filter_cfg.correlation_xy_step,
    )
    yaw_offsets = np.arange(
        -filter_cfg.correlation_yaw_window,
        filter_cfg.correlation_yaw_window + filter_cfg.correlation_yaw_step / 2.0,
        filter_cfg.correlation_yaw_step,
    )
    resolution = _x_coordinates[1] - _x_coordinates[0]
    best_scores = cp.full(particles.shape[0], -cp.inf)
    best_offsets = cp.zeros_like(particles)

    for yaw_offset in yaw_offsets:
        heading = particles[:, 2, None] + yaw_offset
        world_x = particles[:, 0, None] + local_x * cp.cos(heading) - local_y * cp.sin(heading)
        world_y = particles[:, 1, None] + local_x * cp.sin(heading) + local_y * cp.cos(heading)
        for x_offset in xy_offsets:
            grid_x = cp.floor((world_x + x_offset - _x_coordinates[0]) / resolution).astype(cp.int32)
            for y_offset in xy_offsets:
                grid_y = cp.floor((world_y + y_offset - _y_coordinates[0]) / resolution).astype(cp.int32)
                in_bounds = (
                    (grid_x >= 0)
                    & (grid_x < map_gpu.shape[0])
                    & (grid_y >= 0)
                    & (grid_y < map_gpu.shape[1])
                )
                safe_x = cp.clip(grid_x, 0, map_gpu.shape[0] - 1)
                safe_y = cp.clip(grid_y, 0, map_gpu.shape[1] - 1)
                scores = cp.where(in_bounds, map_gpu[safe_x, safe_y], -10.0).mean(axis=1)
                improved = scores > best_scores
                best_scores = cp.where(improved, scores, best_scores)
                best_offsets[improved] = cp.asarray((x_offset, y_offset, yaw_offset))

    particles += best_offsets
    logits = filter_cfg.likelihood_temperature * (best_scores - cp.max(best_scores))
    weights = cp.exp(cp.clip(logits, -700.0, 0.0))
    weight_sum = weights.sum()
    if not bool(cp.isfinite(weight_sum).item()) or float(weight_sum.item()) <= 0:
        return cp.full(particles.shape[0], 1.0 / particles.shape[0])
    return weights / weight_sum


def compute_neff(weights: cp.ndarray) -> float:
    return float((1.0 / cp.sum(weights**2)).item())


def resample_particles(
    particles: cp.ndarray, weights: cp.ndarray, rng: cp.random.RandomState
) -> cp.ndarray:
    positions = (rng.uniform() + cp.arange(len(particles))) / len(particles)
    indices = cp.searchsorted(cp.cumsum(weights), positions, side="right")
    return particles[indices]


def particle_filter_gpu(
    data: Mapping[str, np.ndarray],
    filter_cfg: ParticleFilterConfig | None = None,
    map_cfg: MapConfig | None = None,
    lidar_cfg: LidarConfig | None = None,
    robot_cfg: RobotConfig | None = None,
    output_file: str | Path | None = None,
    diagnostics: dict[str, list[float]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the GPU particle filter and return host-side result arrays."""
    filter_cfg = filter_cfg or ParticleFilterConfig()
    map_cfg = map_cfg or MapConfig()
    lidar_cfg = lidar_cfg or LidarConfig()
    robot_cfg = robot_cfg or RobotConfig()
    rng = cp.random.RandomState(filter_cfg.seed)

    particles = rng.normal(0.0, 0.1, (filter_cfg.num_particles, 3))
    weights = cp.full(filter_cfg.num_particles, 1.0 / filter_cfg.num_particles)
    grid = np.zeros((map_cfg.sizex, map_cfg.sizey), dtype=np.float32)
    trajectory: list[np.ndarray] = []
    x_coordinates = np.arange(map_cfg.xmin, map_cfg.xmax + map_cfg.res, map_cfg.res)
    y_coordinates = np.arange(map_cfg.ymin, map_cfg.ymax + map_cfg.res, map_cfg.res)
    angles = lidar_cfg.angle_min + np.arange(data["lidar"].shape[0]) * lidar_cfg.angle_increment

    sync_times = data["sync_times"]
    if diagnostics is not None:
        diagnostics.update(
            neff=[], max_weight=[], valid_beams=[], resampled=[], position_spread=[]
        )
    for index in tqdm(range(1, sync_times.size), desc="GPU particle SLAM"):
        dt = float(sync_times[index] - sync_times[index - 1])
        if dt <= 0:
            continue
        motion_update(
            particles,
            data["encoder_counts"],
            data["imu_angular_velocity"],
            index,
            dt,
            robot_cfg,
            rng,
            filter_cfg,
        )
        weights = measurement_update_gpu(
            particles,
            data["lidar"][:, index],
            grid,
            x_coordinates,
            y_coordinates,
            lidar_cfg,
            filter_cfg,
            rng,
        )
        effective_particles = compute_neff(weights)
        max_particle_weight = float(weights.max().item())
        did_resample = effective_particles < (
            filter_cfg.num_particles * filter_cfg.resample_threshold
        )
        if did_resample:
            particles = resample_particles(particles, weights, rng)
            weights.fill(1.0 / filter_cfg.num_particles)

        if diagnostics is not None:
            scan = data["lidar"][:, index]
            valid_beams = np.count_nonzero(
                (scan > lidar_cfg.rmin) & (scan < lidar_cfg.rmax_used)
            )
            particles_xy = cp.asnumpy(particles[:, :2])
            diagnostics["neff"].append(effective_particles)
            diagnostics["max_weight"].append(max_particle_weight)
            diagnostics["valid_beams"].append(float(valid_beams))
            diagnostics["resampled"].append(float(did_resample))
            diagnostics["position_spread"].append(
                float(np.mean(np.std(particles_xy, axis=0)))
            )

        estimated_pose = cp.asnumpy(cp.average(particles, axis=0, weights=weights))
        trajectory.append(estimated_pose)
        update_occupancy_grid_vectorized(
            grid,
            estimated_pose,
            data["lidar"][:, index],
            angles,
            lidar_cfg,
            map_cfg,
        )

    trajectory_array = np.asarray(trajectory)
    particles_array = cp.asnumpy(particles)
    if output_file is not None:
        visualize_particles(grid, trajectory_array, map_cfg, particles_array, output_file)
    return trajectory_array, grid, particles_array
