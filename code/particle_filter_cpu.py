"""NumPy particle-filter backend."""

from pathlib import Path
from typing import Mapping

import numpy as np
from tqdm import tqdm

from config import LidarConfig, MapConfig, ParticleFilterConfig, RobotConfig
from occupancy_grid import update_occupancy_grid_vectorized
from visualize import visualize_particles


def compute_differential_drive_update(
    encoder_counts: np.ndarray,
    imu_yaw_rate: float,
    dt: float,
    robot_cfg: RobotConfig,
) -> tuple[float, float]:
    """Convert four wheel increments and IMU yaw rate into planar velocity."""
    front_right, front_left, rear_right, rear_left = encoder_counts
    right_distance = (front_right + rear_right) * robot_cfg.tick_to_meter / 2.0
    left_distance = (front_left + rear_left) * robot_cfg.tick_to_meter / 2.0
    return (right_distance + left_distance) / (2.0 * dt), float(imu_yaw_rate)


def motion_update(
    particles: np.ndarray,
    encoder_counts: np.ndarray,
    imu_angular_velocity: np.ndarray,
    index: int,
    dt: float,
    robot_cfg: RobotConfig,
    rng: np.random.Generator | None = None,
    filter_cfg: ParticleFilterConfig | None = None,
) -> np.ndarray:
    rng = rng or np.random.default_rng()
    filter_cfg = filter_cfg or ParticleFilterConfig()
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
    particles[:, 0] += sampled_velocity * dt * np.cos(heading)
    particles[:, 1] += sampled_velocity * dt * np.sin(heading)
    particles[:, 2] += sampled_yaw_rate * dt
    return particles


def transform_lidar_to_world_cpu(
    scan: np.ndarray, particles: np.ndarray, lidar_cfg: LidarConfig
) -> tuple[np.ndarray, np.ndarray]:
    angles = lidar_cfg.angle_min + np.arange(scan.size) * lidar_cfg.angle_increment
    valid = (scan > lidar_cfg.rmin) & (scan < lidar_cfg.rmax)
    local_x = scan[valid] * np.cos(angles[valid]) + lidar_cfg.x
    local_y = scan[valid] * np.sin(angles[valid]) + lidar_cfg.y
    headings = particles[:, 2, None]
    world_x = particles[:, 0, None] + local_x * np.cos(headings) - local_y * np.sin(headings)
    world_y = particles[:, 1, None] + local_x * np.sin(headings) + local_y * np.cos(headings)
    return world_x, world_y


def measurement_update_cpu(
    particles: np.ndarray,
    scan: np.ndarray,
    grid: np.ndarray,
    _x_coordinates: np.ndarray,
    _y_coordinates: np.ndarray,
    lidar_cfg: LidarConfig,
    filter_cfg: ParticleFilterConfig | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Correlate each particle's scan endpoints with a local map neighborhood."""
    del rng  # Kept in the signature for compatibility with older callers.
    filter_cfg = filter_cfg or ParticleFilterConfig()
    scan = np.asarray(scan)[:: filter_cfg.correlation_beam_stride]
    angles = (
        lidar_cfg.angle_min
        + np.arange(0, np.asarray(scan).size * filter_cfg.correlation_beam_stride,
                    filter_cfg.correlation_beam_stride)
        * lidar_cfg.angle_increment
    )
    valid_ranges = (scan > lidar_cfg.rmin) & (scan < lidar_cfg.rmax_used)
    scan = scan[valid_ranges]
    angles = angles[valid_ranges]
    if scan.size < filter_cfg.min_valid_beams or not np.any(grid):
        return np.full(particles.shape[0], 1.0 / particles.shape[0])

    local_x = scan * np.cos(angles) + lidar_cfg.x
    local_y = scan * np.sin(angles) + lidar_cfg.y
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

    best_scores = np.full(particles.shape[0], -np.inf)
    best_offsets = np.zeros_like(particles)
    for yaw_offset in yaw_offsets:
        heading = particles[:, 2, None] + yaw_offset
        world_x = particles[:, 0, None] + local_x * np.cos(heading) - local_y * np.sin(heading)
        world_y = particles[:, 1, None] + local_x * np.sin(heading) + local_y * np.cos(heading)
        for x_offset in xy_offsets:
            grid_x = np.floor(
                (world_x + x_offset - _x_coordinates[0])
                / (_x_coordinates[1] - _x_coordinates[0])
            ).astype(int)
            for y_offset in xy_offsets:
                grid_y = np.floor(
                    (world_y + y_offset - _y_coordinates[0])
                    / (_y_coordinates[1] - _y_coordinates[0])
                ).astype(int)
                in_bounds = (
                    (grid_x >= 0)
                    & (grid_x < grid.shape[0])
                    & (grid_y >= 0)
                    & (grid_y < grid.shape[1])
                )
                safe_x = np.clip(grid_x, 0, grid.shape[0] - 1)
                safe_y = np.clip(grid_y, 0, grid.shape[1] - 1)
                endpoint_values = np.where(in_bounds, grid[safe_x, safe_y], -10.0)
                scores = endpoint_values.mean(axis=1)
                improved = scores > best_scores
                best_scores[improved] = scores[improved]
                best_offsets[improved] = (x_offset, y_offset, yaw_offset)

    particles += best_offsets
    logits = filter_cfg.likelihood_temperature * (best_scores - np.max(best_scores))
    weights = np.exp(np.clip(logits, -700.0, 0.0))
    weight_sum = weights.sum()
    if not np.isfinite(weight_sum) or weight_sum <= 0:
        return np.full(particles.shape[0], 1.0 / particles.shape[0])
    return weights / weight_sum


def compute_neff(weights: np.ndarray) -> float:
    return float(1.0 / np.sum(weights**2))


def resample_particles(
    particles: np.ndarray,
    weights: np.ndarray,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Systematic resampling with lower variance than independent draws."""
    rng = rng or np.random.default_rng()
    positions = (rng.random() + np.arange(len(particles))) / len(particles)
    indices = np.searchsorted(np.cumsum(weights), positions, side="right")
    return particles[indices]


def particle_filter_cpu(
    data: Mapping[str, np.ndarray] | None = None,
    filter_cfg: ParticleFilterConfig | None = None,
    map_cfg: MapConfig | None = None,
    lidar_cfg: LidarConfig | None = None,
    robot_cfg: RobotConfig | None = None,
    output_file: str | Path | None = None,
    num_particles: int | None = None,
    diagnostics: dict[str, list[float]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the CPU particle filter and return trajectory, grid, and particles."""
    if data is None:
        with np.load("synced_data.npz") as archive:
            data = {key: archive[key] for key in archive.files}

    filter_cfg = filter_cfg or ParticleFilterConfig(num_particles=num_particles or 1000)
    map_cfg = map_cfg or MapConfig()
    lidar_cfg = lidar_cfg or LidarConfig()
    robot_cfg = robot_cfg or RobotConfig()
    rng = np.random.default_rng(filter_cfg.seed)

    particles = rng.normal(0.0, 0.1, (filter_cfg.num_particles, 3))
    weights = np.full(filter_cfg.num_particles, 1.0 / filter_cfg.num_particles)
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
    for index in tqdm(range(1, sync_times.size), desc="CPU particle SLAM"):
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
        weights = measurement_update_cpu(
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
        max_particle_weight = float(weights.max())
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
            diagnostics["neff"].append(effective_particles)
            diagnostics["max_weight"].append(max_particle_weight)
            diagnostics["valid_beams"].append(float(valid_beams))
            diagnostics["resampled"].append(float(did_resample))
            diagnostics["position_spread"].append(
                float(np.mean(np.std(particles[:, :2], axis=0)))
            )

        estimated_pose = np.average(particles, axis=0, weights=weights)
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
    if output_file is not None:
        visualize_particles(grid, trajectory_array, map_cfg, particles, output_file)
    return trajectory_array, grid, particles
