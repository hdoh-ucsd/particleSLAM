"""GTSAM pose-graph optimization with ICP-validated LiDAR loop closures."""

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm

from config import LidarConfig, MapConfig, PoseGraphConfig
from occupancy_grid import update_occupancy_grid_vectorized


def wrap_angle(angle: float | np.ndarray) -> float | np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def relative_pose(origin: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return the SE(2) pose of ``target`` expressed in ``origin``."""
    delta = target[:2] - origin[:2]
    cosine, sine = np.cos(origin[2]), np.sin(origin[2])
    return np.array(
        [
            cosine * delta[0] + sine * delta[1],
            -sine * delta[0] + cosine * delta[1],
            wrap_angle(target[2] - origin[2]),
        ]
    )


def compose_pose(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    cosine, sine = np.cos(first[2]), np.sin(first[2])
    return np.array(
        [
            first[0] + cosine * second[0] - sine * second[1],
            first[1] + sine * second[0] + cosine * second[1],
            wrap_angle(first[2] + second[2]),
        ]
    )


def transform_points(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    cosine, sine = np.cos(pose[2]), np.sin(pose[2])
    rotation = np.array([[cosine, -sine], [sine, cosine]])
    return points @ rotation.T + pose[:2]


def scan_to_body_points(scan: np.ndarray, lidar_cfg: LidarConfig, stride: int) -> np.ndarray:
    beam_indices = np.arange(0, scan.size, stride)
    ranges = scan[beam_indices]
    angles = lidar_cfg.angle_min + beam_indices * lidar_cfg.angle_increment
    valid = (ranges > lidar_cfg.rmin) & (ranges < lidar_cfg.rmax_used)
    ranges, angles = ranges[valid], angles[valid]
    sensor_points = np.column_stack((ranges * np.cos(angles), ranges * np.sin(angles)))
    return transform_points(sensor_points, np.array([lidar_cfg.x, lidar_cfg.y, lidar_cfg.yaw]))


def _rigid_alignment(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    covariance = (source - source_center).T @ (target - target_center)
    u_matrix, _singular_values, vt_matrix = np.linalg.svd(covariance)
    rotation = vt_matrix.T @ u_matrix.T
    if np.linalg.det(rotation) < 0:
        vt_matrix[-1] *= -1
        rotation = vt_matrix.T @ u_matrix.T
    translation = target_center - rotation @ source_center
    return np.array(
        [translation[0], translation[1], np.arctan2(rotation[1, 0], rotation[0, 0])]
    )


def run_icp(
    source: np.ndarray,
    target: np.ndarray,
    initial_pose: np.ndarray,
    config: PoseGraphConfig,
) -> tuple[np.ndarray, float, float]:
    """Align source into target coordinates and return pose, RMSE, and overlap."""
    if min(len(source), len(target)) < 10:
        return initial_pose.copy(), np.inf, 0.0
    target_tree = cKDTree(target)
    estimate = initial_pose.copy()
    previous_rmse = np.inf

    for _ in range(config.icp_max_iterations):
        transformed = transform_points(source, estimate)
        distances, indices = target_tree.query(transformed, k=1)
        accepted = distances < config.icp_max_correspondence_distance
        if np.count_nonzero(accepted) < 10:
            break
        correction = _rigid_alignment(transformed[accepted], target[indices[accepted]])
        estimate = compose_pose(correction, estimate)
        rmse = float(np.sqrt(np.mean(distances[accepted] ** 2)))
        if abs(previous_rmse - rmse) < 1e-4:
            break
        previous_rmse = rmse

    transformed = transform_points(source, estimate)
    distances, _indices = target_tree.query(transformed, k=1)
    accepted = distances < config.icp_max_correspondence_distance
    overlap = float(np.mean(accepted))
    rmse = (
        float(np.sqrt(np.mean(distances[accepted] ** 2)))
        if np.any(accepted)
        else np.inf
    )
    return estimate, rmse, overlap


@dataclass(frozen=True)
class LoopClosure:
    source: int
    target: int
    relative_pose: np.ndarray
    rmse: float
    overlap: float
    method: str


@dataclass(frozen=True)
class PoseGraphResult:
    optimized_trajectory: np.ndarray
    keyframe_indices: np.ndarray
    optimized_keyframes: np.ndarray
    loop_closures: tuple[LoopClosure, ...]
    candidate_count: int


def _candidate_pairs(
    keyframe_poses: np.ndarray, config: PoseGraphConfig
) -> list[tuple[int, int, str]]:
    candidates: dict[tuple[int, int], str] = {}
    for target in range(
        config.fixed_loop_interval, len(keyframe_poses), config.fixed_loop_interval
    ):
        source = target - config.fixed_loop_interval
        candidates[(source, target)] = "fixed"

    proximity_candidates: list[tuple[float, int, int]] = []
    for source in range(len(keyframe_poses)):
        for target in range(source + config.minimum_loop_separation, len(keyframe_poses)):
            distance = float(
                np.linalg.norm(keyframe_poses[source, :2] - keyframe_poses[target, :2])
            )
            if distance <= config.proximity_distance:
                proximity_candidates.append((distance, source, target))
    proximity_candidates.sort()
    for _distance, source, target in proximity_candidates[: config.max_proximity_candidates]:
        candidates.setdefault((source, target), "proximity")
    return [(source, target, method) for (source, target), method in candidates.items()]


def optimize_pose_graph(
    trajectory: np.ndarray,
    lidar_scans: np.ndarray,
    lidar_cfg: LidarConfig,
    config: PoseGraphConfig | None = None,
) -> PoseGraphResult:
    """Optimize PF poses using sequential factors and accepted ICP loop closures."""
    import gtsam

    config = config or PoseGraphConfig()
    if len(trajectory) < 2:
        raise ValueError("pose-graph optimization requires at least two poses")
    keyframe_indices = np.arange(0, len(trajectory), config.keyframe_interval, dtype=int)
    if keyframe_indices[-1] != len(trajectory) - 1:
        keyframe_indices = np.append(keyframe_indices, len(trajectory) - 1)
    keyframe_poses = trajectory[keyframe_indices]
    keyframe_scans = [lidar_scans[:, index + 1] for index in keyframe_indices]
    point_clouds = [
        scan_to_body_points(scan, lidar_cfg, config.icp_beam_stride)
        for scan in keyframe_scans
    ]

    graph = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()
    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.asarray(config.prior_sigmas))
    odometry_noise = gtsam.noiseModel.Diagonal.Sigmas(np.asarray(config.odometry_sigmas))
    loop_base_noise = gtsam.noiseModel.Diagonal.Sigmas(
        np.asarray(config.loop_sigmas)
    )
    loop_noise = gtsam.noiseModel.Robust.Create(
        gtsam.noiseModel.mEstimator.Huber.Create(config.robust_huber_k),
        loop_base_noise,
    )

    for index, pose in enumerate(keyframe_poses):
        initial.insert(index, gtsam.Pose2(*pose))
    graph.add(gtsam.PriorFactorPose2(0, gtsam.Pose2(*keyframe_poses[0]), prior_noise))
    for index in range(1, len(keyframe_poses)):
        measurement = relative_pose(keyframe_poses[index - 1], keyframe_poses[index])
        graph.add(
            gtsam.BetweenFactorPose2(
                index - 1, index, gtsam.Pose2(*measurement), odometry_noise
            )
        )

    accepted_closures: list[LoopClosure] = []
    candidates = _candidate_pairs(keyframe_poses, config)
    for source, target, method in tqdm(candidates, desc="ICP loop closures"):
        initial_relative = relative_pose(keyframe_poses[source], keyframe_poses[target])
        measurement, rmse, overlap = run_icp(
            point_clouds[target], point_clouds[source], initial_relative, config
        )
        if rmse > config.icp_max_rmse or overlap < config.icp_min_overlap:
            continue
        correction = relative_pose(initial_relative, measurement)
        if (
            np.linalg.norm(correction[:2]) > config.icp_max_translation_correction
            or abs(correction[2]) > config.icp_max_yaw_correction
        ):
            continue
        closure = LoopClosure(source, target, measurement, rmse, overlap, method)
        accepted_closures.append(closure)
        graph.add(
            gtsam.BetweenFactorPose2(
                source, target, gtsam.Pose2(*measurement), loop_noise
            )
        )

    result = gtsam.LevenbergMarquardtOptimizer(graph, initial).optimize()
    optimized_keyframes = np.array(
        [
            [result.atPose2(i).x(), result.atPose2(i).y(), result.atPose2(i).theta()]
            for i in range(len(keyframe_poses))
        ]
    )
    all_indices = np.arange(len(trajectory))
    optimized_trajectory = np.column_stack(
        (
            np.interp(all_indices, keyframe_indices, optimized_keyframes[:, 0]),
            np.interp(all_indices, keyframe_indices, optimized_keyframes[:, 1]),
            wrap_angle(
                np.interp(
                    all_indices,
                    keyframe_indices,
                    np.unwrap(optimized_keyframes[:, 2]),
                )
            ),
        )
    )
    return PoseGraphResult(
        optimized_trajectory,
        keyframe_indices,
        optimized_keyframes,
        tuple(accepted_closures),
        len(candidates),
    )


def rebuild_occupancy_grid(
    data: Mapping[str, np.ndarray],
    trajectory: np.ndarray,
    map_cfg: MapConfig,
    lidar_cfg: LidarConfig,
) -> np.ndarray:
    """Reconstruct an occupancy grid using an externally estimated trajectory."""
    grid = np.zeros((map_cfg.sizex, map_cfg.sizey), dtype=np.float32)
    angles = lidar_cfg.angle_min + np.arange(data["lidar"].shape[0]) * lidar_cfg.angle_increment
    for index, pose in enumerate(tqdm(trajectory, desc="Rebuilding optimized map"), start=1):
        update_occupancy_grid_vectorized(
            grid, pose, data["lidar"][:, index], angles, lidar_cfg, map_cfg
        )
    return grid
