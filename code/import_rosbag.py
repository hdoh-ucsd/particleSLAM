"""Convert supported ROS 1 datasets into particleSLAM's synchronized NPZ format."""

import argparse
import json
from pathlib import Path

import numpy as np

from config import RobotConfig


PROFILES = {
    "mit-stata": {
        "lidar": "/base_scan",
        "imu": "/torso_lift_imu/data",
        "odometry": "/base_odometry/odm",
        "odometry_is_ground_truth": False,
    },
    "uhumans2": {
        "lidar": "/tesse/front_lidar/scan",
        "imu": "/tesse/imu/noisy/imu",
        "odometry": "/tesse/odom",
        "odometry_is_ground_truth": True,
    },
}


def yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def poses_to_encoder_increments(poses: np.ndarray, robot: RobotConfig) -> np.ndarray:
    """Create equal wheel increments from planar odometry translation.

    Heading is supplied independently by the IMU in the current motion model.
    """
    deltas = np.diff(poses[:, :2], axis=0, prepend=poses[:1, :2])
    headings = poses[:, 2]
    signed_distance = deltas[:, 0] * np.cos(headings) + deltas[:, 1] * np.sin(headings)
    ticks = signed_distance / robot.tick_to_meter
    return np.tile(ticks, (4, 1))


def build_synced_dataset(
    lidar_times: np.ndarray,
    lidar_ranges: np.ndarray,
    lidar_metadata: dict[str, float],
    imu_times: np.ndarray,
    angular_velocity: np.ndarray,
    linear_acceleration: np.ndarray,
    odom_times: np.ndarray,
    odom_poses: np.ndarray,
    source: str,
    controls_are_ground_truth: bool,
) -> dict[str, np.ndarray]:
    """Interpolate ROS streams to LiDAR time and return canonical arrays."""
    if min(len(lidar_times), len(imu_times), len(odom_times)) < 2:
        raise ValueError("LiDAR, IMU, and odometry topics must each contain at least two messages")
    if lidar_ranges.ndim != 2 or lidar_ranges.shape[0] != len(lidar_times):
        raise ValueError("LiDAR scans must have a constant beam count")
    pose = np.column_stack([
        np.interp(lidar_times, odom_times, odom_poses[:, 0]),
        np.interp(lidar_times, odom_times, odom_poses[:, 1]),
        np.interp(lidar_times, odom_times, np.unwrap(odom_poses[:, 2])),
    ])
    angular = np.vstack([
        np.interp(lidar_times, imu_times, angular_velocity[:, axis]) for axis in range(3)
    ])
    acceleration = np.vstack([
        np.interp(lidar_times, imu_times, linear_acceleration[:, axis]) for axis in range(3)
    ])
    return {
        "sync_times": lidar_times,
        "pose": pose.T,
        "lidar": lidar_ranges.T,
        "encoder_counts": poses_to_encoder_increments(pose, RobotConfig()),
        "imu_angular_velocity": angular,
        "imu_linear_acceleration": acceleration,
        "lidar_angle_min": np.asarray(lidar_metadata["angle_min"]),
        "lidar_angle_increment": np.asarray(lidar_metadata["angle_increment"]),
        "lidar_range_min": np.asarray(lidar_metadata["range_min"]),
        "lidar_range_max": np.asarray(lidar_metadata["range_max"]),
        "ground_truth_pose": pose.T if controls_are_ground_truth else np.empty((3, 0)),
        "source": np.asarray(source),
        "controls_are_ground_truth": np.asarray(controls_are_ground_truth),
    }


def read_rosbag(path: Path, profile: dict[str, object]) -> dict[str, np.ndarray]:
    try:
        from rosbags.highlevel import AnyReader
    except ImportError as error:
        raise SystemExit("ROS bag import requires: python -m pip install rosbags") from error

    topics = {str(profile[name]) for name in ("lidar", "imu", "odometry")}
    lidar_times, scans = [], []
    imu_times, angular, acceleration = [], [], []
    odom_times, poses = [], []
    lidar_metadata = None
    with AnyReader([path]) as reader:
        connections = [connection for connection in reader.connections if connection.topic in topics]
        found = {connection.topic for connection in connections}
        missing = topics - found
        if missing:
            raise ValueError(f"bag is missing required topics: {', '.join(sorted(missing))}")
        for connection, timestamp, rawdata in reader.messages(connections=connections):
            message = reader.deserialize(rawdata, connection.msgtype)
            time_seconds = timestamp * 1e-9
            if connection.topic == profile["lidar"]:
                lidar_times.append(time_seconds)
                scans.append(np.asarray(message.ranges, dtype=float))
                lidar_metadata = {
                    "angle_min": float(message.angle_min),
                    "angle_increment": float(message.angle_increment),
                    "range_min": float(message.range_min),
                    "range_max": float(message.range_max),
                }
            elif connection.topic == profile["imu"]:
                imu_times.append(time_seconds)
                angular.append([message.angular_velocity.x, message.angular_velocity.y, message.angular_velocity.z])
                acceleration.append([message.linear_acceleration.x, message.linear_acceleration.y, message.linear_acceleration.z])
            else:
                odom_times.append(time_seconds)
                position = message.pose.pose.position
                orientation = message.pose.pose.orientation
                poses.append([position.x, position.y, yaw_from_quaternion(
                    orientation.x, orientation.y, orientation.z, orientation.w
                )])
    return build_synced_dataset(
        np.asarray(lidar_times), np.asarray(scans), lidar_metadata or {},
        np.asarray(imu_times), np.asarray(angular), np.asarray(acceleration),
        np.asarray(odom_times), np.asarray(poses), str(profile["name"]),
        bool(profile["odometry_is_ground_truth"]),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=tuple(PROFILES), required=True)
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-ground-truth-controls", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    profile = {**PROFILES[args.profile], "name": args.profile}
    if profile["odometry_is_ground_truth"] and not args.allow_ground_truth_controls:
        raise SystemExit(
            "uHumans2 provides ground-truth odometry, not wheel odometry. "
            "Pass --allow-ground-truth-controls for functional mapping only; "
            "do not report localization accuracy from that configuration."
        )
    data = read_rosbag(args.bag, profile)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **data)
    summary = {
        "source": args.profile,
        "scans": int(data["lidar"].shape[1]),
        "beams": int(data["lidar"].shape[0]),
        "duration_s": float(data["sync_times"][-1] - data["sync_times"][0]),
        "controls_are_ground_truth": bool(data["controls_are_ground_truth"]),
    }
    print(json.dumps(summary, indent=2))
    print(f"Saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
