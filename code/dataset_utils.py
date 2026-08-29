"""Dataset loading and sensor time synchronization."""

from pathlib import Path
from typing import Mapping

import numpy as np

from config import RobotConfig
from odom import DifferentialDrive


def _load_npz(path: Path, required_keys: tuple[str, ...]) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    with np.load(path) as archive:
        missing = [key for key in required_keys if key not in archive]
        if missing:
            raise KeyError(f"{path} is missing required arrays: {', '.join(missing)}")
        return {key: archive[key] for key in archive.files}


def load_dataset(
    dataset: int = 20,
    data_dir: str | Path = "../data",
    robot_config: RobotConfig | None = None,
) -> dict[str, np.ndarray]:
    """Load one numbered encoder/LiDAR/IMU dataset and integrate odometry."""
    data_dir = Path(data_dir)
    encoder = _load_npz(
        data_dir / f"Encoders{dataset}.npz",
        ("counts", "time_stamps"),
    )
    lidar = _load_npz(
        data_dir / f"Hokuyo{dataset}.npz",
        (
            "angle_min",
            "angle_max",
            "angle_increment",
            "range_min",
            "range_max",
            "ranges",
            "time_stamps",
        ),
    )
    imu = _load_npz(
        data_dir / f"Imu{dataset}.npz",
        ("angular_velocity", "linear_acceleration", "time_stamps"),
    )

    encoder_counts = encoder["counts"]
    encoder_stamps = encoder["time_stamps"]
    pose_enc = DifferentialDrive(robot_config or RobotConfig()).integrate_odometry(
        encoder_stamps,
        encoder_counts,
    )

    return {
        "encoder_counts": encoder_counts,
        "encoder_stamps": encoder_stamps,
        "lidar_angle_min": lidar["angle_min"],
        "lidar_angle_max": lidar["angle_max"],
        "lidar_angle_increment": lidar["angle_increment"],
        "lidar_range_min": lidar["range_min"],
        "lidar_range_max": lidar["range_max"],
        "lidar_ranges": lidar["ranges"],
        "lidar_stamps": lidar["time_stamps"],
        "imu_angular_velocity": imu["angular_velocity"],
        "imu_linear_acceleration": imu["linear_acceleration"],
        "imu_stamps": imu["time_stamps"],
        "pose_enc": pose_enc,
    }


def synchronize_dataset(data: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Interpolate encoder, odometry, and IMU streams onto LiDAR timestamps."""
    sync_times = data["lidar_stamps"]

    def interpolate_rows(values: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
        return np.vstack(
            [np.interp(sync_times, timestamps, row) for row in np.asarray(values)]
        )

    cumulative_counts = np.cumsum(data["encoder_counts"], axis=1)
    synchronized_cumulative_counts = interpolate_rows(
        cumulative_counts, data["encoder_stamps"]
    )
    synchronized_count_increments = np.diff(
        synchronized_cumulative_counts,
        axis=1,
        prepend=synchronized_cumulative_counts[:, :1],
    )

    synchronized_imu = interpolate_rows(
        data["imu_angular_velocity"], data["imu_stamps"]
    )
    synchronized_imu[2] = low_pass_filter(
        synchronized_imu[2], sync_times, cutoff_hz=10.0
    )

    return {
        "sync_times": sync_times,
        "pose": interpolate_rows(data["pose_enc"], data["encoder_stamps"]),
        "lidar": data["lidar_ranges"],
        "encoder_counts": synchronized_count_increments,
        "imu_angular_velocity": synchronized_imu,
        "imu_linear_acceleration": interpolate_rows(
            data["imu_linear_acceleration"], data["imu_stamps"]
        ),
        "lidar_angle_min": np.asarray(data["lidar_angle_min"]),
        "lidar_angle_increment": np.asarray(data["lidar_angle_increment"]),
        "lidar_range_min": np.asarray(data["lidar_range_min"]),
        "lidar_range_max": np.asarray(data["lidar_range_max"]),
    }


def low_pass_filter(
    values: np.ndarray, timestamps: np.ndarray, cutoff_hz: float = 10.0
) -> np.ndarray:
    """Apply a first-order low-pass filter to an irregularly sampled signal."""
    values = np.asarray(values, dtype=float)
    timestamps = np.asarray(timestamps, dtype=float)
    if values.shape != timestamps.shape:
        raise ValueError("values and timestamps must have matching shapes")
    if cutoff_hz <= 0:
        raise ValueError("cutoff_hz must be positive")
    if values.size < 2:
        return values.copy()

    filtered = np.empty_like(values)
    filtered[0] = values[0]
    time_constant = 1.0 / (2.0 * np.pi * cutoff_hz)
    for index in range(1, values.size):
        dt = timestamps[index] - timestamps[index - 1]
        if dt <= 0:
            raise ValueError("timestamps must be strictly increasing")
        alpha = dt / (time_constant + dt)
        filtered[index] = filtered[index - 1] + alpha * (
            values[index] - filtered[index - 1]
        )
    return filtered


def save_synced_dataset(
    data: Mapping[str, np.ndarray], output_file: str | Path = "synced_data.npz"
) -> dict[str, np.ndarray]:
    """Synchronize a loaded dataset, save it, and return the synchronized arrays."""
    synced = synchronize_dataset(data)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_file, **synced)
    return synced


def load_synced_data(npz_file: str | Path) -> dict[str, np.ndarray]:
    """Load a previously synchronized dataset into an ordinary dictionary."""
    with np.load(npz_file) as archive:
        return {key: archive[key] for key in archive.files}
