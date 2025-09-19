import numpy as np
from robot import RobotConfig, DifferentialDrive

def load_dataset(dataset=20):
    # Load encoder data
    with np.load(f"../data/Encoders{dataset}.npz") as data:
        encoder_counts = data["counts"]  # 4 x n
        encoder_stamps = data["time_stamps"]

    # Load lidar data
    with np.load(f"../data/Hokuyo{dataset}.npz") as data:
        lidar_angle_min = data["angle_min"]
        lidar_angle_max = data["angle_max"]
        lidar_angle_increment = data["angle_increment"]
        lidar_range_min = data["range_min"]
        lidar_range_max = data["range_max"]
        lidar_ranges = data["ranges"]
        lidar_stamps = data["time_stamps"]

    # Load IMU data
    with np.load(f"../data/Imu{dataset}.npz") as data:
        imu_angular_velocity = data["angular_velocity"]
        imu_linear_acceleration = data["linear_acceleration"]
        imu_stamps = data["time_stamps"]

    # Integrate odometry
    drv = DifferentialDrive(RobotConfig())
    pose_enc = drv.integrate_odometry(encoder_stamps, encoder_counts)

    # Interpolate pose to lidar scan times
    lidar_stamps = np.asarray(lidar_stamps, float)
    pose_L = np.vstack([
        np.interp(lidar_stamps, encoder_stamps, pose_enc[0]),
        np.interp(lidar_stamps, encoder_stamps, pose_enc[1]),
        np.interp(lidar_stamps, encoder_stamps, pose_enc[2])
    ])

    return {
        "encoder_counts": encoder_counts,
        "encoder_stamps": encoder_stamps,
        "lidar_angle_min": lidar_angle_min,
        "lidar_angle_max": lidar_angle_max,
        "lidar_angle_increment": lidar_angle_increment,
        "lidar_range_min": lidar_range_min,
        "lidar_range_max": lidar_range_max,
        "lidar_ranges": lidar_ranges,
        "lidar_stamps": lidar_stamps,
        "imu_angular_velocity": imu_angular_velocity,
        "imu_linear_acceleration": imu_linear_acceleration,
        "imu_stamps": imu_stamps,
        "pose_enc": pose_enc,
        "pose_L": pose_L,
    }