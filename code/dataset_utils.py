import numpy as np
from config import MapConfig, LidarConfig, RobotConfig
from odom import DifferentialDrive

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

def save_synced_dataset(dataset=20, output_file="synced_data.npz"):
    # Load raw data
    data = load_dataset(dataset)

    # Use LiDAR scan times as common sync base
    sync_times = data["lidar_stamps"]
    # Pose_L is already interpolated to lidar_stamps
    synced_pose = data["pose_L"]
    synced_lidar = data["lidar_ranges"]
    
    # Interpolate encoder and IMU data to LiDAR times if needed
    encoder_counts_interp = np.array([
        np.interp(sync_times, data["encoder_stamps"], data["encoder_counts"][ch])
        for ch in range(data["encoder_counts"].shape[0])
    ])
    
    imu_av_interp = np.array([
        np.interp(sync_times, data["imu_stamps"], data["imu_angular_velocity"][i])
        for i in range(data["imu_angular_velocity"].shape[0])
    ])
    imu_la_interp = np.array([
        np.interp(sync_times, data["imu_stamps"], data["imu_linear_acceleration"][i])
        for i in range(data["imu_linear_acceleration"].shape[0])
    ])

    # Save all synced data as .npz file
    np.savez(
        output_file,
        sync_times=sync_times,
        pose=synced_pose,
        lidar=synced_lidar,
        encoder_counts=encoder_counts_interp,
        imu_angular_velocity=imu_av_interp,
        imu_linear_acceleration=imu_la_interp,
    )
    print(f"Synced data saved to {output_file}")

def load_synced_data(npz_file):
    data = np.load(npz_file)
    # Convert arrays from npz to a dict for legacy compatibility
    return {key: data[key] for key in data}