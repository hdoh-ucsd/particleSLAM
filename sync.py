import numpy as np
from load_data import load_dataset

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

if __name__ == "__main__":
    # save_synced_dataset()\
    
    data = np.load('synced_data.npz')
    for name in data:
        arr = data[name]
        print(f"\n{name}:")
        print(f"  shape    = {arr.shape}")
        print(f"  dtype    = {arr.dtype}")
        print(f"  ndim     = {arr.ndim}")
        print(f"  size     = {arr.size}")
        # Optionally: print first few values
        print(f"  first 5 values: {arr.flat[:5] if arr.size >= 5 else arr.flat[:]}")
