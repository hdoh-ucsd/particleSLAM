import numpy as np
from config import MapConfig, LidarConfig, RobotConfig
from odom import DifferentialDrive, bresenham2D_vec
from occupancy_grid import update_occupancy_grid_vectorized
from visualize import visualize_cpu
from tqdm import tqdm

def compute_differential_drive_update(enc_counts, imu_wz, dt, robot_cfg):
    FR, FL, RR, RL = enc_counts
    right_dist = ((FR+RR)/2.0)*0.0022
    left_dist = ((FL+RL)/2.0)*0.0022
    v = (right_dist + left_dist) / (2.0*dt)
    omega = imu_wz
    return v, omega

def motion_update(particles, encoder_counts, imu_ang_vel, t, dt, robot_cfg):
    v, omega = compute_differential_drive_update(encoder_counts[:, t], imu_ang_vel[2, t], dt, robot_cfg)
    v_noise, omega_noise = np.random.normal(0, 0.02, particles.shape[0]), np.random.normal(0, 0.01, particles.shape[0])
    v_sample, omega_sample = v + v_noise, omega + omega_noise
    theta = particles[:, 2]
    particles[:, 0] += v_sample * dt * np.cos(theta)
    particles[:, 1] += v_sample * dt * np.sin(theta)
    particles[:, 2] += omega_sample * dt
    return particles

def transform_lidar_to_world_cpu(scan_np, particles, lidar_cfg):
    angle_min, angle_inc = -2.356194490192345, 0.00436332
    n_beams = scan_np.size
    angles = angle_min + np.arange(n_beams) * angle_inc

    valid = np.logical_and(scan_np > lidar_cfg.rmin, scan_np < lidar_cfg.rmax)
    scan_valid = scan_np[valid]
    angles_valid = angles[valid]

    xs = scan_valid * np.cos(angles_valid) + lidar_cfg.x
    ys = scan_valid * np.sin(angles_valid) + lidar_cfg.y

    part_theta = particles[:, 2][:, None]
    part_x = particles[:, 0][:, None]
    part_y = particles[:, 1][:, None]

    xw = part_x + xs * np.cos(part_theta) - ys * np.sin(part_theta)
    yw = part_y + xs * np.sin(part_theta) + ys * np.cos(part_theta)
    return xw, yw

def measurement_update_cpu(particles, scan, grid, x_im, y_im, lidar_cfg):
    scan_np = np.array(scan)
    xw, yw = transform_lidar_to_world_cpu(scan_np, particles, lidar_cfg)
    # Substitute with a real scoring function based on OGM if available
    scores = np.random.uniform(1.0, 2.0, particles.shape[0])
    scores = np.maximum(scores, 1e-9)
    weights = scores / np.sum(scores)
    return weights

def compute_neff(weights): return 1. / np.sum(weights ** 2)

def resample_particles(particles, weights):
    indices = np.random.choice(len(particles), size=len(particles), p=weights)
    return particles[indices]

# ==== PARTICLE FILTER CPU SLAM ====
def particle_filter_cpu(NUM_PARTICLES=1000):
    global x_im, y_im, grid, particles, weights, traj_estimates
    np.random.seed(42)
    particles = np.random.normal(0, 0.1, (NUM_PARTICLES, 3))
    weights = np.ones(NUM_PARTICLES) / NUM_PARTICLES

    map_cfg, lidar_cfg, robot_cfg = MapConfig(), LidarConfig(), RobotConfig()
    traj_estimates = []
    data = np.load('synced_data.npz')
    sync_times = data['sync_times']
    lidar_scans = data['lidar']
    encoder_counts = data['encoder_counts']
    imu_angular_velocity = data['imu_angular_velocity']
    grid = np.zeros((map_cfg.sizex, map_cfg.sizey), dtype=np.float32)
    x_im = np.arange(map_cfg.xmin, map_cfg.xmax + map_cfg.res, map_cfg.res)
    y_im = np.arange(map_cfg.ymin, map_cfg.ymax + map_cfg.res, map_cfg.res)
    rmin, rmax, angle_min, angle_inc = lidar_cfg.rmin, lidar_cfg.rmax, -2.356194490192345, 0.00436332

    for t in tqdm(range(1, sync_times.shape[0]), desc="PF SLAM Progress"):
        dt = sync_times[t] - sync_times[t-1]
        particles = motion_update(particles, encoder_counts, imu_angular_velocity, t, dt, robot_cfg)
        weights = measurement_update_cpu(particles, lidar_scans[:, t], grid, x_im, y_im, lidar_cfg)
        if compute_neff(weights) < NUM_PARTICLES/2:
            particles = resample_particles(particles, weights)
            weights[:] = 1.0 / NUM_PARTICLES
        est_pose = np.average(particles, axis=0, weights=weights)
        traj_estimates.append(est_pose)
        scan_t = lidar_scans[:, t]
        valid = np.logical_and(scan_t > rmin, scan_t < rmax)
        scan_t_valid = scan_t[valid]
        angles = angle_min + np.arange(len(scan_t)) * angle_inc
        angles = angles[valid]
        update_occupancy_grid_vectorized(grid, est_pose, scan_t_valid, angles, lidar_cfg, map_cfg)
        # if t % 100 == 0:
        #     visualize_cpu(grid, traj_estimates, map_cfg, particles, t)
    # --- Visualize using this config
    visualize_cpu(grid, traj_estimates, map_cfg, particles, t)
    print("PF SLAM complete.")
    print("Estimated trajectory shape:", np.array(traj_estimates).shape)
