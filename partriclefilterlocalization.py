import numpy as np
import matplotlib
matplotlib.use('Agg')  # disables GUI, enables PNG/SVG plot saving
import matplotlib.pyplot as plt
from robot import MapConfig, LidarConfig, RobotConfig
from ogm import update_occupancy_grid, mapCorrelation
from tqdm import tqdm

# ---- Parameters ----
NUM_PARTICLES = 500
np.random.seed(42)

# ---- Load Data ----
data = np.load('synced_data.npz')
sync_times = data['sync_times']
lidar_scans = data['lidar']
poses = data['pose']
encoder_counts = data['encoder_counts']
imu_angular_velocity = data['imu_angular_velocity']

map_cfg = MapConfig()
lidar_cfg = LidarConfig()
robot_cfg = RobotConfig()
grid = np.load('ogm_grid.npz')['grid']
print("Loaded precomputed occupancy grid from 'ogm_grid.npz'")

# ---- Particle Initialization ----
particles = np.zeros((NUM_PARTICLES, 3))
particles += np.random.normal(0, 0.1, particles.shape)
weights = np.ones(NUM_PARTICLES) / NUM_PARTICLES

traj_estimates = []

def compute_differential_drive_update(enc_counts, imu_wz, dt, robot_cfg):
    FR, FL, RR, RL = enc_counts
    right_dist = ((FR + RR)/2.0) * 0.0022  # meters
    left_dist  = ((FL + RL)/2.0) * 0.0022
    v = (right_dist + left_dist) / (2.0 * dt)
    omega = imu_wz
    return v, omega

def motion_update(particles, encoder_counts, imu_ang_vel, t, dt, robot_cfg):
    v, omega = compute_differential_drive_update(encoder_counts[:,t], imu_ang_vel[2,t], dt, robot_cfg)
    v_noise = np.random.normal(0, 0.02, NUM_PARTICLES)
    omega_noise = np.random.normal(0, 0.01, NUM_PARTICLES)
    v_sample = v + v_noise
    omega_sample = omega + omega_noise
    theta = particles[:, 2]
    particles[:, 0] += v_sample * dt * np.cos(theta)
    particles[:, 1] += v_sample * dt * np.sin(theta)
    particles[:, 2] += omega_sample * dt
    return particles

def transform_lidar_to_world(scan, pose, lidar_cfg):
    rmin, rmax = lidar_cfg.rmin, lidar_cfg.rmax
    angle_min = getattr(lidar_cfg, 'angle_min', -2.356194490192345)
    angle_inc = getattr(lidar_cfg, 'angle_increment', 0.00436332)
    valid = np.logical_and(scan > rmin, scan < rmax)
    scan = scan[valid]
    angles = angle_min + np.arange(len(scan)) * angle_inc
    xs = scan * np.cos(angles)
    ys = scan * np.sin(angles)
    xs_body = xs + lidar_cfg.x
    ys_body = ys + lidar_cfg.y
    xw = pose[0] + xs_body * np.cos(pose[2]) - ys_body * np.sin(pose[2])
    yw = pose[1] + xs_body * np.sin(pose[2]) + ys_body * np.cos(pose[2])
    return xw, yw

def measurement_update(particles, scan, grid, x_im, y_im, lidar_cfg):
    scores = np.zeros(len(particles))
    for i, pose in enumerate(particles):
        xw, yw = transform_lidar_to_world(scan, pose, lidar_cfg)
        corr_surface = mapCorrelation(grid, x_im, y_im, xw, yw, pose)
        scores[i] = np.max(corr_surface)
    scores = np.maximum(scores, 1e-9)
    weights = scores / np.sum(scores)
    return weights

def compute_neff(weights):
    return 1.0 / np.sum(weights ** 2)

def resample_particles(particles, weights):
    indices = np.random.choice(len(particles), size=len(particles), p=weights)
    return particles[indices]

def visualize(grid, particles, trajectory, map_cfg):
    plt.figure(figsize=(10,8))
    # Mask cells that were never updated (e.g., still at initial value, usually 0)
    # Change the threshold as appropriate for your log-odds initialization
    visited_mask = np.abs(grid) > 1e-6
    # Optionally, set unexplored cells to transparent
    plt.imshow(grid_plot, origin='lower', cmap='gray')
    res = map_cfg.res
    x0, y0 = map_cfg.xmin, map_cfg.ymin
    traj_x = ((trajectory[:,0] - x0) / res).astype(int)
    traj_y = ((trajectory[:,1] - y0) / res).astype(int)
    part_x = ((particles[:,0] - x0) / res).astype(int)
    part_y = ((particles[:,1] - y0) / res).astype(int)
    plt.plot(traj_x, traj_y, color='deepskyblue', linewidth=2, label='Trajectory')
    plt.scatter(part_x, part_y, color='b', s=1, alpha=0.3, label='Particles')
    plt.legend()
    plt.title("Particle Filter")
    plt.xlabel("X (grid cells)")
    plt.ylabel("Y (grid cells)")
    plt.grid(True, which='both', ls='--', alpha=0.3)
    plt.savefig(f'slam_iter_{t:04d}.png')
    plt.close()



x_im = np.arange(map_cfg.xmin, map_cfg.xmax + map_cfg.res, map_cfg.res)
y_im = np.arange(map_cfg.ymin, map_cfg.ymax + map_cfg.res, map_cfg.res)
rmin, rmax = lidar_cfg.rmin, lidar_cfg.rmax
angle_min = getattr(lidar_cfg, 'angle_min', -2.356194490192345)
angle_inc = getattr(lidar_cfg, 'angle_increment', 0.00436332)
print("before running tqdm for loop")
for t in tqdm(range(1, sync_times.shape[0]), desc="PF SLAM Progress"):
    dt = sync_times[t] - sync_times[t-1]
    particles = motion_update(particles, encoder_counts, imu_angular_velocity, t, dt, robot_cfg)
    weights = measurement_update(particles, lidar_scans[:, t], grid, x_im, y_im, lidar_cfg)
    neff = compute_neff(weights)
    if neff < NUM_PARTICLES / 2:
        particles = resample_particles(particles, weights)
        weights[:] = 1.0 / NUM_PARTICLES
    est_pose = np.average(particles, axis=0, weights=weights)
    traj_estimates.append(est_pose)

    # Compute valid angles and scan values for grid update    
    scan_t = lidar_scans[:, t]
    valid = np.logical_and(scan_t > rmin, scan_t < rmax)
    scan_t_valid = scan_t[valid]
    angles = angle_min + np.arange(len(scan_t)) * angle_inc
    angles = angles[valid]

    update_occupancy_grid(grid, est_pose, scan_t_valid, angles, lidar_cfg, map_cfg)
    
    if t % 50 == 0:
        visualize(grid, particles, np.array(traj_estimates), map_cfg)
    
visualize(grid, particles, np.array(traj_estimates), map_cfg)       

traj_estimates = np.array(traj_estimates)
print("PF SLAM complete.")
print("Estimated trajectory shape:", traj_estimates.shape)
