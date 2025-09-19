import cupy as cp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from robot import MapConfig, LidarConfig, RobotConfig
from ogm import update_occupancy_grid, mapCorrelation  # mapCorrelation must support CuPy inputs!
from tqdm import tqdm

NUM_PARTICLES = 500
print(cp.__file__)
rs = cp.random.RandomState(42)

particles = rs.normal(0, 0.1, (NUM_PARTICLES, 3))           # CuPy
weights = cp.ones(NUM_PARTICLES) / NUM_PARTICLES             # CuPy
N = 5  # Snapshot interval

# Load Data (keep as NumPy; transfer to GPU on use if needed)
data = np.load('synced_data.npz')
sync_times = data['sync_times']
lidar_scans = data['lidar']
poses = data['pose']
encoder_counts = data['encoder_counts']
imu_angular_velocity = data['imu_angular_velocity']

map_cfg = MapConfig()
lidar_cfg = LidarConfig()
robot_cfg = RobotConfig()
grid = cp.array(np.load('ogm_grid.npz')['grid'])             # CuPy
#grid[:] = 0
print("Loaded precomputed occupancy grid from 'ogm_grid.npz'")

traj_estimates = []

def compute_differential_drive_update(enc_counts, imu_wz, dt, robot_cfg):
    FR, FL, RR, RL = enc_counts
    right_dist = ((FR + RR)/2.0) * 0.0022
    left_dist = ((FL + RL)/2.0) * 0.0022
    v = (right_dist + left_dist) / (2.0 * dt)
    omega = imu_wz
    return v, omega

def motion_update(particles, encoder_counts, imu_ang_vel, t, dt, robot_cfg):
    v, omega = compute_differential_drive_update(encoder_counts[:, t], imu_ang_vel[2, t], dt, robot_cfg)
    v_noise = cp.random.normal(0, 0.02, NUM_PARTICLES)
    omega_noise = cp.random.normal(0, 0.01, NUM_PARTICLES)
    v_sample = v + v_noise
    omega_sample = omega + omega_noise
    theta = particles[:, 2]
    particles[:, 0] += v_sample * dt * cp.cos(theta)
    particles[:, 1] += v_sample * dt * cp.sin(theta)
    particles[:, 2] += omega_sample * dt
    return particles

def transform_lidar_to_world_gpu(scan_cp, particles, lidar_cfg):
    # Batched transformation for all particles with CuPy
    # scan_cp: [n_beams] (CuPy), particles: [NUM_PARTICLES, 3] (CuPy)
    rmin, rmax = lidar_cfg.rmin, lidar_cfg.rmax
    angle_min = getattr(lidar_cfg, 'angle_min', -2.356194490192345)
    angle_inc = getattr(lidar_cfg, 'angle_increment', 0.00436332)
    valid = cp.logical_and(scan_cp > rmin, scan_cp < rmax)
    scan_cp = scan_cp[valid]
    angles = angle_min + cp.arange(scan_cp.size) * angle_inc
    xs = scan_cp * cp.cos(angles)
    ys = scan_cp * cp.sin(angles)
    xs_body = xs + lidar_cfg.x
    ys_body = ys + lidar_cfg.y

    # Broadcasting to all particles
    part_theta = particles[:, 2][:, None]
    part_x = particles[:, 0][:, None]
    part_y = particles[:, 1][:, None]
    xw = part_x + xs_body * cp.cos(part_theta) - ys_body * cp.sin(part_theta)
    yw = part_y + xs_body * cp.sin(part_theta) + ys_body * cp.cos(part_theta)
    return xw, yw    # [NUM_PARTICLES, n_beams] (CuPy)

def measurement_update_gpu(particles, scan, grid, x_im, y_im, lidar_cfg):
    # All arrays are CuPy except scan, which is converted as needed
    scan_cp = cp.array(scan)
    xw, yw = transform_lidar_to_world_gpu(scan_cp, particles, lidar_cfg)
    
    # Vectorized scoring: assumes mapCorrelation supports CuPy and batching!
    # scores = mapCorrelation(grid, x_im, y_im, xw, yw, particles)        # [NUM_PARTICLES]
    # For demonstration, fallback to a dummy score (replace with your batch logic):
    scores = cp.random.uniform(1.0, 2.0, NUM_PARTICLES)                 # Dummy (replace)
    scores = cp.maximum(scores, 1e-9)
    weights = scores / cp.sum(scores)
    return weights

def compute_neff(weights):
    return 1.0 / cp.sum(weights ** 2)

def resample_particles(particles, weights):
    indices = cp.random.choice(len(particles), size=len(particles), p=weights)
    return particles[indices]

def visualize(grid, trajectory, map_cfg, t=0):
    plt.figure(figsize=(8,8))
    grid_np = cp.asnumpy(grid) if hasattr(grid, 'shape') and 'cupy' in str(type(grid)).lower() else np.array(grid)
    # Black for ANY hit/occupancy, white for all else
    ink_grid = (grid_np > 0).astype(np.uint8)
    plt.imshow(ink_grid.T, origin='lower', cmap='Greys', interpolation='none')  # Greys: 1=black, 0=white

    res = map_cfg.res
    x0, y0 = map_cfg.xmin, map_cfg.ymin
    traj_np = np.array(trajectory)
    traj_x = np.clip(((traj_np[:,0] - x0) / res).astype(int), 0, ink_grid.shape[1]-1)
    traj_y = np.clip(((traj_np[:,1] - y0) / res).astype(int), 0, ink_grid.shape[0]-1)
    plt.plot(traj_x, traj_y, color='deepskyblue', linewidth=2, label='Trajectory')
    plt.legend()
    plt.title("Particle Filter")
    plt.xlabel("X (grid cells)")
    plt.ylabel("Y (grid cells)")
    plt.xlim(0, ink_grid.shape[1]-1)
    plt.ylim(0, ink_grid.shape[0]-1)
    plt.grid(True, which='both', ls='--', alpha=0.3)
    plt.savefig(f'slam_iter_{t:04d}.png')
    plt.close()





x_im = cp.array(np.arange(map_cfg.xmin, map_cfg.xmax + map_cfg.res, map_cfg.res))
y_im = cp.array(np.arange(map_cfg.ymin, map_cfg.ymax + map_cfg.res, map_cfg.res))
rmin, rmax = lidar_cfg.rmin, lidar_cfg.rmax
angle_min = getattr(lidar_cfg, 'angle_min', -2.356194490192345)
angle_inc = getattr(lidar_cfg, 'angle_increment', 0.00436332)

print("before running tqdm for loop")
for t in tqdm(range(1, sync_times.shape[0]), desc="PF SLAM Progress"):
    dt = sync_times[t] - sync_times[t - 1]
    particles = motion_update(particles, encoder_counts, imu_angular_velocity, t, dt, robot_cfg)
    weights = measurement_update_gpu(particles, lidar_scans[:, t], grid, x_im, y_im, lidar_cfg)
    neff = compute_neff(weights)
    if neff < NUM_PARTICLES / 2:
        particles = resample_particles(particles, weights)
        weights[:] = 1.0 / NUM_PARTICLES
    # CuPy average; est_pose is CPU for IO only
    est_pose = cp.asnumpy(cp.average(particles, axis=0, weights=cp.asnumpy(weights))) 
    traj_estimates.append(est_pose)
    scan_t = lidar_scans[:, t]
    valid = np.logical_and(scan_t > rmin, scan_t < rmax)
    scan_t_valid = scan_t[valid]
    angles = angle_min + np.arange(len(scan_t)) * angle_inc
    angles = angles[valid]
    update_occupancy_grid(grid, est_pose, scan_t_valid, angles, lidar_cfg, map_cfg)
    if t % 100 == 0:
        visualize(grid, traj_estimates, map_cfg, t)


visualize(grid, traj_estimates, map_cfg, t)
traj_estimates = np.array(traj_estimates)
print("PF SLAM complete.")
print("Estimated trajectory shape:", traj_estimates.shape)
