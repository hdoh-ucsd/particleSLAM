import cupy as cp
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from dataclasses import dataclass
from tqdm import tqdm

# ==== CONFIGS ====
@dataclass(frozen=True)
class MapConfig:
    res: float = 0.05
    xmin: float = -10.0
    xmax: float = 30.0
    ymin: float = -10.0
    ymax: float = 30.0
    @property
    def sizex(self): return int(np.ceil((self.xmax - self.xmin) / self.res)) + 1
    @property
    def sizey(self): return int(np.ceil((self.ymax - self.ymin) / self.res)) + 1

@dataclass(frozen=True)
class RobotConfig:
    wheel_base: float = 0.5842
    wheel_radius: float = 0.127
    encoder_resolution: int = 360
    gear_ratio: float = 1.0
    baseline: float = 0.16
    ticks_per_rev = encoder_resolution * gear_ratio
    meters_per_tick = 2 * np.pi * wheel_radius / ticks_per_rev
    @property
    def tick_to_meter(self): return (2.0 * np.pi * self.wheel_radius) / float(self.ticks_per_rev)

@dataclass(frozen=True)
class LidarConfig:
    x: float = 0.30183
    y: float = 0.0
    yaw: float = 0.0
    rmin: float = 0.05
    rmax: float = 30.0
    rmax_used: float = 10.0

    def sensor_world_pose(self, base_xyz: np.ndarray) -> tuple[float, float, float]:
        xw, yw, th = float(base_xyz[0]), float(base_xyz[1]), float(base_xyz[2])
        sx = xw + self.x*np.cos(th) - self.y*np.sin(th)
        sy = yw + self.x*np.sin(th) + self.y*np.cos(th)
        syaw = th + self.yaw
        return sx, sy, syaw

class DifferentialDrive:
    def __init__(self, config):
        self.wheel_base = config.wheel_base      # [meters] distance between wheels
        self.wheel_radius = config.wheel_radius  # [meters]
        self.enc_res = config.encoder_resolution # [ticks per revolution]
        self.gear_ratio = config.gear_ratio      # if present

    def integrate_odometry(self, encoder_stamps, encoder_counts):
        # encoder_counts: shape (4, N), [rl, rr, fl, fr] or [lefts, rights...]
        # Use two wheels for computation (choose the rears, commonly)
        # Convert encoder counts to distance
        # Assume rear left=0, rear right=1 (adjust if your robot differs)
        left_counts = encoder_counts[3]  # shape (N,)
        right_counts = encoder_counts[2] # shape (N,)

        # Ticks to meters
        ticks_per_rev = self.enc_res * self.gear_ratio if hasattr(self, 'gear_ratio') else self.enc_res
        meters_per_tick = 2 * np.pi * self.wheel_radius / ticks_per_rev

        left_dist = left_counts * meters_per_tick
        right_dist = right_counts * meters_per_tick

        x, y, theta = [0.0], [0.0], [0.0]
        for i in range(1, len(encoder_stamps)):
            dl = left_dist[i] - left_dist[i-1]
            dr = right_dist[i] - right_dist[i-1]
            d_center = (dr + dl) / 2.0
            d_theta = (dr - dl) / self.wheel_base

            x_new = x[-1] + d_center * np.cos(theta[-1] + d_theta/2)
            y_new = y[-1] + d_center * np.sin(theta[-1] + d_theta/2)
            theta_new = theta[-1] + d_theta

            x.append(x_new)
            y.append(y_new)
            theta.append(theta_new)
        return np.vstack((x, y, theta))  # shape (3, N)

# ==== util functions ====
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

def logodds_to_prob(grid): 
    clipped = np.clip(grid, -10, 10)  # or another reasonable range
    return 1.0 - 1.0 / (1.0 + np.exp(clipped))

def bresenham2D(x0, y0, x1, y1):
    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
    dx, dy, sx, sy = abs(x1-x0), abs(y1-y0), (1 if x0<x1 else -1), (1 if y0<y1 else -1)
    err, xs, ys = dx-dy, [], []
    while True:
        xs.append(x0)
        ys.append(y0)
        if x0 == x1 and y0 == y1: break
        e2 = 2*err
        if e2 > -dy: err -= dy; x0 += sx
        if e2 < dx:  err += dx; y0 += sy
    return np.array(xs), np.array(ys)

# ==== OGM: Bresenham and Map Update ====
def load_synced_data(npz_file):
    data = np.load(npz_file)
    # Convert arrays from npz to a dict for legacy compatibility
    return {key: data[key] for key in data}

def build_occupancy_grid(data, map_cfg, lidar_cfg):
    grid = np.zeros((map_cfg.sizex, map_cfg.sizey), dtype=np.float32)
    x_im = np.arange(map_cfg.xmin, map_cfg.xmax + map_cfg.res, map_cfg.res)
    y_im = np.arange(map_cfg.ymin, map_cfg.ymax + map_cfg.res, map_cfg.res)
    lidar_angles = getattr(lidar_cfg, "angle_min", -2.356194490192345) + np.arange(data["lidar"].shape[0]) * getattr(lidar_cfg, "angle_increment", 0.00436332)
    lidar_angles = lidar_angles.reshape((-1,))

    for k in range(data["sync_times"].size):
        pose = data["pose"][:, k]
        scan = data["lidar"][:, k]
        update_occupancy_grid(grid, pose, scan, lidar_angles, lidar_cfg, map_cfg)
    return grid

def update_occupancy_grid(grid, robot_pose, scan, angles, lidar_cfg, map_cfg):
    free_val, occ_val = 1.0, 2.0
    # Compute the world pose of the LiDAR sensor
    sx, sy, syaw = lidar_cfg.sensor_world_pose(robot_pose)
    x0 = int((sx - map_cfg.xmin) / map_cfg.res)
    y0 = int((sy - map_cfg.ymin) / map_cfg.res)
    for i in range(len(scan)):
        r = scan[i]
        if r < lidar_cfg.rmin or r > lidar_cfg.rmax:
            continue
        angle = angles[i]
        end_x = sx + r * np.cos(syaw + angle)
        end_y = sy + r * np.sin(syaw + angle)
        x1 = int((end_x - map_cfg.xmin) / map_cfg.res)
        y1 = int((end_y - map_cfg.ymin) / map_cfg.res)
        free_x, free_y = bresenham2D(x0, y0, x1, y1)
        free_x = np.clip(free_x, 0, grid.shape[0] - 1)
        free_y = np.clip(free_y, 0, grid.shape[1] - 1)
        grid[free_x, free_y] -= free_val
        x1c = np.clip(x1, 0, grid.shape[0] - 1)
        y1c = np.clip(y1, 0, grid.shape[1] - 1)
        grid[x1c, y1c] += occ_val

def translate_grid(grid, old_cfg, new_cfg):
    new_grid = np.zeros((new_cfg.sizex, new_cfg.sizey), dtype=grid.dtype)
    # Iterate over ALL cell indices in the old grid
    for ix in range(old_cfg.sizex):
        for iy in range(old_cfg.sizey):
            value = grid[ix, iy]
            if value == 0:
                continue
            # Convert old grid index to world coordinates
            x_world = old_cfg.xmin + ix * old_cfg.res
            y_world = old_cfg.ymin + iy * old_cfg.res
            # Map world coordinates to new grid indices
            new_ix = int(round((x_world - new_cfg.xmin) / new_cfg.res))
            new_iy = int(round((y_world - new_cfg.ymin) / new_cfg.res))
            # Place the value if in bounds
            if 0 <= new_ix < new_cfg.sizex and 0 <= new_iy < new_cfg.sizey:
                new_grid[new_ix, new_iy] = value
    return new_grid

def visualize_ogm(grid, map_cfg):
    prob = logodds_to_prob(grid)
    extent = [map_cfg.ymin, map_cfg.ymax, map_cfg.xmin, map_cfg.xmax]
    plt.figure(figsize=(8,8))
    plt.imshow(prob.T, origin='lower', cmap='gray', extent=extent)
    plt.title("Occupancy Grid Map")
    plt.colorbar(label="Occupancy Probability")
    plt.xlabel("Y [m]")
    plt.ylabel("X [m]")
    plt.tight_layout()
    plt.show(block=True)

def ogm_plot_vectorized(grid, x, y, occupied=False, scale=1, bound=10):
    # Only update valid cells
    valid = (
        (x >= 0) & (x < grid.shape[0]) &
        (y >= 0) & (y < grid.shape[1])
    )
    x = x[valid]
    y = y[valid]
    if x.size == 0 or y.size == 0:
        return
    confidence = 0.9
    odds = confidence / (1 - confidence) if occupied else (1 - confidence) / confidence
    increment = np.log(odds) * scale
    grid[x, y] = np.clip(grid[x, y] + increment, -bound, bound)

def bresenham2D_vec(x0, y0, x1s, y1s, max_length=300):
    # Vectorized Bresenham for many rays using fixed max_length
    xs_all = []
    ys_all = []
    for x1, y1 in zip(x1s, y1s):
        xs, ys = bresenham2D(x0, y0, x1, y1)
        xs_all.append(xs[:-1])  # exclude hit cell from free cells
        ys_all.append(ys[:-1])
    xs_vec = np.concatenate(xs_all)
    ys_vec = np.concatenate(ys_all)
    return xs_vec, ys_vec

def update_occupancy_grid_vectorized(grid, pose, scan, angles, lidar_cfg, map_cfg, scale=1, bound=10):
    # Transform robot pose to sensor pose
    x_sensor, y_sensor, theta_sensor = lidar_cfg.x, lidar_cfg.y, lidar_cfg.yaw
    sx = pose[0] + x_sensor*np.cos(pose[2]) - y_sensor*np.sin(pose[2])
    sy = pose[1] + x_sensor*np.sin(pose[2]) + y_sensor*np.cos(pose[2])
    syaw = pose[2] + theta_sensor

    x0 = int(np.floor((sx - map_cfg.xmin) / map_cfg.res))
    y0 = int(np.floor((sy - map_cfg.ymin) / map_cfg.res))

    # Precompute all endpoints in world and cell coordinates
    valid = (scan > lidar_cfg.rmin) & (scan < lidar_cfg.rmax)
    scan_valid, angles_valid = scan[valid], angles[valid]
    if scan_valid.size == 0:
        return

    range_end_x = sx + scan_valid * np.cos(syaw + angles_valid)
    range_end_y = sy + scan_valid * np.sin(syaw + angles_valid)
    x1s = np.floor((range_end_x - map_cfg.xmin) / map_cfg.res).astype(int)
    y1s = np.floor((range_end_y - map_cfg.ymin) / map_cfg.res).astype(int)
    x1s = np.clip(x1s, 0, grid.shape[0]-1)
    y1s = np.clip(y1s, 0, grid.shape[1]-1)

    # Vectorized free cells: collect all cells along each ray except endpoint
    xs_free, ys_free = bresenham2D_vec(x0, y0, x1s, y1s)
    ogm_plot_vectorized(grid, xs_free, ys_free, occupied=False, scale=scale, bound=bound)
    ogm_plot_vectorized(grid, x1s, y1s, occupied=True, scale=scale, bound=bound)

def compute_differential_drive_update(enc_counts, imu_wz, dt, robot_cfg):
    FR, FL, RR, RL = enc_counts
    right_dist = ((FR+RR)/2.0)*0.0022
    left_dist = ((FL+RL)/2.0)*0.0022
    v = (right_dist + left_dist) / (2.0*dt)
    omega = imu_wz
    return v, omega

def motion_update(particles, encoder_counts, imu_ang_vel, t, dt, robot_cfg):
    v, omega = compute_differential_drive_update(encoder_counts[:, t], imu_ang_vel[2, t], dt, robot_cfg)
    v_noise, omega_noise = cp.random.normal(0, 0.02, particles.shape[0]), cp.random.normal(0, 0.01, particles.shape[0])
    v_sample, omega_sample = v + v_noise, omega + omega_noise
    theta = particles[:, 2]
    particles[:, 0] += v_sample * dt * cp.cos(theta)
    particles[:, 1] += v_sample * dt * cp.sin(theta)
    particles[:, 2] += omega_sample * dt
    return particles

def transform_lidar_to_world_gpu(scan_cp, particles, lidar_cfg):
    angle_min, angle_inc = -2.356194490192345, 0.00436332
    n_beams = scan_cp.size
    angles = angle_min + cp.arange(n_beams) * angle_inc

    # Mask both scan and angles using the same valid indices
    valid = cp.logical_and(scan_cp > lidar_cfg.rmin, scan_cp < lidar_cfg.rmax)
    scan_valid = scan_cp[valid]
    angles_valid = angles[valid]

    xs = scan_valid * cp.cos(angles_valid) + lidar_cfg.x
    ys = scan_valid * cp.sin(angles_valid) + lidar_cfg.y

    part_theta = particles[:, 2][:, None]
    part_x = particles[:, 0][:, None]
    part_y = particles[:, 1][:, None]

    xw = part_x + xs * cp.cos(part_theta) - ys * cp.sin(part_theta)
    yw = part_y + xs * cp.sin(part_theta) + ys * cp.cos(part_theta)
    return xw, yw


def measurement_update_gpu(particles, scan, grid, x_im, y_im, lidar_cfg):
    scan_cp = cp.array(scan)
    xw, yw = transform_lidar_to_world_gpu(scan_cp, particles, lidar_cfg)
    scores = cp.random.uniform(1.0, 2.0, particles.shape[0])
    scores = cp.maximum(scores, 1e-9)
    weights = scores / cp.sum(scores)
    return weights

def compute_neff(weights): return 1. / cp.sum(weights ** 2)

def resample_particles(particles, weights):
    indices = cp.random.choice(len(particles), size=len(particles), p=weights)
    return particles[indices]

def visualize(grid, trajectory, map_cfg, t=0):
    """
    plt.figure(figsize=(8,8)); ink_grid = (grid > 0).astype(np.uint8)
    plt.imshow(ink_grid.T, origin='lower', cmap='Greys', interpolation='none')
    res = map_cfg.res; x0, y0 = map_cfg.xmin, map_cfg.ymin
    traj_np = np.array(trajectory)
    traj_x = np.clip(((traj_np[:,0] - x0)/res).astype(int), 0, ink_grid.shape[1]-1)
    traj_y = np.clip(((traj_np[:,1] - y0)/res).astype(int), 0, ink_grid.shape[0]-1)
    plt.plot(traj_x, traj_y, color='deepskyblue', linewidth=2, label='Trajectory')
    plt.legend(); plt.title("Particle Filter")
    plt.xlabel("X (grid cells)"); plt.ylabel("Y (grid cells)")
    plt.xlim(0, ink_grid.shape[1]-1); plt.ylim(0, ink_grid.shape[0]-1)
    plt.grid(True, which='both', ls='--', alpha=0.3)
    plt.savefig(f'slam_iter_{t:04d}.png'); plt.close()
    """
    plt.figure(figsize=(8,8))

    # Define min/max bounds in meters (x: width, y: height)
    extent = [map_cfg.xmin, map_cfg.xmax, map_cfg.ymin, map_cfg.ymax]

    # Plot grid so axes are in meters
    plt.imshow(grid.T, origin='lower', cmap='gray', interpolation='none', extent=extent)

    # Prepare trajectory for overlay in meters
    traj_np = np.array(trajectory)
    plt.plot(traj_np[:,0], traj_np[:,1], color='red', linewidth=2, label='Trajectory')  # [x, y] in meters

    # Particles in meters
    particles_np = particles.get()
    plt.scatter(particles_np[:,0], particles_np[:,1], color='blue', s=3, alpha=0.5, label='Particles')

    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    plt.title('Occupancy Grid (Position in meters)')
    plt.xlim(map_cfg.xmin, map_cfg.xmax)
    plt.ylim(map_cfg.ymin, map_cfg.ymax)
    plt.legend()
    plt.grid(True, which='both', ls='--', alpha=0.3)
    plt.savefig(f'pf_grid_cells_{t:04d}.png')
    plt.close()

def particle_slam():
    global x_im, y_im, grid, particles, weights, traj_estimates, NUM_PARTICLES
    save_synced_dataset()
    data = load_synced_data("synced_data.npz")

    # Print shape/type information
    print("--- Synced data keys and shapes ---")
    for key, value in data.items():
        print(f"{key}: shape={value.shape}, dtype={value.dtype}")
    print("-----------------------------------")

    map_cfg = MapConfig()
    lidar_cfg = LidarConfig()
    grid = build_occupancy_grid(data, map_cfg, lidar_cfg)
    np.savez("ogm_grid.npz", grid=grid, map_cfg=vars(map_cfg))
    print("OGM grid saved to ogm_grid.npz")
    #visualize_ogm(grid, map_cfg)
    
    x_im = np.arange(map_cfg.xmin, map_cfg.xmax + map_cfg.res, map_cfg.res)
    y_im = np.arange(map_cfg.ymin, map_cfg.ymax + map_cfg.res, map_cfg.res)
    rmin, rmax, angle_min, angle_inc = lidar_cfg.rmin, lidar_cfg.rmax, -2.356194490192345, 0.00436332
    
    # ==== PARTICLE FILTER GPU SLAM ====
    NUM_PARTICLES = 1000
    rs = cp.random.RandomState(42)
    particles = rs.normal(0, 0.1, (NUM_PARTICLES, 3))
    weights = cp.ones(NUM_PARTICLES) / NUM_PARTICLES

    map_cfg, lidar_cfg, robot_cfg = MapConfig(), LidarConfig(), RobotConfig()
    traj_estimates = []
    data = np.load('synced_data.npz')
    sync_times = data['sync_times']
    lidar_scans = data['lidar']
    encoder_counts = data['encoder_counts']
    imu_angular_velocity = data['imu_angular_velocity']
    grid = np.zeros((map_cfg.sizex, map_cfg.sizey), dtype=np.float32)

    for t in tqdm(range(1, sync_times.shape[0]), desc="PF SLAM Progress"):
        dt = sync_times[t] - sync_times[t-1]
        particles = motion_update(particles, encoder_counts, imu_angular_velocity, t, dt, robot_cfg)
        weights = measurement_update_gpu(particles, lidar_scans[:, t], grid, x_im, y_im, lidar_cfg)
        if compute_neff(weights) < NUM_PARTICLES/2:
            particles = resample_particles(particles, weights)
            weights[:] = 1.0/NUM_PARTICLES
        est_pose = cp.asnumpy(cp.average(particles, axis=0, weights=cp.asnumpy(weights)))
        traj_estimates.append(est_pose)
        scan_t = lidar_scans[:, t]
        valid = np.logical_and(scan_t > rmin, scan_t < rmax)
        scan_t_valid = scan_t[valid]
        angles = angle_min + np.arange(len(scan_t)) * angle_inc
        angles = angles[valid]
        update_occupancy_grid_vectorized(grid, est_pose, scan_t_valid, angles, lidar_cfg, map_cfg)
        
        # if t % 100 == 0:
        #    visualize(grid, traj_estimates, map_cfg, t)

    # --- Visualize using this config
    visualize(grid, traj_estimates, map_cfg, t)
    print("PF SLAM complete.")
    print("Estimated trajectory shape:", np.array(traj_estimates).shape)
