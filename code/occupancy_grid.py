import numpy as np
from config import MapConfig, LidarConfig
from odom import DifferentialDrive, bresenham2D, bresenham2D_vec

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
    if not (0 <= x0 < grid.shape[0] and 0 <= y0 < grid.shape[1]):
        return
    for i in range(len(scan)):
        r = scan[i]
        if r < lidar_cfg.rmin or r > min(lidar_cfg.rmax, lidar_cfg.rmax_used):
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
        if 0 <= x1 < grid.shape[0] and 0 <= y1 < grid.shape[1]:
            grid[x1, y1] += occ_val

def update_occupancy_grid_vectorized(grid, pose, scan, angles, lidar_cfg, map_cfg, scale=1, bound=10):
    # Transform robot pose to sensor pose
    x_sensor, y_sensor, theta_sensor = lidar_cfg.x, lidar_cfg.y, lidar_cfg.yaw
    sx = pose[0] + x_sensor*np.cos(pose[2]) - y_sensor*np.sin(pose[2])
    sy = pose[1] + x_sensor*np.sin(pose[2]) + y_sensor*np.cos(pose[2])
    syaw = pose[2] + theta_sensor

    x0 = int(np.floor((sx - map_cfg.xmin) / map_cfg.res))
    y0 = int(np.floor((sy - map_cfg.ymin) / map_cfg.res))
    if not (0 <= x0 < grid.shape[0] and 0 <= y0 < grid.shape[1]):
        return

    # Precompute all endpoints in world and cell coordinates
    valid = (scan > lidar_cfg.rmin) & (
        scan < min(lidar_cfg.rmax, lidar_cfg.rmax_used)
    )
    scan_valid, angles_valid = scan[valid], angles[valid]
    if scan_valid.size == 0:
        return

    range_end_x = sx + scan_valid * np.cos(syaw + angles_valid)
    range_end_y = sy + scan_valid * np.sin(syaw + angles_valid)
    x1s = np.floor((range_end_x - map_cfg.xmin) / map_cfg.res).astype(int)
    y1s = np.floor((range_end_y - map_cfg.ymin) / map_cfg.res).astype(int)
    endpoints_in_bounds = (
        (x1s >= 0)
        & (x1s < grid.shape[0])
        & (y1s >= 0)
        & (y1s < grid.shape[1])
    )
    ray_x1s = np.clip(x1s, 0, grid.shape[0] - 1)
    ray_y1s = np.clip(y1s, 0, grid.shape[1] - 1)

    # Vectorized free cells: collect all cells along each ray except endpoint
    xs_free, ys_free = bresenham2D_vec(x0, y0, ray_x1s, ray_y1s)
    ogm_plot_vectorized(grid, xs_free, ys_free, occupied=False, scale=scale, bound=bound)
    ogm_plot_vectorized(
        grid,
        x1s[endpoints_in_bounds],
        y1s[endpoints_in_bounds],
        occupied=True,
        scale=scale,
        bound=bound,
    )

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
