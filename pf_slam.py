#!/usr/bin/env python3
"""
PF-SLAM (Particle Filter SLAM) — configuration-driven via robot.py

- Config: MapConfig, RobotConfig, LidarConfig, DifferentialDrive (robot.py)
- Prediction: differential drive (v from encoders, ω from IMU)
- Update: scan–grid correlation against log-odds OGM
- Resampling: systematic, when Neff < neff_frac * N
- Mapping: update OGM with the best particle each frame (LiDAR extrinsics applied)
- Progress: tqdm bar (auto ETA)
- Optional: CuPy for beam trig math (map stays on CPU)

Run:
  python pf_slam.py --step 10                 # CPU
  python pf_slam.py --step 10 --gpu-beam      # use CuPy trig if available
"""

import os
import math
import time
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# progress bar
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

# ---- local configuration (robot, map, lidar) ----
from robot import MapConfig, RobotConfig, LidarConfig, DifferentialDrive

# ---------------- Backend selection (CuPy optional) ----------------
def init_backend(use_gpu_beam: bool = False):
    """
    If use_gpu_beam=True, try CuPy for beam trig; otherwise use NumPy.
    Returns (xp_module, using_gpu_bool).
    """
    if use_gpu_beam:
        try:
            import cupy as cp_module
            try:
                ndev = cp_module.cuda.runtime.getDeviceCount()
                print(f"CuPy enabled for beam trig, CUDA devices: {ndev}")
            except Exception:
                print("CuPy imported, but CUDA device probe failed.")
            return cp_module, True
        except ImportError:
            print("CuPy not available; falling back to NumPy for beams.")
    import numpy as np_module
    return np_module, False

# ---------------- CLI ----------------
def parse_args():
    p = argparse.ArgumentParser(description="Particle-Filter SLAM (config-driven)")
    p.add_argument("--dataset", type=int, default=20, help="dataset index (e.g., 20 or 21)")
    p.add_argument("--step", type=int, default=10, help="subsample LiDAR frames (>=1)")
    p.add_argument("--gpu-beam", action="store_true", help="use CuPy for beam trig (map remains on CPU)")
    # PF knobs
    p.add_argument("--N", type=int, default=50, help="number of particles")
    p.add_argument("--yaw-deg", type=float, default=2.0, help="± yaw degrees to test in correlation")
    p.add_argument("--grid-range", type=float, default=0.20, help="XY corr half-range (meters)")
    p.add_argument("--grid-step", type=float, default=0.05, help="XY corr step (meters)")
    p.add_argument("--alpha", type=float, default=0.02, help="weight sharpness for correlation")
    p.add_argument("--neff-frac", type=float, default=0.5, help="resample when Neff < frac*N")
    # output
    p.add_argument("--out", default="pf_output", help="output folder for map images/arrays")
    return p.parse_args()

# ---------------- Utilities bound to MapConfig ----------------
def build_world_to_map(map_cfg: MapConfig):
    def world_to_map(x, y):
        ix = int(np.floor((x - map_cfg.xmin) / map_cfg.res))
        iy = int(np.floor((y - map_cfg.ymin) / map_cfg.res))
        return ix, iy
    return world_to_map

def bresenham(start, end):
    x0, y0 = int(start[0]), int(start[1])
    x1, y1 = int(end[0]), int(end[1])
    points = []
    swapped = False
    steep = abs(y1 - y0) > abs(x1 - x0)
    if steep:
        x0, y0, x1, y1 = y0, x0, y1, x1
    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0
        swapped = True
    dx = x1 - x0
    dy = abs(y1 - y0)
    err = dx / 2.0
    ystep = 1 if y0 < y1 else -1
    y = y0
    for x in range(x0, x1 + 1):
        pt = (y, x) if steep else (x, y)
        points.append(pt)
        err -= dy
        if err < 0:
            y += ystep
            err += dx
    if swapped:
        points.reverse()
    return points

# ---------------- OGM update (CPU) ----------------
def update_map_with_scan(log_odds: np.ndarray,
                         map_cfg: MapConfig,
                         lidar_cfg: LidarConfig,
                         best_base_pose: np.ndarray,
                         scan_ranges: np.ndarray,
                         angles: np.ndarray):
    """
    CPU log-odds update for one scan, from the best particle's base pose,
    applying LiDAR extrinsics to get sensor world pose.
    """
    # Sensor world pose
    sx, sy, syaw = lidar_cfg.sensor_world_pose(best_base_pose)

    # Range validity (clamp long rays via lidar_cfg.rmax_used to stabilize mapping)
    rmax_used = min(lidar_cfg.rmax, lidar_cfg.rmax_used)
    valid = (scan_ranges > lidar_cfg.rmin) & (scan_ranges < rmax_used)
    if not np.any(valid):
        return
    r = scan_ranges[valid]
    a = angles[valid]

    # World endpoints for valid beams
    wx = sx + r * np.cos(a + syaw)
    wy = sy + r * np.sin(a + syaw)

    world_to_map = build_world_to_map(map_cfg)
    sx_i, sy_i = world_to_map(sx, sy)

    H, W = map_cfg.sizex, map_cfg.sizey
    LOG_OCC = float(np.log(0.8 / 0.2))
    LOG_FREE = -LOG_OCC
    CLAMP_MIN, CLAMP_MAX = -10.0, 10.0

    for xw, yw in zip(wx, wy):
        ex_i, ey_i = world_to_map(xw, yw)
        # clamp start/end to map bounds
        sx_i_c = int(np.clip(sx_i, 0, H-1)); sy_i_c = int(np.clip(sy_i, 0, W-1))
        ex_i_c = int(np.clip(ex_i, 0, H-1)); ey_i_c = int(np.clip(ey_i, 0, W-1))
        cells = bresenham((sx_i_c, sy_i_c), (ex_i_c, ey_i_c))
        if not cells:
            continue
        # free cells (all but last), then occupied endpoint
        for (ix, iy) in cells[:-1]:
            if 0 <= ix < H and 0 <= iy < W:
                log_odds[ix, iy] += LOG_FREE
        ix_f, iy_f = cells[-1]
        if 0 <= ix_f < H and 0 <= iy_f < W:
            log_odds[ix_f, iy_f] += LOG_OCC

    np.clip(log_odds, CLAMP_MIN, CLAMP_MAX, out=log_odds)

# ---------------- Correlation scoring ----------------
def correlation_score(log_map: np.ndarray,
                      map_cfg: MapConfig,
                      xw: np.ndarray, yw: np.ndarray) -> float:
    """
    Binary-occupancy correlation: count how many beam endpoints land in log>0 cells.
    """
    m = np.isfinite(xw) & np.isfinite(yw)
    if not np.any(m):
        return 0.0
    xw = xw[m]; yw = yw[m]
    ix = np.floor((xw - map_cfg.xmin) / map_cfg.res).astype(int)
    iy = np.floor((yw - map_cfg.ymin) / map_cfg.res).astype(int)
    H, W = map_cfg.sizex, map_cfg.sizey
    valid = (ix >= 0) & (ix < H) & (iy >= 0) & (iy < W)
    if not np.any(valid):
        return 0.0
    return float((log_map[ix[valid], iy[valid]] > 0.0).sum())

def systematic_resample(weights: np.ndarray) -> np.ndarray:
    N = weights.size
    positions = (np.arange(N) + np.random.rand()) / N
    indexes = np.zeros(N, dtype=int)
    cumsum = np.cumsum(weights)
    i = 0; j = 0
    while i < N:
        if positions[i] < cumsum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
    return indexes

# ---------------- Main ----------------
if __name__ == "__main__":
    args = parse_args()
    xp, use_gpu = init_backend(use_gpu_beam=args.gpu_beam)

    # ---- Build configs (map, robot, lidar) ----
    map_cfg   = MapConfig()      # adjust defaults in robot.py if you want different bounds/res
    robot_cfg = RobotConfig()
    lidar_cfg = LidarConfig()

    # ---- Load dataset ----
    dataset = args.dataset
    enc = np.load(f"../data/Encoders{dataset}.npz")
    lid = np.load(f"../data/Hokuyo{dataset}.npz")
    imu = np.load(f"../data/Imu{dataset}.npz")

    encoder_counts = enc["counts"]           # (4, Ne)
    encoder_stamps = enc["time_stamps"]      # (Ne,)

    lidar_ranges = lid["ranges"]             # (M, K)
    lidar_angle_min = lid["angle_min"].item()
    lidar_angle_inc = lid["angle_increment"].item()
    lidar_range_min = lid["range_min"].item()
    lidar_range_max = lid["range_max"].item()
    lidar_stamps = lid["time_stamps"]        # (K,)

    imu_angvel = imu["angular_velocity"]     # (3, Ni) or (Ni,)
    imu_stamps = imu["time_stamps"]          # (Ni,)

    print("Encoder files are loaded")
    print("Hokuyo files are loaded")
    print("Imu files are loaded")

    # Sync LiDAR range limits from dataset into lidar_cfg (so PF uses your real sensor limits)
    lidar_cfg = LidarConfig(
        x=lidar_cfg.x, y=lidar_cfg.y, yaw=lidar_cfg.yaw,
        rmin=float(lidar_range_min),
        rmax=float(lidar_range_max),
        rmax_used=min(float(lidar_range_max), lidar_cfg.rmax_used)
    )

    # ---- Angles vector (match dataset convention: inclusive end) ----
    beams = lidar_ranges.shape[0]
    n_from_limits = int(np.floor((lidar_range_max - lidar_range_min) / 1.0))  # placeholder; angles depend on min/max angles
    # Correct angles from angle_min / angle_increment:
    M_by_inc = int(np.floor((lid["angle_max"].item() - lidar_angle_min) / lidar_angle_inc)) + 1
    angles_full = lidar_angle_min + np.arange(M_by_inc) * lidar_angle_inc
    # Ensure length = number of beams in ranges
    if angles_full.size != beams:
        # If mismatch, trim/pad to match ranges
        if angles_full.size > beams:
            angles = angles_full[:beams]
        else:
            # pad by repeating last value (rare)
            pad = np.full(beams - angles_full.size, angles_full[-1], float)
            angles = np.concatenate([angles_full, pad])
    else:
        angles = angles_full

    # ---- OGM (CPU) ----
    H, W = map_cfg.sizex, map_cfg.sizey
    log_odds = np.zeros((H, W), dtype=np.float32)

    # ---- PF state ----
    N = int(args.N)
    pose0 = np.array([0.0, 0.0, 0.0], dtype=float)     # world origin; map bounds from MapConfig
    particles = np.tile(pose0, (N, 1)).astype(float)
    weights   = np.ones(N, dtype=float) / N

    # Motion noise (tunable)
    sigma_v = 0.03   # m/s
    sigma_w = 0.05   # rad/s

    # Correlation search grids
    yaw_grid = np.deg2rad(np.array([-args.yaw_deg, 0.0, args.yaw_deg], dtype=float))
    xs_off = np.arange(-args.grid_range, args.grid_range + 1e-9, args.grid_step)
    ys_off = np.arange(-args.grid_range, args.grid_range + 1e-9, args.grid_step)

    # IMU yaw-rate series (z-axis)
    imu_yaw = imu_angvel
    if imu_yaw.ndim == 2:
        if imu_yaw.shape[0] == 3:
            imu_yaw = imu_yaw[2]
        elif imu_yaw.shape[1] == 3:
            imu_yaw = imu_yaw[:, 2]
    imu_yaw = np.asarray(imu_yaw, dtype=float)

    def interpolate_imu(ts):
        idx = np.searchsorted(imu_stamps, ts)
        idx = np.clip(idx, 1, len(imu_yaw) - 1)
        t0, t1 = imu_stamps[idx - 1], imu_stamps[idx]
        v0, v1 = float(imu_yaw[idx - 1]), float(imu_yaw[idx])
        return v0 + (v1 - v0) * (ts - t0) / (t1 - t0 + 1e-9)

    # ---- Bootstrap map with first scan at identity ----
    if lidar_ranges.shape[1] > 0:
        update_map_with_scan(
            log_odds, map_cfg, lidar_cfg, pose0,
            lidar_ranges[:, 0].astype(float),
            angles
        )

    # ---- Frame selection (subsample) ----
    K_all = min(lidar_ranges.shape[1], len(lidar_stamps))
    step  = max(1, int(args.step))
    frame_indices = np.arange(1, K_all, step)
    iterator = tqdm(frame_indices, total=len(frame_indices), desc="Frames", unit="f") if tqdm else frame_indices

    # ---- Main loop ----
    prev_counts = None
    trajectory = []
    t0 = time.time()

    # Precompute per-frame sensor-frame points once per frame (then rotate/translate by particle)
    for t in iterator:
        dt = float(lidar_stamps[t] - lidar_stamps[t-1])
        if dt <= 0:
            continue

        # linear velocity from encoders aligned to this lidar stamp
        i_enc = int(np.argmin(np.abs(encoder_stamps - lidar_stamps[t])))
        counts = encoder_counts[:, i_enc]
        # right = avg(FR, RR), left = avg(FL, RL)
        right_c = (counts[0] + counts[2]) / 2.0
        left_c  = (counts[1] + counts[3]) / 2.0
        if prev_counts is None:
            prev_counts = counts.copy()
        pr = (prev_counts[0] + prev_counts[2]) / 2.0
        pl = (prev_counts[1] + prev_counts[3]) / 2.0
        dr_ticks = (right_c - pr)
        dl_ticks = (left_c  - pl)
        # meters traveled per wheel = ticks * circumference_per_tick
        tick_to_meter = robot_cfg.tick_to_meter
        dr = dr_ticks * tick_to_meter
        dl = dl_ticks * tick_to_meter
        v  = float((dr + dl) / 2.0 / dt)
        w  = float(interpolate_imu(lidar_stamps[t]))
        prev_counts = counts.copy()

        # current scan
        scan = lidar_ranges[:, t].astype(float)

        # -------- Prediction (all particles) --------
        v_noisy = v + np.random.randn(N) * sigma_v
        w_noisy = w + np.random.randn(N) * sigma_w
        th = particles[:, 2]
        particles[:, 0] += v_noisy * dt * np.cos(th)
        particles[:, 1] += v_noisy * dt * np.sin(th)
        particles[:, 2] += w_noisy * dt
        particles[:, 2] = (particles[:, 2] + np.pi) % (2*np.pi) - np.pi

        # -------- Update via scan-grid correlation --------
        # Precompute sensor-frame beam endpoints (xs0, ys0)
        if use_gpu:
            scan_gpu = xp.asarray(scan)
            mask = (scan_gpu > lidar_cfg.rmin) & (scan_gpu < lidar_cfg.rmax)
            scan_gpu = xp.where(mask, scan_gpu, xp.nan)
            a_gpu = xp.asarray(angles)
            xs0 = (scan_gpu * xp.cos(a_gpu)).get()
            ys0 = (scan_gpu * xp.sin(a_gpu)).get()
        else:
            mask = (scan > lidar_cfg.rmin) & (scan < lidar_cfg.rmax)
            scan = np.where(mask, scan, np.nan)
            xs0 = scan * np.cos(angles)
            ys0 = scan * np.sin(angles)

        best_scores = np.full(N, -np.inf, dtype=float)
        best_dx = np.zeros(N, dtype=float)
        best_dy = np.zeros(N, dtype=float)
        best_dth = np.zeros(N, dtype=float)

        for i in range(N):
            px, py, pth = particles[i]
            # Sensor pose for this particle (apply extrinsics!)
            sx, sy, syaw = lidar_cfg.sensor_world_pose(np.array([px, py, pth], float))
            best_c = -1e9; bdx = 0.0; bdy = 0.0; bdth = 0.0
            for dth in yaw_grid:
                cth = math.cos(syaw + dth)
                sth = math.sin(syaw + dth)
                xw = sx + xs0 * cth - ys0 * sth
                yw = sy + xs0 * sth + ys0 * cth
                # sweep small XY offsets
                for dx in xs_off:
                    for dy in ys_off:
                        c = correlation_score(log_odds, map_cfg, xw + dx, yw + dy)
                        if c > best_c:
                            best_c = c; bdx = dx; bdy = dy; bdth = dth
            best_scores[i] = best_c
            best_dx[i] = bdx; best_dy[i] = bdy; best_dth[i] = bdth

        # Weights (softmax-like, stabilized)
        s = args.alpha * best_scores
        s -= s.max()
        w_new = weights * np.exp(s)
        ssum = w_new.sum()
        if ssum > 0 and np.isfinite(ssum):
            weights = w_new / ssum
        else:
            weights[:] = 1.0 / N

        # Gentle nudge toward the best local correlation
        nudge = 0.5
        particles[:, 0] += nudge * best_dx
        particles[:, 1] += nudge * best_dy
        particles[:, 2] += nudge * best_dth
        particles[:, 2] = (particles[:, 2] + np.pi) % (2*np.pi) - np.pi

        # Resample if Neff low
        neff = 1.0 / np.sum(weights**2)
        if neff < args.neff_frac * N:
            idx = systematic_resample(weights)
            particles = particles[idx]
            weights.fill(1.0 / N)

        # Best particle → map + trajectory
        best_idx = int(np.argmax(weights))
        best_pose = particles[best_idx].copy()
        trajectory.append(best_pose.copy())

        update_map_with_scan(
            log_odds, map_cfg, lidar_cfg, best_pose,
            scan, angles
        )

    # ----- Finalize & show/save -----
    print(f"\nCompleted {len(frame_indices)} frames in {time.time()-t0:.1f}s")
    CLAMP_MIN, CLAMP_MAX = -10.0, 10.0
    L = np.clip(log_odds, CLAMP_MIN, CLAMP_MAX)
    prob_map = 1.0 - 1.0 / (1.0 + np.exp(L))

    traj = np.array(trajectory)
    # plotting backend
    try:
        matplotlib.use("TkAgg", force=True)
    except Exception:
        matplotlib.use("Agg", force=True)

    # Show
    extent = [map_cfg.ymin, map_cfg.ymax, map_cfg.xmin, map_cfg.xmax]
    plt.figure(figsize=(8, 8))
    plt.imshow(prob_map.T, origin="lower", cmap="gray", vmin=0, vmax=1, extent=extent)
    if traj.size:
        plt.plot(traj[:,1], traj[:,0], "r-", linewidth=2)  # note extent axes (Y on x-axis)
    plt.title("PF-SLAM: Occupancy Map & Best-Particle Trajectory")
    plt.xlabel("Y [m]"); plt.ylabel("X [m]")
    plt.tight_layout()
    plt.show()

    # Save outputs
    os.makedirs(args.out, exist_ok=True)
    np.save(os.path.join(args.out, "logodds.npy"), L)
    np.save(os.path.join(args.out, "prob.npy"), prob_map)
    if traj.size:
        np.save(os.path.join(args.out, "trajectory.npy"), traj)
    plt.imsave(os.path.join(args.out, "ogm.png"), prob_map.T, cmap="gray", vmin=0, vmax=1)
    print("Saved to:", os.path.abspath(args.out))
