from __future__ import annotations
from dataclasses import dataclass
import numpy as np

# ---- Map configuration ----
@dataclass(frozen=True)
class MapConfig:
    # World bounds in meters and resolution (meters/cell)
    res: float = 0.05
    xmin: float = -20.0
    xmax: float = 20.0
    ymin: float = -20.0
    ymax: float = 20.0

    @property
    def sizex(self) -> int:
        return int(np.ceil((self.xmax - self.xmin) / self.res)) + 1

    @property
    def sizey(self) -> int:
        return int(np.ceil((self.ymax - self.ymin) / self.res)) + 1

    # Helpers (optional)
    def meters_to_cells(self, x: float, y: float) -> tuple[int, int]:
        ix = int(np.floor((x - self.xmin) / self.res))
        iy = int(np.floor((y - self.ymin) / self.res))
        ix = np.clip(ix, 0, self.sizex - 1)
        iy = np.clip(iy, 0, self.sizey - 1)
        return ix, iy

@dataclass(frozen=True)
class RobotConfig:
    wheel_base: float = 0.5842
    wheel_radius: float = 0.127    # m (radius; diameter 0.254 m in PDF)
    encoder_resolution: int = 360
    gear_ratio: float = 1.0
    baseline: float = 0.16         # m (wheel separation; reasonable default)
    ticks_per_rev = encoder_resolution * gear_ratio
    meters_per_tick = 2 * np.pi * wheel_radius / ticks_per_rev

    @property
    def tick_to_meter(self) -> float:
        return (2.0 * np.pi * self.wheel_radius) / float(self.ticks_per_rev)

@dataclass(frozen=True)
class LidarConfig:
    # LiDAR mount wrt robot base (meters / radians)
    x: float = 0.30183     # PDF: 301.83 mm = 0.30183 m
    y: float = 0.0         # PDF does not specify y offset (assume 0)
    yaw: float = 0.0       # PDF does not specify yaw offset (assume 0)
    # valid ranges (defaults; overwrite from dataset at runtime)
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
