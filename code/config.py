from dataclasses import dataclass
import numpy as np
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