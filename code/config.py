import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class MapConfig:
    res: float = 0.05
    xmin: float = -10.0
    xmax: float = 30.0
    ymin: float = -10.0
    ymax: float = 30.0

    @property
    def sizex(self) -> int:
        return int(np.ceil((self.xmax - self.xmin) / self.res)) + 1

    @property
    def sizey(self) -> int:
        return int(np.ceil((self.ymax - self.ymin) / self.res)) + 1


@dataclass(frozen=True)
class RobotConfig:
    wheel_base: float = 0.5842
    wheel_radius: float = 0.127
    encoder_resolution: int = 360
    gear_ratio: float = 1.0
    baseline: float = 0.16

    @property
    def ticks_per_rev(self) -> float:
        return self.encoder_resolution * self.gear_ratio

    @property
    def tick_to_meter(self) -> float:
        return (2.0 * np.pi * self.wheel_radius) / self.ticks_per_rev


@dataclass(frozen=True)
class LidarConfig:
    x: float = 0.30183
    y: float = 0.0
    yaw: float = 0.0
    rmin: float = 0.05
    rmax: float = 30.0
    rmax_used: float = 10.0
    angle_min: float = -2.356194490192345
    angle_increment: float = 0.00436332

    def sensor_world_pose(self, base_xyz: np.ndarray) -> tuple[float, float, float]:
        xw, yw, th = float(base_xyz[0]), float(base_xyz[1]), float(base_xyz[2])
        sx = xw + self.x*np.cos(th) - self.y*np.sin(th)
        sy = yw + self.x*np.sin(th) + self.y*np.cos(th)
        syaw = th + self.yaw
        return sx, sy, syaw


@dataclass(frozen=True)
class ParticleFilterConfig:
    num_particles: int = 1000
    seed: int = 42
    resample_threshold: float = 0.5
    linear_noise_std: float = 0.02
    angular_noise_std: float = 0.01
    correlation_xy_window: float = 0.10
    correlation_xy_step: float = 0.05
    correlation_yaw_window: float = 0.02
    correlation_yaw_step: float = 0.02
    correlation_beam_stride: int = 4
    likelihood_temperature: float = 4.0
