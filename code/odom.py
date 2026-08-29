import numpy as np
from config import RobotConfig

class DifferentialDrive:
    def __init__(self, config):
        self.config = config
        self.wheel_base = config.wheel_base      # [meters] distance between wheels
        self.wheel_radius = config.wheel_radius  # [meters]
        self.enc_res = config.encoder_resolution # [ticks per revolution]
        self.gear_ratio = config.gear_ratio      # if present

    def integrate_odometry(self, encoder_stamps, encoder_counts):
        """Integrate per-reading wheel counts ordered as [FR, FL, RR, RL].

        The hardware resets each encoder counter after every measurement, so
        each column is already an increment and must not be differentiated.
        """
        if encoder_counts.shape[0] != 4:
            raise ValueError("encoder_counts must have shape (4, N)")
        if encoder_counts.shape[1] != len(encoder_stamps):
            raise ValueError("encoder timestamps and counts must have equal lengths")

        front_right, front_left, rear_right, rear_left = encoder_counts
        right_dist = (front_right + rear_right) * self.config.tick_to_meter / 2.0
        left_dist = (front_left + rear_left) * self.config.tick_to_meter / 2.0

        x, y, theta = [0.0], [0.0], [0.0]
        for i in range(1, len(encoder_stamps)):
            dl = left_dist[i]
            dr = right_dist[i]
            d_center = (dr + dl) / 2.0
            d_theta = (dr - dl) / self.wheel_base

            x_new = x[-1] + d_center * np.cos(theta[-1] + d_theta/2)
            y_new = y[-1] + d_center * np.sin(theta[-1] + d_theta/2)
            theta_new = theta[-1] + d_theta

            x.append(x_new)
            y.append(y_new)
            theta.append(theta_new)
        return np.vstack((x, y, theta))  # shape (3, N)

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
