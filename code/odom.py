import numpy as np
from config import RobotConfig

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