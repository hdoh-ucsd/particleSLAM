import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

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

def visualize(grid, trajectory, map_cfg, particles, t=0):
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