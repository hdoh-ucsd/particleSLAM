"""Visualization helpers for occupancy-grid and particle-filter results."""

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import MapConfig
from odom import logodds_to_prob


def visualize_ogm(grid: np.ndarray, map_cfg: MapConfig, output_file: str | Path) -> None:
    """Save an occupancy-probability map."""
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(8, 8))
    image = axis.imshow(
        logodds_to_prob(grid).T,
        origin="lower",
        cmap="gray",
        extent=[map_cfg.xmin, map_cfg.xmax, map_cfg.ymin, map_cfg.ymax],
    )
    axis.set(title="Occupancy Grid Map", xlabel="X [m]", ylabel="Y [m]")
    figure.colorbar(image, ax=axis, label="Occupancy probability")
    figure.tight_layout()
    figure.savefig(output_file)
    plt.close(figure)


def visualize_particles(
    grid: np.ndarray,
    trajectory: np.ndarray,
    map_cfg: MapConfig,
    particles: np.ndarray,
    output_file: str | Path,
) -> None:
    """Save an occupancy grid with the estimated trajectory and particle cloud."""
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(8, 8))
    extent = [map_cfg.xmin, map_cfg.xmax, map_cfg.ymin, map_cfg.ymax]
    axis.imshow(grid.T, origin="lower", cmap="gray", interpolation="none", extent=extent)

    trajectory = np.asarray(trajectory)
    particles = np.asarray(particles)
    if trajectory.size:
        axis.plot(trajectory[:, 0], trajectory[:, 1], color="red", linewidth=2, label="Trajectory")
    if particles.size:
        axis.scatter(
            particles[:, 0],
            particles[:, 1],
            color="blue",
            s=3,
            alpha=0.5,
            label="Particles",
        )

    axis.set(
        title="Particle-filter occupancy grid",
        xlabel="X [m]",
        ylabel="Y [m]",
        xlim=(map_cfg.xmin, map_cfg.xmax),
        ylim=(map_cfg.ymin, map_cfg.ymax),
    )
    axis.grid(True, linestyle="--", alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_file)
    plt.close(figure)


def visualize_cpu(grid, trajectory, map_cfg, particles, t=0):
    """Backward-compatible wrapper for the previous CPU plotting API."""
    visualize_particles(grid, trajectory, map_cfg, particles, f"pf_grid_cells_{t:04d}.png")


def visualize_gpu(grid, trajectory, map_cfg, particles, t=0):
    """Backward-compatible wrapper for the previous GPU plotting API."""
    visualize_particles(
        grid,
        trajectory,
        map_cfg,
        particles.get(),
        f"pf_grid_cells_{t:04d}.png",
    )
