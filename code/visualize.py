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
    title: str = "Particle-filter occupancy grid",
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
        title=title,
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


def visualize_comparison(
    dead_reckoning: np.ndarray,
    particle_slam: np.ndarray,
    map_cfg: MapConfig,
    grid: np.ndarray,
    output_file: str | Path,
) -> None:
    """Save dead-reckoning and particle-SLAM trajectories on common axes."""
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 8))
    extent = [map_cfg.xmin, map_cfg.xmax, map_cfg.ymin, map_cfg.ymax]
    axis.imshow(grid.T, origin="lower", cmap="gray", interpolation="none", extent=extent)
    axis.plot(
        dead_reckoning[:, 0],
        dead_reckoning[:, 1],
        color="deepskyblue",
        linewidth=1.5,
        label="Dead reckoning",
    )
    axis.plot(
        particle_slam[:, 0],
        particle_slam[:, 1],
        color="red",
        linewidth=1.5,
        label="Particle SLAM",
    )
    axis.set(
        title="Dead reckoning vs. particle SLAM",
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


def visualize_optimization(
    original: np.ndarray,
    optimized: np.ndarray,
    optimized_grid: np.ndarray,
    map_cfg: MapConfig,
    output_file: str | Path,
) -> None:
    """Save particle-filter and pose-graph trajectories over the rebuilt map."""
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 8))
    extent = [map_cfg.xmin, map_cfg.xmax, map_cfg.ymin, map_cfg.ymax]
    axis.imshow(
        optimized_grid.T,
        origin="lower",
        cmap="gray",
        interpolation="none",
        extent=extent,
    )
    axis.plot(
        original[:, 0], original[:, 1], color="orange", linewidth=1.2, label="Particle SLAM"
    )
    axis.plot(
        optimized[:, 0], optimized[:, 1], color="blue", linewidth=1.8, label="GTSAM optimized"
    )
    axis.set(
        title="Pose-graph optimization and rebuilt map",
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


def visualize_diagnostics(
    diagnostics: dict[str, np.ndarray], output_file: str | Path
) -> None:
    """Save effective sample size, weights, scan quality, and spread histories."""
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    axes[0].plot(diagnostics["neff"], color="navy")
    axes[0].set_ylabel("Effective N")
    axes[1].plot(diagnostics["max_weight"], color="darkred")
    axes[1].set_ylabel("Max weight")
    axes[2].plot(diagnostics["valid_beams"], color="darkgreen")
    axes[2].set_ylabel("Valid beams")
    axes[3].plot(diagnostics["position_spread"], color="purple")
    axes[3].set_ylabel("XY spread [m]")
    axes[3].set_xlabel("Filter update")
    for axis in axes:
        axis.grid(True, linestyle="--", alpha=0.3)
    figure.suptitle(
        f"Particle-filter diagnostics ({int(np.sum(diagnostics['resampled']))} resamples)"
    )
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
