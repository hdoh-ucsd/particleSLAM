"""Command-line entry point for the particleSLAM pipeline."""

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np

from config import LidarConfig, MapConfig, ParticleFilterConfig, RobotConfig
from dataset_utils import load_dataset, save_synced_dataset
from occupancy_grid import build_occupancy_grid


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run sensor synchronization, occupancy mapping, and particle SLAM."
    )
    parser.add_argument("--dataset", type=int, default=20, help="numbered dataset to load")
    parser.add_argument(
        "--backend",
        choices=("cpu", "gpu"),
        default="cpu",
        help="particle-filter compute backend",
    )
    parser.add_argument(
        "--particles", type=int, default=1000, help="number of particles (default: 1000)"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data",
        help="directory containing the numbered NPZ sensor files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "build",
        help="directory for generated datasets, maps, and figures",
    )
    parser.add_argument(
        "--skip-reference-map",
        action="store_true",
        help="skip the odometry-based occupancy-grid pass",
    )
    return parser


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.particles < 1:
        parser.error("--particles must be at least 1")


def _print_dataset_summary(data: dict[str, np.ndarray]) -> None:
    print("Synchronized dataset:")
    for name, values in data.items():
        print(f"  {name:<24} shape={values.shape!s:<16} dtype={values.dtype}")


def run(args: argparse.Namespace) -> dict[str, Path]:
    """Execute the configured pipeline and return the generated artifact paths."""
    args.output_dir.mkdir(parents=True, exist_ok=True)
    map_cfg = MapConfig()
    robot_cfg = RobotConfig()
    filter_cfg = ParticleFilterConfig(num_particles=args.particles, seed=args.seed)

    raw_data = load_dataset(args.dataset, args.data_dir, robot_cfg)
    synced_path = args.output_dir / f"synced_data_{args.dataset}.npz"
    synced_data = save_synced_dataset(raw_data, synced_path)
    lidar_cfg = LidarConfig(
        rmin=float(np.asarray(synced_data["lidar_range_min"]).squeeze()),
        rmax=float(np.asarray(synced_data["lidar_range_max"]).squeeze()),
        angle_min=float(np.asarray(synced_data["lidar_angle_min"]).squeeze()),
        angle_increment=float(
            np.asarray(synced_data["lidar_angle_increment"]).squeeze()
        ),
    )
    _print_dataset_summary(synced_data)

    artifacts = {"synced_data": synced_path}
    if not args.skip_reference_map:
        reference_grid = build_occupancy_grid(synced_data, map_cfg, lidar_cfg)
        map_path = args.output_dir / f"occupancy_grid_{args.dataset}.npz"
        np.savez(map_path, grid=reference_grid, map_cfg=vars(map_cfg))
        artifacts["reference_map"] = map_path

    result_path = args.output_dir / f"particle_slam_{args.dataset}_{args.backend}.png"
    if args.backend == "cpu":
        from particle_filter_cpu import particle_filter_cpu

        trajectory, grid, _particles = particle_filter_cpu(
            synced_data,
            filter_cfg,
            map_cfg,
            lidar_cfg,
            robot_cfg,
            result_path,
        )
    else:
        try:
            from particle_filter_gpu import particle_filter_gpu
        except ImportError as exc:
            raise RuntimeError(
                "The GPU backend requires a CUDA-compatible CuPy installation."
            ) from exc

        trajectory, grid, _particles = particle_filter_gpu(
            synced_data,
            filter_cfg,
            map_cfg,
            lidar_cfg,
            robot_cfg,
            result_path,
        )

    result_data_path = args.output_dir / f"particle_slam_{args.dataset}_{args.backend}.npz"
    np.savez(result_data_path, trajectory=trajectory, grid=grid)
    artifacts.update(result_figure=result_path, result_data=result_data_path)

    print("Generated artifacts:")
    for name, path in artifacts.items():
        print(f"  {name:<16} {path}")
    return artifacts


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(args, parser)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
