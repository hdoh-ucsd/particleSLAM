"""Command-line entry point for the particleSLAM pipeline."""

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np

from config import (
    LidarConfig,
    MapConfig,
    ParticleFilterConfig,
    PoseGraphConfig,
    RobotConfig,
)
from dataset_utils import load_dataset, save_synced_dataset
from occupancy_grid import build_occupancy_grid


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run sensor synchronization, occupancy mapping, and particle SLAM."
    )
    parser.add_argument("--dataset", type=int, default=20, help="numbered dataset to load")
    parser.add_argument(
        "--mode",
        choices=("slam", "dead-reckoning", "compare"),
        default="slam",
        help="run particle SLAM, the deterministic baseline, or both",
    )
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
        "--max-steps",
        type=int,
        default=None,
        help="process only the first N synchronized timestamps (for diagnostics)",
    )
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
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="run GTSAM pose-graph optimization and rebuild the occupancy map",
    )
    parser.add_argument(
        "--keyframe-interval",
        type=int,
        default=20,
        help="pose-graph keyframe spacing in filter updates",
    )
    parser.add_argument(
        "--proximity-distance",
        type=float,
        default=1.5,
        help="maximum distance for proximity loop-closure candidates",
    )
    return parser


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.particles < 1:
        parser.error("--particles must be at least 1")
    if args.max_steps is not None and args.max_steps < 2:
        parser.error("--max-steps must be at least 2")
    if args.keyframe_interval < 1:
        parser.error("--keyframe-interval must be at least 1")
    if args.proximity_distance <= 0:
        parser.error("--proximity-distance must be positive")
    if args.optimize and args.mode == "dead-reckoning":
        parser.error("--optimize requires --mode slam or --mode compare")


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

    if args.max_steps is not None:
        full_length = synced_data["sync_times"].size
        step_count = min(args.max_steps, full_length)
        synced_data = {
            name: (
                values[..., :step_count]
                if values.ndim > 0 and values.shape[-1] == full_length
                else values
            )
            for name, values in synced_data.items()
        }
        print(f"Processing the first {step_count} synchronized timestamps.")

    artifacts = {"synced_data": synced_path}
    if not args.skip_reference_map:
        reference_grid = build_occupancy_grid(synced_data, map_cfg, lidar_cfg)
        map_path = args.output_dir / f"occupancy_grid_{args.dataset}.npz"
        np.savez(map_path, grid=reference_grid, map_cfg=vars(map_cfg))
        artifacts["reference_map"] = map_path

    dead_reckoning_trajectory = None
    dead_reckoning_grid = None
    if args.mode in ("dead-reckoning", "compare"):
        from dead_reckoning import run_dead_reckoning
        from visualize import visualize_particles

        dead_reckoning_trajectory, dead_reckoning_grid = run_dead_reckoning(
            synced_data, map_cfg, lidar_cfg, robot_cfg
        )
        dead_reckoning_data_path = (
            args.output_dir / f"dead_reckoning_{args.dataset}.npz"
        )
        dead_reckoning_figure_path = (
            args.output_dir / f"dead_reckoning_{args.dataset}.png"
        )
        np.savez(
            dead_reckoning_data_path,
            trajectory=dead_reckoning_trajectory,
            grid=dead_reckoning_grid,
        )
        visualize_particles(
            dead_reckoning_grid,
            dead_reckoning_trajectory,
            map_cfg,
            np.empty((0, 3)),
            dead_reckoning_figure_path,
            title="Dead-reckoning occupancy grid",
        )
        artifacts.update(
            dead_reckoning_data=dead_reckoning_data_path,
            dead_reckoning_figure=dead_reckoning_figure_path,
        )
        if args.mode == "dead-reckoning":
            _print_artifacts(artifacts)
            return artifacts

    result_path = args.output_dir / f"particle_slam_{args.dataset}_{args.backend}.png"
    diagnostics: dict[str, list[float]] = {}
    if args.backend == "cpu":
        from particle_filter_cpu import particle_filter_cpu

        trajectory, grid, _particles = particle_filter_cpu(
            synced_data,
            filter_cfg,
            map_cfg,
            lidar_cfg,
            robot_cfg,
            result_path,
            diagnostics=diagnostics,
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
            diagnostics=diagnostics,
        )

    result_data_path = args.output_dir / f"particle_slam_{args.dataset}_{args.backend}.npz"
    np.savez(result_data_path, trajectory=trajectory, grid=grid)
    artifacts.update(result_figure=result_path, result_data=result_data_path)

    diagnostic_arrays = {
        name: np.asarray(values) for name, values in diagnostics.items()
    }
    diagnostics_data_path = (
        args.output_dir / f"particle_slam_{args.dataset}_{args.backend}_diagnostics.npz"
    )
    diagnostics_figure_path = (
        args.output_dir / f"particle_slam_{args.dataset}_{args.backend}_diagnostics.png"
    )
    np.savez(diagnostics_data_path, **diagnostic_arrays)
    from visualize import visualize_diagnostics

    visualize_diagnostics(diagnostic_arrays, diagnostics_figure_path)
    artifacts.update(
        diagnostics_data=diagnostics_data_path,
        diagnostics_figure=diagnostics_figure_path,
    )

    if args.mode == "compare":
        from visualize import visualize_comparison

        comparison_path = args.output_dir / f"comparison_{args.dataset}_{args.backend}.png"
        visualize_comparison(
            dead_reckoning_trajectory,
            trajectory,
            map_cfg,
            grid,
            comparison_path,
        )
        endpoint_separation = np.linalg.norm(
            dead_reckoning_trajectory[-1, :2] - trajectory[-1, :2]
        )
        comparison_data_path = (
            args.output_dir / f"comparison_{args.dataset}_{args.backend}.npz"
        )
        np.savez(
            comparison_data_path,
            dead_reckoning=dead_reckoning_trajectory,
            particle_slam=trajectory,
            endpoint_separation=endpoint_separation,
        )
        artifacts.update(
            comparison_figure=comparison_path,
            comparison_data=comparison_data_path,
        )

    if args.optimize:
        from evaluation import evaluate_optimization, save_evaluation, write_run_report
        from pose_graph import optimize_pose_graph, rebuild_occupancy_grid
        from visualize import visualize_optimization

        graph_cfg = PoseGraphConfig(
            keyframe_interval=args.keyframe_interval,
            proximity_distance=args.proximity_distance,
        )
        graph_result = optimize_pose_graph(
            trajectory, synced_data["lidar"], lidar_cfg, graph_cfg
        )
        optimized_grid = rebuild_occupancy_grid(
            synced_data,
            graph_result.optimized_trajectory,
            map_cfg,
            lidar_cfg,
        )
        optimized_data_path = (
            args.output_dir / f"optimized_slam_{args.dataset}_{args.backend}.npz"
        )
        loop_sources = np.asarray(
            [closure.source for closure in graph_result.loop_closures], dtype=int
        )
        loop_targets = np.asarray(
            [closure.target for closure in graph_result.loop_closures], dtype=int
        )
        loop_rmse = np.asarray(
            [closure.rmse for closure in graph_result.loop_closures], dtype=float
        )
        loop_overlap = np.asarray(
            [closure.overlap for closure in graph_result.loop_closures], dtype=float
        )
        loop_methods = np.asarray(
            [closure.method for closure in graph_result.loop_closures]
        )
        np.savez(
            optimized_data_path,
            trajectory=graph_result.optimized_trajectory,
            keyframe_indices=graph_result.keyframe_indices,
            optimized_keyframes=graph_result.optimized_keyframes,
            grid=optimized_grid,
            loop_sources=loop_sources,
            loop_targets=loop_targets,
            loop_rmse=loop_rmse,
            loop_overlap=loop_overlap,
            loop_methods=loop_methods,
        )
        optimized_figure_path = (
            args.output_dir / f"optimized_slam_{args.dataset}_{args.backend}.png"
        )
        visualize_optimization(
            trajectory,
            graph_result.optimized_trajectory,
            optimized_grid,
            map_cfg,
            optimized_figure_path,
        )
        metrics = evaluate_optimization(
            trajectory,
            graph_result.optimized_trajectory,
            graph_result.keyframe_indices,
            graph_result.loop_closures,
            grid,
            optimized_grid,
            graph_result.candidate_count,
        )
        evaluation_path = (
            args.output_dir / f"evaluation_{args.dataset}_{args.backend}.json"
        )
        report_path = args.output_dir / f"run_report_{args.dataset}_{args.backend}.md"
        save_evaluation(metrics, evaluation_path)
        write_run_report(metrics, args.dataset, args.backend, report_path)
        artifacts.update(
            optimized_data=optimized_data_path,
            optimized_figure=optimized_figure_path,
            evaluation=evaluation_path,
            run_report=report_path,
        )

    _print_artifacts(artifacts)
    return artifacts


def _print_artifacts(artifacts: dict[str, Path]) -> None:
    print("Generated artifacts:")
    for name, path in artifacts.items():
        print(f"  {name:<16} {path}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(args, parser)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
