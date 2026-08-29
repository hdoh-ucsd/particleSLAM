"""Optimize an existing particle-SLAM result without rerunning the filter."""

import argparse
from pathlib import Path

import numpy as np

from config import LidarConfig, MapConfig, PoseGraphConfig
from dataset_utils import load_synced_data
from evaluation import evaluate_optimization, save_evaluation, write_run_report
from pose_graph import optimize_pose_graph, rebuild_occupancy_grid
from visualize import visualize_optimization


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synced-data", type=Path, required=True)
    parser.add_argument("--particle-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=int, default=20)
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--keyframe-interval", type=int, default=20)
    parser.add_argument("--fixed-loop-interval", type=int, default=10)
    parser.add_argument("--proximity-distance", type=float, default=1.5)
    parser.add_argument("--max-proximity-candidates", type=int, default=30)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = load_synced_data(args.synced_data)
    with np.load(args.particle_result) as archive:
        trajectory = archive["trajectory"]
        original_grid = archive["grid"]
    lidar_cfg = LidarConfig(
        rmin=float(np.asarray(data["lidar_range_min"]).squeeze()),
        rmax=float(np.asarray(data["lidar_range_max"]).squeeze()),
        angle_min=float(np.asarray(data["lidar_angle_min"]).squeeze()),
        angle_increment=float(np.asarray(data["lidar_angle_increment"]).squeeze()),
    )
    map_cfg = MapConfig()
    graph_cfg = PoseGraphConfig(
        keyframe_interval=args.keyframe_interval,
        fixed_loop_interval=args.fixed_loop_interval,
        proximity_distance=args.proximity_distance,
        max_proximity_candidates=args.max_proximity_candidates,
    )
    result = optimize_pose_graph(trajectory, data["lidar"], lidar_cfg, graph_cfg)
    optimized_grid = rebuild_occupancy_grid(
        data, result.optimized_trajectory, map_cfg, lidar_cfg
    )
    stem = f"optimized_slam_{args.dataset}_{args.backend}"
    data_path = args.output_dir / f"{stem}.npz"
    figure_path = args.output_dir / f"{stem}.png"
    evaluation_path = args.output_dir / f"evaluation_{args.dataset}_{args.backend}.json"
    report_path = args.output_dir / f"run_report_{args.dataset}_{args.backend}.md"
    np.savez(
        data_path,
        trajectory=result.optimized_trajectory,
        keyframe_indices=result.keyframe_indices,
        optimized_keyframes=result.optimized_keyframes,
        grid=optimized_grid,
        loop_sources=np.asarray([item.source for item in result.loop_closures]),
        loop_targets=np.asarray([item.target for item in result.loop_closures]),
        loop_rmse=np.asarray([item.rmse for item in result.loop_closures]),
        loop_overlap=np.asarray([item.overlap for item in result.loop_closures]),
        loop_methods=np.asarray([item.method for item in result.loop_closures]),
    )
    visualize_optimization(
        trajectory, result.optimized_trajectory, optimized_grid, map_cfg, figure_path
    )
    metrics = evaluate_optimization(
        trajectory,
        result.optimized_trajectory,
        result.keyframe_indices,
        result.loop_closures,
        original_grid,
        optimized_grid,
        result.candidate_count,
    )
    save_evaluation(metrics, evaluation_path)
    write_run_report(metrics, args.dataset, args.backend, report_path)
    print(f"Accepted {len(result.loop_closures)}/{result.candidate_count} loop candidates")
    print(f"Saved {data_path}")
    print(f"Saved {figure_path}")
    print(f"Saved {evaluation_path}")
    print(f"Saved {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
