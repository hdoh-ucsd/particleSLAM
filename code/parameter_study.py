"""Compare pose-graph settings on saved particle-SLAM trajectories."""

import argparse
import csv
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from config import LidarConfig, PoseGraphConfig
from dataset_utils import load_synced_data
from evaluation import path_length
from pose_graph import optimize_pose_graph


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=int, action="append", required=True)
    parser.add_argument("--input-dir", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, default=Path("build/parameter_study.json"))
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    return parser


def _lidar_config(data: dict[str, np.ndarray]) -> LidarConfig:
    return LidarConfig(
        rmin=float(np.asarray(data["lidar_range_min"]).squeeze()),
        rmax=float(np.asarray(data["lidar_range_max"]).squeeze()),
        angle_min=float(np.asarray(data["lidar_angle_min"]).squeeze()),
        angle_increment=float(np.asarray(data["lidar_angle_increment"]).squeeze()),
    )


def main() -> int:
    args = build_parser().parse_args()
    if len(args.dataset) != len(args.input_dir):
        raise SystemExit("provide one --input-dir for each --dataset")
    base = PoseGraphConfig()
    variants = {
        "conservative": replace(base, proximity_distance=1.0, max_proximity_candidates=15),
        "default": base,
        "wide_proximity": replace(base, proximity_distance=2.0, max_proximity_candidates=45),
    }
    records: list[dict[str, object]] = []
    for dataset, input_dir in zip(args.dataset, args.input_dir):
        data = load_synced_data(input_dir / f"synced_data_{dataset}.npz")
        with np.load(input_dir / f"particle_slam_{dataset}_{args.backend}.npz") as archive:
            trajectory = archive["trajectory"]
        for name, config in variants.items():
            result = optimize_pose_graph(trajectory, data["lidar"], _lidar_config(data), config)
            shifts = np.linalg.norm(result.optimized_trajectory[:, :2] - trajectory[:, :2], axis=1)
            records.append({
                "dataset": dataset, "variant": name,
                "candidates": result.candidate_count,
                "accepted": len(result.loop_closures),
                "pf_path_m": path_length(trajectory),
                "optimized_path_m": path_length(result.optimized_trajectory),
                "mean_shift_m": float(np.mean(shifts)),
                "max_shift_m": float(np.max(shifts)),
                "config": asdict(config),
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    csv_path = args.output.with_suffix(".csv")
    fields = ["dataset", "variant", "candidates", "accepted", "pf_path_m", "optimized_path_m", "mean_shift_m", "max_shift_m"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: record[field] for field in fields} for record in records)
    print(f"Saved {args.output}")
    print(f"Saved {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
