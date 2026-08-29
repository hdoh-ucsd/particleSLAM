"""Evaluation helpers for trajectory and occupancy-map artifacts."""

import json
from pathlib import Path

import numpy as np

from pose_graph import LoopClosure, relative_pose, wrap_angle


def path_length(trajectory: np.ndarray) -> float:
    if len(trajectory) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(trajectory[:, :2], axis=0), axis=1).sum())


def _closure_residual(
    trajectory: np.ndarray,
    keyframe_indices: np.ndarray,
    closure: LoopClosure,
) -> float:
    predicted = relative_pose(
        trajectory[keyframe_indices[closure.source]],
        trajectory[keyframe_indices[closure.target]],
    )
    delta = predicted - closure.relative_pose
    delta[2] = wrap_angle(delta[2])
    return float(np.linalg.norm(delta))


def evaluate_optimization(
    original: np.ndarray,
    optimized: np.ndarray,
    keyframe_indices: np.ndarray,
    closures: tuple[LoopClosure, ...],
    original_grid: np.ndarray,
    optimized_grid: np.ndarray,
    candidate_count: int | None = None,
) -> dict[str, float | int]:
    displacement = np.linalg.norm(optimized[:, :2] - original[:, :2], axis=1)
    before_residuals = [
        _closure_residual(original, keyframe_indices, closure) for closure in closures
    ]
    after_residuals = [
        _closure_residual(optimized, keyframe_indices, closure) for closure in closures
    ]
    return {
        "pose_count": int(len(original)),
        "keyframe_count": int(len(keyframe_indices)),
        "accepted_loop_closures": int(len(closures)),
        "loop_closure_candidates": int(
            len(closures) if candidate_count is None else candidate_count
        ),
        "original_path_length_m": path_length(original),
        "optimized_path_length_m": path_length(optimized),
        "mean_pose_shift_m": float(displacement.mean()),
        "max_pose_shift_m": float(displacement.max()),
        "endpoint_shift_m": float(displacement[-1]),
        "mean_loop_residual_before": float(np.mean(before_residuals))
        if before_residuals
        else 0.0,
        "mean_loop_residual_after": float(np.mean(after_residuals))
        if after_residuals
        else 0.0,
        "original_observed_cells": int(np.count_nonzero(original_grid)),
        "optimized_observed_cells": int(np.count_nonzero(optimized_grid)),
        "optimized_occupied_cells": int(np.count_nonzero(optimized_grid > 0)),
        "optimized_free_cells": int(np.count_nonzero(optimized_grid < 0)),
    }


def save_evaluation(metrics: dict[str, float | int], output_file: str | Path) -> None:
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")


def write_run_report(
    metrics: dict[str, float | int],
    dataset: int,
    backend: str,
    output_file: str | Path,
) -> None:
    """Write a report-ready run summary without claiming unavailable ground truth."""
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    rows = "\n".join(
        f"| {name.replace('_', ' ').title()} | {value:.4f} |"
        if isinstance(value, float)
        else f"| {name.replace('_', ' ').title()} | {value} |"
        for name, value in metrics.items()
    )
    output_file.write_text(
        f"""# particleSLAM Run Report

## Configuration

- Dataset: {dataset}
- Particle backend: {backend}
- Optimization: GTSAM Pose2 graph with ICP-gated loop closures

## Results

| Metric | Value |
| --- | ---: |
{rows}

## Interpretation

These metrics compare the particle-filter trajectory with its pose-graph refinement. They are internal consistency measurements, not absolute accuracy measurements, because no ground-truth trajectory is included with the dataset.

## Texture map status

A texture map was not generated because the repository datasets contain encoder, IMU, and 2-D LiDAR measurements but no RGB or RGB-D frames. Texture reconstruction requires synchronized camera data and calibrated camera extrinsics.
""",
        encoding="utf-8",
    )
