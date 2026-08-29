# particleSLAM Project Report

## Outcome

The project now provides an end-to-end, reproducible 2-D SLAM pipeline for both supplied robot runs. It synchronizes encoder, IMU, and LiDAR measurements; produces a dead-reckoning baseline; estimates trajectory and occupancy with a particle filter; validates loop candidates with ICP; optimizes an SE(2) graph with GTSAM; rebuilds the map; and emits machine-readable diagnostics.

No private assignment documents are included or linked. This report describes only the implementation and reproducible results in this repository.

## Implemented work

- Encoder-aware differential-drive motion with filtered IMU yaw rate
- LiDAR-time synchronization and explicit handling of the IMU tail
- CPU and optional CuPy particle-filter backends
- Scan-to-grid correlation, systematic resampling, and filter diagnostics
- Dead-reckoning comparison mode
- Keyframe graph with ICP quality and consistency gates
- Robust Huber loop factors and GTSAM optimization
- Occupancy-grid reconstruction from optimized poses
- JSON, Markdown, image, and NPZ output artifacts
- Repeatable parameter study across datasets 20 and 21

## Reproducible full-run results

Both runs used the CPU backend, 100 particles, seed 42, 20-scan keyframes, and the default robust pose-graph configuration.

| Metric | Dataset 20 | Dataset 21 |
| --- | ---: | ---: |
| PF poses | 4,961 | 4,784 |
| GTSAM keyframes | 249 | 241 |
| Accepted / candidate loop constraints | 23 / 54 | 37 / 54 |
| PF path length | 97.38 m | 95.53 m |
| Optimized path length | 95.16 m | 90.92 m |
| Mean / maximum pose shift | 0.49 / 0.81 m | 0.48 / 1.07 m |
| Loop residual before / after | 0.072 / 0.016 | 0.023 / 0.006 |
| Rebuilt observed cells | 110,017 | 103,081 |

Residuals and pose shifts measure internal consistency, not absolute accuracy. The supplied sensor runs do not contain a ground-truth trajectory.

## Validation and limitations

The stricter closure policy corrected an earlier over-constrained dataset 20 result: candidates are bounded, fixed candidates are sampled once per interval, ICP must satisfy overlap/RMSE and correction limits, and accepted factors use a robust loss. Unit tests cover synchronization, odometry, resampling, scan correlation, and SE(2) pose operations.

The trajectory-only parameter study produced the following sensitivity results. Candidate count changes substantially, while optimized length and maximum shift remain stable; this supports retaining the middle setting as the default.

| Dataset | Policy | Accepted / candidates | Optimized path | Maximum shift |
| --- | --- | ---: | ---: | ---: |
| 20 | Conservative | 14 / 39 | 95.14 m | 0.81 m |
| 20 | Default | 23 / 54 | 95.16 m | 0.81 m |
| 20 | Wide proximity | 26 / 69 | 95.16 m | 0.81 m |
| 21 | Conservative | 22 / 39 | 90.94 m | 1.08 m |
| 21 | Default | 37 / 54 | 90.92 m | 1.07 m |
| 21 | Wide proximity | 39 / 69 | 90.93 m | 1.07 m |

Absolute trajectory error cannot be reported until an external ground-truth trajectory and its coordinate/time alignment are supplied. Texture mapping cannot be implemented from these inputs because they contain no synchronized RGB or RGB-D frames or camera extrinsics. These are data limitations, not unimplemented switches in the current LiDAR SLAM pipeline.

## Reproduction

See the root README for environment setup and commands. Use `code/main.py` for an end-to-end run, `code/optimize_result.py` to retune a saved particle-filter result, and `code/parameter_study.py` to compare graph settings without rerunning the particle filter.
