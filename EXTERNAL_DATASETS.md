# External Dataset Guide

Large third-party ROS bags are intentionally excluded from Git. Download them from their official project pages, retain their original licenses and citations, and place them under `data/external/`.

## MIT Stata Center

Official source: [MIT Stata Center Data Set](https://projects.csail.mit.edu/stata/downloads.html)

This is the preferred real-world extension because its bags include a 40 Hz Hokuyo 2-D scan, 44 Hz raw wheel odometry, and 100 Hz IMU. Ground-truth laser poses are separately available for 17 bags on the official site.

```bash
python -m pip install rosbags
python code/import_rosbag.py \
  --profile mit-stata \
  --bag data/external/mit-stata/example.bag \
  --output build/mit-stata/synced.npz

python code/main.py \
  --dataset 100 \
  --synced-input build/mit-stata/synced.npz \
  --mode compare \
  --particles 100 \
  --skip-reference-map \
  --output-dir build/mit-stata/run
```

The imported motion input is derived from the bag's raw wheel-odometry pose. Heading propagation continues to use the IMU, matching the active project motion model. Ground-truth files are not yet assumed to share a universal layout; retain them alongside the bag until their particular format and frame convention have been inspected.

## uHumans2

Official source: [MIT SPARK uHumans2](https://web.mit.edu/sparklab/datasets/uHumans2/)

Start with an office sequence with zero humans for a static mapping baseline, then repeat with dynamic agents. The bags provide front 2-D LiDAR, noisy and clean IMU, stereo/depth images, and exact simulator odometry.

uHumans2 does **not** provide independent wheel odometry. Its `/tesse/odom` topic is ground truth. Consequently, the importer refuses to use it as motion input unless explicitly acknowledged:

```bash
python code/import_rosbag.py \
  --profile uhumans2 \
  --bag data/external/uhumans2/uHumans2_office_s1_00h.bag \
  --output build/uhumans2/office_00h.npz \
  --allow-ground-truth-controls

python code/main.py \
  --dataset 200 \
  --synced-input build/uhumans2/office_00h.npz \
  --mode compare \
  --particles 100 \
  --skip-reference-map \
  --output-dir build/uhumans2/run
```

That configuration is useful for exercising mapping under static/dynamic scans, but it is **not** a valid localization-accuracy benchmark because ground truth drives the motion proposal. The imported file records `controls_are_ground_truth=true` and retains `ground_truth_pose` to make this limitation auditable.

## Canonical output

`import_rosbag.py` extracts only the streams used by the planar SLAM pipeline and writes the same synchronized NPZ contract as the original datasets. It also records the source profile and whether ground truth was used as a control. Camera topics remain in the original bag for a future texture-mapping adapter.
