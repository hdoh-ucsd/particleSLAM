"""Regression tests for SE(2), ICP, and pose-graph optimization."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from config import LidarConfig, PoseGraphConfig
from pose_graph import (
    compose_pose,
    optimize_pose_graph,
    relative_pose,
    run_icp,
    transform_points,
)


class PoseMathTests(unittest.TestCase):
    def test_relative_pose_reconstructs_target(self):
        origin = np.array([1.0, 2.0, 0.4])
        target = np.array([3.0, -1.0, -0.2])
        reconstructed = compose_pose(origin, relative_pose(origin, target))
        np.testing.assert_allclose(reconstructed, target, atol=1e-10)

    def test_icp_recovers_known_alignment(self):
        target = np.array(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [1.5, 1.0]] * 4
        )
        expected = np.array([0.3, -0.2, 0.1])
        inverse = relative_pose(expected, np.zeros(3))
        source = transform_points(target, inverse)
        estimate, rmse, overlap = run_icp(
            source,
            target,
            np.array([0.25, -0.15, 0.08]),
            PoseGraphConfig(icp_max_correspondence_distance=1.0),
        )
        np.testing.assert_allclose(estimate, expected, atol=1e-3)
        self.assertLess(rmse, 1e-3)
        self.assertGreater(overlap, 0.99)


class PoseGraphTests(unittest.TestCase):
    def test_sequential_graph_preserves_trajectory_without_closures(self):
        trajectory = np.column_stack(
            (np.linspace(0.0, 1.0, 6), np.zeros(6), np.zeros(6))
        )
        scans = np.full((12, 7), 100.0)
        result = optimize_pose_graph(
            trajectory,
            scans,
            LidarConfig(),
            PoseGraphConfig(
                keyframe_interval=1,
                fixed_loop_interval=100,
                minimum_loop_separation=100,
            ),
        )
        np.testing.assert_allclose(result.optimized_trajectory, trajectory, atol=1e-8)
        self.assertEqual(len(result.loop_closures), 0)


if __name__ == "__main__":
    unittest.main()
