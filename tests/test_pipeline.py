"""Focused regression tests for sensor processing and particle weighting."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from config import LidarConfig, MapConfig, ParticleFilterConfig, RobotConfig
from dataset_utils import low_pass_filter, synchronize_dataset
from odom import DifferentialDrive
from dead_reckoning import run_dead_reckoning
from particle_filter_cpu import (
    measurement_update_cpu,
    particle_filter_cpu,
    resample_particles,
)


class OdometryTests(unittest.TestCase):
    def test_encoder_readings_are_per_sample_increments(self):
        config = RobotConfig()
        timestamps = np.array([0.0, 0.1, 0.2])
        counts = np.array(
            [
                [0.0, 1.0, 1.0],
                [0.0, 1.0, 1.0],
                [0.0, 1.0, 1.0],
                [0.0, 1.0, 1.0],
            ]
        )

        trajectory = DifferentialDrive(config).integrate_odometry(timestamps, counts)

        self.assertAlmostEqual(trajectory[0, -1], 2.0 * config.tick_to_meter)
        self.assertAlmostEqual(trajectory[1, -1], 0.0)
        self.assertAlmostEqual(trajectory[2, -1], 0.0)


class SynchronizationTests(unittest.TestCase):
    def test_cumulative_counts_are_interpolated_then_differenced(self):
        timestamps = np.array([0.0, 0.1, 0.2])
        counts = np.ones((4, 3))
        data = {
            "encoder_counts": counts,
            "encoder_stamps": timestamps,
            "pose_enc": np.zeros((3, 3)),
            "lidar_stamps": timestamps,
            "lidar_ranges": np.ones((2, 3)),
            "imu_stamps": timestamps,
            "imu_angular_velocity": np.zeros((3, 3)),
            "imu_linear_acceleration": np.zeros((3, 3)),
            "lidar_angle_min": np.array(0.0),
            "lidar_angle_increment": np.array(0.1),
            "lidar_range_min": np.array(0.05),
            "lidar_range_max": np.array(30.0),
        }

        synchronized = synchronize_dataset(data)

        np.testing.assert_allclose(
            synchronized["encoder_counts"],
            np.column_stack((np.zeros(4), np.ones(4), np.ones(4))),
        )

    def test_low_pass_filter_attenuates_alternating_noise(self):
        timestamps = np.arange(20) * 0.01
        signal = np.tile([1.0, -1.0], 10)
        filtered = low_pass_filter(signal, timestamps, cutoff_hz=10.0)
        self.assertLess(np.std(filtered[5:]), np.std(signal[5:]))


class MeasurementUpdateTests(unittest.TestCase):
    def test_map_aligned_particle_receives_more_weight(self):
        grid = np.zeros((5, 5), dtype=np.float32)
        grid[3, 2] = 5.0
        particles = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        coordinates = np.arange(-2.0, 3.0)
        lidar_config = LidarConfig(
            x=0.0,
            rmax_used=2.0,
            angle_min=0.0,
            angle_increment=1.0,
        )
        filter_config = ParticleFilterConfig(
            num_particles=2,
            correlation_xy_window=0.0,
            correlation_xy_step=1.0,
            correlation_yaw_window=0.0,
            correlation_yaw_step=1.0,
            correlation_beam_stride=1,
            likelihood_temperature=1.0,
            min_valid_beams=1,
        )

        weights = measurement_update_cpu(
            particles,
            np.array([1.0]),
            grid,
            coordinates,
            coordinates,
            lidar_config,
            filter_config,
        )

        self.assertGreater(weights[0], weights[1])
        self.assertAlmostEqual(float(weights.sum()), 1.0)

    def test_systematic_resampling_favors_dominant_particle(self):
        particles = np.arange(12, dtype=float).reshape(4, 3)
        weights = np.array([0.85, 0.05, 0.05, 0.05])
        sampled = resample_particles(
            particles, weights, np.random.default_rng(7)
        )
        dominant_count = np.count_nonzero(np.all(sampled == particles[0], axis=1))
        self.assertGreaterEqual(dominant_count, 3)


class DeadReckoningTests(unittest.TestCase):
    def test_straight_motion_produces_forward_trajectory(self):
        data = {
            "sync_times": np.array([0.0, 0.1, 0.2]),
            "encoder_counts": np.column_stack(
                (np.zeros(4), np.ones(4), np.ones(4))
            ),
            "imu_angular_velocity": np.zeros((3, 3)),
            "lidar": np.full((2, 3), 100.0),
        }
        trajectory, _grid = run_dead_reckoning(data)
        self.assertGreater(trajectory[-1, 0], 0.0)
        self.assertAlmostEqual(trajectory[-1, 1], 0.0)
        self.assertAlmostEqual(trajectory[-1, 2], 0.0)


class PipelineDiagnosticsTests(unittest.TestCase):
    def test_particle_filter_records_each_diagnostic_update(self):
        data = {
            "sync_times": np.array([0.0, 0.1, 0.2]),
            "encoder_counts": np.zeros((4, 3)),
            "imu_angular_velocity": np.zeros((3, 3)),
            "lidar": np.full((4, 3), 1.0),
        }
        diagnostics = {}
        trajectory, _grid, _particles = particle_filter_cpu(
            data,
            ParticleFilterConfig(
                num_particles=5,
                correlation_xy_window=0.0,
                correlation_xy_step=1.0,
                correlation_yaw_window=0.0,
                correlation_yaw_step=1.0,
                correlation_beam_stride=1,
                min_valid_beams=1,
            ),
            MapConfig(res=1.0, xmin=-2.0, xmax=2.0, ymin=-2.0, ymax=2.0),
            LidarConfig(x=0.0, angle_min=0.0, angle_increment=0.1),
            diagnostics=diagnostics,
        )
        self.assertEqual(trajectory.shape, (2, 3))
        for values in diagnostics.values():
            self.assertEqual(len(values), 2)


if __name__ == "__main__":
    unittest.main()
