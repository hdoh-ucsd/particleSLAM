"""Focused regression tests for sensor processing and particle weighting."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from config import LidarConfig, ParticleFilterConfig, RobotConfig
from dataset_utils import low_pass_filter, synchronize_dataset
from odom import DifferentialDrive
from particle_filter_cpu import measurement_update_cpu


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


if __name__ == "__main__":
    unittest.main()
