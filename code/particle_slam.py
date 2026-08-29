"""Backward-compatible launcher for the particleSLAM command-line pipeline.

New code should invoke ``main.py`` directly. This module remains so existing
commands that call ``python particle_slam.py`` continue to work.
"""

from main import main


if __name__ == "__main__":
    raise SystemExit(main())
