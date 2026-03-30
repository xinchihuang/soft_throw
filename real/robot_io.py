#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Minimal real-robot IO interface.

Default implementation is a dry-run printer. Replace methods with your robot API.
"""

from typing import Tuple
import numpy as np

from core.throw_params import INIT_ARM


class RealRobotIO:
    """Dry-run robot IO. Override methods to connect to real hardware."""

    def __init__(self, print_only: bool = True):
        self.print_only = bool(print_only)
        self._q7 = INIT_ARM.copy()

    def get_joint_pos7(self) -> np.ndarray:
        """Return current 7-DOF joint positions in radians."""
        return self._q7.copy()

    def send_joint_targets(self, q_cmd7: np.ndarray, qdot7: np.ndarray) -> None:
        """Send joint position/velocity targets (radians, rad/s)."""
        if self.print_only:
            vals = []
            for i in range(7):
                vals.append(float(np.degrees(q_cmd7[i])))
                vals.append(float(np.degrees(qdot7[i])))
            print(f"[real_robot] pos_deg/vel_deg_s x7 = {vals}", flush=True)
        self._q7 = np.asarray(q_cmd7, dtype=float).copy()

    def get_ball_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (ball_pos_xyz, ball_vel_xyz). Replace with vision/estimator."""
        return np.zeros(3, dtype=float), np.zeros(3, dtype=float)
