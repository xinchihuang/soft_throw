#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Minimal throw controller.

Inputs: target world XYZ and hard-coded (r, z, rdot, zdot) in yaw-aligned throwing plane.
Outputs: joint position/velocity targets for 7-DoF arm (only joints 2,4,6 move).
"""

import math
from dataclasses import dataclass
from typing import Tuple
import numpy as np

from core.throw_params import (
    INIT_ARM,
    JOINT0_YAW_LIMIT,
    QDOT_LIMITS_7,
    QDDOT_LIMITS_7,
    Q_LIMITS_7,
    DT_CONTROL,
)
from core.kinematics_pin import PinKinematics


# Hard-coded (r, z, rdot, zdot) in throwing plane (user-specified baseline)
HARD_R = 1
HARD_Z = 1.75
HARD_RDOT = 1.2
HARD_ZDOT = 1

# Robot base position in world (used as origin for r,z)
BASE_POS_W = np.array([0.0, 0.0, 0.0], dtype=float)

# Duration to apply throw command (seconds)
THROW_DURATION_SEC = 0.6


@dataclass
class ThrowPlan:
    target_xyz: np.ndarray
    yaw: float
    throw_start_q: np.ndarray
    release_pos_w: np.ndarray
    release_vel_w: np.ndarray


class MinimalThrowController:
    def __init__(self):
        self.pin = PinKinematics()
        self.plan = None
        self._qdot_prev7 = np.zeros(7, dtype=float)

    def plan_throw(self, target_xyz: np.ndarray) -> ThrowPlan:
        target_xyz = np.asarray(target_xyz, dtype=float).reshape(3)
        # yaw based on direction from robot base to target
        tgt_rel = target_xyz - BASE_POS_W
        yaw = math.atan2(float(tgt_rel[1]), float(tgt_rel[0]))
        yaw = float(np.clip(yaw, -JOINT0_YAW_LIMIT, JOINT0_YAW_LIMIT))

        throw_start_q = INIT_ARM.copy()
        throw_start_q[0] = yaw

        e_r = np.array([math.cos(yaw), math.sin(yaw), 0.0], dtype=float)
        e_z = np.array([0.0, 0.0, 1.0], dtype=float)

        release_pos_w = BASE_POS_W + HARD_R * e_r + HARD_Z * e_z
        release_vel_w = HARD_RDOT * e_r + HARD_ZDOT * e_z

        self.plan = ThrowPlan(
            target_xyz=target_xyz,
            yaw=yaw,
            throw_start_q=throw_start_q,
            release_pos_w=release_pos_w,
            release_vel_w=release_vel_w,
        )
        self._qdot_prev7[:] = 0.0
        return self.plan

    def get_init_q(self) -> np.ndarray:
        return INIT_ARM.copy()

    def step_throw(self, q_cmd7: np.ndarray, t: float) -> Tuple[np.ndarray, np.ndarray]:
        """Return (q_cmd7_next, qdot7) for time t during throw phase."""
        assert self.plan is not None, "Call plan_throw() first."
        # constant end-effector velocity command
        v_cmd = self.plan.release_vel_w
        qdot7_des = self.pin.compute_qdot7_for_v(q_cmd7, v_cmd)

        # Acceleration limits (smoothness)
        max_dq = QDDOT_LIMITS_7 * DT_CONTROL
        dq = qdot7_des - self._qdot_prev7
        dq = np.clip(dq, -max_dq, max_dq)
        qdot7 = self._qdot_prev7 + dq

        # Velocity limits
        qdot7 = np.clip(qdot7, -QDOT_LIMITS_7, QDOT_LIMITS_7)

        q_next = np.asarray(q_cmd7, dtype=float).copy()
        q_next = q_next + qdot7 * DT_CONTROL

        # Position limits
        q_min = Q_LIMITS_7[:, 0]
        q_max = Q_LIMITS_7[:, 1]
        q_next = np.clip(q_next, q_min, q_max)

        self._qdot_prev7 = qdot7.copy()

        return q_next, qdot7
