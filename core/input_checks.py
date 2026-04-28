#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Input validation helpers (no collision).

Centralize checks for scripts like `tube_throw.py` so constraints evolve in one place.
All functions raise `ValueError` with a human-readable message on failure.
"""

from __future__ import annotations

import numpy as np

from .throw_params import (
    EE_VEL_MAX,
    EE_WORKSPACE_R_MAX,
    EE_WORKSPACE_Z_MIN,
    EE_WORKSPACE_Z_MAX,
)


def validate_ee_position_world(p_xyz: np.ndarray, *, name: str = "ee_pos_w") -> None:
    p_xyz = np.asarray(p_xyz, dtype=float).reshape(3)
    r = float(np.linalg.norm(p_xyz))
    z = float(p_xyz[2])

    if r > float(EE_WORKSPACE_R_MAX) + 1e-12:
        raise ValueError(
            f"{name} outside conservative workspace: ||p||={r:.4g} > "
            f"EE_WORKSPACE_R_MAX={float(EE_WORKSPACE_R_MAX):.4g} (p={p_xyz.tolist()})"
        )
    if z < float(EE_WORKSPACE_Z_MIN) - 1e-12 or z > float(EE_WORKSPACE_Z_MAX) + 1e-12:
        raise ValueError(
            f"{name} outside conservative workspace: z={z:.4g} not in "
            f"[{float(EE_WORKSPACE_Z_MIN):.4g}, {float(EE_WORKSPACE_Z_MAX):.4g}] (p={p_xyz.tolist()})"
        )


def validate_ee_linear_velocity(v_xyz: np.ndarray, *, name: str = "ee_vel_w", v_max: float = EE_VEL_MAX) -> None:
    v_xyz = np.asarray(v_xyz, dtype=float).reshape(3)
    speed = float(np.linalg.norm(v_xyz))
    if speed > float(v_max) + 1e-12:
        raise ValueError(f"{name} too large: ||v||={speed:.4g} > {float(v_max):.4g} (v={v_xyz.tolist()})")


def validate_release_pose(release_xyz: np.ndarray) -> None:
    validate_ee_position_world(release_xyz, name="release_pose_w")

