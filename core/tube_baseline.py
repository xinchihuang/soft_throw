#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
from typing import Tuple

from .throw_params import (
    BALLISTIC_T_MIN,
    BALLISTIC_T_MAX,
    BALLISTIC_T_STEP,
    TUBE_CONE_DEG,
    TUBE_V_MIN_RATIO,
    TUBE_V_MAX_RATIO,
    EE_VEL_MAX,
    TUBE_DECEL_SEC,
)


def solve_ballistic_velocity(p_r: np.ndarray, p_t: np.ndarray) -> Tuple[np.ndarray, float]:
    """Solve ballistic release velocity v given release point p_r and target point p_t in 3D."""
    p_r = np.asarray(p_r, dtype=float).reshape(3)
    p_t = np.asarray(p_t, dtype=float).reshape(3)
    g = 9.81
    best = None  # (cost, v, T)
    for T in np.arange(BALLISTIC_T_MIN, BALLISTIC_T_MAX + 1e-9, BALLISTIC_T_STEP):
        d = p_t - p_r
        vx, vy = d[0] / T, d[1] / T
        vz = (d[2] + 0.5 * g * T * T) / T
        v = np.array([vx, vy, vz], dtype=float)
        speed = float(np.linalg.norm(v))
        if speed < 1e-6:
            continue
        if speed > EE_VEL_MAX * 0.98:
            continue
        cost = speed + 0.2 * T
        if (best is None) or (cost < best[0]):
            best = (cost, v, float(T))
    if best is None:
        T = 0.45
        d = p_t - p_r
        vx, vy = d[0] / T, d[1] / T
        vz = (d[2] + 0.5 * g * T * T) / T
        v = np.array([vx, vy, vz], dtype=float)
        speed = float(np.linalg.norm(v))
        if speed > EE_VEL_MAX and speed > 1e-6:
            v = v * (EE_VEL_MAX / speed)
        return v, float(T)
    return best[1], best[2]


def smoothstep01(x: float) -> float:
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def project_velocity_to_tube(
    v: np.ndarray,
    v_star: np.ndarray,
    v_hat: np.ndarray,
    tube_cone_deg: float = TUBE_CONE_DEG,
    tube_v_min_ratio: float = TUBE_V_MIN_RATIO,
    tube_v_max_ratio: float = TUBE_V_MAX_RATIO,
    ee_vel_max: float = EE_VEL_MAX,
) -> np.ndarray:
    """Project/clip a velocity into tube constraints (cone + speed bounds)."""
    v = np.asarray(v, dtype=float).reshape(3)
    v_star = np.asarray(v_star, dtype=float).reshape(3)
    v_hat = np.asarray(v_hat, dtype=float).reshape(3)
    v_star_norm = float(np.linalg.norm(v_star))

    vmin = tube_v_min_ratio * v_star_norm
    vmax = tube_v_max_ratio * v_star_norm
    vmax = min(vmax, ee_vel_max)

    speed = float(np.linalg.norm(v))
    if speed < 1e-9:
        return v_hat * max(0.0, min(vmin, ee_vel_max))

    # cone clamp
    cos_alpha = math.cos(math.radians(tube_cone_deg))
    dir_v = v / speed
    cosang = float(np.dot(dir_v, v_hat))
    if cosang < cos_alpha:
        parallel = cosang * v_hat
        perp = dir_v - parallel
        perp_norm = float(np.linalg.norm(perp))
        if perp_norm < 1e-9:
            dir_clamped = v_hat
        else:
            perp_unit = perp / perp_norm
            # boundary direction
            sin_alpha = math.sin(math.radians(tube_cone_deg))
            dir_clamped = cos_alpha * v_hat + sin_alpha * perp_unit
            dir_clamped = dir_clamped / float(np.linalg.norm(dir_clamped))
        v = dir_clamped * speed

    # speed clamp
    speed = float(np.linalg.norm(v))
    speed_clamped = float(np.clip(speed, vmin, vmax))
    if speed > 1e-9:
        v = v * (speed_clamped / speed)
    return v


def tube_velocity_command(
    t: float,
    v_star: np.ndarray,
    v_hat: np.ndarray,
    tube_t0: float,
    tube_t1: float,
    tube_decel_sec: float = TUBE_DECEL_SEC,
) -> np.ndarray:
    """Baseline desired EE velocity as a function of episode time."""
    t = float(t)
    v_star = np.asarray(v_star, dtype=float).reshape(3)
    if t <= tube_t0:
        s = smoothstep01(t / max(1e-6, tube_t0))
        v = s * v_star
    elif t <= tube_t1:
        v = v_star.copy()
    else:
        td = t - tube_t1
        if td >= tube_decel_sec:
            v = np.zeros(3, dtype=float)
        else:
            s = 1.0 - smoothstep01(td / max(1e-6, tube_decel_sec))
            v = s * v_star
    return project_velocity_to_tube(v, v_star, v_hat)
