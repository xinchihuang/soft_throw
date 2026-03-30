#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
from typing import Optional

from .throw_params import (
    TUBE_ACCELERATION_T_BACK,
    TUBE_ACCELERATION_DT,
    TUBE_ACCELERATION_N_LANDING,
    TUBE_ACCELERATION_RDOT_MIN,
    TUBE_ACCELERATION_RDOT_MAX,
    TUBE_ACCELERATION_ZDOT_MIN,
    TUBE_ACCELERATION_ZDOT_MAX,
    TUBE_ACCELERATION_EPS_R,
    TUBE_ACCELERATION_EPS_Z,
    TUBE_ACCELERATION_MAX_ABS_RDOT,
    TUBE_ACCELERATION_MAX_ABS_ZDOT,
    TUBE_ACCELERATION_SEED,
    TUBE_ACCELERATION_MAX_CANDIDATES,
    TUBE_ACCELERATION_Z_MIN,
    TUBE_ACCELERATION_Z_MAX,
    TUBE_ACCELERATION_Z_PREF,
    TUBE_ACCELERATION_MAX_RELEASE_OFFSET,
    TUBE_ACCELERATION_W_RANGE,
    TUBE_ACCELERATION_W_MARGIN,
    TUBE_ACCELERATION_W_HEIGHT,
    TUBE_ACCELERATION_W_OFFSET,
    TUBE_ACCELERATION_DEBUG,
)

_TUBE_ACCELERATION_BRT_CACHE = None


def _unit(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        return v * 0.0
    return v / n


def _tube_acceleration_sample_landing(n: int, rng: np.random.Generator) -> np.ndarray:
    r = rng.uniform(-TUBE_ACCELERATION_EPS_R, TUBE_ACCELERATION_EPS_R, size=n)
    z = rng.uniform(-TUBE_ACCELERATION_EPS_Z, TUBE_ACCELERATION_EPS_Z, size=n)
    rdot = rng.uniform(TUBE_ACCELERATION_RDOT_MIN, TUBE_ACCELERATION_RDOT_MAX, size=n)
    zdot = rng.uniform(TUBE_ACCELERATION_ZDOT_MIN, TUBE_ACCELERATION_ZDOT_MAX, size=n)
    return np.stack([r, z, rdot, zdot], axis=1).astype(float)


def tube_acceleration_generate_brt_states() -> np.ndarray:
    global _TUBE_ACCELERATION_BRT_CACHE
    if _TUBE_ACCELERATION_BRT_CACHE is not None:
        return _TUBE_ACCELERATION_BRT_CACHE

    rng = np.random.default_rng(TUBE_ACCELERATION_SEED)
    landing = _tube_acceleration_sample_landing(TUBE_ACCELERATION_N_LANDING, rng)
    t_steps = int(max(1, round(TUBE_ACCELERATION_T_BACK / TUBE_ACCELERATION_DT)))
    g = 9.81
    trajs = []
    for x0 in landing:
        r, z, rdot, zdot = [float(v) for v in x0]
        for _ in range(t_steps + 1):
            trajs.append([r, z, rdot, zdot])
            # backward integrate (projectile): reverse time for position, gravity still increases zdot
            r += -rdot * TUBE_ACCELERATION_DT
            z += -zdot * TUBE_ACCELERATION_DT
            zdot += g * TUBE_ACCELERATION_DT

    X = np.array(trajs, dtype=float)
    mask = (np.abs(X[:, 2]) <= TUBE_ACCELERATION_MAX_ABS_RDOT) & (np.abs(X[:, 3]) <= TUBE_ACCELERATION_MAX_ABS_ZDOT)
    X = X[mask]
    _TUBE_ACCELERATION_BRT_CACHE = X
    return X


def tube_acceleration_vmax_for_direction(J_lin: np.ndarray, qdot_limits: np.ndarray, d_world: np.ndarray) -> float:
    # J_lin: 3x3 linear Jacobian for learned joints in world-aligned frame
    d_world = _unit(np.asarray(d_world, dtype=float).reshape(3))
    if float(np.linalg.norm(d_world)) < 1e-9:
        return 0.0
    J_pinv = np.linalg.pinv(J_lin, rcond=1e-6)
    qdot_dir = J_pinv @ d_world
    s_hi = float("inf")
    for i in range(3):
        a = float(qdot_dir[i])
        if abs(a) < 1e-9:
            continue
        limit = float(qdot_limits[i])
        s_hi = min(s_hi, limit / abs(a))
    if not math.isfinite(s_hi) or s_hi <= 0.0:
        return 0.0
    return float(s_hi)


def tube_acceleration_select_release(
    target_xyz: np.ndarray,
    yaw: float,
    p_release_nom: np.ndarray,
    J_lin: np.ndarray,
    qdot_limits: np.ndarray,
) -> Optional[dict]:
    """
    Tube-acceleration candidate selection in the yaw-aligned throwing plane.
    Returns dict with release_pos_w, release_vel_w, brt_state, score, vmax, speed, margin.
    """
    X = tube_acceleration_generate_brt_states()
    if X.size == 0:
        return None

    # fixed plane by joint0 yaw
    e_r = np.array([math.cos(yaw), math.sin(yaw), 0.0], dtype=float)
    e_z = np.array([0.0, 0.0, 1.0], dtype=float)
    target_xyz = np.asarray(target_xyz, dtype=float).reshape(3)
    p_release_nom = np.asarray(p_release_nom, dtype=float).reshape(3)

    if X.shape[0] > TUBE_ACCELERATION_MAX_CANDIDATES:
        stride = max(1, X.shape[0] // TUBE_ACCELERATION_MAX_CANDIDATES)
        X = X[::stride]

    best = None
    best_near = None
    cnt_total = 0
    cnt_z = 0
    cnt_vmax0 = 0
    cnt_speed = 0
    cnt_offset = 0
    z_pass = []
    for x in X:
        cnt_total += 1
        r, z, rdot, zdot = [float(v) for v in x]
        z_out = (z < TUBE_ACCELERATION_Z_MIN) or (z > TUBE_ACCELERATION_Z_MAX)
        if z_out:
            cnt_z += 1
        else:
            z_pass.append(z)

        speed = math.hypot(rdot, zdot)
        if speed < 1e-6:
            continue

        # direction in plane
        gamma = math.atan2(zdot, rdot)
        d = math.cos(gamma) * e_r + math.sin(gamma) * e_z
        vmax = tube_acceleration_vmax_for_direction(J_lin, qdot_limits, d)
        vmax0 = vmax <= 1e-6
        if vmax0:
            cnt_vmax0 += 1
        speed_gt = speed > vmax
        if speed_gt:
            cnt_speed += 1

        # map to world release
        E = target_xyz - r * e_r + z * e_z
        v = rdot * e_r + zdot * e_z
        offset = float(np.linalg.norm(E - p_release_nom))
        offset_out = offset > TUBE_ACCELERATION_MAX_RELEASE_OFFSET
        if offset_out:
            cnt_offset += 1

        # track nearest candidate even if infeasible
        if TUBE_ACCELERATION_DEBUG:
            pen = 0.0
            if z_out:
                z_clamped = min(max(z, TUBE_ACCELERATION_Z_MIN), TUBE_ACCELERATION_Z_MAX)
                pen += abs(z - z_clamped)
            if vmax0:
                pen += 10.0
            if speed_gt:
                pen += max(0.0, speed - vmax)
            if offset_out:
                pen += max(0.0, offset - TUBE_ACCELERATION_MAX_RELEASE_OFFSET)
            if (best_near is None) or (pen < best_near["penalty"]):
                best_near = {
                    "brt_state": np.array([r, z, rdot, zdot], dtype=float),
                    "release_pos_w": E,
                    "release_vel_w": v,
                    "vmax": float(vmax),
                    "speed": float(speed),
                    "offset": float(offset),
                    "penalty": float(pen),
                    "z_out": bool(z_out),
                    "vmax0": bool(vmax0),
                    "speed_gt_vmax": bool(speed_gt),
                    "offset_out": bool(offset_out),
                }

        # feasibility filters
        if z_out or vmax0 or speed_gt or offset_out:
            continue

        margin = vmax - speed
        score = (
            TUBE_ACCELERATION_W_RANGE * abs(r) +
            TUBE_ACCELERATION_W_MARGIN * (1.0 / (1e-6 + margin)) +
            TUBE_ACCELERATION_W_HEIGHT * abs(z - TUBE_ACCELERATION_Z_PREF) +
            TUBE_ACCELERATION_W_OFFSET * offset
        )
        cand = {
            "release_pos_w": E,
            "release_vel_w": v,
            "brt_state": np.array([r, z, rdot, zdot], dtype=float),
            "score": float(score),
            "vmax": float(vmax),
            "speed": float(speed),
            "margin": float(margin),
        }
        if (best is None) or (cand["score"] < best["score"]):
            best = cand

    if (best is None) and TUBE_ACCELERATION_DEBUG:
        z_all = X[:, 1] if X.size > 0 else np.zeros((0,), dtype=float)
        z_min = float(np.min(z_all)) if z_all.size > 0 else 0.0
        z_max = float(np.max(z_all)) if z_all.size > 0 else 0.0
        if z_all.size > 0 and z_max > z_min:
            bins = np.linspace(z_min, z_max, 11)
            hist_all, _ = np.histogram(z_all, bins=bins)
            hist_pass, _ = np.histogram(np.asarray(z_pass, dtype=float), bins=bins)
            bins_list = [float(b) for b in bins]
            hist_all_list = [int(v) for v in hist_all]
            hist_pass_list = [int(v) for v in hist_pass]
        else:
            bins_list = []
            hist_all_list = []
            hist_pass_list = []
        print(
            f"[tube_acceleration] no candidate: total={cnt_total} "
            f"z_out={cnt_z} vmax0={cnt_vmax0} speed_gt_vmax={cnt_speed} offset_out={cnt_offset}",
            flush=True,
        )
        print(
            f"[tube_acceleration] z_range_all=[{z_min:.3f}, {z_max:.3f}] "
            f"z_bins={bins_list} z_hist_all={hist_all_list} z_hist_pass={hist_pass_list}",
            flush=True,
        )
        if best_near is not None:
            r, z, rdot, zdot = [float(v) for v in best_near["brt_state"]]
            print(
                "[tube_acceleration] nearest candidate "
                f"r={r:.3f} z={z:.3f} rdot={rdot:.3f} zdot={zdot:.3f} "
                f"speed={best_near['speed']:.3f} vmax={best_near['vmax']:.3f} "
                f"offset={best_near['offset']:.3f} penalty={best_near['penalty']:.3f} "
                f"flags=z_out:{best_near['z_out']} vmax0:{best_near['vmax0']} "
                f"speed_gt_vmax:{best_near['speed_gt_vmax']} offset_out:{best_near['offset_out']}",
                flush=True,
            )
    if (best is not None) and TUBE_ACCELERATION_DEBUG:
        r, z, rdot, zdot = [float(v) for v in best["brt_state"]]
        print(
            f"[tube_acceleration] selected r={r:.3f} z={z:.3f} rdot={rdot:.3f} zdot={zdot:.3f}",
            flush=True,
        )
    return best
