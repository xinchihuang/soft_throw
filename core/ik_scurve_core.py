#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Jerk-limited S-curve trajectory generation for joint-space control.

This module is intentionally separate from `core/ik_poly5_core.py`.

Design goals:
- Enforce per-joint limits on |qdot|, |qddot|, |qdddot| (velocity/accel/jerk).
- Support boundary velocities (v0, v1) with boundary accelerations fixed to 0.
- Provide multi-joint synchronization (common time base) like `poly5_trajectory`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np


@dataclass(frozen=True)
class SCurveLimits:
    vmax: float
    amax: float
    jmax: float


def stopping_distance_jerk_limited(v0: float, *, amax: float, jmax: float) -> float:
    """Minimal stopping distance for decelerating from speed |v0| to 0.

    Assumes start/end acceleration are 0 and uses an S-curve (piecewise-constant jerk).
    Returns a non-negative distance.
    """
    v0 = abs(float(v0))
    if v0 <= 0.0:
        return 0.0
    t_j, t_a = _solve_accel_phase_times(v0, float(amax), float(jmax))
    dq, _dt = _accel_phase_displacement(0.0, v0, t_j, t_a, float(jmax))
    return float(abs(dq))


def _as_1d(x) -> float:
    return float(np.asarray(x, dtype=float).reshape(()))


def _solve_accel_phase_times(dv: float, amax: float, jmax: float) -> Tuple[float, float]:
    """Minimal-time accel phase (start/end accel=0) for a velocity change dv>=0.

    Returns (t_j, t_a) where:
    - jerk +j for t_j, then accel hold for t_a, then jerk -j for t_j.
    - if t_a==0 -> triangular acceleration profile (no accel plateau).
    """
    dv = float(dv)
    amax = float(amax)
    jmax = float(jmax)
    if dv <= 0.0:
        return 0.0, 0.0
    if amax <= 0.0 or jmax <= 0.0:
        raise ValueError("amax and jmax must be positive")

    dv_tri_max = (amax * amax) / jmax
    if dv >= dv_tri_max:
        t_j = amax / jmax
        t_a = (dv - dv_tri_max) / amax
        return float(t_j), float(t_a)
    t_j = np.sqrt(dv / jmax)
    return float(t_j), 0.0


def _accel_phase_displacement(v0: float, dv: float, t_j: float, t_a: float, j: float) -> Tuple[float, float]:
    """Displacement/time for accel phase, starting at velocity v0 and accel=0.

    Uses jerk profile: +j for t_j, 0 for t_a, -j for t_j.
    Returns (dq, dt).
    """
    v0 = float(v0)
    dv = float(dv)
    t_j = float(t_j)
    t_a = float(t_a)
    j = float(j)
    if dv <= 0.0:
        return 0.0, 0.0
    if t_j <= 0.0:
        return 0.0, 0.0

    if t_a <= 0.0:
        # triangular (no accel plateau); dv = j * t_j^2
        # dq = v0 * (2 t_j) + j * t_j^3
        dq = v0 * (2.0 * t_j) + j * (t_j ** 3)
        return float(dq), float(2.0 * t_j)

    # trapezoidal accel profile; amax = j * t_j, dv = j*t_j^2 + amax*t_a
    amax = j * t_j
    # segment 1 (+j)
    dq1 = v0 * t_j + (j / 6.0) * (t_j ** 3)
    v1 = v0 + 0.5 * j * (t_j ** 2)
    # segment 2 (a=amax)
    dq2 = v1 * t_a + 0.5 * amax * (t_a ** 2)
    v2 = v1 + amax * t_a
    # segment 3 (-j)
    # starting accel amax, ending accel 0
    dq3 = v2 * t_j + 0.5 * amax * (t_j ** 2) - (j / 6.0) * (t_j ** 3)
    return float(dq1 + dq2 + dq3), float(2.0 * t_j + t_a)


def _sample_profile(
    *,
    q0: float,
    v0: float,
    a0: float,
    segments: Iterable[Tuple[float, float]],
    dt: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample piecewise-constant jerk segments.

    segments: iterable of (duration, jerk_value).
    """
    dt = float(dt)
    if dt <= 0.0:
        raise ValueError("dt must be > 0")
    q = float(q0)
    v = float(v0)
    a = float(a0)

    t_hist = [0.0]
    q_hist = [q]
    v_hist = [v]
    a_hist = [a]

    t_abs = 0.0
    for duration, jerk in segments:
        duration = float(duration)
        jerk = float(jerk)
        if duration <= 0.0:
            continue
        n = max(1, int(np.ceil(duration / dt)))
        # Use fixed dt samples, but clamp final to end exactly at segment boundary.
        for k in range(n):
            h = min(dt, duration - k * dt)
            if h <= 0.0:
                break
            # integrate jerk over h
            q = q + v * h + 0.5 * a * (h ** 2) + (1.0 / 6.0) * jerk * (h ** 3)
            v = v + a * h + 0.5 * jerk * (h ** 2)
            a = a + jerk * h
            t_abs += h
            t_hist.append(t_abs)
            q_hist.append(q)
            v_hist.append(v)
            a_hist.append(a)

    return (
        np.asarray(t_hist, dtype=float),
        np.asarray(q_hist, dtype=float),
        np.asarray(v_hist, dtype=float),
        np.asarray(a_hist, dtype=float),
    )


def scurve_trajectory_1d(
    *,
    q_start: float,
    q_goal: float,
    v_start: float,
    v_goal: float,
    limits: SCurveLimits,
    dt: float,
    a_start: float = 0.0,
    a_goal: float = 0.0,
    time_optimal: bool = True,
    max_iter: int = 80,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate a jerk-limited S-curve for 1 DOF.

    Boundary accelerations are supported but the intended use is a_start=a_goal=0.
    """
    if abs(float(a_start)) > 1e-9 or abs(float(a_goal)) > 1e-9:
        raise NotImplementedError("Only a_start=a_goal=0 is supported in this implementation")

    q0 = _as_1d(q_start)
    q1 = _as_1d(q_goal)
    v0 = _as_1d(v_start)
    v1 = _as_1d(v_goal)
    vmax = float(limits.vmax)
    amax = float(limits.amax)
    jmax = float(limits.jmax)
    if vmax <= 0.0 or amax <= 0.0 or jmax <= 0.0:
        raise ValueError("limits must be positive")

    dq = q1 - q0
    if abs(dq) < 1e-12 and abs(v0 - v1) < 1e-12:
        return np.array([0.0]), np.array([q0]), np.array([v0]), np.array([0.0])

    # Normalize direction so dq >= 0.
    s = 1.0 if dq >= 0.0 else -1.0
    dq = abs(dq)
    q0n = 0.0
    q1n = dq
    v0n = s * v0
    v1n = s * v1

    # Clamp boundary velocities to within vmax to avoid impossible constraints.
    v0n = float(np.clip(v0n, -vmax, vmax))
    v1n = float(np.clip(v1n, -vmax, vmax))
    v0n_req = float(v0n)
    v1n_req = float(v1n)

    # We solve for a feasible peak velocity v_peak in [max(v0n,v1n), vmax]
    # such that displacement matches dq with (accel to v_peak) + (cruise) + (decel).
    v_low = max(v0n, v1n)
    v_high = vmax
    if v_low > v_high + 1e-9:
        raise ValueError("Boundary velocities exceed vmax and cannot be satisfied")

    def _disp_for_vpeak(vp: float) -> Tuple[float, float, Tuple[float, float], Tuple[float, float]]:
        vp = float(vp)
        dv_acc = max(0.0, vp - v0n)
        dv_dec = max(0.0, vp - v1n)
        tja, taa = _solve_accel_phase_times(dv_acc, amax, jmax)
        tjd, tad = _solve_accel_phase_times(dv_dec, amax, jmax)
        dq_acc, dt_acc = _accel_phase_displacement(v0n, dv_acc, tja, taa, jmax)
        # decel phase from vp to v1n is equivalent to accel from v1n to vp, run backwards.
        dq_dec, dt_dec = _accel_phase_displacement(v1n, dv_dec, tjd, tad, jmax)
        # During decel, average velocity is higher, but displacement from vp down to v1n
        # equals displacement of accel from v1n up to vp with initial velocity v1n.
        # Total displacement without cruise:
        dq_min = dq_acc + dq_dec
        dt_min = dt_acc + dt_dec
        return dq_min, dt_min, (tja, taa), (tjd, tad)

    dq_min_low, _dt_low, _ta, _td = _disp_for_vpeak(v_low)
    if dq_min_low > dq + 1e-9:
        # Not enough distance to satisfy boundary velocities with a monotone profile.
        # Relax boundary velocities by scaling them down towards 0 until feasible.
        if not time_optimal:
            raise ValueError("Infeasible: not enough distance for given boundary velocities")
        scale_lo = 0.0
        scale_hi = 1.0
        v0_base = float(v0n_req)
        v1_base = float(v1n_req)

        def _feasible_for_scale(scale: float) -> bool:
            nonlocal v0n, v1n
            v0n = float(scale) * float(v0_base)
            v1n = float(scale) * float(v1_base)
            v0n = float(np.clip(v0n, -vmax, vmax))
            v1n = float(np.clip(v1n, -vmax, vmax))
            v_low_local = max(v0n, v1n)
            dq_min_local, _dt_local, _ta_local, _td_local = _disp_for_vpeak(v_low_local)
            return dq_min_local <= dq + 1e-9

        if not _feasible_for_scale(scale_hi):
            for _ in range(int(max_iter)):
                mid = 0.5 * (scale_lo + scale_hi)
                if _feasible_for_scale(mid):
                    scale_lo = mid
                else:
                    scale_hi = mid
                if (scale_hi - scale_lo) <= 1e-6:
                    break
            _feasible_for_scale(scale_lo)

        v_low = max(v0n, v1n)
        dq_min_low, _dt_low, _ta, _td = _disp_for_vpeak(v_low)

    dq_min_high, _dt_high, _ta, _td = _disp_for_vpeak(v_high)
    if dq_min_high < dq - 1e-9:
        # Need cruise at vmax.
        v_peak = v_high
        dq_min, _dt_min, (tja, taa), (tjd, tad) = _disp_for_vpeak(v_peak)
        t4 = (dq - dq_min) / max(1e-9, v_peak)
    else:
        # Find v_peak that makes dq_min(v_peak) == dq (no cruise).
        lo = v_low
        hi = v_high
        v_peak = hi
        for _ in range(int(max_iter)):
            mid = 0.5 * (lo + hi)
            dq_mid, _dt_mid, _ta, _td = _disp_for_vpeak(mid)
            if dq_mid > dq:
                hi = mid
            else:
                lo = mid
            v_peak = 0.5 * (lo + hi)
            if abs(hi - lo) <= 1e-9:
                break
        dq_min, _dt_min, (tja, taa), (tjd, tad) = _disp_for_vpeak(v_peak)
        t4 = 0.0

    # Build jerk segments.
    # Accel to v_peak:
    dv_acc = max(0.0, v_peak - v0n)
    tja, taa = _solve_accel_phase_times(dv_acc, amax, jmax)
    # Decel from v_peak to v1n:
    dv_dec = max(0.0, v_peak - v1n)
    tjd, tad = _solve_accel_phase_times(dv_dec, amax, jmax)

    seg = []
    # accel (+j, 0, -j)
    if tja > 0:
        seg.append((tja, +jmax))
        if taa > 0:
            seg.append((taa, 0.0))
        seg.append((tja, -jmax))
    # cruise
    if t4 > 0:
        seg.append((t4, 0.0))
    # decel (-j, 0, +j) (mirror, to reduce velocity)
    if tjd > 0:
        seg.append((tjd, -jmax))
        if tad > 0:
            seg.append((tad, 0.0))
        seg.append((tjd, +jmax))

    t, qn, vn, an = _sample_profile(q0=q0n, v0=v0n, a0=0.0, segments=seg, dt=dt)

    # Restore sign and absolute position.
    q = q0 + s * qn
    v = s * vn
    a = s * an

    # Avoid forcing q/v independently at the final sample; that can introduce
    # inconsistent (q, qdot, qddot) tuples. Keep the integrated terminal state,
    # but snap acceleration to 0 (should already be ~0).
    a[-1] = 0.0

    return t, q, v, a


def scurve_trajectory(
    q_start: np.ndarray,
    q_goal: np.ndarray,
    qdot_start: np.ndarray,
    qdot_goal: np.ndarray,
    qdot_limits: np.ndarray,
    qddot_limits: np.ndarray,
    qdddot_limits: np.ndarray,
    control_dt: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Multi-joint jerk-limited S-curve trajectory (synchronized)."""
    q0 = np.asarray(q_start, dtype=float).reshape(7)
    q1 = np.asarray(q_goal, dtype=float).reshape(7)
    v0 = np.asarray(qdot_start, dtype=float).reshape(7)
    v1 = np.asarray(qdot_goal, dtype=float).reshape(7)
    vmax = np.asarray(qdot_limits, dtype=float).reshape(7)
    amax = np.asarray(qddot_limits, dtype=float).reshape(7)
    jmax = np.asarray(qdddot_limits, dtype=float).reshape(7)
    dt = float(control_dt)

    # First pass: compute individual joint durations.
    per = []
    durations = []
    for j in range(7):
        lim = SCurveLimits(vmax=float(vmax[j]), amax=float(amax[j]), jmax=float(jmax[j]))
        tj, qj, vj, aj = scurve_trajectory_1d(
            q_start=float(q0[j]),
            q_goal=float(q1[j]),
            v_start=float(v0[j]),
            v_goal=float(v1[j]),
            limits=lim,
            dt=dt,
            a_start=0.0,
            a_goal=0.0,
        )
        per.append((tj, qj, vj, aj))
        durations.append(float(tj[-1]) if tj.size else 0.0)

    T = float(max(durations))
    if T <= 0.0:
        t = np.array([0.0], dtype=float)
        q = q0.reshape(1, 7)
        v = v0.reshape(1, 7)
        a = np.zeros((1, 7), dtype=float)
        return t, q, v, a

    # Second pass: resample all joints on a common time grid.
    # We synchronize by re-solving faster joints with a reduced vmax so their
    # duration matches the slowest joint, while keeping boundary velocities.
    n = max(2, int(np.ceil(T / dt)) + 1)
    t_common = np.linspace(0.0, T, n)
    q_out = np.zeros((n, 7), dtype=float)
    v_out = np.zeros((n, 7), dtype=float)
    a_out = np.zeros((n, 7), dtype=float)

    for j in range(7):
        # If joint is already the slowest (or close), interpolate directly.
        tj, qj, vj, aj = per[j]
        if not tj.size:
            q_out[:, j] = q0[j]
            v_out[:, j] = v0[j]
            a_out[:, j] = 0.0
            continue
        if abs(float(tj[-1]) - T) <= 1e-6:
            q_out[:, j] = np.interp(t_common, tj, qj)
            v_out[:, j] = np.interp(t_common, tj, vj)
            a_out[:, j] = np.interp(t_common, tj, aj)
            continue

        # Otherwise, slow down by reducing vmax (but never below boundary speeds).
        base_vmax = float(vmax[j])
        bound_speed = float(max(abs(float(v0[j])), abs(float(v1[j]))))
        scale_min = min(1.0, max(1e-6, bound_speed / max(1e-12, base_vmax)))

        def _duration_for_scale(scale: float) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            lim = SCurveLimits(vmax=base_vmax * float(scale), amax=float(amax[j]), jmax=float(jmax[j]))
            tt, qq, vv, aa = scurve_trajectory_1d(
                q_start=float(q0[j]),
                q_goal=float(q1[j]),
                v_start=float(v0[j]),
                v_goal=float(v1[j]),
                limits=lim,
                dt=dt,
            )
            dur = float(tt[-1]) if tt.size else 0.0
            return dur, tt, qq, vv, aa

        lo = scale_min
        hi = 1.0
        dur_hi, tt_hi, qq_hi, vv_hi, aa_hi = _duration_for_scale(hi)
        if dur_hi >= T - 1e-6:
            tt, qq, vv, aa = tt_hi, qq_hi, vv_hi, aa_hi
        else:
            # Binary search for a scale that makes duration >= T.
            best = (dur_hi, tt_hi, qq_hi, vv_hi, aa_hi)
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                dur_mid, tt_mid, qq_mid, vv_mid, aa_mid = _duration_for_scale(mid)
                if dur_mid >= T:
                    lo = mid
                    best = (dur_mid, tt_mid, qq_mid, vv_mid, aa_mid)
                else:
                    hi = mid
            _dur, tt, qq, vv, aa = best

        q_out[:, j] = np.interp(t_common, tt, qq)
        v_out[:, j] = np.interp(t_common, tt, vv)
        a_out[:, j] = np.interp(t_common, tt, aa)

    return t_common, q_out, v_out, a_out
