#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Jerk-limited control through a sequence of q/dq waypoints.

This module is for execution trajectory generation in the refactor path. It
accepts a series of target joint positions and velocities, then reaches them in
order using jerk-limited S-curve segments. No extra deceleration segment is
added after the final target.
"""

from __future__ import annotations

from dataclasses import dataclass
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

try:
    from .config import params
    from .trajectory_csv import TrajectoryList, read_trajectory_csv, write_trajectory_csv
except ImportError:
    _ROOT = Path(__file__).resolve().parents[1]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    from refractor.config import params
    from refractor.trajectory_csv import TrajectoryList, read_trajectory_csv, write_trajectory_csv


@dataclass(frozen=True)
class JointSCurveLimits:
    dq_max: float
    ddq_max: float
    jerk_max: float


def _as_scalar(x) -> float:
    return float(np.asarray(x, dtype=float).reshape(()))


def _solve_accel_phase_times(dv: float, ddq_max: float, jerk_max: float) -> tuple[float, float]:
    """Return jerk-ramp time and accel-hold time for a velocity change."""
    dv = float(dv)
    ddq_max = float(ddq_max)
    jerk_max = float(jerk_max)
    if dv <= 0.0:
        return 0.0, 0.0
    if ddq_max <= 0.0 or jerk_max <= 0.0:
        raise ValueError("ddq_max and jerk_max must be positive")

    dv_triangular_max = (ddq_max * ddq_max) / jerk_max
    if dv >= dv_triangular_max:
        t_jerk = ddq_max / jerk_max
        t_accel = (dv - dv_triangular_max) / ddq_max
        return float(t_jerk), float(t_accel)

    return float(np.sqrt(dv / jerk_max)), 0.0


def _accel_phase_displacement(v0: float, dv: float, t_jerk: float, t_accel: float, jerk: float) -> tuple[float, float]:
    """Return displacement/time for a +jerk, hold, -jerk acceleration phase."""
    v0 = float(v0)
    dv = float(dv)
    t_jerk = float(t_jerk)
    t_accel = float(t_accel)
    jerk = float(jerk)
    if dv <= 0.0 or t_jerk <= 0.0:
        return 0.0, 0.0

    if t_accel <= 0.0:
        dq = v0 * (2.0 * t_jerk) + jerk * (t_jerk ** 3)
        return float(dq), float(2.0 * t_jerk)

    accel_max = jerk * t_jerk
    dq1 = v0 * t_jerk + (jerk / 6.0) * (t_jerk ** 3)
    v1 = v0 + 0.5 * jerk * (t_jerk ** 2)
    dq2 = v1 * t_accel + 0.5 * accel_max * (t_accel ** 2)
    v2 = v1 + accel_max * t_accel
    dq3 = v2 * t_jerk + 0.5 * accel_max * (t_jerk ** 2) - (jerk / 6.0) * (t_jerk ** 3)
    return float(dq1 + dq2 + dq3), float(2.0 * t_jerk + t_accel)


def _sample_jerk_segments(
    *,
    q0: float,
    dq0: float,
    ddq0: float,
    segments: Sequence[tuple[float, float]],
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Integrate piecewise-constant jerk segments."""
    dt = float(dt)
    if dt <= 0.0:
        raise ValueError("dt must be > 0")

    q = float(q0)
    dq = float(dq0)
    ddq = float(ddq0)
    t_abs = 0.0

    t_hist = [0.0]
    q_hist = [q]
    dq_hist = [dq]
    ddq_hist = [ddq]

    for duration, jerk in segments:
        duration = float(duration)
        jerk = float(jerk)
        if duration <= 0.0:
            continue
        n = max(1, int(np.ceil(duration / dt)))
        for k in range(n):
            h = min(dt, duration - k * dt)
            if h <= 0.0:
                break
            q = q + dq * h + 0.5 * ddq * (h ** 2) + (1.0 / 6.0) * jerk * (h ** 3)
            dq = dq + ddq * h + 0.5 * jerk * (h ** 2)
            ddq = ddq + jerk * h
            t_abs += h
            t_hist.append(t_abs)
            q_hist.append(q)
            dq_hist.append(dq)
            ddq_hist.append(ddq)

    return (
        np.asarray(t_hist, dtype=float),
        np.asarray(q_hist, dtype=float),
        np.asarray(dq_hist, dtype=float),
        np.asarray(ddq_hist, dtype=float),
    )


def jerk_limited_segment_1d(
    *,
    q_start: float,
    q_goal: float,
    dq_start: float,
    dq_goal: float,
    limits: JointSCurveLimits,
    dt: float,
    max_iter: int = 80,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate one 1-DOF jerk-limited segment from (q,dq) to (q,dq)."""
    q0 = _as_scalar(q_start)
    q1 = _as_scalar(q_goal)
    v0 = _as_scalar(dq_start)
    v1 = _as_scalar(dq_goal)
    dq_max = float(limits.dq_max)
    ddq_max = float(limits.ddq_max)
    jerk_max = float(limits.jerk_max)
    if dq_max <= 0.0 or ddq_max <= 0.0 or jerk_max <= 0.0:
        raise ValueError("dq/ddq/jerk limits must be positive")

    distance = q1 - q0
    if abs(distance) < 1e-12 and abs(v0 - v1) < 1e-12:
        return np.array([0.0]), np.array([q0]), np.array([v0]), np.array([0.0])

    direction = 1.0 if distance >= 0.0 else -1.0
    dist = abs(distance)
    v0n = float(np.clip(direction * v0, -dq_max, dq_max))
    v1n = float(np.clip(direction * v1, -dq_max, dq_max))

    def displacement_for_peak(v_peak: float) -> tuple[float, tuple[float, float], tuple[float, float]]:
        v_peak = float(v_peak)
        dv_acc = max(0.0, v_peak - v0n)
        dv_dec = max(0.0, v_peak - v1n)
        tja, taa = _solve_accel_phase_times(dv_acc, ddq_max, jerk_max)
        tjd, tad = _solve_accel_phase_times(dv_dec, ddq_max, jerk_max)
        dq_acc, _ = _accel_phase_displacement(v0n, dv_acc, tja, taa, jerk_max)
        dq_dec, _ = _accel_phase_displacement(v1n, dv_dec, tjd, tad, jerk_max)
        return float(dq_acc + dq_dec), (tja, taa), (tjd, tad)

    v_low = max(v0n, v1n)
    v_high = dq_max
    if v_low > v_high + 1e-9:
        raise ValueError("Boundary velocities exceed dq limit")

    min_dist, _acc_times, _dec_times = displacement_for_peak(v_low)
    if min_dist > dist + 1e-9:
        raise ValueError(
            f"Segment infeasible: not enough distance for boundary velocities "
            f"(need {min_dist:.12g}, have {dist:.12g})"
        )

    high_dist, _acc_times, _dec_times = displacement_for_peak(v_high)
    if high_dist < dist - 1e-9:
        v_peak = v_high
        base_dist, acc_times, dec_times = displacement_for_peak(v_peak)
        t_cruise = (dist - base_dist) / max(1e-12, v_peak)
    else:
        lo = v_low
        hi = v_high
        for _ in range(int(max_iter)):
            mid = 0.5 * (lo + hi)
            mid_dist, _acc_times, _dec_times = displacement_for_peak(mid)
            if mid_dist > dist:
                hi = mid
            else:
                lo = mid
            if abs(hi - lo) <= 1e-9:
                break
        v_peak = 0.5 * (lo + hi)
        _base_dist, acc_times, dec_times = displacement_for_peak(v_peak)
        t_cruise = 0.0

    tja, taa = acc_times
    tjd, tad = dec_times
    segments: list[tuple[float, float]] = []
    if tja > 0.0:
        segments.append((tja, +jerk_max))
        if taa > 0.0:
            segments.append((taa, 0.0))
        segments.append((tja, -jerk_max))
    if t_cruise > 0.0:
        segments.append((t_cruise, 0.0))
    if tjd > 0.0:
        segments.append((tjd, -jerk_max))
        if tad > 0.0:
            segments.append((tad, 0.0))
        segments.append((tjd, +jerk_max))

    t, qn, dqn, ddqn = _sample_jerk_segments(q0=0.0, dq0=v0n, ddq0=0.0, segments=segments, dt=dt)
    q = q0 + direction * qn
    dq = direction * dqn
    ddq = direction * ddqn
    ddq[-1] = 0.0
    return t, q, dq, ddq


def jerk_limited_segment_7d(
    *,
    q_start: np.ndarray,
    q_goal: np.ndarray,
    dq_start: np.ndarray,
    dq_goal: np.ndarray,
    dt: float = params.DEFAULT_JERK_CONTROL_DT,
    qdot_limits: np.ndarray = params.QDOT_LIMITS_7,
    qddot_limits: np.ndarray = params.QDDOT_LIMITS_7,
    jerk_limits: np.ndarray = params.JERK_LIMITS_7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate a synchronized 7-DOF segment from start (q,dq) to goal (q,dq)."""
    q0 = np.asarray(q_start, dtype=float).reshape(params.TRAJECTORY_DOF)
    q1 = np.asarray(q_goal, dtype=float).reshape(params.TRAJECTORY_DOF)
    v0 = np.asarray(dq_start, dtype=float).reshape(params.TRAJECTORY_DOF)
    v1 = np.asarray(dq_goal, dtype=float).reshape(params.TRAJECTORY_DOF)
    dq_max = np.asarray(qdot_limits, dtype=float).reshape(params.TRAJECTORY_DOF)
    ddq_max = np.asarray(qddot_limits, dtype=float).reshape(params.TRAJECTORY_DOF)
    jerk_max = np.asarray(jerk_limits, dtype=float).reshape(params.TRAJECTORY_DOF)
    dt = float(dt)

    per_joint = []
    durations = []
    for j in range(params.TRAJECTORY_DOF):
        limits = JointSCurveLimits(dq_max=float(dq_max[j]), ddq_max=float(ddq_max[j]), jerk_max=float(jerk_max[j]))
        tj, qj, dqj, ddqj = jerk_limited_segment_1d(
            q_start=float(q0[j]),
            q_goal=float(q1[j]),
            dq_start=float(v0[j]),
            dq_goal=float(v1[j]),
            limits=limits,
            dt=dt,
        )
        per_joint.append((tj, qj, dqj, ddqj))
        durations.append(float(tj[-1]) if tj.size else 0.0)

    duration = float(max(durations))
    if duration <= 0.0:
        t = np.array([0.0], dtype=float)
        return t, q0.reshape(1, -1), v0.reshape(1, -1), np.zeros((1, params.TRAJECTORY_DOF), dtype=float)

    n = max(2, int(np.ceil(duration / dt)) + 1)
    t_common = np.linspace(0.0, duration, n, dtype=float)
    q = np.zeros((n, params.TRAJECTORY_DOF), dtype=float)
    dq = np.zeros_like(q)
    ddq = np.zeros_like(q)

    for j, (tj, qj, dqj, ddqj) in enumerate(per_joint):
        if abs(float(tj[-1]) - duration) <= 1e-6:
            q[:, j] = np.interp(t_common, tj, qj)
            dq[:, j] = np.interp(t_common, tj, dqj)
            ddq[:, j] = np.interp(t_common, tj, ddqj)
            continue

        base_dq_max = float(dq_max[j])
        boundary_speed = float(max(abs(v0[j]), abs(v1[j])))
        scale_min = min(1.0, max(1e-6, boundary_speed / max(1e-12, base_dq_max)))

        def solve_for_scale(scale: float) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            limits = JointSCurveLimits(
                dq_max=base_dq_max * float(scale),
                ddq_max=float(ddq_max[j]),
                jerk_max=float(jerk_max[j]),
            )
            tt, qq, vv, aa = jerk_limited_segment_1d(
                q_start=float(q0[j]),
                q_goal=float(q1[j]),
                dq_start=float(v0[j]),
                dq_goal=float(v1[j]),
                limits=limits,
                dt=dt,
            )
            return float(tt[-1]) if tt.size else 0.0, tt, qq, vv, aa

        dur_hi, tt_hi, qq_hi, vv_hi, aa_hi = solve_for_scale(1.0)
        if dur_hi >= duration - 1e-6:
            tt, qq, vv, aa = tt_hi, qq_hi, vv_hi, aa_hi
        else:
            lo = scale_min
            hi = 1.0
            best = (dur_hi, tt_hi, qq_hi, vv_hi, aa_hi)
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                dur_mid, tt_mid, qq_mid, vv_mid, aa_mid = solve_for_scale(mid)
                if dur_mid >= duration:
                    lo = mid
                    best = (dur_mid, tt_mid, qq_mid, vv_mid, aa_mid)
                else:
                    hi = mid
            _dur, tt, qq, vv, aa = best

        q[:, j] = np.interp(t_common, tt, qq)
        dq[:, j] = np.interp(t_common, tt, vv)
        ddq[:, j] = np.interp(t_common, tt, aa)
    return t_common, q, dq, ddq


def _waypoints_from_trajectory(trajectory: TrajectoryList) -> tuple[np.ndarray, np.ndarray]:
    if not trajectory:
        raise ValueError("trajectory is empty")
    q = np.asarray([point["q"] for point in trajectory], dtype=float).reshape(-1, params.TRAJECTORY_DOF)
    dq = np.asarray([point.get("dq", np.zeros(params.TRAJECTORY_DOF)) for point in trajectory], dtype=float).reshape(-1, params.TRAJECTORY_DOF)
    return q, dq


def generate_jerk_limited_trajectory(
    waypoints: TrajectoryList | Sequence[dict],
    *,
    dt: float = params.DEFAULT_JERK_CONTROL_DT,
) -> TrajectoryList:
    """Reach each input ``q/dq`` waypoint in order using jerk-limited control."""
    q_wp, dq_wp = _waypoints_from_trajectory(list(waypoints))
    if q_wp.shape[0] == 1:
        return [{"t": 0.0, "q": q_wp[0].copy(), "dq": dq_wp[0].copy(), "ddq": np.zeros(params.TRAJECTORY_DOF)}]

    out: TrajectoryList = []
    t_offset = 0.0
    for segment_index in range(q_wp.shape[0] - 1):
        try:
            t_seg, q_seg, dq_seg, ddq_seg = jerk_limited_segment_7d(
                q_start=q_wp[segment_index],
                q_goal=q_wp[segment_index + 1],
                dq_start=dq_wp[segment_index],
                dq_goal=dq_wp[segment_index + 1],
                dt=dt,
            )
        except Exception as exc:
            raise ValueError(f"jerk-limited segment {segment_index}->{segment_index + 1} failed: {exc}") from exc
        start_i = 0 if segment_index == 0 else 1
        for i in range(start_i, t_seg.size):
            out.append(
                {
                    "t": float(t_offset + t_seg[i]),
                    "q": q_seg[i].copy(),
                    "dq": dq_seg[i].copy(),
                    "ddq": ddq_seg[i].copy(),
                }
            )
        t_offset = float(out[-1]["t"])
    print(f"[jerk_control] waypoints={q_wp.shape[0]} commands={len(out)} duration={t_offset:.12g}", flush=True)
    return out


def main() -> int:
    trajectory = read_trajectory_csv(params.DEFAULT_TRAJECTORY_CSV)
    q0 = np.asarray(trajectory[0]["q"], dtype=float).reshape(params.TRAJECTORY_DOF)
    q1 = np.asarray(trajectory[-1]["q"], dtype=float).reshape(params.TRAJECTORY_DOF)
    demo_waypoints = [
        {"t": 0.0, "q": q0, "dq": np.zeros(params.TRAJECTORY_DOF), "ddq": np.zeros(params.TRAJECTORY_DOF)},
        {"t": 0.0, "q": q1, "dq": np.zeros(params.TRAJECTORY_DOF), "ddq": np.zeros(params.TRAJECTORY_DOF)},
    ]
    generated = generate_jerk_limited_trajectory(demo_waypoints, dt=params.DEFAULT_JERK_CONTROL_DT)
    write_trajectory_csv(params.DEFAULT_JERK_CONTROL_OUTPUT_CSV, generated)
    print(f"[ok] wrote {params.DEFAULT_JERK_CONTROL_OUTPUT_CSV}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
