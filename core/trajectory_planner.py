#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Trajectory planning helpers for joint-space control."""

from __future__ import annotations

from typing import List, Tuple, Optional

import numpy as np

from .ik_poly5_core import solve_ik_for_q_goal, solve_ik_for_reset_pos, poly5_trajectory
from .ik_scurve_core import scurve_trajectory, stopping_distance_jerk_limited
from .kinematics_pin import PinKinematics, pin
from . import tube_method


Point = Tuple[np.ndarray, np.ndarray, np.ndarray, float]


def _effective_q_limits(q_limits: np.ndarray, margin: float) -> Tuple[np.ndarray, np.ndarray]:
    q_min = q_limits[:, 0] + float(margin)
    q_max = q_limits[:, 1] - float(margin)
    return q_min, q_max


def _clamp_qdot_at_limits(
    q: np.ndarray,
    qdot: np.ndarray,
    q_min: np.ndarray,
    q_max: np.ndarray,
    eps: float = 1e-9,
) -> np.ndarray:
    q = np.asarray(q, dtype=float).reshape(7)
    qdot = np.asarray(qdot, dtype=float).reshape(7).copy()
    q_min = np.asarray(q_min, dtype=float).reshape(7)
    q_max = np.asarray(q_max, dtype=float).reshape(7)

    at_min = q <= (q_min + float(eps))
    at_max = q >= (q_max - float(eps))
    qdot[at_min] = np.maximum(qdot[at_min], 0.0)
    qdot[at_max] = np.minimum(qdot[at_max], 0.0)
    return qdot


def build_waypoints(
    p_start: np.ndarray,
    p_goal: np.ndarray,
    v_start: np.ndarray,
    v_goal: np.ndarray,
    n_wp: int,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    n_wp = max(2, int(n_wp))
    waypoints = [
        (1.0 - a) * p_start + a * p_goal
        for a in np.linspace(0.0, 1.0, n_wp)
    ]
    v_wp = [
        (1.0 - a) * v_start + a * v_goal
        for a in np.linspace(0.0, 1.0, n_wp)
    ]
    return waypoints, v_wp


def compute_reset_q7(
    pin_model: PinKinematics,
    q_init7: np.ndarray,
    reset_pos_world: np.ndarray,
    q_limits: np.ndarray,
    qdot_limits: np.ndarray,
    qddot_limits: np.ndarray,
    control_dt: float,
    target_frame: str = "panda_link7",
    max_iter: int = 300,
) -> np.ndarray:
    reset_frame_id = pin_model.model.getFrameId(target_frame)
    if reset_frame_id == len(pin_model.model.frames):
        raise RuntimeError(f"[pin] reset frame not found: {target_frame}")
    q_reset, _ = solve_ik_for_reset_pos(
        pin_model,
        q_init7,
        reset_pos_world,
        q_limits,
        qdot_limits,
        qddot_limits,
        control_dt=control_dt,
        max_iter=max_iter,
        target_frame_id=reset_frame_id,
    )
    return q_reset


def _append_point(points: List[Point], positions, velocities, time_from_start: float) -> None:
    points.append(
        (
            np.asarray(positions, dtype=float).copy(),
            np.asarray(velocities, dtype=float).copy(),
            np.zeros(7, dtype=float),
            float(time_from_start),
        )
    )


def _append_hold(
    points: List[Point],
    positions: np.ndarray,
    duration_sec: float,
    time_from_start: float,
    control_dt: float,
    waypoint_density: int,
) -> float:
    dt_waypoint = control_dt / float(waypoint_density)
    n_steps = max(1, int(np.ceil(float(duration_sec) / dt_waypoint)))
    for _ in range(n_steps):
        time_from_start += dt_waypoint
        _append_point(points, positions, np.zeros(7, dtype=float), time_from_start)
    return time_from_start


def _append_interp(
    points: List[Point],
    q_from: np.ndarray,
    q_to: np.ndarray,
    time_from_start: float,
    control_dt: float,
    waypoint_density: int,
    qdot_limits: np.ndarray,
) -> Tuple[np.ndarray, float]:
    q_from = np.asarray(q_from, dtype=float).copy()
    q_to = np.asarray(q_to, dtype=float).copy()
    dt_waypoint = control_dt / float(waypoint_density)
    dq = np.abs(q_to - q_from)
    step_counts = np.ceil(dq / np.maximum(1e-9, qdot_limits * dt_waypoint)).astype(int)
    n_steps = max(1, int(np.max(step_counts)))

    for k in range(n_steps):
        u = float(k + 1) / float(n_steps)
        alpha = u * u * (3.0 - 2.0 * u)
        q = (1.0 - alpha) * q_from + alpha * q_to
        q_prev = q_from if k == 0 else points[-1][0]
        qdot = (q - q_prev) / dt_waypoint
        time_from_start += dt_waypoint
        _append_point(points, q, qdot, time_from_start)

    return q_to.copy(), time_from_start


def _postprocess_trajectory(
    points: List[Point],
    q_limits: np.ndarray,
    joint_limit_margin: float,
    smoothing_window: int,
) -> List[Point]:
    if not points:
        return points

    q = np.asarray([pt[0] for pt in points], dtype=float)
    t = np.asarray([pt[3] for pt in points], dtype=float)
    n = len(points)

    if smoothing_window and n >= smoothing_window:
        pad = smoothing_window // 2
        kernel = np.ones(smoothing_window, dtype=float) / float(smoothing_window)
        q_smooth = q.copy()
        for j in range(q.shape[1]):
            series = np.pad(q[:, j], (pad, pad), mode="edge")
            q_smooth[:, j] = np.convolve(series, kernel, mode="valid")
        q = q_smooth
        q_min, q_max = _effective_q_limits(q_limits, joint_limit_margin)
        q = np.clip(q, q_min, q_max)

    qdot = np.zeros_like(q)
    if n >= 2:
        dt_f = np.diff(t)
        dq_f = np.diff(q, axis=0)
        valid_f = dt_f > 1e-9
        qdot[:-1][valid_f] = dq_f[valid_f] / dt_f[valid_f, None]
        qdot[-1] = np.zeros(7, dtype=float)

    qddot = np.zeros_like(q)
    if n >= 2:
        dt_v = np.diff(t)
        dqdot = np.diff(qdot, axis=0)
        valid_v = dt_v > 1e-9
        qddot[:-1][valid_v] = dqdot[valid_v] / dt_v[valid_v, None]
        qddot[-1] = np.zeros(7, dtype=float)

    out = []
    for i in range(n):
        out.append((q[i].copy(), qdot[i].copy(), qddot[i].copy(), float(t[i])))
    return out


def build_reset_trajectory(
    start_q7: np.ndarray,
    reset_q7: np.ndarray,
    control_dt: float,
    waypoint_density: int,
    settle_sec: float,
    extra_hold_sec: float,
    q_limits: np.ndarray,
    qdot_limits: np.ndarray,
    limit_scale: float = 1.0,
    joint_limit_margin: float = 0.0,
    smoothing_window: int = 0,
) -> Tuple[List[Point], np.ndarray, int]:
    q_min, q_max = _effective_q_limits(q_limits, joint_limit_margin)
    q_init = np.clip(np.asarray(reset_q7, dtype=float).reshape(7), q_min, q_max)
    q_cmd7 = np.clip(np.asarray(start_q7, dtype=float).reshape(7), q_min, q_max)

    points: List[Point] = []
    time_from_start = 0.0
    q_cmd7, time_from_start = _append_interp(
        points,
        q_cmd7,
        q_init,
        time_from_start,
        control_dt,
        waypoint_density,
        limit_scale * qdot_limits,
    )
    time_from_start = _append_hold(
        points,
        q_cmd7,
        settle_sec,
        time_from_start,
        control_dt,
        waypoint_density,
    )
    split_index = len(points)
    time_from_start = _append_hold(
        points,
        q_cmd7,
        extra_hold_sec,
        time_from_start,
        control_dt,
        waypoint_density,
    )
    points = _postprocess_trajectory(points, q_limits, joint_limit_margin, smoothing_window)
    return points, q_cmd7, split_index


def build_execution_trajectory(
    target_pose_vel: np.ndarray,
    start_q7: np.ndarray,
    q_limits: np.ndarray,
    qdot_limits: np.ndarray,
    qddot_limits: np.ndarray,
    control_dt: float,
    waypoint_density: int,
    path_plan_waypoints: int,
    segment_method: str = "poly5",
    qdddot_limits: Optional[np.ndarray] = None,
    target_frame: str = "panda_link7",
    pin_model: Optional[PinKinematics] = None,
    qdot_start: Optional[np.ndarray] = None,
    joint_limit_margin: float = 0.0,
    ik_max_iter: int = 300,
    poly5_max_samples: Optional[int] = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dt_waypoint = control_dt / float(waypoint_density)
    pin_model = pin_model or PinKinematics()

    target_frame_id = pin_model.model.getFrameId(target_frame)
    if target_frame_id == len(pin_model.model.frames):
        raise RuntimeError(f"[pin] target frame not found: {target_frame}")

    pose = np.asarray(target_pose_vel, dtype=float).reshape(6)
    p_j7_des = pose[:3]
    v_j7_des = pose[3:]

    q_min, q_max = _effective_q_limits(q_limits, joint_limit_margin)
    q_start = np.clip(np.asarray(start_q7, dtype=float).reshape(7), q_min, q_max)
    qdot_seg_start = np.zeros_like(q_start) if qdot_start is None else np.asarray(qdot_start, dtype=float).reshape(7)

    q_full = pin_model.make_q_full_from_arm7(q_start)
    pin.forwardKinematics(pin_model.model, pin_model.data, q_full)
    pin.updateFramePlacements(pin_model.model, pin_model.data)
    T0 = pin_model.data.oMf[target_frame_id]
    p_start = np.array(T0.translation, dtype=float)

    waypoints, v_wp = build_waypoints(
        p_start,
        p_j7_des,
        np.zeros(3, dtype=float),
        np.asarray(v_j7_des, dtype=float).reshape(3),
        path_plan_waypoints,
    )

    t, q, qdot, qddot, u, _q_end, _qdot_end = build_execution_trajectory_from_waypoints(
        waypoints=waypoints,
        v_wp=v_wp,
        start_q7=q_start,
        q_limits=q_limits,
        qdot_limits=qdot_limits,
        qddot_limits=qddot_limits,
        control_dt=control_dt,
        waypoint_density=waypoint_density,
        segment_method=segment_method,
        qdddot_limits=qdddot_limits,
        target_frame=target_frame,
        pin_model=pin_model,
        qdot_start=qdot_seg_start,
        joint_limit_margin=joint_limit_margin,
        ik_max_iter=ik_max_iter,
        poly5_max_samples=poly5_max_samples,
        verbose=verbose,
    )
    return t, q, qdot, qddot, u


def build_execution_trajectory_from_waypoints(
    waypoints: List[np.ndarray],
    v_wp: List[np.ndarray],
    start_q7: np.ndarray,
    q_limits: np.ndarray,
    qdot_limits: np.ndarray,
    qddot_limits: np.ndarray,
    control_dt: float,
    waypoint_density: int,
    segment_method: str = "poly5",
    qdddot_limits: Optional[np.ndarray] = None,
    target_frame: str = "panda_link7",
    pin_model: Optional[PinKinematics] = None,
    qdot_start: Optional[np.ndarray] = None,
    joint_limit_margin: float = 0.0,
    ik_max_iter: int = 300,
    poly5_max_samples: Optional[int] = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dt_waypoint = control_dt / float(waypoint_density)
    pin_model = pin_model or PinKinematics()

    target_frame_id = pin_model.model.getFrameId(target_frame)
    if target_frame_id == len(pin_model.model.frames):
        raise RuntimeError(f"[pin] target frame not found: {target_frame}")

    q_min, q_max = _effective_q_limits(q_limits, joint_limit_margin)
    q_start = np.clip(np.asarray(start_q7, dtype=float).reshape(7), q_min, q_max)
    qdot_seg_start = np.zeros_like(q_start) if qdot_start is None else np.asarray(qdot_start, dtype=float).reshape(7)
    qdot_seg_start = _clamp_qdot_at_limits(q_start, qdot_seg_start, q_min, q_max)

    q_full = pin_model.make_q_full_from_arm7(q_start)
    pin.forwardKinematics(pin_model.model, pin_model.data, q_full)
    pin.updateFramePlacements(pin_model.model, pin_model.data)
    T0 = pin_model.data.oMf[target_frame_id]
    R_j7_des = np.array(T0.rotation, dtype=float)

    t_list = []
    q_list = []
    qdot_list = []
    qddot_list = []
    u_list = []
    t_offset = 0.0
    q_seg_start = q_start.copy()

    for i in range(len(waypoints) - 1):
        p_next = waypoints[i + 1]
        v_next = v_wp[i + 1]
        if verbose:
            print(f"[tube] IK waypoint {i + 1}/{len(waypoints) - 1}")
        q_goal, qdot_goal = solve_ik_for_q_goal(
            pin_model,
            target_frame_id,
            q_seg_start,
            p_next,
            R_j7_des,
            q_limits,
            qdot_limits,
            qddot_limits,
            control_dt=control_dt,
            max_iter=int(ik_max_iter),
            kp_pos=2.0,
            kp_rot=1.0,
            v_j7_des=v_next,
        )
        q_goal = np.clip(np.asarray(q_goal, dtype=float).reshape(7), q_min, q_max)
        qdot_goal = _clamp_qdot_at_limits(q_goal, qdot_goal, q_min, q_max)
        # Ensure the end-of-segment joint velocities are deceleratable without
        # hitting joint limits (otherwise a following decel-to-zero segment with
        # a fixed position goal will be infeasible and will cause velocity jumps).
        if segment_method.lower() in ("scurve", "s_curve", "s-curve") and qdddot_limits is not None:
            jmax = np.asarray(qdddot_limits, dtype=float).reshape(7)
            qdot_goal_safe = qdot_goal.copy()
            for j in range(7):
                vj = float(qdot_goal[j])
                if abs(vj) <= 1e-12:
                    continue
                if vj > 0.0:
                    avail = float(q_max[j] - q_goal[j])
                else:
                    avail = float(q_goal[j] - q_min[j])
                avail = max(0.0, avail)
                # If we cannot even move, force velocity to 0 at the waypoint.
                if avail <= 1e-12:
                    qdot_goal_safe[j] = 0.0
                    continue
                # If stopping distance exceeds available distance, reduce boundary velocity.
                d_stop = stopping_distance_jerk_limited(vj, amax=float(qddot_limits[j]), jmax=float(jmax[j]))
                if d_stop <= avail + 1e-12:
                    continue
                # Binary search vmax such that d_stop(v) ~= avail.
                lo = 0.0
                hi = abs(vj)
                for _ in range(60):
                    mid = 0.5 * (lo + hi)
                    if stopping_distance_jerk_limited(mid, amax=float(qddot_limits[j]), jmax=float(jmax[j])) <= avail:
                        lo = mid
                    else:
                        hi = mid
                qdot_goal_safe[j] = float(np.sign(vj) * lo)
            qdot_goal = qdot_goal_safe

        segment_method = str(segment_method).lower()
        if segment_method == "poly5":
            if verbose:
                print(f"[tube] poly5 segment {i + 1}/{len(waypoints) - 1}")
            t_seg, q_seg, qdot_seg, qddot_seg, u_seg = poly5_trajectory(
                q_seg_start,
                q_goal,
                q_limits,
                qdot_limits,
                qddot_limits,
                control_dt=dt_waypoint,
                qdot_start=qdot_seg_start,
                qdot_goal=qdot_goal,
                max_samples=poly5_max_samples,
                check_max_samples=2000,
                progress=bool(verbose),
                progress_prefix=f"[poly5 {i + 1}/{len(waypoints) - 1}]",
            )
        elif segment_method in ("scurve", "s_curve", "s-curve"):
            if qdddot_limits is None:
                raise ValueError("qdddot_limits is required when segment_method='scurve'")
            if verbose:
                print(f"[tube] scurve segment {i + 1}/{len(waypoints) - 1}")
            t_seg, q_seg, qdot_seg, qddot_seg = scurve_trajectory(
                q_start=q_seg_start,
                q_goal=q_goal,
                qdot_start=qdot_seg_start,
                qdot_goal=qdot_goal,
                qdot_limits=qdot_limits,
                qddot_limits=qddot_limits,
                qdddot_limits=qdddot_limits,
                control_dt=dt_waypoint,
            )
            u_seg = qdot_seg.copy()
        else:
            raise ValueError(f"Unsupported segment_method: {segment_method!r} (expected 'poly5' or 'scurve')")
        if verbose:
            print(f"[tube] {segment_method} segment {i + 1}/{len(waypoints) - 1} done")

        start_idx = 1 if i > 0 else 0
        for k in range(start_idx, len(t_seg)):
            t_list.append(float(t_seg[k] + t_offset))
            q_list.append(q_seg[k].copy())
            qdot_list.append(qdot_seg[k].copy())
            qddot_list.append(qddot_seg[k].copy())
            u_list.append(u_seg[k].copy())

        t_offset = float(t_list[-1]) if t_list else t_offset
        # Carry forward the *actual* terminal state of the generated segment.
        # For poly5, this matches (q_goal, qdot_goal). For scurve, the solver may
        # relax boundary velocities for feasibility/synchronization, so we must
        # propagate the realized end velocity to avoid discontinuities.
        if len(q_seg):
            q_seg_start = np.asarray(q_seg[-1], dtype=float).reshape(7).copy()
        else:
            q_seg_start = q_goal.copy()
        if len(qdot_seg):
            qdot_seg_start = np.asarray(qdot_seg[-1], dtype=float).reshape(7).copy()
        else:
            qdot_seg_start = qdot_goal.copy()

    t = np.asarray(t_list, dtype=float)
    q = np.asarray(q_list, dtype=float)
    qdot = np.asarray(qdot_list, dtype=float)
    qddot = np.asarray(qddot_list, dtype=float)
    u = np.asarray(u_list, dtype=float)
    return t, q, qdot, qddot, u, q_seg_start.copy(), qdot_seg_start.copy()


def build_decel_trajectory(
    start_q7: np.ndarray,
    start_qdot7: np.ndarray,
    goal_q7: Optional[np.ndarray],
    q_limits: np.ndarray,
    qdot_limits: np.ndarray,
    qddot_limits: np.ndarray,
    control_dt: float,
    waypoint_density: int,
    poly5_max_samples: Optional[int] = None,
    segment_method: str = "poly5",
    qdddot_limits: Optional[np.ndarray] = None,
    min_duration_sec: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dt_waypoint = control_dt / float(waypoint_density)
    v0 = np.asarray(start_qdot7, dtype=float).reshape(7)
    q0 = np.asarray(start_q7, dtype=float).reshape(7)
    q_goal = q0.copy()

    # `goal_q7` can be:
    # - None: all joints are auto-braking (goal computed)
    # - array(7,): per-joint goals; NaN entries mean "auto" for that joint.
    goal = None if goal_q7 is None else np.asarray(goal_q7, dtype=float).reshape(7)
    goal_is_fixed = np.zeros(7, dtype=bool)
    if goal is not None:
        goal_is_fixed = ~np.isnan(goal)
        q_goal[goal_is_fixed] = goal[goal_is_fixed]

    # Auto-braking for joints without a fixed goal.
    if np.any(~goal_is_fixed):
        dq_need = np.zeros(7, dtype=float)
        if segment_method.lower() in ("scurve", "s_curve", "s-curve") and qdddot_limits is not None:
            jmax = np.asarray(qdddot_limits, dtype=float).reshape(7)
            for j in range(7):
                if goal_is_fixed[j]:
                    continue
                dv = abs(float(v0[j]))
                if dv <= 1e-12:
                    continue
                dq_stop = stopping_distance_jerk_limited(dv, amax=float(qddot_limits[j]), jmax=float(jmax[j]))
                dq_need[j] = np.sign(float(v0[j])) * float(dq_stop)
        else:
            dt_est = float(control_dt) / float(waypoint_density)
            dq_need = v0 * (5.0 * dt_est)
        q_goal = np.where(goal_is_fixed, q_goal, q0 + dq_need)

    q_goal = np.clip(q_goal, q_limits[:, 0], q_limits[:, 1])
    qdot_goal = np.zeros(7, dtype=float)

    segment_method = str(segment_method).lower()
    if segment_method == "poly5":
        t_seg, q_seg, qdot_seg, qddot_seg, u_seg = poly5_trajectory(
            np.asarray(start_q7, dtype=float).reshape(7),
            q_goal,
            q_limits,
            qdot_limits,
            qddot_limits,
            control_dt=dt_waypoint,
            qdot_start=np.asarray(start_qdot7, dtype=float).reshape(7),
            qdot_goal=qdot_goal,
            max_samples=poly5_max_samples,
            progress=False,
            progress_prefix="[poly5 decel]",
        )
    elif segment_method in ("scurve", "s_curve", "s-curve"):
        if qdddot_limits is None:
            raise ValueError("qdddot_limits is required when segment_method='scurve'")
        t_seg, q_seg, qdot_seg, qddot_seg = scurve_trajectory(
            q_start=np.asarray(start_q7, dtype=float).reshape(7),
            q_goal=q_goal,
            qdot_start=np.asarray(start_qdot7, dtype=float).reshape(7),
            qdot_goal=qdot_goal,
            qdot_limits=qdot_limits,
            qddot_limits=qddot_limits,
            qdddot_limits=qdddot_limits,
            control_dt=dt_waypoint,
        )
        u_seg = qdot_seg.copy()
    else:
        raise ValueError(f"Unsupported segment_method: {segment_method!r} (expected 'poly5' or 'scurve')")
    if min_duration_sec is not None and t_seg.size and t_seg[-1] > 1e-9:
        min_duration_sec = float(min_duration_sec)
        if t_seg[-1] < min_duration_sec:
            scale = min_duration_sec / float(t_seg[-1])
            t_seg = t_seg * scale
            qdot_seg = qdot_seg / scale
            qddot_seg = qddot_seg / (scale * scale)
    return t_seg, q_seg, qdot_seg, qddot_seg, u_seg


def build_tube_throw_trajectory(
    target_xyz: np.ndarray,
    release_pos_w: np.ndarray,
    hold_sec: float = 0.5,
    method: str = "middle",
    n_window_samples: Optional[int] = None,
    flight_time_range: Tuple[float, float] = (0.45, 0.85),
    n_family: int = 21,
    nominal_flight_time: float = 0.65,
    control_dt: float = 0.01,
    waypoint_density: int = 10,
    q_limits: Optional[np.ndarray] = None,
    qdot_limits: Optional[np.ndarray] = None,
    qddot_limits: Optional[np.ndarray] = None,
    path_plan_waypoints: int = 5,
    segment_method: str = "poly5",
    qdddot_limits: Optional[np.ndarray] = None,
    target_frame: str = "panda_link7",
    pin_model: Optional[PinKinematics] = None,
    start_q7: Optional[np.ndarray] = None,
    qdot_start: Optional[np.ndarray] = None,
    joint_limit_margin: float = 0.0,
    ik_max_iter: int = 300,
    poly5_max_samples: Optional[int] = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    if q_limits is None or qdot_limits is None or qddot_limits is None:
        raise ValueError("q_limits, qdot_limits, qddot_limits are required")

    window = tube_method.generate_tube_window_trajectory(
        target_point=target_xyz,
        release_point=release_pos_w,
        hold_sec=hold_sec,
        n_window_samples=n_window_samples,
        method=method,
        flight_time_range=flight_time_range,
        n_family=n_family,
        nominal_flight_time=nominal_flight_time,
        g=tube_method.G,
    )

    p_list = window["release_points"]
    v_list = window["release_velocities"]
    meta = window["meta"]

    pin_model = pin_model or PinKinematics()
    if start_q7 is None:
        start_q7 = np.zeros(7, dtype=float)
    if qdot_start is None:
        qdot_start = np.zeros(7, dtype=float)

    if verbose:
        print("[tube] pre segment start")
    pre_target = np.concatenate([p_list[0], v_list[0]], axis=0)
    t_pre, q_pre, qdot_pre, qddot_pre, _u_pre = build_execution_trajectory(
        target_pose_vel=pre_target,
        start_q7=start_q7,
        q_limits=q_limits,
        qdot_limits=qdot_limits,
        qddot_limits=qddot_limits,
        control_dt=control_dt,
        waypoint_density=waypoint_density,
        path_plan_waypoints=path_plan_waypoints,
        segment_method=segment_method,
        qdddot_limits=qdddot_limits,
        target_frame=target_frame,
        pin_model=pin_model,
        qdot_start=qdot_start,
        joint_limit_margin=joint_limit_margin,
        ik_max_iter=ik_max_iter,
        poly5_max_samples=poly5_max_samples,
        verbose=verbose,
    )
    if verbose:
        print("[tube] pre segment done")

    if verbose:
        print("[tube] window segment start")
    t_win, q_win, qdot_win, qddot_win, _u_win, q_end, qdot_end = build_execution_trajectory_from_waypoints(
        waypoints=[p_list[i] for i in range(len(p_list))],
        v_wp=[v_list[i] for i in range(len(v_list))],
        start_q7=q_pre[-1] if len(q_pre) else start_q7,
        q_limits=q_limits,
        qdot_limits=qdot_limits,
        qddot_limits=qddot_limits,
        control_dt=control_dt,
        waypoint_density=waypoint_density,
        segment_method=segment_method,
        qdddot_limits=qdddot_limits,
        target_frame=target_frame,
        pin_model=pin_model,
        qdot_start=qdot_pre[-1] if len(qdot_pre) else qdot_start,
        joint_limit_margin=joint_limit_margin,
        ik_max_iter=ik_max_iter,
        poly5_max_samples=poly5_max_samples,
        verbose=verbose,
    )
    if verbose:
        print("[tube] window segment done")

    if verbose:
        print("[tube] decel segment start")
    t_dec, q_dec, qdot_dec, qddot_dec, _u_dec = build_decel_trajectory(
        start_q7=q_end,
        start_qdot7=qdot_end,
        goal_q7=None,
        q_limits=q_limits,
        qdot_limits=qdot_limits,
        qddot_limits=qddot_limits,
        control_dt=control_dt,
        waypoint_density=waypoint_density,
        poly5_max_samples=poly5_max_samples,
        segment_method=segment_method,
        qdddot_limits=qdddot_limits,
    )
    if verbose:
        print("[tube] decel segment done")

    def _concat_with_offset(t1, q1, qd1, qdd1, t2, q2, qd2, qdd2):
        if t1.size == 0:
            return t2, q2, qd2, qdd2
        t2 = t2 + float(t1[-1])
        return (
            np.concatenate([t1, t2]),
            np.concatenate([q1, q2]),
            np.concatenate([qd1, qd2]),
            np.concatenate([qdd1, qdd2]),
        )

    t, q, qdot, qddot = _concat_with_offset(t_pre, q_pre, qdot_pre, qddot_pre, t_win, q_win, qdot_win, qddot_win)
    t, q, qdot, qddot = _concat_with_offset(t, q, qdot, qddot, t_dec, q_dec, qdot_dec, qddot_dec)

    return t, q, qdot, qddot, meta


def solve_release_velocity(
    release_pos_w: np.ndarray,
    target_pos_w: np.ndarray,
) -> Tuple[np.ndarray, float]:
    v_release, t_flight = tube_method.solve_ballistic_velocity(
        np.asarray(release_pos_w, dtype=float).reshape(3),
        np.asarray(target_pos_w, dtype=float).reshape(3),
    )
    return np.asarray(v_release, dtype=float).reshape(3), float(t_flight)


def write_trace_csv(csv_path: str, t, q, qdot, qddot, *, time_unit: str = "s") -> None:
    t = np.asarray(t, dtype=float).reshape(-1)
    if time_unit == "ms":
        t_out = t * 1000.0
    elif time_unit == "s":
        t_out = t
    else:
        raise ValueError(f"Unsupported time_unit: {time_unit!r} (expected 's' or 'ms')")

    header = ["t"] + [f"q{j}" for j in range(7)] + [f"dq{j}" for j in range(7)] + [f"ddq{j}" for j in range(7)]
    data = np.column_stack([t_out, q, qdot, qddot])
    np.savetxt(csv_path, data, delimiter=",", header=",".join(header), comments="")


def write_trace_with_tau_csv(csv_path: str, t, q, qdot, qddot, *, time_unit: str = "s") -> None:
    t = np.asarray(t, dtype=float).reshape(-1)
    q = np.asarray(q, dtype=float)
    qdot = np.asarray(qdot, dtype=float)
    qddot = np.asarray(qddot, dtype=float)

    if time_unit == "ms":
        t_out = t * 1000.0
    elif time_unit == "s":
        t_out = t
    else:
        raise ValueError(f"Unsupported time_unit: {time_unit!r} (expected 's' or 'ms')")

    if t.size < 2:
        tau = np.zeros_like(qddot)
    else:
        tau = np.gradient(qddot, t, axis=0)
    header = (
        ["t"]
        + [f"q{j}" for j in range(7)]
        + [f"dq{j}" for j in range(7)]
        + [f"ddq{j}" for j in range(7)]
        + [f"tau{j}" for j in range(7)]
    )
    data = np.column_stack([t_out, q, qdot, qddot, tau])
    np.savetxt(csv_path, data, delimiter=",", header=",".join(header), comments="")
