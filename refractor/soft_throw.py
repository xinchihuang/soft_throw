#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Generate a direct soft-throw trajectory from a world release command.

This refactor entrypoint intentionally removes tube-window/ballistic planning.
It accepts a desired release position and release velocity in world
coordinates, then produces:

1. acceleration/prep segment from the reset joint pose to the release state
2. deceleration segment after release

The stopping/deceleration goal is still the current conservative joint-space
backoff behavior; choosing a task-space stopping position is left for later.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import numpy as np


_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from refractor.config import params
from refractor.release_candidates import generate_release_candidates, select_reachable_release_candidate
from refractor.trajectory_afterprocess import assert_valid_trajectory
from refractor.trajectory_csv import write_trajectory_csv


def _load_env_from_file(path: str | Path) -> None:
    path = Path(path)
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("export "):
            continue
        match = re.match(r"export\s+([A-Za-z_][A-Za-z0-9_]*)=(.*)", line)
        if not match:
            continue
        key, raw_value = match.groups()
        value = raw_value.strip().strip('"').strip("'")
        if key not in os.environ:
            os.environ[key] = value


if "SOFT_THROW_URDF" not in os.environ:
    _load_env_from_file(params.DEFAULT_ENV_SH)

from core.ik_poly5_core import poly5_trajectory, solve_ik_for_q_goal
from core.ik_scurve_core import scurve_trajectory
from core.ik_scurve_core import stopping_distance_jerk_limited
from core.kinematic_forward import PinKinematics
from core.trajectory_planner import build_decel_trajectory, build_execution_trajectory
from core.input_checks import validate_release_pose
from core.throw_params import Q_LIMITS_7, QDOT_LIMITS_7, QDDOT_LIMITS_7, QDDDOT_LIMITS_7


def _scale_q_limits(q_limits: np.ndarray, scale: float) -> np.ndarray:
    q_limits = np.asarray(q_limits, dtype=float).reshape(params.TRAJECTORY_DOF, 2)
    scale = float(scale)
    if not (0.0 < scale <= 1.0):
        raise ValueError(f"limit scale must be in (0, 1], got {scale}")
    q_min = q_limits[:, 0]
    q_max = q_limits[:, 1]
    center = 0.5 * (q_min + q_max)
    half = 0.5 * (q_max - q_min) * scale
    return np.stack([center - half, center + half], axis=1)


def _shrink_q_limits_abs(q_limits: np.ndarray, margin: float) -> np.ndarray:
    q_limits = np.asarray(q_limits, dtype=float).reshape(params.TRAJECTORY_DOF, 2)
    margin = max(0.0, float(margin))
    out = q_limits.copy()
    out[:, 0] += margin
    out[:, 1] -= margin
    if np.any(out[:, 0] >= out[:, 1]):
        raise ValueError(f"q limit margin too large: {margin}")
    return out


def _concat_with_offset(t1, q1, dq1, ddq1, t2, q2, dq2, ddq2):
    if np.asarray(t1).size == 0:
        return t2, q2, dq2, ddq2
    if np.asarray(t2).size:
        t2 = t2[1:]
        q2 = q2[1:]
        dq2 = dq2[1:]
        ddq2 = ddq2[1:]
        if np.asarray(t2).size == 0:
            return t1, q1, dq1, ddq1
    t2 = t2 + float(t1[-1])
    return (
        np.concatenate([t1, t2]),
        np.concatenate([q1, q2]),
        np.concatenate([dq1, dq2]),
        np.concatenate([ddq1, ddq2]),
    )


def _backcompute_start_q_from_release(
    *,
    q_release: np.ndarray,
    dq_release: np.ndarray,
    q_limits: np.ndarray,
    ddq_limits: np.ndarray,
    jerk_limits: np.ndarray,
) -> np.ndarray:
    q_release = np.asarray(q_release, dtype=float).reshape(params.TRAJECTORY_DOF)
    dq_release = np.asarray(dq_release, dtype=float).reshape(params.TRAJECTORY_DOF)
    q_limits = np.asarray(q_limits, dtype=float).reshape(params.TRAJECTORY_DOF, 2)
    ddq_limits = np.asarray(ddq_limits, dtype=float).reshape(params.TRAJECTORY_DOF)
    jerk_limits = np.asarray(jerk_limits, dtype=float).reshape(params.TRAJECTORY_DOF)

    q_min = q_limits[:, 0]
    q_max = q_limits[:, 1]
    q_start = q_release.copy()
    for joint in range(params.TRAJECTORY_DOF):
        v = float(dq_release[joint])
        if abs(v) <= 1e-12:
            continue
        avail = float(q_release[joint] - q_min[joint]) if v > 0.0 else float(q_max[joint] - q_release[joint])
        need = stopping_distance_jerk_limited(abs(v), amax=float(ddq_limits[joint]), jmax=float(jerk_limits[joint]))
        q_start[joint] = q_release[joint] - np.sign(v) * min(max(0.0, avail), need)
    return np.clip(q_start, q_min, q_max)


def _arrays_to_trajectory_list(t: np.ndarray, q: np.ndarray, dq: np.ndarray, ddq: np.ndarray) -> list[dict]:
    trajectory = []
    for i in range(len(t)):
        trajectory.append(
            {
                "t": float(t[i]),
                "q": np.asarray(q[i], dtype=float).reshape(params.TRAJECTORY_DOF),
                "dq": np.asarray(dq[i], dtype=float).reshape(params.TRAJECTORY_DOF),
                "ddq": np.asarray(ddq[i], dtype=float).reshape(params.TRAJECTORY_DOF),
            }
        )
    return trajectory


def _build_joint_release_trajectory(
    *,
    start_q: np.ndarray,
    q_release: np.ndarray,
    dq_release: np.ndarray,
    q_limits: np.ndarray,
    dq_limits: np.ndarray,
    ddq_limits: np.ndarray,
    jerk_limits: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    method = str(params.DEFAULT_SOFT_THROW_SEGMENT_METHOD).lower()
    dt = float(params.DEFAULT_SOFT_THROW_CONTROL_DT) / float(params.DEFAULT_SOFT_THROW_WAYPOINT_DENSITY)
    start_q = np.asarray(start_q, dtype=float).reshape(params.TRAJECTORY_DOF)
    q_release = np.asarray(q_release, dtype=float).reshape(params.TRAJECTORY_DOF)
    dq_release = np.asarray(dq_release, dtype=float).reshape(params.TRAJECTORY_DOF)
    if method == "poly5":
        t, q, dq, ddq, _u = poly5_trajectory(
            start_q,
            q_release,
            q_limits,
            dq_limits,
            ddq_limits,
            control_dt=dt,
            qdot_start=np.zeros(params.TRAJECTORY_DOF, dtype=float),
            qdot_goal=dq_release,
            max_samples=None,
            check_max_samples=int(params.DEFAULT_SOFT_THROW_POLY5_CHECK_MAX_SAMPLES),
        )
        return t, q, dq, ddq
    if method in ("scurve", "s_curve", "s-curve"):
        return scurve_trajectory(
            q_start=start_q,
            q_goal=q_release,
            qdot_start=np.zeros(params.TRAJECTORY_DOF, dtype=float),
            qdot_goal=dq_release,
            qdot_limits=dq_limits,
            qddot_limits=ddq_limits,
            qdddot_limits=jerk_limits,
            control_dt=dt,
        )
    raise ValueError(f"Unsupported segment_method: {params.DEFAULT_SOFT_THROW_SEGMENT_METHOD!r}")


def _write_phase_sidecar(path: str | Path, *, accel_end: float, end: float) -> Path:
    out = Path(str(path) + params.DEFAULT_PHASE_SIDECAR_SUFFIX)
    payload = {
        "time_unit": "s",
        "with_window": False,
        "phases": [
            {"name": "accel", "t_start": 0.0, "t_end": float(accel_end)},
            {"name": "decel", "t_start": float(accel_end), "t_end": float(end)},
        ],
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return out


def _release_joint_state(
    *,
    pin_model: PinKinematics,
    release_pos: np.ndarray,
    release_vel: np.ndarray,
    q_limits: np.ndarray,
    dq_limits: np.ndarray,
    ddq_limits: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    target_frame = str(params.DEFAULT_SOFT_THROW_TARGET_FRAME)
    target_frame_id = pin_model.model.getFrameId(target_frame)
    if target_frame_id == len(pin_model.model.frames):
        raise RuntimeError(f"pin frame not found: {target_frame}")
    init_arm = np.asarray(params.DEFAULT_INIT_ARM, dtype=float).reshape(params.TRAJECTORY_DOF)
    _p0, r_des = pin_model.get_frame_world_pose(init_arm, target_frame)
    return solve_ik_for_q_goal(
        pin_model,
        target_frame_id,
        init_arm,
        np.asarray(release_pos, dtype=float).reshape(3),
        np.asarray(r_des, dtype=float).reshape(3, 3),
        q_limits,
        dq_limits,
        ddq_limits,
        control_dt=float(params.DEFAULT_SOFT_THROW_CONTROL_DT),
        max_iter=int(params.DEFAULT_SOFT_THROW_IK_MAX_ITER),
        kp_pos=float(params.DEFAULT_SOFT_THROW_IK_KP_POS),
        kp_rot=float(params.DEFAULT_SOFT_THROW_IK_KP_ROT),
        v_j7_des=np.asarray(release_vel, dtype=float).reshape(3),
    )


def generate_soft_throw_trajectory(
    *,
    release_pos: np.ndarray | None = None,
    release_vel: np.ndarray | None = None,
    landing_pos: np.ndarray | None = None,
    reference_q: np.ndarray | None = None,
    use_release_selector: bool | None = None,
    output_csv: str | Path | None = None,
) -> list[dict]:
    """Generate accel-to-release plus decel trajectory.

    ``release_pos`` and ``release_vel`` are in world coordinates and apply to
    the current planning frame, ``panda_link7``.
    """
    q_hard_limits = np.asarray(Q_LIMITS_7, dtype=float).reshape(params.TRAJECTORY_DOF, 2)
    q_exec_limits = _shrink_q_limits_abs(q_hard_limits, params.DEFAULT_SOFT_THROW_Q_LIMIT_MARGIN_ABS)
    q_limits = _scale_q_limits(Q_LIMITS_7, params.DEFAULT_SOFT_THROW_LIMIT_SCALE)
    dq_limits = np.asarray(QDOT_LIMITS_7, dtype=float).reshape(params.TRAJECTORY_DOF) * float(params.DEFAULT_SOFT_THROW_LIMIT_SCALE)
    ddq_limits = np.asarray(QDDOT_LIMITS_7, dtype=float).reshape(params.TRAJECTORY_DOF) * float(params.DEFAULT_SOFT_THROW_QDDOT_SCALE)
    jerk_limits = np.asarray(QDDDOT_LIMITS_7, dtype=float).reshape(params.TRAJECTORY_DOF)

    pin_model = PinKinematics()
    use_selector = bool(params.DEFAULT_SOFT_THROW_USE_RELEASE_SELECTOR if use_release_selector is None else use_release_selector)
    selection = None
    if release_pos is None and release_vel is None and use_selector:
        candidates = generate_release_candidates(
            landing_pos=params.DEFAULT_RELEASE_LANDING_POS if landing_pos is None else landing_pos,
            initial_pos=params.DEFAULT_RELEASE_INITIAL_POS,
            candidate_count=int(params.DEFAULT_RELEASE_CANDIDATE_COUNT),
        )
        selection = select_reachable_release_candidate(
            candidates,
            reference_q=params.DEFAULT_RELEASE_REFERENCE_Q if reference_q is None else reference_q,
            pin_model=pin_model,
            frame_name=str(params.DEFAULT_RELEASE_SELECTOR_FRAME),
            object_offset=params.DEFAULT_RELEASE_SELECTOR_OBJECT_OFFSET,
            object_offset_mode=str(params.DEFAULT_RELEASE_SELECTOR_OBJECT_OFFSET_MODE),
            q_limits=q_exec_limits,
            dq_limits=dq_limits,
            ddq_limits=ddq_limits,
            jerk_limits=jerk_limits,
            pos_tol=float(params.DEFAULT_RELEASE_SELECTOR_POS_TOL),
            vel_tol=float(params.DEFAULT_RELEASE_SELECTOR_VEL_TOL),
            q_weight=float(params.DEFAULT_RELEASE_SELECTOR_Q_WEIGHT),
            pos_weight=float(params.DEFAULT_RELEASE_SELECTOR_POS_WEIGHT),
            vel_weight=float(params.DEFAULT_RELEASE_SELECTOR_VEL_WEIGHT),
            ik_max_iter=int(params.DEFAULT_RELEASE_SELECTOR_IK_MAX_ITER),
            require_decel_feasible=bool(params.DEFAULT_RELEASE_SELECTOR_REQUIRE_DECEL_FEASIBLE),
            decel_margin=float(params.DEFAULT_RELEASE_SELECTOR_DECEL_MARGIN),
        )
        release_pos = selection.candidate.release_pos.copy()
        release_vel = selection.candidate.release_vel.copy()
        q_release = selection.q_release.copy()
        dq_release = selection.dq_release.copy()
        q_limits = q_exec_limits
    else:
        release_pos = np.asarray(params.DEFAULT_SOFT_THROW_RELEASE if release_pos is None else release_pos, dtype=float).reshape(3)
        release_vel = np.asarray(
            params.DEFAULT_SOFT_THROW_RELEASE_VELOCITY if release_vel is None else release_vel,
            dtype=float,
        ).reshape(3)
        validate_release_pose(release_pos)
        q_release, dq_release = _release_joint_state(
            pin_model=pin_model,
            release_pos=release_pos,
            release_vel=release_vel,
            q_limits=q_limits,
            dq_limits=dq_limits,
            ddq_limits=ddq_limits,
        )

    start_q = np.asarray(params.DEFAULT_INIT_ARM, dtype=float).reshape(params.TRAJECTORY_DOF)
    if bool(params.DEFAULT_SOFT_THROW_AUTO_START):
        start_q = _backcompute_start_q_from_release(
            q_release=q_release,
            dq_release=dq_release,
            q_limits=q_limits,
            ddq_limits=ddq_limits,
            jerk_limits=jerk_limits,
        )

    print(
        f"[soft_throw] release_pos={np.round(release_pos, 6).tolist()} "
        f"release_vel={np.round(release_vel, 6).tolist()}",
        flush=True,
    )
    if selection is not None:
        print(
            f"[soft_throw] release selector: score={selection.score:.6g} "
            f"pos_error={selection.pos_error:.6g} vel_error={selection.vel_error:.6g} "
            f"q_distance={selection.q_distance:.6g}",
            flush=True,
        )
        print(f"[soft_throw] release_q={np.round(q_release, 6).tolist()}", flush=True)
        print(f"[soft_throw] release_dq={np.round(dq_release, 6).tolist()}", flush=True)
    print("[soft_throw] accel segment start", flush=True)
    if selection is not None:
        t_acc, q_acc, dq_acc, ddq_acc = _build_joint_release_trajectory(
            start_q=start_q,
            q_release=q_release,
            dq_release=dq_release,
            q_limits=q_limits,
            dq_limits=dq_limits,
            ddq_limits=ddq_limits,
            jerk_limits=jerk_limits,
        )
    else:
        target_pose_vel = np.concatenate([release_pos, release_vel], axis=0)
        t_acc, q_acc, dq_acc, ddq_acc, _u = build_execution_trajectory(
            target_pose_vel=target_pose_vel,
            start_q7=start_q,
            q_limits=q_limits,
            qdot_limits=dq_limits,
            qddot_limits=ddq_limits,
            control_dt=float(params.DEFAULT_SOFT_THROW_CONTROL_DT),
            waypoint_density=int(params.DEFAULT_SOFT_THROW_WAYPOINT_DENSITY),
            path_plan_waypoints=int(params.DEFAULT_SOFT_THROW_PATH_PLAN_WAYPOINTS),
            segment_method=str(params.DEFAULT_SOFT_THROW_SEGMENT_METHOD),
            qdddot_limits=jerk_limits if str(params.DEFAULT_SOFT_THROW_SEGMENT_METHOD) == "scurve" else None,
            pin_model=pin_model,
            qdot_start=None,
            joint_limit_margin=float(params.DEFAULT_SOFT_THROW_JOINT_LIMIT_MARGIN),
            ik_max_iter=int(params.DEFAULT_SOFT_THROW_IK_MAX_ITER),
            poly5_max_samples=None,
            verbose=False,
        )

    q_end = q_acc[-1].copy() if len(q_acc) else q_release.copy()
    dq_end = dq_acc[-1].copy() if len(dq_acc) else dq_release.copy()

    print("[soft_throw] decel segment start", flush=True)
    decel_goal = np.full(params.TRAJECTORY_DOF, np.nan, dtype=float)
    decel_joint = int(params.DEFAULT_SOFT_THROW_DECEL_JOINT_INDEX)
    raw_decel_goal = float(q_end[decel_joint]) - float(params.DEFAULT_SOFT_THROW_DECEL_J6_BACKOFF_RAD)
    decel_goal[decel_joint] = float(np.clip(raw_decel_goal, q_limits[decel_joint, 0], q_limits[decel_joint, 1]))
    t_dec, q_dec, dq_dec, ddq_dec, _u_dec = build_decel_trajectory(
        start_q7=q_end,
        start_qdot7=dq_end,
        goal_q7=decel_goal,
        q_limits=q_limits,
        qdot_limits=dq_limits,
        qddot_limits=ddq_limits,
        control_dt=float(params.DEFAULT_SOFT_THROW_CONTROL_DT),
        waypoint_density=int(params.DEFAULT_SOFT_THROW_WAYPOINT_DENSITY),
        poly5_max_samples=None,
        segment_method=str(params.DEFAULT_SOFT_THROW_SEGMENT_METHOD),
        qdddot_limits=jerk_limits if str(params.DEFAULT_SOFT_THROW_SEGMENT_METHOD) == "scurve" else None,
    )

    t, q, dq, ddq = _concat_with_offset(t_acc, q_acc, dq_acc, ddq_acc, t_dec, q_dec, dq_dec, ddq_dec)
    trajectory = _arrays_to_trajectory_list(t, q, dq, ddq)
    assert_valid_trajectory(trajectory)

    out = Path(params.DEFAULT_SOFT_THROW_OUTPUT_CSV if output_csv is None else output_csv)
    write_trajectory_csv(out, trajectory)
    accel_end = float(t_acc[-1]) if len(t_acc) else 0.0
    phase_path = _write_phase_sidecar(out, accel_end=accel_end, end=float(t[-1]))
    print(f"[soft_throw] saved trajectory: {out}", flush=True)
    print(f"[soft_throw] saved phases: {phase_path}", flush=True)
    print(f"[soft_throw] samples={len(trajectory)} duration={float(t[-1]):.6g}s", flush=True)
    return trajectory


def main() -> int:
    generate_soft_throw_trajectory()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
