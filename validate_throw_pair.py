#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass

import numpy as np


def _load_env_from_file(path: str) -> None:
    if not os.path.isfile(path):
        return
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line.startswith("export "):
                continue
            match = re.match(r'export\s+([A-Za-z_][A-Za-z0-9_]*)=(.*)', line)
            if not match:
                continue
            key, raw_value = match.groups()
            value = raw_value.strip().strip('"').strip("'")
            if key not in os.environ:
                os.environ[key] = value


def _parse_vec3(text: str) -> np.ndarray:
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Expected 3 comma-separated values, e.g. 2.0,0.0,0.0")
    return np.array([float(p) for p in parts], dtype=float)


@dataclass(frozen=True)
class FeasibilityResult:
    feasible: bool
    reason: str


@dataclass(frozen=True)
class ThrowPairCheck:
    feasible: bool
    reason: str
    v_release: np.ndarray
    t_flight: float
    q_release: np.ndarray | None = None
    qdot_release: np.ndarray | None = None
    v_actual: np.ndarray | None = None
    angle_deg: float | None = None
    v_parallel_max: float | None = None
    init_runup_ok: bool | None = None
    auto_start_q7: np.ndarray | None = None


def _frame_linear_jacobian_arm7(pin_model, frame_id: int) -> np.ndarray:
    """Return the world-aligned linear Jacobian (3x7) for panda joints 1..7."""
    import pinocchio as pin

    q_full = pin.neutral(pin_model.model).copy()
    # Caller should have already updated kinematics; we compute fresh for safety.
    # `pin_model` is the PinKinematics wrapper from core/kinematic_forward.py.
    # It exposes `model`/`data` compatible with pinocchio.
    # We assume arm7 joints are panda_joint1..panda_joint7.
    J = pin.computeFrameJacobian(
        pin_model.model,
        pin_model.data,
        q_full,
        frame_id,
        pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
    )
    # Map to arm7.
    J_arm = np.zeros((3, 7), dtype=float)
    for j in range(1, 8):
        jid = pin_model.model.getJointId(f"panda_joint{j}")
        idx_v = int(pin_model.model.joints[jid].idx_v)
        J_arm[:, j - 1] = J[:3, idx_v]
    return J_arm


def _backcompute_start_q_from_release(
    *,
    q_release: np.ndarray,
    qdot_release: np.ndarray,
    q_limits: np.ndarray,
    qddot_limits: np.ndarray,
    qdddot_limits: np.ndarray,
) -> np.ndarray:
    from core.ik_scurve_core import stopping_distance_jerk_limited

    q_release = np.asarray(q_release, dtype=float).reshape(7)
    qdot_release = np.asarray(qdot_release, dtype=float).reshape(7)
    q_limits = np.asarray(q_limits, dtype=float).reshape(7, 2)
    qddot_limits = np.asarray(qddot_limits, dtype=float).reshape(7)
    qdddot_limits = np.asarray(qdddot_limits, dtype=float).reshape(7)

    q_min = q_limits[:, 0]
    q_max = q_limits[:, 1]
    q_start = q_release.copy()
    for j in range(7):
        v = float(qdot_release[j])
        if abs(v) <= 1e-12:
            continue
        if v > 0.0:
            avail = float(q_release[j] - q_min[j])
        else:
            avail = float(q_max[j] - q_release[j])
        avail = max(0.0, avail)
        need = stopping_distance_jerk_limited(abs(v), amax=float(qddot_limits[j]), jmax=float(qdddot_limits[j]))
        need = min(float(need), avail)
        q_start[j] = q_release[j] - np.sign(v) * need
    return np.clip(q_start, q_min, q_max)


def _check_reach_release_from_start(
    *,
    q_start: np.ndarray,
    q_release: np.ndarray,
    qdot_release: np.ndarray,
    qddot_limits: np.ndarray,
    qdddot_limits: np.ndarray,
) -> FeasibilityResult:
    from core.ik_scurve_core import stopping_distance_jerk_limited

    q_start = np.asarray(q_start, dtype=float).reshape(7)
    q_release = np.asarray(q_release, dtype=float).reshape(7)
    qdot_release = np.asarray(qdot_release, dtype=float).reshape(7)
    qddot_limits = np.asarray(qddot_limits, dtype=float).reshape(7)
    qdddot_limits = np.asarray(qdddot_limits, dtype=float).reshape(7)

    for j in range(7):
        vj = float(qdot_release[j])
        if abs(vj) <= 1e-12:
            continue
        need = stopping_distance_jerk_limited(abs(vj), amax=float(qddot_limits[j]), jmax=float(qdddot_limits[j]))
        avail = abs(float(q_release[j] - q_start[j]))
        if avail + 1e-9 < float(need):
            return FeasibilityResult(
                False,
                f"insufficient run-up for joint{j+1}: need >= {need:.6g} rad, have {avail:.6g} rad",
            )
    return FeasibilityResult(True, "ok")


def check_throw_pair(
    *,
    landing: np.ndarray,
    release: np.ndarray,
    v_release_override: np.ndarray | None = None,
    limit_scale: float = 0.9,
    qddot_scale: float | None = None,
    ik_max_iter: int = 50,
    use_auto_start: bool = False,
) -> ThrowPairCheck:
    from core.throw_params import (
        INIT_ARM,
        Q_LIMITS_7,
        QDOT_LIMITS_7,
        QDDOT_LIMITS_7,
        QDDDOT_LIMITS_7,
        PLANNER_QDDOT_LIMIT_SCALE,
        BALLISTIC_T_MIN,
        BALLISTIC_T_MAX,
        BALLISTIC_T_STEP,
        EE_VEL_MAX,
    )
    from core.kinematic_forward import PinKinematics
    from core.ik_poly5_core import solve_ik_for_q_goal
    from core import tube_method

    landing = np.asarray(landing, dtype=float).reshape(3)
    release = np.asarray(release, dtype=float).reshape(3)
    limit_scale = float(limit_scale)
    if not (0.0 < limit_scale <= 1.0):
        raise ValueError("limit_scale must be in (0,1]")
    if qddot_scale is None:
        qddot_scale = float(PLANNER_QDDOT_LIMIT_SCALE)
    qddot_scale = float(qddot_scale)
    if not (0.0 < qddot_scale <= 1.0):
        raise ValueError("qddot_scale must be in (0,1]")

    q_limits = np.asarray(Q_LIMITS_7, dtype=float).copy()
    q_min = q_limits[:, 0]
    q_max = q_limits[:, 1]
    center = 0.5 * (q_min + q_max)
    half = 0.5 * (q_max - q_min) * limit_scale
    q_limits = np.stack([center - half, center + half], axis=1)
    qdot_limits = np.asarray(QDOT_LIMITS_7, dtype=float) * limit_scale
    qddot_limits = np.asarray(QDDOT_LIMITS_7, dtype=float) * qddot_scale
    qdddot_limits = np.asarray(QDDDOT_LIMITS_7, dtype=float)

    # 1) Ballistic feasibility unless overridden.
    if v_release_override is None:
        best = None
        for T in np.arange(float(BALLISTIC_T_MIN), float(BALLISTIC_T_MAX) + 1e-9, float(BALLISTIC_T_STEP)):
            v = tube_method.velocity_from_target_release_and_time(landing, release, float(T), g=tube_method.G)
            speed = float(np.linalg.norm(v))
            if best is None or speed < best[0]:
                best = (speed, v, float(T))
        if best is None:
            return ThrowPairCheck(False, "ballistic_invalid_time_range", np.zeros(3), float("nan"))
        min_speed, v_release, t_flight = best
        if float(min_speed) > float(EE_VEL_MAX) + 1e-9:
            return ThrowPairCheck(
                False,
                f"ballistic_requires_speed>{EE_VEL_MAX:.6g} (min={min_speed:.6g})",
                np.asarray(v_release, dtype=float).reshape(3),
                float(t_flight),
            )
        v_release = np.asarray(v_release, dtype=float).reshape(3)
    else:
        v_release = np.asarray(v_release_override, dtype=float).reshape(3)
        t_flight = float("nan")
        if float(np.linalg.norm(v_release)) > float(EE_VEL_MAX) + 1e-9:
            return ThrowPairCheck(
                False,
                f"release_speed>{EE_VEL_MAX:.6g}",
                v_release,
                t_flight,
            )

    # 2) IK at release (position + desired EE linear velocity)
    pin_model = PinKinematics()
    target_frame = "panda_link7"
    target_frame_id = pin_model.model.getFrameId(target_frame)
    if target_frame_id == len(pin_model.model.frames):
        return ThrowPairCheck(False, f"pin_missing_frame:{target_frame}", v_release, t_flight)
    _p0, R_des = pin_model.get_frame_world_pose(np.asarray(INIT_ARM, dtype=float).reshape(7), target_frame)

    try:
        q_rel, qdot_rel = solve_ik_for_q_goal(
            pin_model,
            target_frame_id,
            np.asarray(INIT_ARM, dtype=float).reshape(7),
            np.asarray(release, dtype=float).reshape(3),
            np.asarray(R_des, dtype=float).reshape(3, 3),
            q_limits,
            qdot_limits,
            qddot_limits,
            control_dt=0.01,
            max_iter=int(ik_max_iter),
            kp_pos=2.0,
            kp_rot=1.0,
            v_j7_des=v_release,
        )
    except Exception as exc:
        return ThrowPairCheck(False, f"ik_failed:{exc}", v_release, t_flight)

    q_rel = np.asarray(q_rel, dtype=float).reshape(7)
    qdot_rel = np.asarray(qdot_rel, dtype=float).reshape(7)
    if np.any(q_rel < q_limits[:, 0] - 1e-9) or np.any(q_rel > q_limits[:, 1] + 1e-9):
        return ThrowPairCheck(False, "q_release_out_of_limits", v_release, t_flight, q_rel, qdot_rel)
    if np.any(np.abs(qdot_rel) > qdot_limits + 1e-9):
        return ThrowPairCheck(False, "qdot_release_out_of_limits", v_release, t_flight, q_rel, qdot_rel)

    # 2b) v_actual + angle + v_parallel_max
    state = pin_model.get_frame_world_state(q7=q_rel, qdot7=qdot_rel, frame_name=target_frame)
    v_actual = np.asarray(state["vel_w"], dtype=float).reshape(3)
    speed_des = float(np.linalg.norm(v_release))
    speed_act = float(np.linalg.norm(v_actual))
    angle_deg = None
    v_parallel_max = None
    if speed_des > 1e-9 and speed_act > 1e-9:
        cosang = float(np.clip(np.dot(v_actual, v_release) / (speed_act * speed_des), -1.0, 1.0))
        angle_deg = float(np.degrees(np.arccos(cosang)))
    if speed_des > 1e-9:
        d = v_release / speed_des
        import pinocchio as pin

        q_full = pin_model.make_q_full_from_arm7(q_rel)
        pin.forwardKinematics(pin_model.model, pin_model.data, q_full)
        pin.updateFramePlacements(pin_model.model, pin_model.data)
        J = pin.computeFrameJacobian(
            pin_model.model,
            pin_model.data,
            q_full,
            target_frame_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )
        J_arm = np.zeros((3, 7), dtype=float)
        for j in range(1, 8):
            jid = pin_model.model.getJointId(f"panda_joint{j}")
            idx_v = int(pin_model.model.joints[jid].idx_v)
            J_arm[:, j - 1] = J[:3, idx_v]
        row = d.reshape(1, 3) @ J_arm
        v_parallel_max = float(np.sum(np.abs(row.reshape(7)) * qdot_limits.reshape(7)))

    # 3) Run-up from INIT_ARM, optionally suggest auto-start.
    init_res = _check_reach_release_from_start(
        q_start=np.asarray(INIT_ARM, dtype=float).reshape(7),
        q_release=q_rel,
        qdot_release=qdot_rel,
        qddot_limits=qddot_limits,
        qdddot_limits=qdddot_limits,
    )
    init_ok = bool(init_res.feasible)
    auto_start_q7 = None
    if not init_ok and use_auto_start:
        q_start_auto = _backcompute_start_q_from_release(
            q_release=q_rel,
            qdot_release=qdot_rel,
            q_limits=q_limits,
            qddot_limits=qddot_limits,
            qdddot_limits=qdddot_limits,
        )
        res2 = _check_reach_release_from_start(
            q_start=q_start_auto,
            q_release=q_rel,
            qdot_release=qdot_rel,
            qddot_limits=qddot_limits,
            qdddot_limits=qdddot_limits,
        )
        if res2.feasible:
            auto_start_q7 = q_start_auto
        else:
            return ThrowPairCheck(
                False,
                f"auto_start_infeasible:{res2.reason}",
                v_release,
                t_flight,
                q_rel,
                qdot_rel,
                v_actual,
                angle_deg,
                v_parallel_max,
                init_ok,
                None,
            )

    feasible = init_ok or (auto_start_q7 is not None)
    reason = "ok" if feasible else f"init_runup_failed:{init_res.reason}"
    return ThrowPairCheck(
        feasible,
        reason,
        v_release,
        t_flight,
        q_rel,
        qdot_rel,
        v_actual,
        angle_deg,
        v_parallel_max,
        init_ok,
        auto_start_q7,
    )


def main() -> int:
    if "SOFT_THROW_URDF" not in os.environ:
        _load_env_from_file(os.path.join(os.path.dirname(__file__), "env.sh"))

    from core.throw_params import INIT_ARM, Q_LIMITS_7, QDOT_LIMITS_7, QDDOT_LIMITS_7, QDDDOT_LIMITS_7, PLANNER_QDDOT_LIMIT_SCALE
    from core.kinematic_forward import PinKinematics
    from core.ik_poly5_core import solve_ik_for_q_goal
    from core import tube_method

    parser = argparse.ArgumentParser(description="Validate feasibility of (landing, release) under joint limits.")
    parser.add_argument("--landing", type=_parse_vec3, required=True, help="Landing XYZ, e.g. 2.0,0.0,0.0")
    parser.add_argument("--release", type=_parse_vec3, required=True, help="Release XYZ, e.g. 0.4,0.0,0.9")
    parser.add_argument("--limit-scale", type=float, default=0.9, help="Safety scale for joint q/qdot limits (<=1).")
    parser.add_argument(
        "--qddot-scale",
        type=float,
        default=float(PLANNER_QDDOT_LIMIT_SCALE),
        help="Safety scale for joint acceleration limits during planning (<=1).",
    )
    parser.add_argument("--ik-max-iter", type=int, default=50)
    parser.add_argument("--use-auto-start", action="store_true", help="Also test back-computed start pose feasibility.")
    args = parser.parse_args()

    landing = np.asarray(args.landing, dtype=float).reshape(3)
    release = np.asarray(args.release, dtype=float).reshape(3)
    limit_scale = float(args.limit_scale)
    if not (0.0 < limit_scale <= 1.0):
        raise ValueError("--limit-scale must be in (0,1]")
    qddot_scale = float(args.qddot_scale)
    if not (0.0 < qddot_scale <= 1.0):
        raise ValueError("--qddot-scale must be in (0,1]")

    q_limits = np.asarray(Q_LIMITS_7, dtype=float).copy()
    # apply limit_scale to q_limits like tube_throw does (center scaling)
    q_min = q_limits[:, 0]
    q_max = q_limits[:, 1]
    center = 0.5 * (q_min + q_max)
    half = 0.5 * (q_max - q_min) * limit_scale
    q_limits = np.stack([center - half, center + half], axis=1)
    qdot_limits = np.asarray(QDOT_LIMITS_7, dtype=float) * limit_scale
    qddot_limits = np.asarray(QDDOT_LIMITS_7, dtype=float) * qddot_scale
    qdddot_limits = np.asarray(QDDDOT_LIMITS_7, dtype=float)

    check = check_throw_pair(
        landing=landing,
        release=release,
        v_release_override=None,
        limit_scale=limit_scale,
        qddot_scale=qddot_scale,
        ik_max_iter=int(args.ik_max_iter),
        use_auto_start=bool(args.use_auto_start),
    )
    if check.t_flight == check.t_flight:
        print("[info] ballistic v_release:", np.round(check.v_release, 4), "t_flight:", round(float(check.t_flight), 4))
    else:
        print("[info] v_release:", np.round(check.v_release, 4))
    if not check.feasible:
        print("[fail]", check.reason)
        return 2
    if check.q_release is not None:
        print("[ok] ik q_release:", np.round(check.q_release, 4))
    if check.qdot_release is not None:
        print("[ok] ik qdot_release:", np.round(check.qdot_release, 4))
    if check.v_actual is not None:
        print("[info] v_actual:", np.round(check.v_actual, 4), "angle_deg:", check.angle_deg)
    if check.v_parallel_max is not None:
        print("[info] v_parallel_max:", round(float(check.v_parallel_max), 4))
    if check.init_runup_ok:
        print("[ok] INIT_ARM run-up ok")
    elif check.auto_start_q7 is not None:
        print("[ok] auto-start feasible, q_start:", np.round(check.auto_start_q7, 4))
    print("[done] feasible")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
