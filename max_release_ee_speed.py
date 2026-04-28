#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Estimate the maximum achievable EE speed at a release joint configuration.

Goal:
- Given a start joint configuration q_start (assume qdot_start=0) and a release joint
  configuration q_release, compute a *conservative* upper bound on the EE linear speed
  at the moment the arm reaches q_release, while respecting:
  - joint position limits (Q_LIMITS_7)
  - joint velocity limits (QDOT_LIMITS_7)
  - joint acceleration limits (QDDOT_LIMITS_7)

Model/assumptions (conservative):
- Per joint, we assume a monotonic move from q_start to q_release (no overshoot).
- Starting joint velocity is 0.
- The maximum end velocity per joint achievable at the target, without overshoot, is:
    v_end_max = min(v_limit, sqrt(2 * a_limit * |dq|))
  This comes from accelerating at max acceleration over distance |dq|.
- End velocity sign is constrained to match sign(dq).
- We then maximize || J_lin(q_release) * qdot_end || over the resulting box constraints.
  Because ||A x|| is convex, the maximum over a box occurs at a vertex, so we enumerate
  2^7 vertices (fast).

Edit defaults in code below, or pass CLI args.
"""

from __future__ import annotations

import argparse
import os
import re
from itertools import product

import numpy as np

from core.kinematic_forward import PinKinematics, pin
from core.throw_params import EE_FRAME, INIT_ARM, Q_LIMITS_7, QDOT_LIMITS_7, QDDOT_LIMITS_7


# ---------------- Defaults (edit in code) ----------------
DEFAULT_Q_START = INIT_ARM.tolist()
DEFAULT_Q_RELEASE = INIT_ARM.tolist()
DEFAULT_FRAME = EE_FRAME


def _load_env_from_file(path: str) -> None:
    if not os.path.isfile(path):
        return
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
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


def _parse_args():
    p = argparse.ArgumentParser(description="Compute conservative max EE speed at q_release.")
    p.add_argument("--q-start", type=float, nargs=7, default=None, help="Start joint positions (rad)")
    p.add_argument("--q-release", type=float, nargs=7, default=None, help="Release joint positions (rad)")
    p.add_argument("--frame", type=str, default=DEFAULT_FRAME, help=f"Frame name (default: {DEFAULT_FRAME})")
    p.add_argument("--eps-dq", type=float, default=1e-6, help="Treat |dq|<eps as zero (no motion)")
    p.add_argument(
        "--strict-limits",
        action="store_true",
        help="Error out if q_start/q_release violate position limits (default: clip and warn).",
    )
    return p.parse_args()


def _check_q_limits(q: np.ndarray, name: str) -> None:
    q = np.asarray(q, dtype=float).reshape(7)
    lo = np.asarray(Q_LIMITS_7[:, 0], dtype=float)
    hi = np.asarray(Q_LIMITS_7[:, 1], dtype=float)
    if np.any(q < lo) or np.any(q > hi):
        bad = np.where((q < lo) | (q > hi))[0].tolist()
        joint_names = [f"panda_joint{i + 1}" for i in range(7)]
        details = []
        for i in bad:
            qi = float(q[i])
            lo_i = float(lo[i])
            hi_i = float(hi[i])
            if qi < lo_i:
                details.append(
                    f"  - j{i + 1} ({joint_names[i]}): q={qi:.6g} < min={lo_i:.6g} (delta={lo_i - qi:.6g})"
                )
            else:
                details.append(
                    f"  - j{i + 1} ({joint_names[i]}): q={qi:.6g} > max={hi_i:.6g} (delta={qi - hi_i:.6g})"
                )
        raise ValueError(
            f"{name} violates joint position limits (count={len(bad)}):\n" + "\n".join(details)
        )

def _clip_q_to_limits(q: np.ndarray, name: str) -> np.ndarray:
    q = np.asarray(q, dtype=float).reshape(7)
    lo = np.asarray(Q_LIMITS_7[:, 0], dtype=float)
    hi = np.asarray(Q_LIMITS_7[:, 1], dtype=float)
    clipped = np.clip(q, lo, hi)
    if not np.allclose(clipped, q):
        bad = np.where(np.abs(clipped - q) > 0.0)[0].tolist()
        joint_names = [f"panda_joint{i + 1}" for i in range(7)]
        print(f"[warn] {name} clipped to joint limits (count={len(bad)}):", flush=True)
        for i in bad:
            qi = float(q[i])
            ci = float(clipped[i])
            lo_i = float(lo[i])
            hi_i = float(hi[i])
            if qi < lo_i:
                why = f"below min {lo_i:.6g}"
            elif qi > hi_i:
                why = f"above max {hi_i:.6g}"
            else:
                why = "unknown"
            print(
                f"  - j{i + 1} ({joint_names[i]}): q={qi:.6g} -> {ci:.6g} ({why})",
                flush=True,
            )
    return clipped


def _linear_jacobian_arm7_world(pin_model: PinKinematics, q7: np.ndarray, frame_name: str) -> np.ndarray:
    q7 = np.asarray(q7, dtype=float).reshape(7)
    frame_id = pin_model.model.getFrameId(frame_name)
    if frame_id == len(pin_model.model.frames):
        raise ValueError(f"Frame not found: {frame_name!r}")

    q_full = pin_model.make_q_full_from_arm7(q7)
    pin.forwardKinematics(pin_model.model, pin_model.data, q_full)
    pin.updateFramePlacements(pin_model.model, pin_model.data)

    J6 = pin.computeFrameJacobian(pin_model.model, pin_model.data, q_full, frame_id, pin.ReferenceFrame.WORLD)  # 6 x nv
    J_lin_full = np.asarray(J6[3:6, :], dtype=float)  # 3 x nv

    J_arm = np.zeros((3, 7), dtype=float)
    for i in range(1, 8):
        jid = pin_model.model.getJointId(f"panda_joint{i}")
        idx_v = int(pin_model.model.joints[jid].idx_v)
        J_arm[:, i - 1] = J_lin_full[:, idx_v]
    return J_arm


def compute_conservative_qdot_end_bounds(q_start: np.ndarray, q_release: np.ndarray, *, eps_dq: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (qdot_max_abs(7,), qdot_sign(7,)) where qdot_end = qdot_sign * x, x in [0, qdot_max_abs]."""
    q_start = np.asarray(q_start, dtype=float).reshape(7)
    q_release = np.asarray(q_release, dtype=float).reshape(7)
    dq = q_release - q_start

    qdot_lim = np.asarray(QDOT_LIMITS_7, dtype=float).reshape(7)
    qddot_lim = np.asarray(QDDOT_LIMITS_7, dtype=float).reshape(7)

    qdot_max = np.zeros(7, dtype=float)
    qdot_sign = np.ones(7, dtype=float)
    for i in range(7):
        d = float(dq[i])
        if abs(d) < float(eps_dq):
            qdot_max[i] = 0.0
            qdot_sign[i] = 1.0
            continue
        qdot_sign[i] = 1.0 if d > 0.0 else -1.0
        v_dist = float(np.sqrt(max(0.0, 2.0 * float(qddot_lim[i]) * abs(d))))
        qdot_max[i] = float(min(float(qdot_lim[i]), v_dist))
    return qdot_max, qdot_sign


def maximize_ee_speed_at_release(
    pin_model: PinKinematics,
    q_release: np.ndarray,
    qdot_max_abs: np.ndarray,
    qdot_sign: np.ndarray,
    *,
    frame_name: str,
) -> dict:
    J = _linear_jacobian_arm7_world(pin_model, q_release, frame_name)  # 3x7
    bounds = np.asarray(qdot_max_abs, dtype=float).reshape(7)
    signs = np.asarray(qdot_sign, dtype=float).reshape(7)

    best = None
    for bits in product([0.0, 1.0], repeat=7):
        x = bounds * np.asarray(bits, dtype=float)
        qdot = signs * x
        v = J @ qdot
        speed = float(np.linalg.norm(v))
        if best is None or speed > best["speed_m_s"]:
            best = {
                "qdot_end": qdot,
                "ee_vel_w": v,
                "speed_m_s": speed,
                "vertex_bits": np.asarray(bits, dtype=float),
            }
    assert best is not None
    return best


def main() -> int:
    if "SOFT_THROW_URDF" not in os.environ:
        _load_env_from_file(os.path.join(os.path.dirname(__file__), "env.sh"))

    args = _parse_args()
    q_start = np.asarray(DEFAULT_Q_START if args.q_start is None else args.q_start, dtype=float).reshape(7)
    q_release = np.asarray(DEFAULT_Q_RELEASE if args.q_release is None else args.q_release, dtype=float).reshape(7)
    frame = str(args.frame)

    if bool(args.strict_limits):
        _check_q_limits(q_start, "q_start")
        _check_q_limits(q_release, "q_release")
    else:
        q_start = _clip_q_to_limits(q_start, "q_start")
        q_release = _clip_q_to_limits(q_release, "q_release")

    pin_model = PinKinematics()
    p_release = pin_model.get_frame_world_pos(q_release, frame)

    qdot_max_abs, qdot_sign = compute_conservative_qdot_end_bounds(q_start, q_release, eps_dq=float(args.eps_dq))
    best = maximize_ee_speed_at_release(
        pin_model,
        q_release,
        qdot_max_abs,
        qdot_sign,
        frame_name=frame,
    )

    print(f"[frame] {frame} pos_w={p_release.tolist()}", flush=True)
    print(f"[dq] q_start={q_start.tolist()} q_release={q_release.tolist()}", flush=True)
    print(f"[qdot_end_bounds_abs] {qdot_max_abs.tolist()}", flush=True)
    print(f"[qdot_end_sign] {qdot_sign.tolist()}", flush=True)
    print(f"[best] qdot_end={best['qdot_end'].round(6).tolist()}", flush=True)
    print(f"[best] ee_vel_w={best['ee_vel_w'].round(6).tolist()} speed_m_s={best['speed_m_s']:.6g}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
