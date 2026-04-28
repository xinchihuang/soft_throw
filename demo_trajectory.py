#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Export a joint-space demo trajectory (poly5) at 1000 Hz.

Input:
- q_start: 7 joint angles (rad)
- q_goal: 7 joint angles (rad)

Output:
- CSV with columns: t,q0..q6,dq0..dq6,ddq0..ddq6,tau0..tau6

Notes:
- The `tau` columns are intended for torque-command executors. By default we
  export them as zeros (safe).
- If you explicitly set `--tau-mode jerk`, the `tau*` columns become jerk
  (time-derivative of joint acceleration), which is useful for offline plots
  but MUST NOT be used as torque commands.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import numpy as np

from core.ik_poly5_core import poly5_trajectory
from core.kinematic_forward import PinKinematics
from core.throw_params import Q_LIMITS_7, QDOT_LIMITS_7, QDDOT_LIMITS_7
from core.throw_params import EE_FRAME


# ---------------- Defaults (edit in code) ----------------
DEFAULT_Q_START = [0.038830, 0.681215, -0.034082, -2.716049, -0.023610, 3.663629, -0.185898]
DEFAULT_Q_GOAL = [0.052365, 0.408409, 0.013253, -1.575655, -0.015081, 2.699226, -0.128522]
# Position limits already use soft limits in `core.throw_params.Q_LIMITS_7`.
# IMPORTANT: do not silently clip q_start/q_goal for execution, otherwise the
# first commanded sample will not match the robot state (can trigger reflex).
Q_LIMIT_SCALE = 1
# Apply conservatism on velocity limits; keep extra tracking headroom below.
QDOT_LIMIT_SCALE = 0.9
# Extra margin to leave tracking headroom (robot may overshoot commanded dq).
TRACKING_QDOT_MARGIN = 0.90

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


def _scale_q_limits(q_limits: np.ndarray, scale: float) -> np.ndarray:
    q_limits = np.asarray(q_limits, dtype=float)
    if q_limits.shape != (7, 2):
        raise ValueError(f"Expected q_limits shape (7,2), got {q_limits.shape}")
    scale = float(scale)
    if not (0.0 < scale <= 1.0):
        raise ValueError(f"limit scale must be in (0,1], got {scale}")
    q_min = q_limits[:, 0]
    q_max = q_limits[:, 1]
    center = 0.5 * (q_min + q_max)
    half = 0.5 * (q_max - q_min) * scale
    return np.stack([center - half, center + half], axis=1)


def _check_or_clip_q(
    q: np.ndarray,
    q_limits: np.ndarray,
    *,
    strict: bool,
    allow_clip: bool,
    name: str,
) -> np.ndarray:
    q = np.asarray(q, dtype=float).reshape(7)
    lo = np.asarray(q_limits[:, 0], dtype=float)
    hi = np.asarray(q_limits[:, 1], dtype=float)
    if np.any(q < lo) or np.any(q > hi):
        bad = np.where((q < lo) | (q > hi))[0].tolist()
        if strict and not allow_clip:
            joint_names = [f"panda_joint{i + 1}" for i in range(7)]
            details = []
            for i in bad:
                details.append(
                    f"j{i + 1} ({joint_names[i]}): q={float(q[i]):.6g} lim=[{float(lo[i]):.6g},{float(hi[i]):.6g}]"
                )
            raise ValueError(f"{name} violates effective joint limits: " + "; ".join(details))
        if not allow_clip:
            joint_names = [f"panda_joint{i + 1}" for i in range(7)]
            details = []
            for i in bad:
                details.append(
                    f"j{i + 1} ({joint_names[i]}): q={float(q[i]):.6g} lim=[{float(lo[i]):.6g},{float(hi[i]):.6g}]"
                )
            raise ValueError(
                f"{name} violates effective joint limits: "
                + "; ".join(details)
                + " (pass --clip-start-goal to clip)"
            )
    clipped = np.clip(q, lo, hi)
    if not np.allclose(clipped, q):
        bad = np.where(np.abs(clipped - q) > 0.0)[0].tolist()
        joint_names = [f"panda_joint{i + 1}" for i in range(7)]
        print(f"[warn] {name} clipped to limits (count={len(bad)}):", flush=True)
        for i in bad:
            qi = float(q[i])
            ci = float(clipped[i])
            print(
                f"  - j{i + 1} ({joint_names[i]}): q={qi:.6g} -> {ci:.6g}  lim=[{float(lo[i]):.6g},{float(hi[i]):.6g}]",
                flush=True,
            )
    return clipped


def _write_trace_with_tau_csv(csv_path: Path, t, q, qdot, qddot, *, tau_mode: str) -> None:
    t = np.asarray(t, dtype=float).reshape(-1)
    q = np.asarray(q, dtype=float)
    qdot = np.asarray(qdot, dtype=float)
    qddot = np.asarray(qddot, dtype=float)

    tau_mode = str(tau_mode)
    if tau_mode == "zero":
        tau = np.zeros_like(qddot)
    elif tau_mode == "jerk":
        if t.size < 2:
            tau = np.zeros_like(qddot)
        else:
            tau = np.gradient(qddot, t, axis=0)
    else:
        raise ValueError(f"Unknown tau_mode: {tau_mode!r}")

    # Use 0-based column naming (q0..q6) to match the online executor CSV convention.
    header = (
        ["t"]
        + [f"q{j}" for j in range(7)]
        + [f"dq{j}" for j in range(7)]
        + [f"ddq{j}" for j in range(7)]
        + [f"tau{j}" for j in range(7)]
    )
    data = np.column_stack([t, q, qdot, qddot, tau])
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(str(csv_path), data, delimiter=",", header=",".join(header), comments="")

def _write_ee_states_csv(csv_path: Path, t: np.ndarray, q: np.ndarray, qdot: np.ndarray, *, frame_name: str) -> None:
    t = np.asarray(t, dtype=float).reshape(-1)
    q = np.asarray(q, dtype=float)
    qdot = np.asarray(qdot, dtype=float)
    if q.shape[0] != t.size or qdot.shape[0] != t.size:
        raise ValueError("t, q, qdot must have the same length")

    pin_model = PinKinematics()
    out = np.zeros((t.size, 1 + 3 + 3 + 6), dtype=float)
    for i in range(t.size):
        state = pin_model.get_frame_world_state(q7=q[i], qdot7=qdot[i], frame_name=frame_name)
        out[i, 0] = float(t[i])
        out[i, 1:4] = state["pos_w"]
        out[i, 4:7] = state["vel_w"]
        out[i, 7:13] = state["twist_w"]

    header = ["t", "x", "y", "z", "vx", "vy", "vz", "wx", "wy", "wz", "tvx", "tvy", "tvz"]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(str(csv_path), out, delimiter=",", header=",".join(header), comments="")


def main() -> int:
    if "SOFT_THROW_URDF" not in os.environ:
        _load_env_from_file(str(Path(__file__).resolve().parent / "env.sh"))

    parser = argparse.ArgumentParser(description="Export poly5 joint demo trajectory at 1000 Hz.")
    parser.add_argument("--q-start", type=float, nargs=7, default=None, help="Start joint angles (rad)")
    parser.add_argument("--q-goal", type=float, nargs=7, default=None, help="Goal joint angles (rad)")
    parser.add_argument("--dt", type=float, default=0.001, help="Sample period (sec), default 0.001 (1000Hz)")
    parser.add_argument("--out", type=str, default="trajectory.csv")
    parser.add_argument("--limit-scale", type=float, default=1.0, help="Safety scale for joint position/velocity limits")
    parser.add_argument("--q-limit-scale", type=float, default=Q_LIMIT_SCALE, help="Extra scale for joint position limits")
    parser.add_argument("--qdot-limit-scale", type=float, default=QDOT_LIMIT_SCALE, help="Extra scale for joint velocity limits")
    parser.add_argument(
        "--tracking-qdot-margin",
        type=float,
        default=TRACKING_QDOT_MARGIN,
        help="Extra margin (<1) applied to qdot limits to leave tracking headroom",
    )
    parser.add_argument("--strict-limits", action="store_true", help="Error if q_start/q_goal violate limits")
    parser.add_argument(
        "--clip-start-goal",
        action="store_true",
        help="Clip q_start/q_goal into effective limits (not recommended for real execution)",
    )
    parser.add_argument("--progress", action="store_true", help="Print coarse poly5 feasibility search progress")
    parser.add_argument("--check-max-samples", type=int, default=2000, help="Max samples used during limit checking")
    parser.add_argument("--plot", dest="plot", action="store_true", help="Generate per-joint q/dq/ddq plots (PNG)")
    parser.add_argument("--no-plot", dest="plot", action="store_false", help="Disable plot generation")
    parser.add_argument("--plot-out-dir", type=str, default="joint_plots", help="Output dir for plots")
    parser.set_defaults(plot=True)
    parser.add_argument(
        "--tau-mode",
        type=str,
        default="zero",
        choices=["zero", "jerk"],
        help="How to export tau columns: 'zero' (safe for torque executors) or 'jerk' (offline only)",
    )
    parser.add_argument("--export-ee", action="store_true", help="Also export EE pose/velocity per timestep as CSV")
    parser.add_argument("--ee-out", type=str, default="ee_states.csv", help="Output CSV path for EE states")
    parser.add_argument("--ee-frame", type=str, default=EE_FRAME, help=f"EE frame name (default: {EE_FRAME})")
    args = parser.parse_args()

    dt = float(args.dt)
    if dt <= 0.0:
        raise ValueError("--dt must be > 0")

    limit_scale = float(args.limit_scale)
    q_limit_scale = float(args.q_limit_scale)
    qdot_limit_scale = float(args.qdot_limit_scale)
    tracking_qdot_margin = float(args.tracking_qdot_margin)
    if not (0.0 < tracking_qdot_margin <= 1.0):
        raise ValueError("--tracking-qdot-margin must be in (0,1]")
    q_limits = _scale_q_limits(Q_LIMITS_7, limit_scale * q_limit_scale)
    qdot_limits = np.asarray(QDOT_LIMITS_7, dtype=float) * limit_scale * qdot_limit_scale * tracking_qdot_margin
    qddot_limits = np.asarray(QDDOT_LIMITS_7, dtype=float)

    q_start_in = DEFAULT_Q_START if args.q_start is None else args.q_start
    q_goal_in = DEFAULT_Q_GOAL if args.q_goal is None else args.q_goal
    q_start = _check_or_clip_q(
        np.asarray(q_start_in, dtype=float),
        q_limits,
        strict=bool(args.strict_limits),
        allow_clip=bool(args.clip_start_goal),
        name="q_start",
    )
    q_goal = _check_or_clip_q(
        np.asarray(q_goal_in, dtype=float),
        q_limits,
        strict=bool(args.strict_limits),
        allow_clip=bool(args.clip_start_goal),
        name="q_goal",
    )

    t, q, qdot, qddot, _u = poly5_trajectory(
        q_start=q_start,
        q_goal=q_goal,
        q_limits=q_limits,
        qdot_limits=qdot_limits,
        qddot_limits=qddot_limits,
        control_dt=dt,
        qdot_start=np.zeros(7, dtype=float),
        qdot_goal=np.zeros(7, dtype=float),
        qddot_start=np.zeros(7, dtype=float),
        qddot_goal=np.zeros(7, dtype=float),
        check_max_samples=int(args.check_max_samples) if args.check_max_samples > 0 else None,
        progress=bool(args.progress),
        progress_prefix="[poly5_1kHz]",
        snap_final_time_to_dt=True,
    )

    out_path = Path(args.out)
    _write_trace_with_tau_csv(out_path, t, q, qdot, qddot, tau_mode=str(args.tau_mode))
    print(f"[ok] wrote {out_path} (samples={len(t)} duration_s={float(t[-1]) if len(t) else 0.0:.6g} dt={dt:g})", flush=True)

    q_min_eff = q_limits[:, 0]
    q_max_eff = q_limits[:, 1]
    max_qdot = np.max(np.abs(qdot), axis=0)
    max_qddot = np.max(np.abs(qddot), axis=0)
    print(
        f"[limits] arg_limit_scale={limit_scale:g} q_limit_scale={q_limit_scale:g} "
        f"qdot_limit_scale={qdot_limit_scale:g} tracking_qdot_margin={tracking_qdot_margin:g}",
        flush=True,
    )
    print(f"[limits] qdot_limits_eff={np.asarray(qdot_limits, dtype=float).round(6).tolist()}", flush=True)
    print(f"[stats] max_abs_qdot={max_qdot.round(6).tolist()}", flush=True)
    print(f"[stats] margin_qdot={(qdot_limits - max_qdot).round(6).tolist()}", flush=True)
    # Best-effort check for any position limit violations in generated samples.
    q_below = np.any(q < q_min_eff[None, :], axis=0)
    q_above = np.any(q > q_max_eff[None, :], axis=0)
    if bool(np.any(q_below) or np.any(q_above)):
        bad = [i + 1 for i in range(7) if (q_below[i] or q_above[i])]
        print(f"[warn] generated q samples exceed effective limits at joints: {bad}", flush=True)
    print(f"[stats] max_abs_qddot={max_qddot.round(6).tolist()}", flush=True)

    if bool(args.export_ee):
        ee_out = Path(args.ee_out)
        _write_ee_states_csv(ee_out, t, q, qdot, frame_name=str(args.ee_frame))
        print(f"[ok] wrote {ee_out} (frame={args.ee_frame})", flush=True)

    if bool(args.plot):
        from sim.plot_joint_traces import plot_from_csv

        plot_out_dir = str(args.plot_out_dir)
        plot_from_csv(str(out_path), plot_out_dir, time_unit="s")
        print(f"[ok] plots saved to {plot_out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
