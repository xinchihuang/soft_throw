#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse

import numpy as np

import os
import re
import json


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


if "SOFT_THROW_URDF" not in os.environ:
    _load_env_from_file(os.path.join(os.path.dirname(__file__), "env.sh"))

from core.throw_params import (
    INIT_ARM,
    Q_LIMITS_7,
    QDOT_LIMITS_7,
    QDDOT_LIMITS_7,
    QDDDOT_LIMITS_7,
    PLANNER_QDDOT_LIMIT_SCALE,
    TUBE_DECEL_J6_BACKOFF_RAD,
)
from core.ik_scurve_core import stopping_distance_jerk_limited
from core.ik_poly5_core import solve_ik_for_q_goal
from validate_throw_pair import check_throw_pair
from core.throw_params import EE_VEL_MAX
from core.kinematic_forward import PinKinematics
from core.trajectory_planner import (
    build_execution_trajectory,
    build_execution_trajectory_from_waypoints,
    build_decel_trajectory,
    write_trace_with_tau_csv,
)
from core import tube_method
from core.input_checks import validate_release_pose


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
    out = np.stack([center - half, center + half], axis=1)
    return out


def _parse_vec3(text: str) -> np.ndarray:
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Expected 3 comma-separated values, e.g. 0.4,0.0,0.9")
    return np.array([float(p) for p in parts], dtype=float)


def _plot_tube_html(
    target: np.ndarray,
    release: np.ndarray,
    hold_sec: float,
    method: str,
    n_window_samples: int | None,
    nominal_flight_time: float,
    out_html: str,
    out_image: str | None,
    open_html: bool,
) -> None:
    import plotly.graph_objects as go

    target = np.asarray(target, dtype=float)
    release = np.asarray(release, dtype=float)

    def _as_traj(arr: np.ndarray | None) -> np.ndarray | None:
        if arr is None:
            return None
        traj = np.asarray(arr, dtype=float)
        if traj.ndim == 1:
            if traj.size == 3:
                traj = traj.reshape(1, 3)
            else:
                return None
        if traj.shape[1] != 3 and traj.shape[0] == 3:
            traj = traj.T
        if traj.shape[1] != 3:
            return None
        return traj

    middle_result = tube_method.generate_middle_trajectory_method(
        target_point=target,
        release_point=release,
        flight_time_range=(0.45, 0.85),
        n_family=21,
        g=tube_method.G,
        n_pts=120,
    )
    fig = go.Figure()

    for i, traj in enumerate(middle_result["family_trajs"]):
        traj = _as_traj(traj)
        if traj is None or len(traj) < 2:
            continue
        fig.add_trace(
            go.Scatter3d(
                x=traj[:, 0],
                y=traj[:, 1],
                z=traj[:, 2],
                mode="lines",
                line=dict(color="royalblue", width=3),
                opacity=0.25,
                name="Middle method family" if i == 0 else None,
                showlegend=(i == 0),
            )
        )

    fig.add_trace(
        go.Scatter3d(
            x=[release[0]],
            y=[release[1]],
            z=[release[2]],
            mode="markers",
            marker=dict(color="black", size=7),
            name="Nominal release",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[target[0]],
            y=[target[1]],
            z=[target[2]],
            mode="markers",
            marker=dict(color="green", size=9, symbol="diamond"),
            name="Target",
        )
    )

    selected = _as_traj(middle_result["selected_traj"])
    if selected is not None and len(selected) >= 2:
        fig.add_trace(
            go.Scatter3d(
                x=selected[:, 0],
                y=selected[:, 1],
                z=selected[:, 2],
                mode="lines",
                line=dict(color="orange", width=8),
                name="Middle selected",
            )
        )

    fig.update_layout(
        title="Middle-method ballistic family",
        scene=dict(
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="z",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, b=0, t=60),
        showlegend=True,
    )

    fig.write_html(out_html, include_plotlyjs="cdn")
    if out_image:
        try:
            import plotly.io as pio
            pio.write_image(fig, out_image)
        except Exception as exc:
            print(f"[tube] Failed to write image '{out_image}': {exc}")
    if open_html:
        import webbrowser
        webbrowser.open(f"file://{os.path.abspath(out_html)}")


def _concat_with_offset(t1, q1, qd1, qdd1, t2, q2, qd2, qdd2):
    if t1.size == 0:
        return t2, q2, qd2, qdd2
    if t2.size:
        t2 = t2[1:]
        q2 = q2[1:]
        qd2 = qd2[1:]
        qdd2 = qdd2[1:]
        if t2.size == 0:
            return t1, q1, qd1, qdd1
    t2 = t2 + float(t1[-1])
    return (
        np.concatenate([t1, t2]),
        np.concatenate([q1, q2]),
        np.concatenate([qd1, qd2]),
        np.concatenate([qdd1, qdd2]),
    )


def _write_phase_sidecar_json(csv_path: str, *, pre_end: float, win_end: float, end: float, with_window: bool) -> str:
    payload = {
        "time_unit": "s",
        "with_window": bool(with_window),
        "phases": [
            {"name": "pre", "t_start": 0.0, "t_end": float(pre_end)},
        ],
    }
    if with_window:
        payload["phases"].append({"name": "window", "t_start": float(pre_end), "t_end": float(win_end)})
    payload["phases"].append({"name": "decel", "t_start": float(win_end), "t_end": float(end)})

    out_path = f"{csv_path}.phases.json"
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return out_path


def _export_ee_states_csv(
    *,
    out_csv: str,
    t: np.ndarray,
    q: np.ndarray,
    qdot: np.ndarray,
    frame_name: str,
    time_unit: str = "s",
) -> None:
    t = np.asarray(t, dtype=float).reshape(-1)
    q = np.asarray(q, dtype=float)
    qdot = np.asarray(qdot, dtype=float)
    if t.size != q.shape[0] or t.size != qdot.shape[0]:
        raise ValueError("t, q, qdot must have the same length")

    if time_unit == "ms":
        t_out = t * 1000.0
    elif time_unit == "s":
        t_out = t
    else:
        raise ValueError(f"Unsupported time_unit: {time_unit!r} (expected 's' or 'ms')")

    pin_model = PinKinematics()
    out = np.zeros((t.size, 1 + 3 + 3 + 6), dtype=float)
    for i in range(t.size):
        state = pin_model.get_frame_world_state(q7=q[i], qdot7=qdot[i], frame_name=frame_name)
        out[i, 0] = float(t_out[i])
        out[i, 1:4] = state["pos_w"]
        out[i, 4:7] = state["vel_w"]
        out[i, 7:13] = state["twist_w"]

    header = ["t", "x", "y", "z", "vx", "vy", "vz", "wx", "wy", "wz", "tvx", "tvy", "tvz"]
    np.savetxt(out_csv, out, delimiter=",", header=",".join(header), comments="")


def _backcompute_start_q_from_release(
    *,
    q_release: np.ndarray,
    qdot_release: np.ndarray,
    q_limits: np.ndarray,
    qddot_limits: np.ndarray,
    qdddot_limits: np.ndarray,
) -> np.ndarray:
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
        # Available distance to accelerate into q_release with end speed v, staying within limits.
        if v > 0.0:
            avail = float(q_release[j] - q_min[j])
        else:
            avail = float(q_max[j] - q_release[j])
        avail = max(0.0, avail)
        # Minimal distance required to go 0 -> |v| under jerk/acc limits.
        need = stopping_distance_jerk_limited(abs(v), amax=float(qddot_limits[j]), jmax=float(qdddot_limits[j]))
        if need > avail + 1e-12:
            # Not enough room: fall back to the closest feasible start (at the limit).
            need = avail
        q_start[j] = q_release[j] - np.sign(v) * need
    return np.clip(q_start, q_min, q_max)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate tube-throw joint trajectory.")
    # Backward-compatible comma-separated args (preferred for configs/scripts).
    parser.add_argument("--target", type=_parse_vec3, required=False, help="Target XYZ in world frame, e.g. 2.0,0.0,0.0")
    parser.add_argument("--release", type=_parse_vec3, required=False, help="Release XYZ in world frame, e.g. 0.4,0.0,0.9")
    # Human-friendly space-separated aliases (e.g. --release-pos 0.4 0 0.9).
    parser.add_argument(
        "--target-pos",
        "--target_pos",
        dest="target_pos",
        type=float,
        nargs=3,
        default=None,
        help="Target XYZ in world frame (3 floats), e.g. --target-pos 2.0 0.0 0.0",
    )
    parser.add_argument(
        "--release-pos",
        "--release_pos",
        dest="release_pos",
        type=float,
        nargs=3,
        default=None,
        help="Release XYZ in world frame (3 floats), e.g. --release-pos 0.4 0.0 0.9",
    )
    parser.add_argument("--hold-sec", type=float, default=0.5, help="Hold duration for tube window (sec)")
    parser.add_argument("--method", type=str, default="middle", choices=["middle", "paper_like"])
    parser.add_argument("--control-dt", type=float, default=0.01)
    parser.add_argument("--waypoint-density", type=int, default=10)
    parser.add_argument("--path-plan-waypoints", type=int, default=2)
    parser.add_argument("--n-window-samples", type=int, default=3)
    parser.add_argument(
        "--window-segments",
        type=int,
        default=1,
        help="How many joint-space segments to use for the tube window (1 uses only start/end waypoints).",
    )
    parser.add_argument("--nominal-flight-time", type=float, default=0.65)
    parser.add_argument(
        "--limit-scale",
        type=float,
        default=0.9,
        help="Safety scale for joint position range and joint velocity limits (<=1.0).",
    )
    parser.add_argument(
        "--qddot-scale",
        type=float,
        default=float(PLANNER_QDDOT_LIMIT_SCALE),
        help="Safety scale for joint acceleration limits during planning (<=1.0).",
    )
    parser.add_argument(
        "--max-pre-duration-sec",
        type=float,
        default=5.0,
        help="If matching release velocity makes the pre segment too long, automatically scale down the velocity.",
    )
    parser.add_argument(
        "--max-duration-sec",
        type=float,
        default=8.0,
        help="If planning makes the full trajectory too long, automatically scale down the window velocities.",
    )
    parser.add_argument("--ik-max-iter", type=int, default=50)
    parser.add_argument(
        "--poly5-max-samples",
        type=int,
        default=0,
        help="Cap samples per poly5 segment (0 disables capping; recommended for 1kHz export).",
    )
    parser.add_argument(
        "--segment-method",
        type=str,
        default="scurve",
        choices=["poly5", "scurve"],
        help="Trajectory segment generator: 'poly5' (existing) or 'scurve' (jerk-limited).",
    )
    parser.add_argument("--verbose", action="store_true", help="Print IK progress per waypoint")
    parser.add_argument("--progress", action="store_true", help="Print coarse progress while planning (IK/poly5)")
    parser.add_argument(
        "--auto-start",
        action="store_true",
        help="If INIT_ARM cannot reach the release (pos,vel) under limits, back-compute a feasible start joint pose.",
    )
    parser.add_argument(
        "--precheck",
        dest="precheck",
        action="store_true",
        help="Run feasibility precheck (ballistic/IK/run-up) before planning; exits early if infeasible.",
    )
    parser.add_argument(
        "--no-precheck",
        dest="precheck",
        action="store_false",
        help="Disable feasibility precheck.",
    )
    parser.add_argument("--save-csv", type=str, default=None, help="Optional path to save t,q,qdot,qddot as CSV")
    parser.add_argument("--export-ee", dest="export_ee", action="store_true", help="Also export EE pose/velocity per timestep as CSV")
    parser.add_argument("--no-export-ee", dest="export_ee", action="store_false", help="Disable EE CSV export")
    parser.add_argument("--ee-out", type=str, default="ee_states.csv", help="Output CSV path for EE states")
    parser.add_argument("--ee-frame", type=str, default="panda_hand", help="EE frame name for EE export")
    parser.add_argument("--plot", dest="plot", action="store_true", help="Generate per-joint plots from saved CSV")
    parser.add_argument("--no-plot", dest="plot", action="store_false", help="Disable per-joint plots")
    parser.add_argument("--plot-out-dir", type=str, default="joint_plots")
    parser.add_argument("--plot-tube", dest="plot_tube", action="store_true", help="Generate tube window HTML animation")
    parser.add_argument("--no-plot-tube", dest="plot_tube", action="store_false", help="Disable tube HTML animation")
    parser.add_argument("--plot-tube-html", type=str, default="tube_window.html")
    parser.add_argument("--plot-tube-image", type=str, default=None, help="Optional PNG/SVG output path for tube plot")
    parser.add_argument("--open-html", dest="open_html", action="store_true", help="Open tube HTML after generation")
    parser.add_argument("--no-open-html", dest="open_html", action="store_false", help="Do not open tube HTML")
    parser.add_argument("--plot-tube-only", action="store_true", help="Only generate tube plot (skip IK/trajectory)")
    parser.add_argument(
        "--no-window",
        dest="with_window",
        action="store_false",
        help="Do not include the moving tube window segment in the planned trajectory (pre + decel only).",
    )
    parser.add_argument("--compare-no-tube", action="store_true", help="Also generate baseline without tube window")
    parser.add_argument("--compare-csv", type=str, default="trajectory_no_tube.csv")
    parser.add_argument("--baseline-decel-min-sec", type=float, default=0.5)
    parser.add_argument(
        "--decel-j6-backoff-rad",
        type=float,
        default=float(TUBE_DECEL_J6_BACKOFF_RAD),
        help="During decel, command joint6 to rotate back by this amount from the end of the window segment.",
    )
    parser.set_defaults(open_html=True, plot_tube=True, plot=True, export_ee=True, with_window=True, precheck=True)
    args = parser.parse_args()

    target = args.target
    if target is None and args.target_pos is not None:
        target = np.asarray(args.target_pos, dtype=float).reshape(3)
    release = args.release
    if release is None and args.release_pos is not None:
        release = np.asarray(args.release_pos, dtype=float).reshape(3)
    if target is None:
        raise SystemExit("tube_throw.py: missing target; use --target x,y,z or --target-pos x y z")
    if release is None:
        raise SystemExit("tube_throw.py: missing release; use --release x,y,z or --release-pos x y z")

    validate_release_pose(release)

    if args.ik_max_iter > 150:
        args.ik_max_iter = 150

    limit_scale = float(args.limit_scale)
    q_limits = _scale_q_limits(Q_LIMITS_7, limit_scale)
    qdot_limits = np.asarray(QDOT_LIMITS_7, dtype=float) * limit_scale
    qddot_scale = float(args.qddot_scale)
    if not (0.0 < qddot_scale <= 1.0):
        raise ValueError(f"--qddot-scale must be in (0,1], got {qddot_scale}")
    qddot_limits = np.asarray(QDDOT_LIMITS_7, dtype=float) * qddot_scale
    qdddot_limits = np.asarray(QDDDOT_LIMITS_7, dtype=float)

    poly5_max_samples = int(args.poly5_max_samples)
    if poly5_max_samples <= 0:
        poly5_max_samples = None
    poly5_max_samples_try = poly5_max_samples
    if poly5_max_samples_try is None:
        poly5_max_samples_try = 2000

    if args.plot and args.save_csv is None:
        args.save_csv = "trajectory.csv"

    if not args.plot_tube_only:
        print("[tube] window start", flush=True)
        window = tube_method.generate_tube_window_trajectory(
            target_point=target,
            release_point=release,
            hold_sec=args.hold_sec,
            n_window_samples=args.n_window_samples,
            method=args.method,
            nominal_flight_time=args.nominal_flight_time,
            g=tube_method.G,
        )
        meta = window["meta"]
        print("[tube] window done", flush=True)

        p_release = window["release_points"][0]
        v_release = window["release_velocities"][0]
        p_list = window["release_points"]
        v_list = window["release_velocities"]
        window_segments = max(1, int(args.window_segments))
        if window_segments == 1 and len(p_list) >= 2:
            p_list = [p_list[0], p_list[-1]]
            v_list = [v_list[0], v_list[-1]]

        # Optional precheck before planning: verify (landing, release) is feasible.
        # IMPORTANT: do NOT just clamp/scale release speed; if the landing is too far
        # to reach under EE_VEL_MAX, we must fail early.
        start_q7 = np.asarray(INIT_ARM, dtype=float).reshape(7)
        v_release_nom = np.asarray(v_release, dtype=float).reshape(3)
        if args.precheck:
            check = check_throw_pair(
                landing=target,
                release=p_release,
                v_release_override=None,  # let validator decide if ballistic is feasible under EE_VEL_MAX
                limit_scale=float(args.limit_scale),
                qddot_scale=float(args.qddot_scale),
                ik_max_iter=int(args.ik_max_iter),
                use_auto_start=bool(args.auto_start),
            )
            if not check.feasible:
                raise SystemExit(f"[tube] precheck failed: {check.reason}")

            # Use the validator's feasible ballistic release velocity (<= EE_VEL_MAX).
            v_release = np.asarray(check.v_release, dtype=float).reshape(3)
            scale = float(np.linalg.norm(v_release)) / max(1e-12, float(np.linalg.norm(v_release_nom)))
            if abs(scale - 1.0) > 1e-6:
                v_list = [np.asarray(v, dtype=float).reshape(3) * scale for v in v_list]
                print(
                    f"[tube] precheck: rescaled window velocities by {scale:.4g} (|v| {np.linalg.norm(v_release_nom):.4g}->{np.linalg.norm(v_release):.4g})",
                    flush=True,
                )
            if check.auto_start_q7 is not None:
                start_q7 = np.asarray(check.auto_start_q7, dtype=float).reshape(7)
                print("[tube] precheck: using auto-start q7", np.round(start_q7, 4), flush=True)

            meta = dict(meta) if isinstance(meta, dict) else {}
            meta["selected_release_point"] = np.asarray(p_release, dtype=float).reshape(3).copy()
            meta["selected_release_velocity"] = np.asarray(v_release, dtype=float).reshape(3).copy()

        pin_model = PinKinematics()
        # Pre-compute a release joint state (q,qdot) via IK so we can back-compute a feasible start pose if needed.
        target_frame = "panda_link7"
        target_frame_id = pin_model.model.getFrameId(target_frame)
        if target_frame_id == len(pin_model.model.frames):
            raise RuntimeError(f"[pin] target frame not found: {target_frame}")
        _p0, R_des = pin_model.get_frame_world_pose(np.asarray(INIT_ARM, dtype=float).reshape(7), target_frame)
        q_rel, qdot_rel = solve_ik_for_q_goal(
            pin_model,
            target_frame_id,
            np.asarray(INIT_ARM, dtype=float).reshape(7),
            np.asarray(p_release, dtype=float).reshape(3),
            np.asarray(R_des, dtype=float).reshape(3, 3),
            q_limits,
            qdot_limits,
            qddot_limits,
            control_dt=args.control_dt,
            max_iter=int(args.ik_max_iter),
            kp_pos=2.0,
            kp_rot=1.0,
            v_j7_des=np.asarray(v_release, dtype=float).reshape(3),
        )
        # Check whether INIT_ARM provides enough joint-space "run-up" to hit qdot_rel at q_rel.
        if args.auto_start:
            q_start_auto = _backcompute_start_q_from_release(
                q_release=q_rel,
                qdot_release=qdot_rel,
                q_limits=q_limits,
                qddot_limits=qddot_limits,
                qdddot_limits=qdddot_limits,
            )
            # If auto differs and INIT_ARM is too close (per-joint), use auto start.
            # Heuristic: if any joint lacks the minimal accel distance margin, switch.
            need_switch = False
            for j in range(7):
                vj = float(qdot_rel[j])
                if abs(vj) <= 1e-12:
                    continue
                need = stopping_distance_jerk_limited(abs(vj), amax=float(qddot_limits[j]), jmax=float(qdddot_limits[j]))
                avail = abs(float(q_rel[j] - start_q7[j]))
                if avail + 1e-9 < need:
                    need_switch = True
                    break
            if need_switch:
                start_q7 = q_start_auto
                print("[tube] auto-start enabled: using back-computed start_q7", np.round(start_q7, 4), flush=True)
        v_scale = 1.0
        last_exc = None
        attempt = 0
        t_pre = q_pre = qdot_pre = qddot_pre = None
        t_win = q_win = qdot_win = qddot_win = None
        t_dec = q_dec = qdot_dec = qddot_dec = None
        q_end = qdot_end = None
        while True:
            attempt += 1
            pre_target = np.concatenate([p_release, v_release * float(v_scale)], axis=0)
            if args.progress:
                print(f"[tube] attempt={attempt} v_scale={v_scale:g}", flush=True)
            try:
                print("[tube] pre segment start", flush=True)
                t_pre, q_pre, qdot_pre, qddot_pre, _u = build_execution_trajectory(
                    target_pose_vel=pre_target,
                    start_q7=start_q7,
                    q_limits=q_limits,
                    qdot_limits=qdot_limits,
                    qddot_limits=qddot_limits,
                    control_dt=args.control_dt,
                    waypoint_density=args.waypoint_density,
                    path_plan_waypoints=args.path_plan_waypoints,
                    segment_method=args.segment_method,
                    qdddot_limits=qdddot_limits if args.segment_method == "scurve" else None,
                    pin_model=pin_model,
                    qdot_start=None,
                    joint_limit_margin=0.0,
                    ik_max_iter=args.ik_max_iter,
                    poly5_max_samples=poly5_max_samples_try,
                    verbose=(args.verbose or args.progress),
                )
                pre_duration = float(t_pre[-1]) if len(t_pre) else 0.0
                if pre_duration > float(args.max_pre_duration_sec) and v_scale > 1e-3:
                    v_scale *= 0.5
                    continue

                if args.with_window:
                    print("[tube] window segment start", flush=True)
                    (
                        t_win,
                        q_win,
                        qdot_win,
                        qddot_win,
                        _u_win,
                        q_end,
                        qdot_end,
                    ) = build_execution_trajectory_from_waypoints(
                        waypoints=[np.asarray(p, dtype=float) for p in p_list],
                        v_wp=[np.asarray(v, dtype=float) * float(v_scale) for v in v_list],
                        start_q7=q_pre[-1] if len(q_pre) else INIT_ARM,
                        q_limits=q_limits,
                        qdot_limits=qdot_limits,
                        qddot_limits=qddot_limits,
                        control_dt=args.control_dt,
                        waypoint_density=args.waypoint_density,
                        segment_method=args.segment_method,
                        qdddot_limits=qdddot_limits if args.segment_method == "scurve" else None,
                        pin_model=pin_model,
                        qdot_start=qdot_pre[-1] if len(qdot_pre) else None,
                        joint_limit_margin=0.0,
                        ik_max_iter=args.ik_max_iter,
                        poly5_max_samples=poly5_max_samples_try,
                        verbose=(args.verbose or args.progress),
                    )
                else:
                    t_win = np.zeros(0, dtype=float)
                    q_win = np.zeros((0, 7), dtype=float)
                    qdot_win = np.zeros((0, 7), dtype=float)
                    qddot_win = np.zeros((0, 7), dtype=float)
                    q_end = q_pre[-1].copy() if len(q_pre) else np.asarray(INIT_ARM, dtype=float).reshape(7)
                    qdot_end = qdot_pre[-1].copy() if len(qdot_pre) else np.zeros(7, dtype=float)

                print("[tube] decel segment start", flush=True)
                decel_goal = np.full(7, np.nan, dtype=float)
                decel_goal[5] = float(q_end[5]) - float(args.decel_j6_backoff_rad)
                t_dec, q_dec, qdot_dec, qddot_dec, _u_dec = build_decel_trajectory(
                    start_q7=q_end,
                    start_qdot7=qdot_end,
                    goal_q7=decel_goal,
                    q_limits=q_limits,
                    qdot_limits=qdot_limits,
                    qddot_limits=qddot_limits,
                    control_dt=args.control_dt,
                    waypoint_density=args.waypoint_density,
                    poly5_max_samples=poly5_max_samples_try,
                    segment_method=args.segment_method,
                    qdddot_limits=qdddot_limits if args.segment_method == "scurve" else None,
                )

                t, q, qdot, qddot = _concat_with_offset(
                    t_pre, q_pre, qdot_pre, qddot_pre, t_win, q_win, qdot_win, qddot_win
                )
                t, q, qdot, qddot = _concat_with_offset(t, q, qdot, qddot, t_dec, q_dec, qdot_dec, qddot_dec)

                duration = float(t[-1]) if len(t) else 0.0
                if duration <= float(args.max_duration_sec) or v_scale <= 1e-3:
                    break
                v_scale *= 0.5
                continue
            except Exception as exc:
                last_exc = exc
                if v_scale <= 1e-3:
                    raise
                v_scale *= 0.5

        if last_exc is not None and v_scale <= 1e-3:
            raise last_exc

        # If we capped samples during feasibility search, regenerate full-resolution export once feasible.
        if poly5_max_samples is None and poly5_max_samples_try is not None:
            pre_target = np.concatenate([p_release, v_release * float(v_scale)], axis=0)
            t_pre, q_pre, qdot_pre, qddot_pre, _u = build_execution_trajectory(
                target_pose_vel=pre_target,
                start_q7=start_q7,
                q_limits=q_limits,
                qdot_limits=qdot_limits,
                qddot_limits=qddot_limits,
                control_dt=args.control_dt,
                waypoint_density=args.waypoint_density,
                path_plan_waypoints=args.path_plan_waypoints,
                segment_method=args.segment_method,
                qdddot_limits=qdddot_limits if args.segment_method == "scurve" else None,
                pin_model=pin_model,
                qdot_start=None,
                joint_limit_margin=0.0,
                ik_max_iter=args.ik_max_iter,
                poly5_max_samples=None,
                verbose=False,
            )
            if args.with_window:
                (
                    t_win,
                    q_win,
                    qdot_win,
                    qddot_win,
                    _u_win,
                    q_end,
                    qdot_end,
                ) = build_execution_trajectory_from_waypoints(
                    waypoints=[np.asarray(p, dtype=float) for p in p_list],
                    v_wp=[np.asarray(v, dtype=float) * float(v_scale) for v in v_list],
                    start_q7=q_pre[-1] if len(q_pre) else INIT_ARM,
                    q_limits=q_limits,
                    qdot_limits=qdot_limits,
                    qddot_limits=qddot_limits,
                    control_dt=args.control_dt,
                    waypoint_density=args.waypoint_density,
                    segment_method=args.segment_method,
                    qdddot_limits=qdddot_limits if args.segment_method == "scurve" else None,
                    pin_model=pin_model,
                    qdot_start=qdot_pre[-1] if len(qdot_pre) else None,
                    joint_limit_margin=0.0,
                    ik_max_iter=args.ik_max_iter,
                    poly5_max_samples=None,
                    verbose=False,
                )
            else:
                t_win = np.zeros(0, dtype=float)
                q_win = np.zeros((0, 7), dtype=float)
                qdot_win = np.zeros((0, 7), dtype=float)
                qddot_win = np.zeros((0, 7), dtype=float)
                q_end = q_pre[-1].copy() if len(q_pre) else np.asarray(INIT_ARM, dtype=float).reshape(7)
                qdot_end = qdot_pre[-1].copy() if len(qdot_pre) else np.zeros(7, dtype=float)
            decel_goal = np.full(7, np.nan, dtype=float)
            decel_goal[5] = float(q_end[5]) - float(args.decel_j6_backoff_rad)
            t_dec, q_dec, qdot_dec, qddot_dec, _u_dec = build_decel_trajectory(
                start_q7=q_end,
                start_qdot7=qdot_end,
                goal_q7=decel_goal,
                q_limits=q_limits,
                qdot_limits=qdot_limits,
                qddot_limits=qddot_limits,
                control_dt=args.control_dt,
                waypoint_density=args.waypoint_density,
                poly5_max_samples=None,
                segment_method=args.segment_method,
                qdddot_limits=qdddot_limits if args.segment_method == "scurve" else None,
            )
            t, q, qdot, qddot = _concat_with_offset(
                t_pre, q_pre, qdot_pre, qddot_pre, t_win, q_win, qdot_win, qddot_win
            )
            t, q, qdot, qddot = _concat_with_offset(t, q, qdot, qddot, t_dec, q_dec, qdot_dec, qddot_dec)
        print("[tube] trajectory done", flush=True)

        print("Trajectory points:", len(t))
        print("Trajectory duration (s):", float(t[-1]) if len(t) else 0.0)
        if t_pre is not None and t_win is not None and t_dec is not None:
            print(f"Pre duration (s): {float(t_pre[-1]) if len(t_pre) else 0.0:.6f} (samples={len(t_pre)})")
            if args.with_window:
                print(
                    f"Window duration (s): {float(t_win[-1]) if len(t_win) else 0.0:.6f} "
                    f"(ref_points={len(p_list)}, segments={max(0, len(p_list) - 1)}, samples={len(t_win)})"
                )
            else:
                print("Window duration (s): 0.000000 (skipped)")
            print(f"Decel duration (s): {float(t_dec[-1]) if len(t_dec) else 0.0:.6f} (samples={len(t_dec)})")
        # Note: `meta` may be overridden when we rescale release speeds for executability.
        print("Selected release point:", np.round(meta.get("selected_release_point", np.asarray(p_release, dtype=float)), 4))
        print("Selected release velocity:", np.round(meta.get("selected_release_velocity", np.asarray(v_release, dtype=float)), 4))
        if v_scale < 0.999:
            print(f"[tube] pre segment release velocity scaled by {v_scale:g}", flush=True)

        t_base = q_base = qdot_base = qddot_base = None
        if args.compare_no_tube:
            print("[tube] baseline start", flush=True)
            decel_goal = np.full(7, np.nan, dtype=float)
            # Baseline has no window; still apply joint6 backoff relative to pre end.
            decel_goal[5] = float((q_pre[-1] if len(q_pre) else INIT_ARM)[5]) - float(args.decel_j6_backoff_rad)
            t_dec, q_dec, qdot_dec, qddot_dec, _u_dec = build_decel_trajectory(
                start_q7=q_pre[-1] if len(q_pre) else np.zeros(7, dtype=float),
                start_qdot7=qdot_pre[-1] if len(qdot_pre) else np.zeros(7, dtype=float),
                goal_q7=decel_goal,
                q_limits=q_limits,
                qdot_limits=qdot_limits,
                qddot_limits=qddot_limits,
                control_dt=args.control_dt,
                waypoint_density=args.waypoint_density,
                poly5_max_samples=poly5_max_samples_try,
                min_duration_sec=args.baseline_decel_min_sec,
                segment_method=args.segment_method,
                qdddot_limits=qdddot_limits if args.segment_method == "scurve" else None,
            )
            t_base, q_base, qdot_base, qddot_base = _concat_with_offset(
                t_pre, q_pre, qdot_pre, qddot_pre, t_dec, q_dec, qdot_dec, qddot_dec
            )
            if len(t_base):
                print("Baseline duration raw (s):", float(t_base[-1]))
            print("[tube] baseline done", flush=True)

        if args.save_csv:
            print("[tube] saving csv", flush=True)
            write_trace_with_tau_csv(args.save_csv, t, q, qdot, qddot, time_unit="s")
            print("Saved:", args.save_csv)
            try:
                pre_end = float(t_pre[-1]) if t_pre is not None and len(t_pre) else 0.0
                win_dur = float(t_win[-1]) if args.with_window and t_win is not None and len(t_win) else 0.0
                win_end = pre_end + win_dur
                end = float(t[-1]) if len(t) else win_end
                sidecar = _write_phase_sidecar_json(
                    args.save_csv,
                    pre_end=pre_end,
                    win_end=win_end,
                    end=end,
                    with_window=bool(args.with_window),
                )
                print("Saved phases:", sidecar)
            except Exception as exc:
                print(f"[tube] Failed to save phase sidecar: {exc}")
            if args.compare_no_tube and t_base is not None:
                write_trace_with_tau_csv(args.compare_csv, t_base, q_base, qdot_base, qddot_base, time_unit="s")
                print("Saved:", args.compare_csv)
                try:
                    end_b = float(t_base[-1]) if len(t_base) else 0.0
                    pre_end_b = float(t_pre[-1]) if t_pre is not None and len(t_pre) else 0.0
                    sidecar_b = _write_phase_sidecar_json(
                        args.compare_csv,
                        pre_end=pre_end_b,
                        win_end=pre_end_b,
                        end=end_b,
                        with_window=False,
                    )
                    print("Saved phases:", sidecar_b)
                except Exception as exc:
                    print(f"[tube] Failed to save baseline phase sidecar: {exc}")
            if args.export_ee:
                try:
                    _export_ee_states_csv(
                        out_csv=str(args.ee_out),
                        t=t,
                        q=q,
                        qdot=qdot,
                        frame_name=str(args.ee_frame),
                        time_unit="s",
                    )
                    print("Saved EE states:", args.ee_out)
                    if args.plot:
                        from sim.plot_ee_traces import plot_ee_from_csv
                        plot_ee_from_csv(
                            str(args.ee_out),
                            args.plot_out_dir,
                            phases_path=f"{args.save_csv}.phases.json" if args.save_csv else None,
                            time_unit="s",
                        )
                except Exception as exc:
                    print(f"[tube] Failed to export EE states: {exc}")
            if args.plot:
                print("[tube] joint plots start", flush=True)
                if args.compare_no_tube and t_base is not None:
                    from sim.plot_joint_traces import plot_compare_from_csv
                    plot_compare_from_csv(args.save_csv, args.compare_csv, args.plot_out_dir, time_unit="s")
                else:
                    from sim.plot_joint_traces import plot_from_csv
                    plot_from_csv(args.save_csv, args.plot_out_dir, time_unit="s")
                print("Plots saved to:", args.plot_out_dir)

    if args.plot_tube:
        print("[tube] tube plot start", flush=True)
        _plot_tube_html(
            target=target,
            release=release,
            hold_sec=args.hold_sec,
            method=args.method,
            n_window_samples=args.n_window_samples,
            nominal_flight_time=args.nominal_flight_time,
            out_html=args.plot_tube_html,
            out_image=args.plot_tube_image,
            open_html=args.open_html,
        )
        print("Tube animation saved to:", args.plot_tube_html)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
