#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Visualize current arm configuration and EE velocity (no collision).

Given joint positions/velocities for Franka Panda (7-DoF), this script:
- computes the chosen frame position (default: EE_FRAME from core.throw_params)
- computes the frame linear velocity from qdot via Pinocchio Jacobian
- plots a simple 3D stick model of the arm and a velocity arrow at the EE

Requirements:
- Pinocchio available (repo already uses it)
- Optional: matplotlib for plotting (otherwise prints numbers only)

Examples:
  python viz_arm_state.py --q 0 0 0 -1.57 0 1.57 0 --qdot 0 0 0 0 0 0 0
  python viz_arm_state.py --q 0.1 -0.5 0.2 -1.2 0.3 1.0 0.0 --qdot 0 0 0 0 0 0 0 --save arm.png
  python viz_arm_state.py --q ... --qdot ... --frame panda_hand
"""

from __future__ import annotations

import argparse
import os
import re
import time
from pathlib import Path

import numpy as np

from core.kinematic_forward import PinKinematics
from core import tube_method
from core.throw_params import EE_FRAME, INIT_ARM

# ---------------- Planned throw defaults (edit in code) ----------------
# These are *not* exposed as CLI args on purpose.
# Set to `None` to disable planned visualization.
DEFAULT_PLANNED_RELEASE_POS_W = [0.4, 0.0, 0.9]
DEFAULT_PLANNED_TARGET_POS_W = [2.0, 0.0, 0.0]
DEFAULT_PLANNED_FLIGHT_TIME_S = 0.55


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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize Panda arm state and EE velocity.")
    parser.add_argument(
        "--q",
        type=float,
        nargs=7,
        default=None,
        help="Joint positions (rad) for panda_joint1..7",
    )
    parser.add_argument(
        "--qdot",
        type=float,
        nargs=7,
        default=None,
        help="Joint velocities (rad/s) for panda_joint1..7",
    )
    parser.add_argument(
        "--frame",
        type=str,
        default=EE_FRAME,
        help=f"Frame to evaluate/plot velocity at (default: {EE_FRAME}).",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "matplotlib", "meshcat"],
        help="Visualization backend (default: auto).",
    )
    parser.add_argument(
        "--arrow-scale",
        type=float,
        default=0.25,
        help="Scale factor applied to velocity arrow for visualization.",
    )
    parser.add_argument(
        "--arrow-style",
        type=str,
        default="line",
        choices=["line"],
        help="Velocity visualization style (line+point).",
    )
    parser.add_argument("--save", type=str, default=None, help="Save figure to path (PNG/SVG/etc).")
    parser.add_argument("--no-show", action="store_true", help="Do not open an interactive window.")
    return parser.parse_args()


def visualize_arm_state(
    q7,
    qdot7=None,
    *,
    frame: str = EE_FRAME,
    backend: str = "meshcat",
    arrow_scale: float = 0.25,
    arrow_style: str = "line",
    planned_release_pos_w=DEFAULT_PLANNED_RELEASE_POS_W,
    planned_target_pos_w=DEFAULT_PLANNED_TARGET_POS_W,
    planned_flight_time: float | None = DEFAULT_PLANNED_FLIGHT_TIME_S,
    planned_arrow_scale: float = 0.25,
    save: str | None = None,
    show: bool = True,
) -> None:
    """Visualize an arm state.

    `q7` and `qdot7` can be Python lists or numpy arrays of shape (7,).
    """
    if "SOFT_THROW_URDF" not in os.environ:
        _load_env_from_file(os.path.join(os.path.dirname(__file__), "env.sh"))

    q7 = np.asarray(q7, dtype=float).reshape(7)
    if qdot7 is None:
        qdot7 = np.zeros(7, dtype=float)
    qdot7 = np.asarray(qdot7, dtype=float).reshape(7)

    pin_model = PinKinematics()
    p_w, v_w = pin_model.get_frame_world_pos_vel_world(q7, qdot7, frame)

    print(f"[frame] name={frame}  pos_w={p_w.tolist()}  vel_w={v_w.tolist()}", flush=True)
    print(f"[frame] speed_m_s={float(np.linalg.norm(v_w)):.6g}", flush=True)
    if frame != EE_FRAME:
        p_ee = pin_model.get_ee_world_pos(q7)
        print(f"[ee_frame] name={EE_FRAME}  pos_w={p_ee.tolist()}", flush=True)

    planned = None
    if planned_release_pos_w is not None and planned_target_pos_w is not None:
        release = np.asarray(planned_release_pos_w, dtype=float).reshape(3)
        target = np.asarray(planned_target_pos_w, dtype=float).reshape(3)
        if planned_flight_time is None:
            v_rel, T = tube_method.solve_ballistic_velocity(release, target)
        else:
            T = float(planned_flight_time)
            v_rel = tube_method.velocity_from_target_release_and_time(target, release, T)
        v_rel = np.asarray(v_rel, dtype=float).reshape(3)
        planned = {"release": release, "target": target, "v": v_rel, "T": float(T)}
        print(
            f"[planned] release={planned['release'].tolist()} target={planned['target'].tolist()} "
            f"T={planned['T']:.4g}s v={planned['v'].tolist()} speed={float(np.linalg.norm(planned['v'])):.6g}",
            flush=True,
        )
        ts = np.linspace(0.0, planned["T"], 60, dtype=float)
        g = float(getattr(tube_method, "G", 9.81))
        traj = planned["release"][None, :] + planned["v"][None, :] * ts[:, None]
        traj[:, 2] = traj[:, 2] - 0.5 * g * (ts**2)
        planned["traj"] = traj

    backend = str(backend)
    if backend == "auto":
        try:
            import meshcat  # noqa: F401
            backend = "meshcat"
        except Exception:
            backend = "matplotlib"

    if backend == "meshcat":
        from pinocchio.visualize import MeshcatVisualizer
        import meshcat.geometry as g
        import meshcat.transformations as tf

        viz = MeshcatVisualizer(pin_model.model, pin_model.robot.collision_model, pin_model.robot.visual_model)
        viz.initViewer(open=bool(show))
        viz.loadViewerModel()
        viz.display(pin_model.make_q_full_from_arm7(q7))

        p0 = p_w.astype(float)
        speed = float(np.linalg.norm(v_w))
        dir_v = (v_w / speed) if speed > 1e-12 else np.array([0.0, 0.0, 1.0], dtype=float)
        arrow_len = float(arrow_scale) * speed

        p1 = (p0 + arrow_len * dir_v).astype(float)
        verts = np.vstack([p0, p1]).T
        geom = g.LineSegments(
            g.PointsGeometry(verts),
            g.LineBasicMaterial(color=0xFF0000, linewidth=4),
        )
        viz.viewer["ee_vel"].set_object(geom)
        viz.viewer["ee_vel"].set_transform(tf.identity_matrix())
        viz.viewer["ee_vel_tip"].set_object(g.Sphere(0.02), g.MeshBasicMaterial(color=0xFF0000, opacity=0.85))
        viz.viewer["ee_vel_tip"].set_transform(tf.translation_matrix(p1.tolist()))
        viz.viewer["ee_pos"].set_object(g.Sphere(0.02), g.MeshBasicMaterial(color=0xFF0000, opacity=0.8))
        viz.viewer["ee_pos"].set_transform(tf.translation_matrix(p0.tolist()))

        if planned is not None:
            blue = g.MeshBasicMaterial(color=0x1F77B4, opacity=0.85)   # release point
            purple = g.MeshBasicMaterial(color=0x9467BD, opacity=0.85) # release velocity direction
            green = g.MeshBasicMaterial(color=0x2CA02C, opacity=0.85)
            cyan = g.LineBasicMaterial(color=0x17BECF, linewidth=3)

            r0 = planned["release"].astype(float)
            rv = planned["v"].astype(float)
            rs = float(np.linalg.norm(rv))
            rdir = (rv / rs) if rs > 1e-12 else np.array([1.0, 0.0, 0.0], dtype=float)
            rlen = float(planned_arrow_scale) * rs
            r1 = (r0 + rlen * rdir).astype(float)

            viz.viewer["planned/release"].set_object(g.Sphere(0.02), blue)
            viz.viewer["planned/release"].set_transform(tf.translation_matrix(r0.tolist()))

            verts2 = np.vstack([r0, r1]).T
            geom2 = g.LineSegments(g.PointsGeometry(verts2), g.LineBasicMaterial(color=0x9467BD, linewidth=4))
            viz.viewer["planned/vel"].set_object(geom2)
            viz.viewer["planned/vel"].set_transform(tf.identity_matrix())
            viz.viewer["planned/vel_tip"].set_object(g.Sphere(0.02), purple)
            viz.viewer["planned/vel_tip"].set_transform(tf.translation_matrix(r1.tolist()))

            viz.viewer["planned/target"].set_object(g.Sphere(0.02), green)
            viz.viewer["planned/target"].set_transform(tf.translation_matrix(planned["target"].astype(float).tolist()))

            traj = np.asarray(planned.get("traj", None), dtype=float) if planned.get("traj", None) is not None else None
            if traj is not None and traj.ndim == 2 and traj.shape[0] >= 2:
                # build line segments as consecutive pairs
                seg = np.zeros((traj.shape[0] - 1) * 2, dtype=int)
                pts = []
                for i in range(traj.shape[0] - 1):
                    pts.append(traj[i])
                    pts.append(traj[i + 1])
                pts = np.asarray(pts, dtype=float)
                verts = pts.T  # 3 x (2*(N-1))
                geom_traj = g.LineSegments(g.PointsGeometry(verts), cyan)
                viz.viewer["planned/traj"].set_object(geom_traj)
                viz.viewer["planned/traj"].set_transform(tf.identity_matrix())

        if save:
            print(f"[warn] meshcat backend does not support --save in this script (requested {save!r})", flush=True)

        print("[ok] meshcat viewer updated (Ctrl+C to exit)", flush=True)
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            return

    # matplotlib fallback
    import matplotlib.pyplot as plt

    joints_w = pin_model.joint_positions_world(q7)
    xs, ys, zs = joints_w[:, 0], joints_w[:, 1], joints_w[:, 2]

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(xs, ys, zs, "-o", lw=2, ms=4, label="joints 1..7")
    ax.scatter([p_w[0]], [p_w[1]], [p_w[2]], s=60, c="tab:red", label=frame)
    speed = float(np.linalg.norm(v_w))
    dir_v = (v_w / speed) if speed > 1e-12 else np.array([0.0, 0.0, 1.0], dtype=float)
    arrow_len = float(arrow_scale) * speed
    p1 = p_w + arrow_len * dir_v
    ax.plot([p_w[0], p1[0]], [p_w[1], p1[1]], [p_w[2], p1[2]], color="tab:red", lw=3)
    ax.scatter([p1[0]], [p1[1]], [p1[2]], s=40, c="tab:red")

    if planned is not None:
        r0 = planned["release"]
        rv = planned["v"]
        rs = float(np.linalg.norm(rv))
        rdir = (rv / rs) if rs > 1e-12 else np.array([1.0, 0.0, 0.0], dtype=float)
        rlen = float(planned_arrow_scale) * rs
        r1 = r0 + rlen * rdir
        ax.scatter([r0[0]], [r0[1]], [r0[2]], s=60, c="tab:blue", label="planned_release")
        ax.plot([r0[0], r1[0]], [r0[1], r1[1]], [r0[2], r1[2]], color="tab:purple", lw=3, label="planned_release_vel_dir")
        ax.scatter([r1[0]], [r1[1]], [r1[2]], s=40, c="tab:purple")
        t = planned["target"]
        ax.scatter([t[0]], [t[1]], [t[2]], s=60, c="tab:green", label="target")
        traj = planned.get("traj", None)
        if traj is not None:
            traj = np.asarray(traj, dtype=float)
            if traj.ndim == 2 and traj.shape[1] == 3 and traj.shape[0] >= 2:
                ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color="tab:cyan", lw=2, label="planned_ballistic")

    pts = np.vstack([joints_w, p_w.reshape(1, 3)])
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    ctr = 0.5 * (mins + maxs)
    half = 0.5 * float(np.max(maxs - mins) + 1e-9)
    ax.set_xlim(ctr[0] - half, ctr[0] + half)
    ax.set_ylim(ctr[1] - half, ctr[1] + half)
    ax.set_zlim(ctr[2] - half, ctr[2] + half)

    ax.set_title("Panda arm (stick model) + EE velocity")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend()
    fig.tight_layout()

    if save:
        out = Path(save)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=180)
        print(f"[ok] saved plot -> {out}", flush=True)

    if show and not save:
        plt.show()
    plt.close(fig)

DEFAULT_PLANNED_RELEASE_POS_W = [0.8, 0.0, 0.9]
DEFAULT_PLANNED_TARGET_POS_W = [1.0, 0.0, 0.0]
DEFAULT_PLANNED_FLIGHT_TIME_S = 0.55
def main() -> int:
    if "SOFT_THROW_URDF" not in os.environ:
        _load_env_from_file(os.path.join(os.path.dirname(__file__), "env.sh"))

    args = _parse_args()
    q7 = INIT_ARM if args.q is None else args.q
    qdot7 = np.zeros(7, dtype=float) if args.qdot is None else args.qdot
    q7=[0.083972, 0.577050, -0.028606, -2.636392, 0.031767, 3.794059, -0.089983]
    qdot7=[2.0, 0.0, 1.0, 0.0, 0.0, 0, 0.0]
    visualize_arm_state(
        q7,
        qdot7,
        frame=str(args.frame),
        backend=str(args.backend),
        arrow_scale=float(args.arrow_scale),
        arrow_style=str(args.arrow_style),
        save=args.save,
        show=not bool(args.no_show),
        planned_release_pos_w=DEFAULT_PLANNED_RELEASE_POS_W,
        planned_target_pos_w=DEFAULT_PLANNED_TARGET_POS_W,
        planned_flight_time=DEFAULT_PLANNED_FLIGHT_TIME_S,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
