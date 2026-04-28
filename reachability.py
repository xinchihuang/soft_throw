#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Reachability sampling + IK reach query (no collision).

This script estimates the arm end-effector positional workspace by:
1) Monte-Carlo sampling joint configurations within limits, and
2) Optionally querying reachability of specific XYZ targets via IK.

Outputs:
- CSV of sampled EE positions
- Summary JSON (min/max bounds, radius stats)
- Plots (if matplotlib is available)
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Iterable

import numpy as np

from core.kinematic_forward import PinKinematics
from core.ik_poly5_core import solve_ik_for_reset_pos
from core.throw_params import (
    DT_CONTROL,
    EE_FRAME,
    INIT_ARM,
    QDDOT_LIMITS_7,
    QDOT_LIMITS_7,
    Q_LIMITS_7,
)

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


if "SOFT_THROW_URDF" not in os.environ:
    _load_env_from_file(os.path.join(os.path.dirname(__file__), "env.sh"))


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


def _parse_vec3(text: str) -> np.ndarray:
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Expected 3 comma-separated values, e.g. 0.4,0.0,0.9")
    return np.array([float(p) for p in parts], dtype=float)


def _sample_q(rng: np.random.Generator, q_limits: np.ndarray, n: int) -> np.ndarray:
    q_min = np.asarray(q_limits[:, 0], dtype=float)
    q_max = np.asarray(q_limits[:, 1], dtype=float)
    u = rng.random((int(n), 7), dtype=float)
    return q_min[None, :] + u * (q_max - q_min)[None, :]


def _compute_workspace_points(pin: PinKinematics, qs: np.ndarray) -> np.ndarray:
    qs = np.asarray(qs, dtype=float).reshape(-1, 7)
    pts = np.zeros((qs.shape[0], 3), dtype=float)
    for i in range(qs.shape[0]):
        pts[i] = pin.get_ee_world_pos(qs[i])
    return pts


def _summarize_points(points: np.ndarray) -> dict:
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    r = np.linalg.norm(points, axis=1)
    return {
        "n": int(points.shape[0]),
        "xyz_min": points.min(axis=0).tolist(),
        "xyz_max": points.max(axis=0).tolist(),
        "r_min": float(r.min()) if r.size else 0.0,
        "r_max": float(r.max()) if r.size else 0.0,
        "r_mean": float(r.mean()) if r.size else 0.0,
        "r_p05": float(np.quantile(r, 0.05)) if r.size else 0.0,
        "r_p95": float(np.quantile(r, 0.95)) if r.size else 0.0,
    }


def _write_csv(csv_path: Path, points: np.ndarray, qs: np.ndarray | None) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    if qs is not None:
        qs = np.asarray(qs, dtype=float).reshape(-1, 7)
        data = np.concatenate([points, qs], axis=1)
        header = ["x", "y", "z"] + [f"q{j + 1}" for j in range(7)]
    else:
        data = points
        header = ["x", "y", "z"]
    np.savetxt(str(csv_path), data, delimiter=",", header=",".join(header), comments="")


def _try_plot(out_dir: Path, points: np.ndarray, *, max_points: int = 40000) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[plot] matplotlib not available, skip plots: {exc}", flush=True)
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    n = points.shape[0]
    if n == 0:
        return

    idx = np.arange(n)
    if n > int(max_points):
        rng = np.random.default_rng(0)
        idx = rng.choice(n, size=int(max_points), replace=False)
    p = points[idx]

    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    r = np.linalg.norm(p, axis=1)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x, y, z, s=1, alpha=0.2)
    ax.set_title("EE workspace samples (3D)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    fig.tight_layout()
    fig.savefig(out_dir / "workspace_3d.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(x, y, s=1, alpha=0.2)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("EE workspace projection (XY)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()
    fig.savefig(out_dir / "workspace_xy.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(r, z, s=1, alpha=0.2)
    ax.set_title("EE workspace projection (r-z)")
    ax.set_xlabel("r = ||[x,y,z]||")
    ax.set_ylabel("z")
    fig.tight_layout()
    fig.savefig(out_dir / "workspace_rz.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(r, bins=80)
    ax.set_title("Radius histogram")
    ax.set_xlabel("r = ||[x,y,z]||")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(out_dir / "radius_hist.png", dpi=180)
    plt.close(fig)


def _iter_queries(values: Iterable[np.ndarray] | None) -> list[np.ndarray]:
    if not values:
        return []
    out = []
    for v in values:
        arr = np.asarray(v, dtype=float).reshape(3)
        out.append(arr)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Estimate arm positional workspace (no collision).")
    parser.add_argument("--samples", type=int, default=50000, help="Number of joint samples")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit-scale", type=float, default=1.0, help="Scale joint position range (<=1.0)")
    parser.add_argument("--out-dir", type=str, default="reachability_out")
    parser.add_argument("--csv", type=str, default="reachability_points.csv")
    parser.add_argument("--no-save-q", action="store_true", help="Do not save sampled joint angles in CSV")
    parser.add_argument("--no-plot", action="store_true", help="Do not generate plots")
    parser.add_argument("--max-plot-points", type=int, default=40000)
    parser.add_argument("--query", type=_parse_vec3, action="append", default=None, help="Query reachability for XYZ")
    parser.add_argument("--ik-max-iter", type=int, default=300)
    parser.add_argument("--ik-tol", type=float, default=2e-3, help="Reachability tolerance (meters)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    q_limits = _scale_q_limits(Q_LIMITS_7, float(args.limit_scale))

    print(f"[pin] building model (ee_frame={EE_FRAME})", flush=True)
    pin = PinKinematics()

    rng = np.random.default_rng(int(args.seed))
    qs = _sample_q(rng, q_limits, int(args.samples))
    points = _compute_workspace_points(pin, qs)

    summary = _summarize_points(points)
    summary.update(
        {
            "ee_frame": str(EE_FRAME),
            "urdf": os.environ.get("SOFT_THROW_URDF", ""),
            "pkg": os.environ.get("SOFT_THROW_PKG", ""),
            "limit_scale": float(args.limit_scale),
            "seed": int(args.seed),
        }
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    csv_path = out_dir / str(args.csv)
    _write_csv(csv_path, points, None if args.no_save_q else qs)
    print(f"[ok] wrote {csv_path}", flush=True)
    print(f"[ok] wrote {out_dir / 'summary.json'}", flush=True)

    if not args.no_plot:
        _try_plot(out_dir, points, max_points=int(args.max_plot_points))
        print(f"[ok] plots in {out_dir}", flush=True)

    queries = _iter_queries(args.query)
    if queries:
        print("\n[query] IK reachability (position-only, fixed orientation from INIT_ARM):", flush=True)
        q_init = np.asarray(INIT_ARM, dtype=float).reshape(7)
        for p_des in queries:
            q_sol, _qdot = solve_ik_for_reset_pos(
                pin,
                q_init,
                p_des,
                q_limits,
                QDOT_LIMITS_7,
                QDDOT_LIMITS_7,
                control_dt=float(DT_CONTROL),
                max_iter=int(args.ik_max_iter),
                kp_pos=2.0,
                kp_rot=1.0,
                target_frame_id=None,
            )
            p_sol = pin.get_ee_world_pos(q_sol)
            err = float(np.linalg.norm(p_sol - p_des))
            ok = err <= float(args.ik_tol)
            print(
                f"  target={p_des.tolist()} reachable={ok} err_m={err:.4g} "
                f"q_sol={np.asarray(q_sol, dtype=float).round(5).tolist()}",
                flush=True,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
