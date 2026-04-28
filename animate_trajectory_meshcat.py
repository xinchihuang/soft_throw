#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Animate a joint trajectory in Meshcat.

Input:
- trajectory CSV with at least: t and q columns (q0..q6 or q1..q7)

Usage:
  python animate_trajectory_meshcat.py
  python animate_trajectory_meshcat.py --speed 0.5   # slower
  python animate_trajectory_meshcat.py --speed 2.0   # faster
  python animate_trajectory_meshcat.py --stride 2    # skip frames
"""

from __future__ import annotations

import argparse
import os
import re
import time
from pathlib import Path

import numpy as np

from core.kinematic_forward import PinKinematics


DEFAULT_CSV_PATH = "demo_trajectory.csv"


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


def _read_trajectory_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    arr = np.genfromtxt(str(path), delimiter=",", names=True)
    if arr is None or len(arr) == 0:
        raise ValueError(f"Empty CSV: {path}")
    names = list(arr.dtype.names or [])
    if all(k in names for k in ["q0", "q1", "q2", "q3", "q4", "q5", "q6"]):
        q_cols = [f"q{i}" for i in range(7)]
    elif all(k in names for k in ["q1", "q2", "q3", "q4", "q5", "q6", "q7"]):
        q_cols = [f"q{i}" for i in range(1, 8)]
    else:
        raise ValueError(f"CSV missing expected q columns. Found: {names}")

    if "t" not in names:
        raise ValueError(f"CSV missing column 't'. Found: {names}")

    t = np.asarray(arr["t"], dtype=float).reshape(-1)
    q = np.column_stack([arr[c] for c in q_cols]).astype(float)
    return t, q


def main() -> int:
    parser = argparse.ArgumentParser(description="Animate trajectory CSV in Meshcat.")
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help=f"Optional path to trajectory CSV (default: {DEFAULT_CSV_PATH})",
    )
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier")
    parser.add_argument("--stride", type=int, default=1, help="Frame stride (skip every N-1 samples)")
    parser.add_argument("--loop", action="store_true", help="Loop playback")
    parser.add_argument(
        "--zmq-url",
        type=str,
        default=None,
        help="Optional Meshcat ZMQ URL to connect to an already-running server (e.g. tcp://127.0.0.1:6000).",
    )
    parser.add_argument("--open", dest="open_viewer", action="store_true", help="Open browser viewer")
    parser.add_argument("--no-open", dest="open_viewer", action="store_false", help="Do not open browser viewer")
    parser.set_defaults(open_viewer=True)
    args = parser.parse_args()

    if "SOFT_THROW_URDF" not in os.environ:
        _load_env_from_file(str(Path(__file__).resolve().parent / "env.sh"))

    csv_path = Path(DEFAULT_CSV_PATH if args.csv is None else args.csv)
    if not csv_path.is_file():
        raise SystemExit(f"trajectory CSV not found: {csv_path}")
    t, q = _read_trajectory_csv(csv_path)
    stride = max(1, int(args.stride))
    speed = float(args.speed)
    if speed <= 0.0:
        raise ValueError("--speed must be > 0")

    # downsample
    t = t[::stride]
    q = q[::stride]
    if len(t) < 2:
        raise ValueError("Need at least 2 samples to animate")

    # normalize time to start at 0
    t = t - float(t[0])
    dt = np.diff(t)
    dt = np.maximum(dt, 0.0)

    from pinocchio.visualize import MeshcatVisualizer

    pin_model = PinKinematics()
    viz = MeshcatVisualizer(pin_model.model, pin_model.robot.collision_model, pin_model.robot.visual_model)
    try:
        viz.initViewer(open=bool(args.open_viewer), zmq_url=args.zmq_url)
    except Exception as exc:
        hint = ""
        if args.zmq_url is None:
            hint = (
                "\nHint: if the meshcat server cannot be started in this environment, start it manually:\n"
                "  python -m meshcat.servers.zmqserver --zmq-url tcp://127.0.0.1:6000 --open\n"
                "Then re-run this script with:\n"
                "  python animate_trajectory_meshcat.py --csv trajectory.csv --zmq-url tcp://127.0.0.1:6000\n"
            )
        raise RuntimeError(f"Failed to initialize Meshcat viewer: {exc}{hint}") from exc
    viz.loadViewerModel()

    print(f"[ok] loaded {csv_path} samples={len(t)} stride={stride} speed={speed:g}", flush=True)
    print("[play] Ctrl+C to stop", flush=True)

    try:
        while True:
            t0_wall = time.time()
            for i in range(len(t)):
                q_full = pin_model.make_q_full_from_arm7(q[i])
                viz.display(q_full)
                if i < len(t) - 1:
                    # use CSV time deltas
                    sleep_s = float(dt[i]) / speed
                    if sleep_s > 0.0:
                        time.sleep(sleep_s)
            if not bool(args.loop):
                break
            # avoid busy looping if trajectory duration is ~0
            if time.time() - t0_wall < 1e-3:
                time.sleep(0.05)
    except KeyboardInterrupt:
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
