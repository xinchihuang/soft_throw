#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Play back a saved joint trajectory CSV in Isaac Sim (same env as run_min_throw_sim.py).

Behavior:
- Loads the same Isaac Sim scene as `sim/run_min_throw_sim.py`.
- Reads a CSV trajectory (t + q columns, optional dq).
- Commands the arm to the first q in the CSV (reset/settle).
- Waits for user input, then plays back the trajectory.

Edit defaults in code:
- DEFAULT_CSV_PATH
- PLAYBACK_DT_S
- RESET_SETTLE_SEC
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

# Ensure repo root is on sys.path when running from sim/
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ---------------- Defaults (edit in code) ----------------
DEFAULT_CSV_PATH = str(_ROOT / "trajectory.csv")
# Playback control period. Keep this at `core.throw_params.DT_CONTROL` unless your CSV is already 100 Hz.
PLAYBACK_DT_S = None  # None => use core.throw_params.DT_CONTROL
RESET_SETTLE_SEC = 2.0
HOLD_AFTER_SEC = 1.0


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


def _resolve_joint_cols(names: list[str]) -> tuple[list[str], list[str] | None]:
    if all(k in names for k in [f"q{i}" for i in range(7)]):
        q_cols = [f"q{i}" for i in range(7)]
        dq_cols = [f"dq{i}" for i in range(7)] if all(f"dq{i}" in names for i in range(7)) else None
        return q_cols, dq_cols
    if all(k in names for k in [f"q{i}" for i in range(1, 8)]):
        q_cols = [f"q{i}" for i in range(1, 8)]
        dq_cols = [f"dq{i}" for i in range(1, 8)] if all(f"dq{i}" in names for i in range(1, 8)) else None
        return q_cols, dq_cols
    raise ValueError(f"Unrecognized q column scheme. Found: {names}")


def _read_trajectory_csv(csv_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.genfromtxt(str(csv_path), delimiter=",", names=True, dtype=None, encoding="utf-8")
    if arr is None or len(arr) == 0:
        raise ValueError(f"Empty CSV: {csv_path}")
    names = list(arr.dtype.names or [])
    if "t" not in names:
        raise ValueError(f"CSV missing 't'. Found: {names}")
    q_cols, dq_cols = _resolve_joint_cols(names)

    t = np.asarray(arr["t"], dtype=float).reshape(-1)
    q = np.column_stack([arr[c] for c in q_cols]).astype(float)
    if dq_cols is None:
        dq = np.zeros_like(q)
    else:
        dq = np.column_stack([arr[c] for c in dq_cols]).astype(float)
    return t, q, dq


def _resample_to_dt(t: np.ndarray, q: np.ndarray, dq: np.ndarray, dt_out: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Downsample by selecting the first sample at/after each output time grid."""
    t = np.asarray(t, dtype=float).reshape(-1)
    if len(t) < 2:
        return t, q, dq
    t0 = float(t[0])
    t = t - t0
    t_end = float(t[-1])
    if dt_out <= 0.0:
        raise ValueError("dt_out must be > 0")

    t_grid = np.arange(0.0, t_end + 0.5 * dt_out, dt_out, dtype=float)
    idx = np.searchsorted(t, t_grid, side="left")
    idx = np.clip(idx, 0, len(t) - 1)
    # Ensure strictly increasing indices to avoid repeated frames if t has coarse steps.
    keep = np.concatenate([[True], idx[1:] > idx[:-1]])
    idx = idx[keep]
    t_grid = t_grid[keep]
    return t_grid, q[idx], dq[idx]


def main() -> int:
    parser = argparse.ArgumentParser(description="Play a saved trajectory CSV in Isaac Sim.")
    parser.add_argument(
        "--csv",
        type=str,
        default=DEFAULT_CSV_PATH,
        help=f"Trajectory CSV path (default: {DEFAULT_CSV_PATH})",
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--auto-start",
        action="store_true",
        help="Do not wait for Enter; start playback immediately after reset/settle.",
    )
    args = parser.parse_args()

    if "SOFT_THROW_URDF" not in os.environ:
        _load_env_from_file(str(_ROOT / "env.sh"))

    from core.throw_params import (
        ROBOT_USD,
        ROBOT_PRIM,
        FRANKA_ROOT,
        DT_CONTROL,
        SIM_UPDATES_PER_STEP,
    )
    from sim.isaac_scene import (
        get_stage,
        ensure_physics_scene,
        spawn_ground,
        spawn_lights,
        add_robot_reference,
        detect_franka_root,
    )
    from sim.isaac_robot_io import (
        find_joint_paths,
        apply_arm_targets,
    )

    csv_path = Path(args.csv).expanduser()
    if not csv_path.is_file():
        raise SystemExit(f"CSV not found: {csv_path}")

    t_csv, q_csv, dq_csv = _read_trajectory_csv(csv_path)
    dt_play = float(DT_CONTROL if PLAYBACK_DT_S is None else PLAYBACK_DT_S)
    t, q, dq = _resample_to_dt(t_csv, q_csv, dq_csv, dt_play)
    if len(t) < 2:
        raise SystemExit("Need at least 2 samples to play back")

    from isaaclab.app import AppLauncher
    import omni.timeline

    app_launcher = AppLauncher(headless=bool(args.headless))
    simulation_app = app_launcher.app

    stage = get_stage()
    ensure_physics_scene(stage)
    spawn_ground(stage)
    spawn_lights(stage)
    add_robot_reference(stage, ROBOT_PRIM, ROBOT_USD)

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()

    for _ in range(120):
        simulation_app.update()

    franka_root = detect_franka_root(stage, FRANKA_ROOT)
    joint_paths = find_joint_paths(stage, franka_root)

    q0 = np.asarray(q[0], dtype=float).reshape(7)

    # Reset/settle to the first CSV pose.
    print(f"[reset] commanding q0 from {csv_path.name}: {np.round(q0, 6).tolist()}", flush=True)
    t0 = time.time()
    while time.time() - t0 < float(RESET_SETTLE_SEC):
        apply_arm_targets(stage, joint_paths, q0, np.zeros(7, dtype=float))
        for _ in range(int(SIM_UPDATES_PER_STEP)):
            simulation_app.update()

    if not bool(args.auto_start):
        input("[ready] arm set to first pose. Press Enter to start playback... ")

    print(f"[play] samples={len(t)} dt={dt_play:g}s source={csv_path.name}", flush=True)
    try:
        for k in range(len(t)):
            apply_arm_targets(stage, joint_paths, q[k], dq[k])
            for _ in range(int(SIM_UPDATES_PER_STEP)):
                simulation_app.update()
    except KeyboardInterrupt:
        return 0

    # Hold after playback.
    n_hold = max(1, int(np.ceil(float(HOLD_AFTER_SEC) / dt_play)))
    for _ in range(n_hold):
        apply_arm_targets(stage, joint_paths, q[-1], np.zeros(7, dtype=float))
        simulation_app.update()

    print("[done] playback complete; simulation left running. Ctrl+C to exit.", flush=True)
    try:
        while True:
            simulation_app.update()
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
