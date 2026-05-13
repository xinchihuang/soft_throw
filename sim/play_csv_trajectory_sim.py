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
import select
import sys
import termios
import time
import tty
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
    parser.add_argument(
        "--repeat",
        type=int,
        default=0,
        help="Number of playback passes. Default 0 means repeat forever until q.",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Repeat playback forever. Equivalent to --repeat 0.",
    )
    parser.add_argument(
        "--repeat-reset-sec",
        type=float,
        default=0.75,
        help="Seconds to command the first CSV pose before each repeated pass.",
    )
    parser.add_argument(
        "--ball-prim",
        type=str,
        default="/World/LMM/Ball",
        help="Ball prim to reset before each playback pass.",
    )
    parser.add_argument(
        "--head-prim",
        type=str,
        default="/World/LMM/Franka/lacrosse",
        help="Optional explicit lacrosse head prim path used for ball reset.",
    )
    parser.add_argument(
        "--ball-offset",
        type=float,
        default=0.10,
        help="Ball reset height above lacrosse head top, in meters.",
    )
    args = parser.parse_args()
    if args.repeat < 0:
        raise SystemExit("--repeat must be >= 0")

    if "SOFT_THROW_URDF" not in os.environ:
        _load_env_from_file(str(_ROOT / "env.sh"))

    from core.throw_params import (
        ROBOT_USD,
        ROBOT_PRIM,
        FRANKA_ROOT,
        DT_CONTROL,
        SIM_UPDATES_PER_STEP,
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

    app_launcher = AppLauncher(headless=bool(args.headless))
    simulation_app = app_launcher.app

    import omni.timeline
    from sim.isaac_scene import (
        get_stage,
        ensure_physics_scene,
        repair_lmm_asset_references,
        spawn_ground,
        spawn_lights,
        add_robot_reference,
        detect_franka_root,
    )
    from sim.isaac_robot_io import (
        find_joint_paths,
        apply_arm_targets,
    )
    from sim.show_asset_sim import _reset_ball_above_lacrosse_head

    stage = get_stage()
    ensure_physics_scene(stage)
    spawn_ground(stage)
    spawn_lights(stage)
    add_robot_reference(stage, ROBOT_PRIM, ROBOT_USD)
    repair_lmm_asset_references(stage, ROBOT_PRIM)

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()

    for _ in range(120):
        simulation_app.update()

    franka_root = detect_franka_root(stage, FRANKA_ROOT)
    joint_paths = find_joint_paths(stage, franka_root)

    q0 = np.asarray(q[0], dtype=float).reshape(7)

    repeat_count = 0 if bool(args.loop) else int(args.repeat)
    repeat_label = "infinite" if repeat_count == 0 else str(repeat_count)
    print(f"[play] samples={len(t)} dt={dt_play:g}s source={csv_path.name} repeat={repeat_label}", flush=True)

    def _command_pose_for(seconds: float, q_cmd: np.ndarray) -> None:
        t_start = time.time()
        while time.time() - t_start < float(seconds):
            apply_arm_targets(stage, joint_paths, q_cmd, np.zeros(7, dtype=float))
            for _ in range(int(SIM_UPDATES_PER_STEP)):
                simulation_app.update()

    def _wait_for_start_key(pass_index: int) -> bool:
        print(f"[ready] pass={pass_index} Press Enter to play, q to quit", flush=True)
        stdin_is_tty = bool(sys.stdin.isatty())
        fd = sys.stdin.fileno()
        old_settings = None
        if stdin_is_tty:
            old_settings = termios.tcgetattr(fd)
            tty.setcbreak(fd)
        try:
            while True:
                ready, _, _ = select.select([sys.stdin], [], [], 0.0)
                if ready:
                    key = sys.stdin.read(1) if stdin_is_tty else sys.stdin.readline()[:1]
                    key = key.lower()
                    if key in ("\r", "\n", " "):
                        return True
                    if key == "q":
                        print("[ready] quit", flush=True)
                        return False
                simulation_app.update()
                time.sleep(0.01)
        finally:
            if old_settings is not None:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def _prepare_pass(pass_index: int, *, wait_for_key: bool) -> bool:
        settle_sec = float(RESET_SETTLE_SEC if pass_index == 1 else args.repeat_reset_sec)
        print(f"[reset] pass={pass_index} commanding q0 from {csv_path.name}: {np.round(q0, 6).tolist()}", flush=True)
        _command_pose_for(settle_sec, q0)
        _reset_ball_above_lacrosse_head(
            stage,
            ball_path=args.ball_prim,
            head_path=args.head_prim,
            offset_m=float(args.ball_offset),
        )
        if wait_for_key:
            return _wait_for_start_key(pass_index)
        return True

    def _play_once(pass_index: int) -> None:
        print(f"[play] pass={pass_index}", flush=True)
        for k in range(len(t)):
            apply_arm_targets(stage, joint_paths, q[k], dq[k])
            for _ in range(int(SIM_UPDATES_PER_STEP)):
                simulation_app.update()

    try:
        pass_index = 1
        while repeat_count == 0 or pass_index <= repeat_count:
            wait_for_key = not (bool(args.auto_start) and pass_index == 1)
            if not _prepare_pass(pass_index, wait_for_key=wait_for_key):
                return 0
            _play_once(pass_index)
            pass_index += 1
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
