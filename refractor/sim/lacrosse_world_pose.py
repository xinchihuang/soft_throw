#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Print the lacrosse head world pose in Isaac Sim."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import numpy as np


_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from refractor.config import params
from refractor.sim.relative_transform import get_world_transform, matrix4d_to_numpy


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


def world_pose_summary(stage, prim_path: str) -> dict:
    matrix = get_world_transform(stage, prim_path)
    translation = matrix.ExtractTranslation()
    return {
        "prim_path": str(prim_path),
        "translation": np.array([translation[0], translation[1], translation[2]], dtype=float),
        "matrix": matrix4d_to_numpy(matrix),
    }


def print_world_pose(summary: dict) -> None:
    print(f"[world_pose] prim={summary['prim_path']}", flush=True)
    print(f"[world_pose] translation={summary['translation'].round(9).tolist()}", flush=True)
    print("[world_pose] matrix=", flush=True)
    print(np.array2string(summary["matrix"], precision=9, suppress_small=False), flush=True)


def main() -> int:
    if "SOFT_THROW_ROBOT_USD" not in os.environ:
        _load_env_from_file(params.DEFAULT_ENV_SH)

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=bool(params.DEFAULT_SIM_HEADLESS))
    simulation_app = app_launcher.app

    import omni.timeline
    from core.throw_params import ROBOT_PRIM, ROBOT_USD
    from sim.isaac_scene import (
        add_robot_reference,
        ensure_physics_scene,
        get_stage,
        repair_lmm_asset_references,
        spawn_ground,
        spawn_lights,
    )

    stage = get_stage()
    ensure_physics_scene(stage)
    spawn_ground(stage)
    spawn_lights(stage)
    add_robot_reference(stage, ROBOT_PRIM, ROBOT_USD)
    repair_lmm_asset_references(stage, ROBOT_PRIM)

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    for _ in range(int(params.DEFAULT_SIM_WARMUP_STEPS)):
        simulation_app.update()

    summary = world_pose_summary(stage, params.DEFAULT_SIM_HEAD_PRIM)
    print_world_pose(summary)

    print("[world_pose] simulation left running. Press Ctrl+C to exit.", flush=True)
    try:
        while True:
            simulation_app.update()
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
