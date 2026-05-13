#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Open the configured Isaac Sim environment for manual editing and saving."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from refractor.config import params


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


def main() -> int:
    if "SOFT_THROW_ROBOT_USD" not in os.environ:
        _load_env_from_file(params.DEFAULT_ENV_SH)

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=bool(params.DEFAULT_SIM_OPEN_ENVIRONMENT_HEADLESS))
    simulation_app = app_launcher.app

    import omni.timeline
    from core.throw_params import ROBOT_PRIM, ROBOT_USD
    from sim.isaac_scene import (
        add_robot_reference,
        ensure_physics_scene,
        get_stage,
        repair_lmm_asset_references,
        set_prim_translate,
        spawn_ground,
        spawn_lights,
    )

    stage = get_stage()
    ensure_physics_scene(stage)
    if bool(params.DEFAULT_SIM_SPAWN_GROUND):
        spawn_ground(stage, color=params.DEFAULT_SIM_GROUND_COLOR)
    spawn_lights(stage)
    add_robot_reference(stage, ROBOT_PRIM, ROBOT_USD)
    if abs(float(params.DEFAULT_SIM_ASSET_Z_OFFSET)) > 0.0:
        set_prim_translate(stage, ROBOT_PRIM, (0.0, 0.0, float(params.DEFAULT_SIM_ASSET_Z_OFFSET)), label="asset root")
    repair_lmm_asset_references(stage, ROBOT_PRIM)

    timeline = omni.timeline.get_timeline_interface()
    timeline.pause()
    timeline.commit()

    for _ in range(int(params.DEFAULT_SIM_WARMUP_STEPS)):
        simulation_app.update()

    print(f"[open_environment] asset={ROBOT_USD}", flush=True)
    print(f"[open_environment] root={ROBOT_PRIM}", flush=True)
    print("[open_environment] environment opened. Adjust in Isaac Sim and use File > Save/Save As.", flush=True)
    print("[open_environment] Press Ctrl+C in this terminal to close.", flush=True)

    try:
        while True:
            simulation_app.update()
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
