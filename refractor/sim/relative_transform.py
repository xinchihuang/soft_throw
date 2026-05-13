#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Relative transform helpers for Isaac Sim USD prims."""

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


def _valid_prim(prim) -> bool:
    return bool(prim and prim.IsValid())


def find_prim_by_name(stage, root_path: str, name: str):
    """Find the first prim named ``name`` under ``root_path``."""
    root = stage.GetPrimAtPath(root_path)
    if not _valid_prim(root):
        raise RuntimeError(f"root prim not found: {root_path}")

    stack = [root]
    while stack:
        prim = stack.pop()
        if prim.GetName() == name:
            return prim
        for child in prim.GetChildren():
            stack.append(child)
    return None


def list_child_names(stage, root_path: str, max_count: int = params.DEFAULT_RELATIVE_CHILD_SNAPSHOT_COUNT) -> list[str]:
    """Return a small prim-name/path snapshot under root for diagnostics."""
    root = stage.GetPrimAtPath(root_path)
    if not _valid_prim(root):
        return []
    out: list[str] = []
    stack = [root]
    while stack and len(out) < int(max_count):
        prim = stack.pop()
        out.append(f"{prim.GetName()} {prim.GetPath()}")
        for child in prim.GetChildren():
            stack.append(child)
    return out


def get_world_transform(stage, prim_path: str):
    """Return a prim's world transform as a USD Gf.Matrix4d."""
    from pxr import Usd, UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not _valid_prim(prim):
        raise RuntimeError(f"prim not found: {prim_path}")
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    return cache.GetLocalToWorldTransform(prim)


def get_relative_transform(stage, parent_path: str, child_path: str):
    """Return ``T_parent_child`` as a USD Gf.Matrix4d."""
    t_world_parent = get_world_transform(stage, parent_path)
    t_world_child = get_world_transform(stage, child_path)
    return t_world_parent.GetInverse() * t_world_child


def matrix4d_to_numpy(matrix) -> np.ndarray:
    """Convert USD Gf.Matrix4d to a 4x4 numpy array."""
    out = np.zeros((4, 4), dtype=float)
    for i in range(4):
        for j in range(4):
            out[i, j] = float(matrix[i][j])
    return out


def relative_pose_summary(stage, parent_path: str, child_path: str) -> dict:
    """Return relative transform matrix, translation, and paths."""
    matrix = get_relative_transform(stage, parent_path, child_path)
    translation = matrix.ExtractTranslation()
    return {
        "parent_path": str(parent_path),
        "child_path": str(child_path),
        "translation": np.array([translation[0], translation[1], translation[2]], dtype=float),
        "matrix": matrix4d_to_numpy(matrix),
    }


def link7_to_lacrosse_head_summary(
    stage,
    *,
    franka_root: str,
    link7_name: str = params.DEFAULT_RELATIVE_LINK7_NAME,
    head_name: str = params.DEFAULT_RELATIVE_HEAD_NAME,
    head_path: str | None = None,
) -> dict:
    """Find link7 and lacrosse head, then return head pose relative to link7."""
    link7 = find_prim_by_name(stage, franka_root, link7_name)
    if link7 is None:
        children = "\n  ".join(list_child_names(stage, franka_root))
        raise RuntimeError(f"cannot find {link7_name!r} under {franka_root}. Prim snapshot:\n  {children}")

    if head_path:
        head = stage.GetPrimAtPath(head_path)
        if not _valid_prim(head):
            raise RuntimeError(f"head prim not found: {head_path}")
    else:
        head = find_prim_by_name(stage, franka_root, head_name)
        if head is None:
            raise RuntimeError(f"cannot find {head_name!r} under {franka_root}")

    return relative_pose_summary(stage, link7.GetPath().pathString, head.GetPath().pathString)


def print_relative_pose(summary: dict) -> None:
    """Print a relative pose summary in a readable format."""
    print(f"[relative] parent={summary['parent_path']}", flush=True)
    print(f"[relative] child={summary['child_path']}", flush=True)
    print(f"[relative] translation={summary['translation'].round(9).tolist()}", flush=True)
    print("[relative] matrix=", flush=True)
    print(np.array2string(summary["matrix"], precision=9, suppress_small=False), flush=True)


def main() -> int:
    if "SOFT_THROW_ROBOT_USD" not in os.environ:
        _load_env_from_file(params.DEFAULT_ENV_SH)

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=bool(params.DEFAULT_SIM_HEADLESS))
    simulation_app = app_launcher.app

    import omni.timeline
    from core.throw_params import FRANKA_ROOT, ROBOT_PRIM, ROBOT_USD
    from sim.isaac_scene import (
        add_robot_reference,
        detect_franka_root,
        ensure_physics_scene,
        get_stage,
        repair_lmm_asset_references,
        spawn_ground,
        spawn_lights,
    )

    print(f"[config] ROBOT_USD={ROBOT_USD}", flush=True)
    print(f"[config] ROBOT_PRIM={ROBOT_PRIM}", flush=True)
    print(f"[config] FRANKA_ROOT={FRANKA_ROOT}", flush=True)

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

    franka_root = detect_franka_root(stage, FRANKA_ROOT)
    summary = link7_to_lacrosse_head_summary(
        stage,
        franka_root=franka_root,
        head_path=params.DEFAULT_SIM_HEAD_PRIM,
    )
    print_relative_pose(summary)
    print("[relative] simulation left running. Press Ctrl+C to exit.", flush=True)
    try:
        while True:
            simulation_app.update()
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
