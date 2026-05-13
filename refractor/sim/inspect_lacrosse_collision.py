#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Inspect collision schemas under the lacrosse head prim in Isaac Sim."""

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


def _has_api(prim, api_cls) -> bool:
    try:
        return bool(prim.HasAPI(api_cls))
    except Exception:
        return False


def inspect_collision(stage, root_path: str) -> None:
    from pxr import PhysxSchema, UsdGeom, UsdPhysics

    root = stage.GetPrimAtPath(root_path)
    if not (root and root.IsValid()):
        raise RuntimeError(f"prim not found: {root_path}")

    total = 0
    mesh_count = 0
    collision_count = 0
    physx_collision_count = 0
    rigid_body_count = 0
    collision_enabled_count = 0
    rows = []

    stack = [root]
    while stack:
        prim = stack.pop()
        total += 1
        is_mesh = bool(prim.IsA(UsdGeom.Mesh))
        has_collision = _has_api(prim, UsdPhysics.CollisionAPI)
        has_physx_collision = _has_api(prim, PhysxSchema.PhysxCollisionAPI)
        has_rigid_body = _has_api(prim, UsdPhysics.RigidBodyAPI) or _has_api(prim, PhysxSchema.PhysxRigidBodyAPI)

        enabled = None
        approximation = None
        if has_collision:
            collision = UsdPhysics.CollisionAPI(prim)
            enabled_attr = collision.GetCollisionEnabledAttr()
            enabled = enabled_attr.Get() if enabled_attr else None
            if enabled is not False:
                collision_enabled_count += 1
        if has_physx_collision:
            physx_collision = PhysxSchema.PhysxCollisionAPI(prim)
            approximation_attr = physx_collision.GetApproximationAttr()
            approximation = approximation_attr.Get() if approximation_attr else None

        mesh_count += int(is_mesh)
        collision_count += int(has_collision)
        physx_collision_count += int(has_physx_collision)
        rigid_body_count += int(has_rigid_body)

        if is_mesh or has_collision or has_physx_collision or has_rigid_body:
            rows.append(
                (
                    prim.GetPath().pathString,
                    prim.GetTypeName(),
                    is_mesh,
                    has_collision,
                    enabled,
                    has_physx_collision,
                    approximation,
                    has_rigid_body,
                    list(prim.GetAppliedSchemas()),
                )
            )

        for child in prim.GetChildren():
            stack.append(child)

    print(f"[collision] root={root_path}", flush=True)
    print(
        f"[collision] prims={total} meshes={mesh_count} collision_api={collision_count} "
        f"enabled_collision={collision_enabled_count} physx_collision_api={physx_collision_count} "
        f"rigid_body_api={rigid_body_count}",
        flush=True,
    )
    if not rows:
        print("[collision] no mesh/collision/rigid-body prims found under root", flush=True)
        return

    for path, type_name, is_mesh, has_collision, enabled, has_physx_collision, approximation, has_rigid_body, schemas in rows:
        print(
            f"[collision] path={path} type={type_name} mesh={is_mesh} "
            f"collision={has_collision} enabled={enabled} "
            f"physx_collision={has_physx_collision} approximation={approximation} "
            f"rigid_body={has_rigid_body} schemas={schemas}",
            flush=True,
        )


def main() -> int:
    if "SOFT_THROW_ROBOT_USD" not in os.environ:
        _load_env_from_file(params.DEFAULT_ENV_SH)

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=bool(params.DEFAULT_SIM_HEADLESS))
    simulation_app = app_launcher.app

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
    for _ in range(int(params.DEFAULT_SIM_WARMUP_STEPS)):
        simulation_app.update()

    inspect_collision(stage, params.DEFAULT_SIM_HEAD_PRIM)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
