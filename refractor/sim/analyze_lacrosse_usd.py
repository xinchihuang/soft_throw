#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Analyze lacrosse-related USD prims, references, meshes, and physics schemas."""

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


def _prim_references(prim) -> list[str]:
    refs: list[str] = []
    try:
        for spec in prim.GetPrimStack():
            info = spec.GetInfo("references")
            if not info:
                continue
            for ref in info.prependedItems:
                refs.append(str(ref))
            for ref in info.addedItems:
                refs.append(str(ref))
            for ref in info.explicitItems:
                refs.append(str(ref))
    except Exception as exc:
        refs.append(f"<reference read failed: {exc}>")
    return refs


def _prim_payloads(prim) -> list[str]:
    payloads: list[str] = []
    try:
        for spec in prim.GetPrimStack():
            info = spec.GetInfo("payload")
            if not info:
                info = spec.GetInfo("payloads")
            if not info:
                continue
            for payload in getattr(info, "prependedItems", []):
                payloads.append(str(payload))
            for payload in getattr(info, "addedItems", []):
                payloads.append(str(payload))
            for payload in getattr(info, "explicitItems", []):
                payloads.append(str(payload))
    except Exception as exc:
        payloads.append(f"<payload read failed: {exc}>")
    return payloads


def _print_prim_line(prim, *, prefix: str = "[usd]") -> None:
    from pxr import PhysxSchema, UsdGeom, UsdPhysics

    path = prim.GetPath().pathString
    type_name = prim.GetTypeName()
    schemas = list(prim.GetAppliedSchemas())
    is_mesh = bool(prim.IsA(UsdGeom.Mesh))
    has_collision = _has_api(prim, UsdPhysics.CollisionAPI)
    has_physx_collision = _has_api(prim, PhysxSchema.PhysxCollisionAPI)
    has_rigid_body = _has_api(prim, UsdPhysics.RigidBodyAPI) or _has_api(prim, PhysxSchema.PhysxRigidBodyAPI)
    refs = _prim_references(prim)
    payloads = _prim_payloads(prim)
    print(
        f"{prefix} path={path} type={type_name} active={prim.IsActive()} loaded={prim.IsLoaded()} "
        f"instance={prim.IsInstance()} mesh={is_mesh} collision={has_collision} "
        f"physx_collision={has_physx_collision} rigid_body={has_rigid_body} schemas={schemas}",
        flush=True,
    )
    if refs:
        print(f"{prefix}   references={refs}", flush=True)
    if payloads:
        print(f"{prefix}   payloads={payloads}", flush=True)


def analyze_stage(stage, root_path: str, keyword: str) -> None:
    from pxr import PhysxSchema, UsdGeom, UsdPhysics

    root = stage.GetPrimAtPath(root_path)
    print(f"[usd] root={root_path} valid={bool(root and root.IsValid())}", flush=True)
    if root and root.IsValid():
        _print_prim_line(root)

    keyword_l = keyword.lower()
    matches = []
    meshes = []
    collisions = []
    rigid_bodies = []

    for prim in stage.TraverseAll():
        path_l = prim.GetPath().pathString.lower()
        name_l = prim.GetName().lower()
        if (
            keyword_l in path_l
            or keyword_l in name_l
            or "lacross" in path_l
            or "lacross" in name_l
            or "lacorsse" in path_l
            or "lacorsse" in name_l
        ):
            matches.append(prim)
        if prim.IsA(UsdGeom.Mesh):
            meshes.append(prim)
        if _has_api(prim, UsdPhysics.CollisionAPI) or _has_api(prim, PhysxSchema.PhysxCollisionAPI):
            collisions.append(prim)
        if _has_api(prim, UsdPhysics.RigidBodyAPI) or _has_api(prim, PhysxSchema.PhysxRigidBodyAPI):
            rigid_bodies.append(prim)

    print(
        f"[usd] totals prims={sum(1 for _ in stage.TraverseAll())} meshes={len(meshes)} "
        f"colliders={len(collisions)} rigid_bodies={len(rigid_bodies)} "
        f"lacrosse_matches={len(matches)}",
        flush=True,
    )

    print("[usd] lacrosse-related prims:", flush=True)
    for prim in matches:
        _print_prim_line(prim, prefix="[usd:lacrosse]")

    print("[usd] mesh prims under lacrosse-related paths:", flush=True)
    lacrosse_meshes = [
        prim
        for prim in meshes
        if "lacrosse" in prim.GetPath().pathString.lower()
        or "lacross" in prim.GetPath().pathString.lower()
        or "lacorsse" in prim.GetPath().pathString.lower()
    ]
    if not lacrosse_meshes:
        print("[usd:mesh] none", flush=True)
    for prim in lacrosse_meshes:
        _print_prim_line(prim, prefix="[usd:mesh]")

    print("[usd] collider prims under lacrosse-related paths:", flush=True)
    lacrosse_collisions = [
        prim
        for prim in collisions
        if "lacrosse" in prim.GetPath().pathString.lower()
        or "lacross" in prim.GetPath().pathString.lower()
        or "lacorsse" in prim.GetPath().pathString.lower()
    ]
    if not lacrosse_collisions:
        print("[usd:collider] none", flush=True)
    for prim in lacrosse_collisions:
        _print_prim_line(prim, prefix="[usd:collider]")

    if not matches:
        print("[usd] no lacrosse keyword match; printing all mesh/collider/rigid body prims:", flush=True)
        for prim in meshes:
            _print_prim_line(prim, prefix="[usd:all_mesh]")
        for prim in collisions:
            if not prim.IsA(UsdGeom.Mesh):
                _print_prim_line(prim, prefix="[usd:all_collider]")
        for prim in rigid_bodies:
            if not prim.IsA(UsdGeom.Mesh):
                _print_prim_line(prim, prefix="[usd:all_rigid_body]")


def main() -> int:
    if "SOFT_THROW_ROBOT_USD" not in os.environ:
        _load_env_from_file(params.DEFAULT_ENV_SH)

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=bool(params.DEFAULT_SIM_HEADLESS))
    simulation_app = app_launcher.app

    from core.throw_params import ROBOT_PRIM, ROBOT_USD
    from sim.isaac_scene import add_robot_reference, get_stage, repair_lmm_asset_references

    stage = get_stage()
    add_robot_reference(stage, ROBOT_PRIM, ROBOT_USD)
    repair_lmm_asset_references(stage, ROBOT_PRIM)
    for _ in range(int(params.DEFAULT_SIM_WARMUP_STEPS)):
        simulation_app.update()

    print(f"[usd] asset={ROBOT_USD}", flush=True)
    analyze_stage(stage, params.DEFAULT_SIM_HEAD_PRIM, keyword=str(params.DEFAULT_USD_ANALYZE_KEYWORD))
    simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
