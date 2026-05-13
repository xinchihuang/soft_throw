#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Load and display a USD asset in Isaac Sim without playing a trajectory."""

from __future__ import annotations

import argparse
import os
import select
import sys
import termios
import time
import tty
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _as_asset_ref(path_or_url: str) -> str:
    if "://" in path_or_url:
        return path_or_url
    path = Path(path_or_url).expanduser()
    if path.is_file():
        return "file://" + str(path.resolve())
    return path_or_url


def _print_stage_summary(stage, root_path: str | None) -> None:
    world = stage.GetPrimAtPath("/World")
    if world and world.IsValid():
        children = [(child.GetPath().pathString, child.GetTypeName()) for child in world.GetChildren()]
        print(f"[asset] /World children={children}", flush=True)

    if root_path is None:
        return

    root = stage.GetPrimAtPath(root_path)
    if not (root and root.IsValid()):
        print(f"[asset] root prim not found: {root_path}", flush=True)
        return

    children = [(child.GetPath().pathString, child.GetTypeName()) for child in root.GetChildren()]
    print(f"[asset] root={root_path}", flush=True)
    print(f"[asset] children={children}", flush=True)

    joint_paths = []
    for prim in stage.Traverse():
        name = prim.GetName()
        if name.startswith("panda_joint"):
            joint_paths.append(prim.GetPath().pathString)
    if joint_paths:
        print(f"[asset] panda joints={joint_paths}", flush=True)


def _find_prim_by_name(stage, predicates):
    for prim in stage.Traverse():
        name = prim.GetName().lower()
        path = prim.GetPath().pathString.lower()
        if any(predicate(name, path) for predicate in predicates):
            return prim
    return None


def _find_ball_prim(stage, explicit_path: str | None):
    for path in [explicit_path, "/World/LMM/Ball", "/World/Ball"]:
        if not path:
            continue
        prim = stage.GetPrimAtPath(path)
        if prim and prim.IsValid():
            return prim
    return _find_prim_by_name(stage, [lambda name, _path: name == "ball"])


def _find_lacrosse_head_prim(stage, explicit_path: str | None):
    for path in [
        explicit_path,
        "/World/LMM/Franka/lacorsse",
        "/World/LMM/Franka/lacorsse/LacrossTrophyHead",
        "/World/LMM/Franka/lacrosse/LacrossTrophyHead",
        "/World/Franka/lacrosse/LacrossTrophyHead",
        "/World/LacrossTrophyHead",
    ]:
        if not path:
            continue
        prim = stage.GetPrimAtPath(path)
        if prim and prim.IsValid():
            return prim
    return _find_prim_by_name(
        stage,
        [
            lambda name, _path: "lacrosstrophyhead" in name,
            lambda name, path: "lacorsse" in path and "head" in name,
            lambda name, path: "lacorsse" in path and "head" in path,
            lambda name, path: "lacrosse" in path and "head" in name,
            lambda name, path: "lacrosse" in path and "head" in path,
        ],
    )


def _set_world_translation(prim, world_pos):
    from pxr import Gf, UsdGeom

    parent = prim.GetParent()
    if parent and parent.IsValid():
        parent_world = UsdGeom.XformCache().GetLocalToWorldTransform(parent)
        local_pos = parent_world.GetInverse().Transform(Gf.Vec3d(*world_pos))
    else:
        local_pos = Gf.Vec3d(*world_pos)

    xf = UsdGeom.Xformable(prim)
    translate_op = None
    for op in xf.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            translate_op = op
            break
    if translate_op is None:
        translate_op = xf.AddTranslateOp()
    translate_op.Set(local_pos)


def _zero_rigid_body_velocity(prim):
    from pxr import Gf, UsdPhysics

    try:
        rb = UsdPhysics.RigidBodyAPI(prim)
        vel_attr = rb.GetVelocityAttr()
        ang_attr = rb.GetAngularVelocityAttr()
        if vel_attr:
            vel_attr.Set(Gf.Vec3f(0.0, 0.0, 0.0))
        if ang_attr:
            ang_attr.Set(Gf.Vec3f(0.0, 0.0, 0.0))
    except Exception:
        pass


def _reset_ball_above_lacrosse_head(stage, *, ball_path: str | None, head_path: str | None, offset_m: float) -> bool:
    from pxr import Usd, UsdGeom

    ball = _find_ball_prim(stage, ball_path)
    if ball is None:
        print("[reset] ball prim not found", flush=True)
        return False

    head = _find_lacrosse_head_prim(stage, head_path)
    if head is None:
        print("[reset] lacrosse head prim not found", flush=True)
        return False

    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy"])
    bbox = bbox_cache.ComputeWorldBound(head).ComputeAlignedBox()
    min_pt = bbox.GetMin()
    max_pt = bbox.GetMax()
    center = (
        0.5 * (float(min_pt[0]) + float(max_pt[0])),
        0.5 * (float(min_pt[1]) + float(max_pt[1])),
        0.5 * (float(min_pt[2]) + float(max_pt[2])),
    )
    target = (float(center[0]), float(center[1]), float(max_pt[2]) + float(offset_m))

    _set_world_translation(ball, target)
    _zero_rigid_body_velocity(ball)
    print(f"[reset] moved {ball.GetPath()} above {head.GetPath()} to {target}", flush=True)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Show a USD asset in Isaac Sim.")
    parser.add_argument(
        "--asset",
        type=str,
        default=os.environ.get("SOFT_THROW_ROBOT_USD", "/home/xinchi/Desktop/lacosse_large_mobie_manipulator_new.usd"),
        help="USD asset path or URL to reference into the stage.",
    )
    parser.add_argument(
        "--prim",
        type=str,
        default=os.environ.get("SOFT_THROW_ROBOT_PRIM", "/World/LMM"),
        help="Stage prim path where the asset is referenced when --reference is used.",
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--reference",
        action="store_true",
        help="Reference the asset under --prim instead of opening the USD as the stage.",
    )
    parser.add_argument("--no-ground", action="store_true", help="Do not add the helper ground plane.")
    parser.add_argument("--no-lights", action="store_true", help="Do not add helper lights.")
    parser.add_argument("--ball-prim", type=str, default=None, help="Optional explicit ball prim path.")
    parser.add_argument(
        "--head-prim",
        type=str,
        default="/World/LMM/Franka/lacrosse",
        help="Optional explicit lacrosse head prim path.",
    )
    parser.add_argument("--ball-offset", type=float, default=0.10, help="Ball reset height above lacrosse head top, in meters.")
    args = parser.parse_args()

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=bool(args.headless))
    simulation_app = app_launcher.app

    import omni.timeline
    import omni.usd
    from sim.isaac_scene import (
        get_stage,
        ensure_physics_scene,
        repair_lmm_asset_references,
        spawn_ground,
        spawn_lights,
        valid,
    )

    asset_ref = _as_asset_ref(args.asset)
    if bool(args.reference):
        stage = get_stage()
    else:
        print(f"[asset] opening stage {asset_ref}", flush=True)
        if not omni.usd.get_context().open_stage(asset_ref):
            raise RuntimeError(f"Failed to open USD stage: {asset_ref}")
        for _ in range(120):
            simulation_app.update()
        stage = get_stage()

    ensure_physics_scene(stage)
    if not bool(args.no_ground):
        spawn_ground(stage)
    if not bool(args.no_lights):
        spawn_lights(stage)

    summary_root = None
    if bool(args.reference):
        if not valid(stage, args.prim):
            stage.DefinePrim(args.prim, "Xform")
        stage.GetPrimAtPath(args.prim).GetReferences().AddReference(asset_ref)
        print(f"[asset] referenced {asset_ref} -> {args.prim}", flush=True)
        repair_lmm_asset_references(stage, args.prim)
        summary_root = args.prim
    else:
        print(f"[asset] opened {asset_ref}", flush=True)

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()

    for _ in range(120):
        simulation_app.update()

    _print_stage_summary(stage, summary_root)
    print("[asset] simulation running. Press Enter to reset ball, Ctrl+C to exit.", flush=True)

    stdin_is_tty = bool(sys.stdin.isatty())
    stdin_fd = sys.stdin.fileno()
    old_settings = None
    if stdin_is_tty:
        old_settings = termios.tcgetattr(stdin_fd)
        tty.setcbreak(stdin_fd)
    try:
        while True:
            ready, _, _ = select.select([sys.stdin], [], [], 0.0)
            if ready:
                key = sys.stdin.read(1) if stdin_is_tty else sys.stdin.readline()[:1]
                if key in ("\r", "\n", " "):
                    _reset_ball_above_lacrosse_head(
                        stage,
                        ball_path=args.ball_prim,
                        head_path=args.head_prim,
                        offset_m=float(args.ball_offset),
                    )
            simulation_app.update()
            time.sleep(0.01)
    except KeyboardInterrupt:
        return 0
    finally:
        if old_settings is not None:
            termios.tcsetattr(stdin_fd, termios.TCSADRAIN, old_settings)


if __name__ == "__main__":
    raise SystemExit(main())
