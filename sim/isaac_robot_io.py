#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics, Gf

from core.throw_params import (
    BALL_PATH,
    BALL_DETACH_DIST,
    BALL_LANDED_Z_THRESH,
    DEBUG_PRINT_ARM_TARGETS,
)
from sim.isaac_scene import valid, get_or_add_translate_op, find_child_prim_by_name, set_drive_target_rad


def find_joint_paths(stage, franka_root: str):
    root = stage.GetPrimAtPath(franka_root)
    out = {}
    for i in range(1, 8):
        nm = f"panda_joint{i}"
        p = find_child_prim_by_name(root, nm)
        if p is None:
            raise RuntimeError(f"[usd] cannot find {nm} under {franka_root}")
        out[nm] = p.GetPath().pathString
    print("[usd] joints ready", flush=True)
    return out


def apply_arm_targets(stage, joint_paths, q_cmd7: np.ndarray, qdot7: np.ndarray):
    if DEBUG_PRINT_ARM_TARGETS:
        vals = []
        for i in range(7):
            vals.append(float(np.degrees(q_cmd7[i])))
            vals.append(float(np.degrees(qdot7[i])))
        print(f"[arm_targets] pos_deg/vel_deg_s x7 = {vals}", flush=True)
    for i in range(1, 8):
        prim = stage.GetPrimAtPath(joint_paths[f"panda_joint{i}"])
        set_drive_target_rad(prim, float(q_cmd7[i - 1]), float(qdot7[i - 1]))


def reset_ball(stage, pos_world):
    if not valid(stage, BALL_PATH):
        print(f"[ball] not found: {BALL_PATH}", flush=True)
        return
    prim = stage.GetPrimAtPath(BALL_PATH)
    xf = UsdGeom.Xformable(prim)
    op = get_or_add_translate_op(xf)
    op.Set(Gf.Vec3d(*pos_world))
    # best-effort zero velocity
    try:
        rb = UsdPhysics.RigidBodyAPI(prim)
        v = rb.GetVelocityAttr()
        w = rb.GetAngularVelocityAttr()
        if v:
            v.Set(Gf.Vec3f(0.0, 0.0, 0.0))
        if w:
            w.Set(Gf.Vec3f(0.0, 0.0, 0.0))
    except Exception:
        pass


def get_ball_world_pos_vel(stage):
    if not valid(stage, BALL_PATH):
        return None, None, None
    prim = stage.GetPrimAtPath(BALL_PATH)
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    M = cache.GetLocalToWorldTransform(prim)
    p = M.Transform(Gf.Vec3d(0.0, 0.0, 0.0))
    pos = np.array([p[0], p[1], p[2]], dtype=float)

    vel = None
    speed = None
    try:
        rb = UsdPhysics.RigidBodyAPI(prim)
        v_attr = rb.GetVelocityAttr()
        if v_attr:
            v = v_attr.Get()
            vel = np.array([v[0], v[1], v[2]], dtype=float)
            speed = float(np.linalg.norm(vel))
    except Exception:
        pass
    return pos, vel, speed


def ball_landed(stage) -> bool:
    pos, _, _ = get_ball_world_pos_vel(stage)
    if pos is None:
        return False
    if pos[2] > BALL_LANDED_Z_THRESH:
        return False
    return True


def ball_detached_from_net(stage, ee_pos: np.ndarray) -> bool:
    bpos, _, _ = get_ball_world_pos_vel(stage)
    if bpos is None:
        return False
    return float(np.linalg.norm(bpos - ee_pos)) > BALL_DETACH_DIST
