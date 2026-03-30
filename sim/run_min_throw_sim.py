#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Minimal pose/velocity control script for Isaac Sim.

Inputs: desired lacrosse target (x,y,z,vx,vy,vz) in world.
Pipeline: reset -> solve IK -> move joint7 to target pose/velocity -> hold.
"""

import argparse
import time
import sys
from pathlib import Path
import numpy as np

# Ensure repo root is on sys.path when running from sim/
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from isaaclab.app import AppLauncher


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pose_lac_vel", type=float, nargs=6, required=True,
                        help="Target lacrosse: x y z vx vy vz (world)")
    parser.add_argument("--hold_only", action="store_true",
                        help="Do not move arm; only reset/hold and print lacrosse/joint7 pose")
    parser.add_argument("--place_ball_on_lacrosse", action="store_true",
                        help="After reset, place ball at lacrosse position and hold (no arm motion)")
    parser.add_argument("--hold_sec", type=float, default=2.0, help="Hold duration for test mode")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    # Launch Isaac Sim
    app_launcher = AppLauncher(headless=bool(args.headless))
    simulation_app = app_launcher.app

    # IMPORTANT: import Omniverse/Isaac modules AFTER SimulationApp is created
    from core.throw_params import (
        ROBOT_USD,
        ROBOT_PRIM,
        FRANKA_ROOT,
        DT_CONTROL,
        SIM_UPDATES_PER_STEP,
        RESET_ARM_SETTLE_SEC,
        BALL_RESET_POS_WORLD,
        RESET_BALL_WAIT_SEC,
        INIT_ARM,
        JOINT0_ALIGN_STEPS,
        QDOT_LIMITS_7,
        QDDOT_LIMITS_7,
        Q_LIMITS_7,
        LACROSSE_PATH,
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
        reset_ball,
    )
    from core.kinematics_pin import PinKinematics
    from pxr import UsdPhysics
    import pinocchio as pin
    import omni.timeline

    def _read_joint_targets(stage, joint_paths):
        q = np.zeros(7, dtype=float)
        for i in range(1, 8):
            prim = stage.GetPrimAtPath(joint_paths[f"panda_joint{i}"])
            drv = UsdPhysics.DriveAPI(prim, "angular")
            attr = drv.GetTargetPositionAttr()
            if attr:
                # stored in degrees
                q[i - 1] = np.deg2rad(float(attr.Get()))
        return q

    def _interp_move(simulation_app, stage, joint_paths, q_from, q_to, steps: int, phase: str):
        q_from = np.asarray(q_from, dtype=float)
        q_to = np.asarray(q_to, dtype=float)
        for k in range(max(1, steps)):
            alpha = (k + 1) / float(max(1, steps))
            q = (1 - alpha) * q_from + alpha * q_to
            qdot = np.zeros(7, dtype=float)
            print(f"[joint_control][{phase}] step={k+1}/{max(1, steps)} q_cmd7={q.tolist()} qdot7={qdot.tolist()}", flush=True)
            apply_arm_targets(stage, joint_paths, q, qdot)
            for _ in range(SIM_UPDATES_PER_STEP):
                simulation_app.update()
        return q_to.copy()

    # Build scene
    stage = get_stage()
    ensure_physics_scene(stage)
    spawn_ground(stage)
    spawn_lights(stage)
    add_robot_reference(stage, ROBOT_PRIM, ROBOT_USD)

    # Auto-play timeline
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()

    # Debug: find and print panda_link8 prim path(s)
    try:
        matches = []
        for prim in stage.Traverse():
            name = prim.GetName()
            if "panda_link8" in name:
                matches.append(prim.GetPath().pathString)
        print(f"[debug] panda_link8 prims: {matches}", flush=True)
    except Exception as e:
        print(f"[debug] failed to find panda_link8 prims: {e}", flush=True)

    # Debug: find and print panda_joint1 world position
    try:
        from pxr import UsdGeom, Usd, Gf
        joint0_paths = []
        for prim in stage.Traverse():
            if prim.GetName() == "panda_joint1":
                joint0_paths.append(prim.GetPath().pathString)
        print(f"[debug] panda_joint1 prims: {joint0_paths}", flush=True)
        if joint0_paths:
            joint0_prim = stage.GetPrimAtPath(joint0_paths[0])
            cache = UsdGeom.XformCache(Usd.TimeCode.Default())
            M = cache.GetLocalToWorldTransform(joint0_prim)
            p = M.Transform(Gf.Vec3d(0.0, 0.0, 0.0))
            print(f"[debug] panda_joint1 world pos: {[p[0], p[1], p[2]]}", flush=True)
    except Exception as e:
        print(f"[debug] failed to get panda_joint1 world pos: {e}", flush=True)

    def _resolve_lacrosse_prim(stage):
        cand_paths = [LACROSSE_PATH, LACROSSE_PATH + "/mesh"]
        for p in cand_paths:
            prim = stage.GetPrimAtPath(p)
            if prim and prim.IsValid():
                return prim
        # fallback: search by name
        for prim in stage.Traverse():
            name = prim.GetName().lower()
            if "lacrosse" in name:
                return prim
        return None

    # Debug: print lacrosse world position
    try:
        from pxr import UsdGeom, Usd, Gf
        lacrosse_prim = _resolve_lacrosse_prim(stage)
        if lacrosse_prim and lacrosse_prim.IsValid():
            cache = UsdGeom.XformCache(Usd.TimeCode.Default())
            M = cache.GetLocalToWorldTransform(lacrosse_prim)
            p = M.Transform(Gf.Vec3d(0.0, 0.0, 0.0))
            print(f"[debug] lacrosse world pos: {[p[0], p[1], p[2]]} path={lacrosse_prim.GetPath().pathString}", flush=True)
            # Euler angles (XYZ) in degrees
            R = M.ExtractRotation()
            euler = R.Decompose(Gf.Vec3d(1, 0, 0), Gf.Vec3d(0, 1, 0), Gf.Vec3d(0, 0, 1))
            print(f"[debug] lacrosse euler xyz deg: {[float(euler[0]), float(euler[1]), float(euler[2])]}", flush=True)
            # Debug: children and xform ops
            children = [c.GetPath().pathString for c in lacrosse_prim.GetChildren()]
            xf = UsdGeom.Xformable(lacrosse_prim)
            ops = [op.GetOpType().name for op in xf.GetOrderedXformOps()]
            print(f"[debug] lacrosse children: {children}", flush=True)
            print(f"[debug] lacrosse xform ops: {ops}", flush=True)
        else:
            print(f"[debug] lacrosse prim not found (searched {LACROSSE_PATH})", flush=True)
    except Exception as e:
        print(f"[debug] failed to get lacrosse world pos: {e}", flush=True)

    for _ in range(120):
        simulation_app.update()

    franka_root = detect_franka_root(stage, FRANKA_ROOT)
    joint_paths = find_joint_paths(stage, franka_root)
    # Debug: print all panda_joint paths
    try:
        all_joints = []
        for prim in stage.Traverse():
            name = prim.GetName()
            if name.startswith("panda_joint"):
                all_joints.append(prim.GetPath().pathString)
        print(f"[debug] panda_joint paths: {all_joints}", flush=True)
    except Exception as e:
        print(f"[debug] failed to list panda joints: {e}", flush=True)

    # Parse target lacrosse position and velocity (world)
    pose = np.asarray(args.pose_lac_vel, dtype=float).reshape(6)
    p_lac_des = pose[:3]
    v_lac_des = pose[3:]

    pin_model = PinKinematics()
    target_frame = "panda_link7"
    target_frame_id = pin_model.model.getFrameId(target_frame)
    if target_frame_id == len(pin_model.model.frames):
        raise RuntimeError(f"[pin] target frame not found: {target_frame}")

    # Fixed rigid transform from joint7 -> lacrosse (from hold-time calibration)
    R_rel = np.array(
        [
            [0.068831, 0.0, -0.997628],
            [-0.997628, 0.0, -0.068831],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    p_rel = np.array([0.08539635, -0.10828736, -1.14e-07], dtype=float)

    # Reset: arm to INIT, settle
    q_cmd7 = INIT_ARM.copy()
    qdot7 = np.zeros(7, dtype=float)
    t0 = time.time()
    while time.time() - t0 < RESET_ARM_SETTLE_SEC:
        apply_arm_targets(stage, joint_paths, q_cmd7, qdot7)
        simulation_app.update()


    # Optional ball reset (kept for scene consistency)
    reset_ball(stage, BALL_RESET_POS_WORLD)
    t1 = time.time()
    while time.time() - t1 < RESET_BALL_WAIT_SEC:
        apply_arm_targets(stage, joint_paths, q_cmd7, qdot7)
        simulation_app.update()

    if args.hold_only:
        print("[test] hold_only enabled: arm will not move.", flush=True)
        # Snap targets to current drive targets to avoid motion
        q_hold = _read_joint_targets(stage, joint_paths)
        # Hold arm steady
        t_hold = time.time()
        while time.time() - t_hold < 2.0:
            apply_arm_targets(stage, joint_paths, q_hold, np.zeros(7, dtype=float))
            simulation_app.update()
        # Print lacrosse and joint7 pose after reset/hold
        try:
            from pxr import UsdGeom, Usd, Gf
            lacrosse_prim = _resolve_lacrosse_prim(stage)
            if lacrosse_prim and lacrosse_prim.IsValid():
                cache = UsdGeom.XformCache(Usd.TimeCode.Default())
                M = cache.GetLocalToWorldTransform(lacrosse_prim)
                p = M.Transform(Gf.Vec3d(0.0, 0.0, 0.0))
                R = M.ExtractRotation()
                euler = R.Decompose(Gf.Vec3d(1, 0, 0), Gf.Vec3d(0, 1, 0), Gf.Vec3d(0, 0, 1))
                print(f"[debug] lacrosse world pos (hold): {[p[0], p[1], p[2]]} path={lacrosse_prim.GetPath().pathString}", flush=True)
                print(f"[debug] lacrosse euler xyz deg (hold): {[float(euler[0]), float(euler[1]), float(euler[2])]}", flush=True)
            joint7_path = None
            for prim in stage.Traverse():
                if prim.GetName() == "panda_joint7":
                    joint7_path = prim.GetPath().pathString
                    break
            if joint7_path:
                joint7_prim = stage.GetPrimAtPath(joint7_path)
                cache = UsdGeom.XformCache(Usd.TimeCode.Default())
                M = cache.GetLocalToWorldTransform(joint7_prim)
                p = M.Transform(Gf.Vec3d(0.0, 0.0, 0.0))
                R = M.ExtractRotation()
                euler = R.Decompose(Gf.Vec3d(1, 0, 0), Gf.Vec3d(0, 1, 0), Gf.Vec3d(0, 0, 1))
                print(f"[debug] joint7 world pos (hold): {[p[0], p[1], p[2]]} path={joint7_path}", flush=True)
                print(f"[debug] joint7 euler xyz deg (hold): {[float(euler[0]), float(euler[1]), float(euler[2])]}", flush=True)
        except Exception as e:
            print(f"[debug] failed to print hold poses: {e}", flush=True)
        # Keep sim running without moving arm
        while True:
            apply_arm_targets(stage, joint_paths, q_hold, np.zeros(7, dtype=float))
            simulation_app.update()
        return
    else:
        # Desired joint7 pose from desired lacrosse position (orientation held)
        # We hold current joint7 orientation to avoid orientation commands.
        q_full = pin_model.make_q_full_from_arm7(q_cmd7)
        pin.forwardKinematics(pin_model.model, pin_model.data, q_full)
        pin.updateFramePlacements(pin_model.model, pin_model.data)
        T0 = pin_model.data.oMf[target_frame_id]
        R_j7_des = np.array(T0.rotation, dtype=float)
        p_j7_des = p_lac_des - R_j7_des @ p_rel

        # IK loop to reach target pose (joint7)
        max_iter = 300
        kp_pos = 2.0
        kp_rot = 1.0
        qdot_prev = np.zeros(7, dtype=float)

        for i in range(max_iter):
            q_full = pin_model.make_q_full_from_arm7(q_cmd7)
            pin.forwardKinematics(pin_model.model, pin_model.data, q_full)
            pin.updateFramePlacements(pin_model.model, pin_model.data)
            T = pin_model.data.oMf[target_frame_id]
            p_cur = np.array(T.translation, dtype=float)
            R_cur = np.array(T.rotation, dtype=float)

            ep = p_j7_des - p_cur
            eR = pin.log3(R_cur.T @ R_j7_des)

            if np.linalg.norm(ep) < 1e-3 and np.linalg.norm(eR) < 1e-3:
                print(f"[ik] converged at iter {i}", flush=True)
                break

            J = pin.computeFrameJacobian(
                pin_model.model, pin_model.data, q_full, target_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
            )  # 6 x nv
            # build 6x7
            J_arm = np.zeros((6, 7), dtype=float)
            for j in range(1, 8):
                jid = pin_model.model.getJointId(f"panda_joint{j}")
                idx_v = int(pin_model.model.joints[jid].idx_v)
                J_arm[:, j - 1] = J[:, idx_v]

            lam = 0.05
            JJt = J_arm @ J_arm.T
            inv = np.linalg.inv(JJt + (lam * lam) * np.eye(6))
            J_pinv = J_arm.T @ inv
            # Desired spatial velocity: linear = v_lac_des + Kp*pos_err, angular = Kp_rot*rot_err
            v_cmd = v_lac_des + kp_pos * ep
            w_cmd = kp_rot * eR
            v6 = np.concatenate([v_cmd, w_cmd])
            qdot_des = J_pinv @ v6

            # accel limits
            max_dq = QDDOT_LIMITS_7 * DT_CONTROL
            dq = np.clip(qdot_des - qdot_prev, -max_dq, max_dq)
            qdot7 = qdot_prev + dq
            qdot7 = np.clip(qdot7, -QDOT_LIMITS_7, QDOT_LIMITS_7)

            q_cmd7 = q_cmd7 + qdot7 * DT_CONTROL
            q_cmd7 = np.clip(q_cmd7, Q_LIMITS_7[:, 0], Q_LIMITS_7[:, 1])
            qdot_prev = qdot7.copy()

            apply_arm_targets(stage, joint_paths, q_cmd7, qdot7)
            for _ in range(SIM_UPDATES_PER_STEP):
                simulation_app.update()

    # Hold at target (or INIT if hold_only)
    t2 = time.time()
    while time.time() - t2 < 1.0:
        apply_arm_targets(stage, joint_paths, INIT_ARM if args.hold_only else q_cmd7, np.zeros(7, dtype=float))
        simulation_app.update()

    # Print lacrosse world position after execution
    try:
        from pxr import UsdGeom, Usd, Gf
        lacrosse_prim = _resolve_lacrosse_prim(stage)
        if lacrosse_prim and lacrosse_prim.IsValid():
            cache = UsdGeom.XformCache(Usd.TimeCode.Default())
            M = cache.GetLocalToWorldTransform(lacrosse_prim)
            p = M.Transform(Gf.Vec3d(0.0, 0.0, 0.0))
            print(f"[debug] lacrosse world pos (after): {[p[0], p[1], p[2]]} path={lacrosse_prim.GetPath().pathString}", flush=True)
            # Euler angles (XYZ) in degrees
            R = M.ExtractRotation()
            euler = R.Decompose(Gf.Vec3d(1, 0, 0), Gf.Vec3d(0, 1, 0), Gf.Vec3d(0, 0, 1))
            print(f"[debug] lacrosse euler xyz deg (after): {[float(euler[0]), float(euler[1]), float(euler[2])]}", flush=True)
        else:
            print(f"[debug] lacrosse prim not found (after)", flush=True)
    except Exception as e:
        print(f"[debug] failed to get lacrosse world pos: {e}", flush=True)

    # Print joint7 world position and angle
    try:
        from pxr import UsdGeom, Usd, Gf
        joint7_path = None
        for prim in stage.Traverse():
            if prim.GetName() == "panda_joint7":
                joint7_path = prim.GetPath().pathString
                break
        if joint7_path:
            joint7_prim = stage.GetPrimAtPath(joint7_path)
            cache = UsdGeom.XformCache(Usd.TimeCode.Default())
            M = cache.GetLocalToWorldTransform(joint7_prim)
            p = M.Transform(Gf.Vec3d(0.0, 0.0, 0.0))
            R = M.ExtractRotation()
            euler = R.Decompose(Gf.Vec3d(1, 0, 0), Gf.Vec3d(0, 1, 0), Gf.Vec3d(0, 0, 1))
            print(f"[debug] joint7 world pos (after): {[p[0], p[1], p[2]]} path={joint7_path}", flush=True)
            print(f"[debug] joint7 euler xyz deg (after): {[float(euler[0]), float(euler[1]), float(euler[2])]}", flush=True)
        else:
            print("[debug] joint7 prim not found", flush=True)
    except Exception as e:
        print(f"[debug] failed to get joint7 world pos: {e}", flush=True)

    # Keep simulation running for inspection
    print("[done] IK reached target. Simulation left running.", flush=True)
    while True:
        simulation_app.update()


if __name__ == "__main__":
    main()
