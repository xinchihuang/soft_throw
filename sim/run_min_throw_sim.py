#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Simulation runner: compute joint trajectory (IK + poly5) and execute.

Inputs: desired joint7 target (x,y,z,vx,vy,vz) in world.
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


def _append_trace_sample(t_hist, q_hist, qdot_hist, qddot_hist, t_now, q_now, qdot_now, qddot_now):
    t_hist.append(float(t_now))
    q_hist.append(np.asarray(q_now, dtype=float).copy())
    qdot_hist.append(np.asarray(qdot_now, dtype=float).copy())
    qddot_hist.append(np.asarray(qddot_now, dtype=float).copy())


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--pose_j7_vel",
        type=float,
        nargs=6,
        help="Target joint7: x y z vx vy vz (world)",
    )
    group.add_argument(
        "--release_pos",
        type=float,
        nargs=3,
        help="Release position in world: x y z (joint7)",
    )
    parser.add_argument(
        "--target_pos",
        type=float,
        nargs=3,
        help="Target landing position in world: x y z (ground z=0 if omitted)",
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--hold_only", action="store_true",
                        help="Do not move arm; only reset/hold and print joint7 pose")
    args = parser.parse_args()

    app_launcher = AppLauncher(headless=bool(args.headless))
    simulation_app = app_launcher.app

    from core.throw_params import (
        ROBOT_USD,
        ROBOT_PRIM,
        FRANKA_ROOT,
        DT_CONTROL,
        WAYPOINT_DENSITY,
        SIM_UPDATES_PER_STEP,
        RESET_ARM_SETTLE_SEC,
        BALL_RESET_POS_WORLD,
        RESET_BALL_WAIT_SEC,
        INIT_ARM,
        RESET_ARM_POS_WORLD,
        QDOT_LIMITS_7,
        QDDOT_LIMITS_7,
        Q_LIMITS_7,
        PATH_PLAN_WAYPOINTS,
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
    from core.ik_poly5_core import solve_ik_for_q_goal, solve_ik_for_reset_pos, poly5_trajectory
    from core.tube_baseline import solve_ballistic_velocity
    from sim.plot_joint_traces import plot_from_csv
    import pinocchio as pin
    import omni.timeline

    def _read_joint_targets(stage, joint_paths):
        from pxr import UsdPhysics
        q = np.zeros(7, dtype=float)
        for i in range(1, 8):
            prim = stage.GetPrimAtPath(joint_paths[f"panda_joint{i}"])
            drv = UsdPhysics.DriveAPI(prim, "angular")
            attr = drv.GetTargetPositionAttr()
            if attr:
                q[i - 1] = np.deg2rad(float(attr.Get()))
        return q

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

    # Build pinocchio model (used for reset IK and later trajectory IK)
    pin_model = PinKinematics()

    # Reset arm (optionally via world-position IK)
    q_cmd7 = INIT_ARM.copy()
    if RESET_ARM_POS_WORLD is not None:
        reset_frame_id = pin_model.model.getFrameId("panda_link7")
        if reset_frame_id == len(pin_model.model.frames):
            raise RuntimeError("[pin] reset frame not found: panda_link7")
        q_cmd7, _ = solve_ik_for_reset_pos(
            pin_model,
            q_cmd7,
            RESET_ARM_POS_WORLD,
            Q_LIMITS_7,
            QDOT_LIMITS_7,
            QDDOT_LIMITS_7,
            control_dt=DT_CONTROL,
            target_frame_id=reset_frame_id,
        )
    qdot7 = np.zeros(7, dtype=float)
    n_reset = max(1, int(np.ceil(RESET_ARM_SETTLE_SEC / DT_CONTROL)))
    for _ in range(n_reset):
        apply_arm_targets(stage, joint_paths, q_cmd7, qdot7)
        simulation_app.update()

    # Ball reset (kept for scene consistency)
    reset_ball(stage, BALL_RESET_POS_WORLD)
    n_wait = max(1, int(np.ceil(RESET_BALL_WAIT_SEC / DT_CONTROL)))
    for _ in range(n_wait):
        apply_arm_targets(stage, joint_paths, q_cmd7, qdot7)
        simulation_app.update()

    if args.hold_only:
        q_hold = _read_joint_targets(stage, joint_paths)
        t_hold = time.time()
        while time.time() - t_hold < 2.0:
            apply_arm_targets(stage, joint_paths, q_hold, np.zeros(7, dtype=float))
            simulation_app.update()
        # Keep sim running without moving arm
        while True:
            apply_arm_targets(stage, joint_paths, q_hold, np.zeros(7, dtype=float))
            simulation_app.update()

    def _link7_world_pos(q7):
        q_full = pin_model.make_q_full_from_arm7(q7)
        pin.forwardKinematics(pin_model.model, pin_model.data, q_full)
        pin.updateFramePlacements(pin_model.model, pin_model.data)
        fid = pin_model.model.getFrameId("panda_link7")
        if fid == len(pin_model.model.frames):
            raise RuntimeError("[pin] target frame not found: panda_link7")
        T = pin_model.data.oMf[fid]
        return np.array(T.translation, dtype=float)

    # Parse target
    if args.release_pos is not None:
        if args.target_pos is None:
            raise RuntimeError("--target_pos required when using --release_pos")
        p_j7_des = np.asarray(args.release_pos, dtype=float).reshape(3)
        target_pos = np.asarray(args.target_pos, dtype=float).reshape(3)
        v_j7_des, T_ball = solve_ballistic_velocity(p_j7_des, target_pos)
        print(f"[ballistic] T={T_ball:.3f} v={v_j7_des.tolist()}", flush=True)
    else:
        pose = np.asarray(args.pose_j7_vel, dtype=float).reshape(6)
        p_j7_des = pose[:3]
        v_j7_des = pose[3:]

    target_frame = "panda_link7"
    target_frame_id = pin_model.model.getFrameId(target_frame)
    if target_frame_id == len(pin_model.model.frames):
        raise RuntimeError(f"[pin] target frame not found: {target_frame}")

    # Use current orientation as target orientation
    q_full = pin_model.make_q_full_from_arm7(q_cmd7)
    pin.forwardKinematics(pin_model.model, pin_model.data, q_full)
    pin.updateFramePlacements(pin_model.model, pin_model.data)
    T0 = pin_model.data.oMf[target_frame_id]
    R_j7_des = np.array(T0.rotation, dtype=float)

    # Task-space path planning: straight-line waypoints from current to target.
    p_start = _link7_world_pos(q_cmd7)
    n_wp = max(2, int(PATH_PLAN_WAYPOINTS))
    waypoints = [
        (1.0 - a) * p_start + a * p_j7_des
        for a in np.linspace(0.0, 1.0, n_wp)
    ]
    v_start = np.zeros(3, dtype=float)
    v_end = np.asarray(v_j7_des, dtype=float).reshape(3)
    v_wp = [
        (1.0 - a) * v_start + a * v_end
        for a in np.linspace(0.0, 1.0, n_wp)
    ]

    # Build segmented joint-space trajectory via IK + poly5 per segment.
    dt_waypoint = DT_CONTROL / float(WAYPOINT_DENSITY)
    t = []
    q = []
    qdot = []
    qddot = []
    u = []
    t_offset = 0.0
    q_seg_start = q_cmd7.copy()
    qdot_seg_start = qdot7.copy()

    for i in range(n_wp - 1):
        p_next = waypoints[i + 1]
        v_next = v_wp[i + 1]
        q_goal, qdot_goal = solve_ik_for_q_goal(
            pin_model,
            target_frame_id,
            q_seg_start,
            p_next,
            R_j7_des,
            Q_LIMITS_7,
            QDOT_LIMITS_7,
            QDDOT_LIMITS_7,
            control_dt=DT_CONTROL,
            max_iter=300,
            kp_pos=2.0,
            kp_rot=1.0,
            v_j7_des=v_next,
        )

        t_seg, q_seg, qdot_seg, qddot_seg, u_seg = poly5_trajectory(
            q_seg_start,
            q_goal,
            Q_LIMITS_7,
            QDOT_LIMITS_7,
            QDDOT_LIMITS_7,
            control_dt=dt_waypoint,
            qdot_start=qdot_seg_start,
            qdot_goal=qdot_goal,
        )

        start_idx = 1 if i > 0 else 0
        for k in range(start_idx, len(t_seg)):
            t.append(float(t_seg[k] + t_offset))
            q.append(q_seg[k].copy())
            qdot.append(qdot_seg[k].copy())
            qddot.append(qddot_seg[k].copy())
            u.append(u_seg[k].copy())

        t_offset = float(t[-1]) if t else t_offset
        q_seg_start = q_goal.copy()
        qdot_seg_start = qdot_goal.copy()

    trace_t = []
    trace_q = []
    trace_qdot = []
    trace_qddot = []

    # Execute trajectory
    for k in range(len(t)):
        apply_arm_targets(stage, joint_paths, q[k], qdot[k])
        _append_trace_sample(trace_t, trace_q, trace_qdot, trace_qddot, t[k], q[k], qdot[k], qddot[k])
        for _ in range(SIM_UPDATES_PER_STEP):
            simulation_app.update()

    # Hold after execution, but do not record it
    n_hold = max(1, int(np.ceil(1.0 / DT_CONTROL)))
    for _ in range(n_hold):
        apply_arm_targets(stage, joint_paths, q[-1], np.zeros(7, dtype=float))
        simulation_app.update()

    # Save CSV for plotting
    out_dir = (Path(__file__).resolve().parents[1] / "outputs").as_posix()
    csv_path = Path(out_dir) / "joint_traces.csv"
    header = ["t"] + [f"q{j}" for j in range(7)] + [f"dq{j}" for j in range(7)] + [f"ddq{j}" for j in range(7)]
    data = np.column_stack([
        np.asarray(trace_t, dtype=float),
        np.asarray(trace_q, dtype=float),
        np.asarray(trace_qdot, dtype=float),
        np.asarray(trace_qddot, dtype=float),
    ])
    np.savetxt(str(csv_path), data, delimiter=",", header=",".join(header), comments="")

    # Plot
    plot_from_csv(str(csv_path), out_dir)

    # Keep sim running
    print("[done] trajectory executed, simulation left running.", flush=True)
    while True:
        simulation_app.update()


if __name__ == "__main__":
    main()
