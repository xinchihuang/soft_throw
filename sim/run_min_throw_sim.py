#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Simulation runner: compute joint trajectory (IK + poly5) and execute.

Inputs:
- desired joint7 target pose/velocity (x,y,z,vx,vy,vz) in world, OR
- release position + landing target position (ballistic solve).
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
    parser.add_argument(
        "--hold_only",
        action="store_true",
        help="Do not move arm; only reset/hold and print joint7 pose",
    )
    parser.add_argument(
        "--print_only",
        action="store_true",
        help="Do not launch sim; only write trajectory.csv with tau",
    )
    parser.add_argument(
        "--ik_max_iter",
        type=int,
        default=300,
        help="Max IK iterations per waypoint (print_only only)",
    )
    parser.add_argument(
        "--poly5_max_samples",
        type=int,
        default=20000,
        help="Max samples per poly5 segment (print_only only)",
    )
    args = parser.parse_args()

    if args.print_only and args.hold_only:
        raise RuntimeError("--print_only cannot be combined with --hold_only")

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
    from core.kinematic_forward import PinKinematics
    from core.trajectory_planner import (
        build_execution_trajectory,
        build_reset_trajectory,
        compute_reset_q7,
        solve_release_velocity,
        write_trace_csv,
        write_trace_with_tau_csv,
    )

    if args.print_only:
        pin_model = PinKinematics()
        q_cmd7 = INIT_ARM.copy()
        if RESET_ARM_POS_WORLD is not None:
            q_cmd7 = compute_reset_q7(
                pin_model,
                q_cmd7,
                RESET_ARM_POS_WORLD,
                Q_LIMITS_7,
                QDOT_LIMITS_7,
                QDDOT_LIMITS_7,
                control_dt=DT_CONTROL,
                target_frame="panda_link7",
                max_iter=int(args.ik_max_iter),
            )

        if args.release_pos is not None:
            if args.target_pos is None:
                raise RuntimeError("--target_pos required when using --release_pos")
            p_j7_des = np.asarray(args.release_pos, dtype=float).reshape(3)
            target_pos = np.asarray(args.target_pos, dtype=float).reshape(3)
            v_j7_des, T_ball = solve_release_velocity(p_j7_des, target_pos)
            print(f"[ballistic] T={T_ball:.3f} v={v_j7_des.tolist()}", flush=True)
            target_pose_vel = np.concatenate([p_j7_des, v_j7_des], axis=0)
        else:
            target_pose_vel = np.asarray(args.pose_j7_vel, dtype=float)

        print("[print_only] planning execution trajectory...", flush=True)
        t_exec, q_exec, qdot_exec, qddot_exec, _u_exec = build_execution_trajectory(
            target_pose_vel=target_pose_vel,
            start_q7=q_cmd7,
            q_limits=Q_LIMITS_7,
            qdot_limits=QDOT_LIMITS_7,
            qddot_limits=QDDOT_LIMITS_7,
            control_dt=DT_CONTROL,
            waypoint_density=WAYPOINT_DENSITY,
            path_plan_waypoints=PATH_PLAN_WAYPOINTS,
            target_frame="panda_link7",
            pin_model=pin_model,
            qdot_start=np.zeros(7, dtype=float),
            joint_limit_margin=0.0,
            ik_max_iter=int(args.ik_max_iter),
            poly5_max_samples=int(args.poly5_max_samples) if args.poly5_max_samples > 0 else None,
        )

        csv_path = _ROOT / "trajectory.csv"
        print("[print_only] writing trajectory.csv (throw segment only)...", flush=True)
        write_trace_with_tau_csv(str(csv_path), t_exec, q_exec, qdot_exec, qddot_exec)
        print(f"[print_only] wrote {csv_path}", flush=True)
        return

    from isaaclab.app import AppLauncher
    from sim.plot_joint_traces import plot_from_csv

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
        reset_ball,
    )

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
    repair_lmm_asset_references(stage, ROBOT_PRIM)

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
        q_cmd7 = compute_reset_q7(
            pin_model,
            q_cmd7,
            RESET_ARM_POS_WORLD,
            Q_LIMITS_7,
            QDOT_LIMITS_7,
            QDDOT_LIMITS_7,
            control_dt=DT_CONTROL,
            target_frame="panda_link7",
        )

    reset_points, q_cmd7, split_idx = build_reset_trajectory(
        start_q7=q_cmd7,
        reset_q7=q_cmd7,
        control_dt=DT_CONTROL,
        waypoint_density=1,
        settle_sec=RESET_ARM_SETTLE_SEC,
        extra_hold_sec=RESET_BALL_WAIT_SEC,
        q_limits=Q_LIMITS_7,
        qdot_limits=QDOT_LIMITS_7,
        limit_scale=1.0,
        joint_limit_margin=0.0,
        smoothing_window=0,
    )
    for positions, velocities, _acc, _t in reset_points[:split_idx]:
        apply_arm_targets(stage, joint_paths, positions, velocities)
        simulation_app.update()

    # Ball reset (kept for scene consistency)
    reset_ball(stage, BALL_RESET_POS_WORLD)
    for positions, velocities, _acc, _t in reset_points[split_idx:]:
        apply_arm_targets(stage, joint_paths, positions, velocities)
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

    # Parse target
    if args.release_pos is not None:
        if args.target_pos is None:
            raise RuntimeError("--target_pos required when using --release_pos")
        p_j7_des = np.asarray(args.release_pos, dtype=float).reshape(3)
        target_pos = np.asarray(args.target_pos, dtype=float).reshape(3)
        v_j7_des, T_ball = solve_release_velocity(p_j7_des, target_pos)
        print(f"[ballistic] T={T_ball:.3f} v={v_j7_des.tolist()}", flush=True)
        target_pose_vel = np.concatenate([p_j7_des, v_j7_des], axis=0)
    else:
        target_pose_vel = np.asarray(args.pose_j7_vel, dtype=float)

    t, q, qdot, qddot, _u = build_execution_trajectory(
        target_pose_vel=target_pose_vel,
        start_q7=q_cmd7,
        q_limits=Q_LIMITS_7,
        qdot_limits=QDOT_LIMITS_7,
        qddot_limits=QDDOT_LIMITS_7,
        control_dt=DT_CONTROL,
        waypoint_density=WAYPOINT_DENSITY,
        path_plan_waypoints=PATH_PLAN_WAYPOINTS,
        target_frame="panda_link7",
        pin_model=pin_model,
        qdot_start=np.zeros(7, dtype=float),
        joint_limit_margin=0.0,
    )

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
    write_trace_with_tau_csv(str(csv_path), trace_t, trace_q, trace_qdot, trace_qddot)

    # Plot
    plot_from_csv(str(csv_path), out_dir)

    # Keep sim running
    print("[done] trajectory executed, simulation left running.", flush=True)
    while True:
        simulation_app.update()


if __name__ == "__main__":
    main()
