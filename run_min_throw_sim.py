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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pose_j7_vel", type=float, nargs=6, required=True,
                        help="Target joint7: x y z vx vy vz (world)")
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
        SIM_UPDATES_PER_STEP,
        RESET_ARM_SETTLE_SEC,
        BALL_RESET_POS_WORLD,
        RESET_BALL_WAIT_SEC,
        INIT_ARM,
        QDOT_LIMITS_7,
        QDDOT_LIMITS_7,
        Q_LIMITS_7,
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
    from core.ik_poly5_core import solve_ik_for_q_goal, poly5_trajectory
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

    # Reset arm
    q_cmd7 = INIT_ARM.copy()
    qdot7 = np.zeros(7, dtype=float)
    t0 = time.time()
    while time.time() - t0 < RESET_ARM_SETTLE_SEC:
        apply_arm_targets(stage, joint_paths, q_cmd7, qdot7)
        simulation_app.update()

    # Ball reset (kept for scene consistency)
    reset_ball(stage, BALL_RESET_POS_WORLD)
    t1 = time.time()
    while time.time() - t1 < RESET_BALL_WAIT_SEC:
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

    # Parse target
    pose = np.asarray(args.pose_j7_vel, dtype=float).reshape(6)
    p_j7_des = pose[:3]
    v_j7_des = pose[3:]

    pin_model = PinKinematics()
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

    # IK solve (no actuation)
    q_goal = solve_ik_for_q_goal(
        pin_model,
        target_frame_id,
        q_cmd7,
        p_j7_des,
        R_j7_des,
        Q_LIMITS_7,
        QDOT_LIMITS_7,
        QDDOT_LIMITS_7,
        control_dt=0.01,
        max_iter=300,
        kp_pos=2.0,
        kp_rot=1.0,
        v_j7_des=v_j7_des,
    )

    # Poly5 trajectory
    t, q, qdot, qddot, u = poly5_trajectory(
        q_cmd7,
        q_goal,
        Q_LIMITS_7,
        QDOT_LIMITS_7,
        QDDOT_LIMITS_7,
        control_dt=0.01,
    )

    # Execute trajectory
    for k in range(len(t)):
        apply_arm_targets(stage, joint_paths, q[k], qdot[k])
        for _ in range(SIM_UPDATES_PER_STEP):
            simulation_app.update()

    # Hold
    t2 = time.time()
    while time.time() - t2 < 1.0:
        apply_arm_targets(stage, joint_paths, q[-1], np.zeros(7, dtype=float))
        simulation_app.update()

    # Save CSV for plotting
    out_dir = (Path(__file__).resolve().parents[1] / "outputs").as_posix()
    csv_path = Path(out_dir) / "joint_traces.csv"
    header = ["t"] + [f"q{j+1}" for j in range(7)] + [f"qdot{j+1}" for j in range(7)] + [f"u{j+1}" for j in range(7)]
    data = np.column_stack([t, q, qdot, u])
    np.savetxt(str(csv_path), data, delimiter=",", header=",".join(header), comments="")

    # Plot
    plot_from_csv(str(csv_path), out_dir)

    # Keep sim running
    print("[done] trajectory executed, simulation left running.", flush=True)
    while True:
        simulation_app.update()


if __name__ == "__main__":
    main()
