#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Move the real Panda so joint7 follows a target pose/velocity profile.

This mirrors the motion logic in sim/run_min_throw_sim.py, but targets
the joint7 frame directly instead of a lacrosse attachment.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Ensure repo root is on sys.path when running from real/
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.throw_params import (
    DT_CONTROL,
    WAYPOINT_DENSITY,
    RESET_ARM_SETTLE_SEC,
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
from sim.plot_joint_traces import plot_from_csv


JOINT_NAMES = [
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
]

DEFAULT_ACTION_SERVER = "/position_joint_trajectory_controller/follow_joint_trajectory"
LIMIT_SCALE = 0.1
JOINT_LIMIT_MARGIN = 0.2
SMOOTHING_WINDOW = 9


def _save_csv_and_plot(t, q, qdot, qddot, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "joint_traces.csv"
    write_trace_with_tau_csv(str(csv_path), t, q, qdot, qddot)
    plot_from_csv(str(csv_path), str(output_dir))


def _send_ros_trajectory(points, action_server: str, start_delay: float):
    import actionlib
    import rospy
    from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

    client = actionlib.SimpleActionClient(action_server, FollowJointTrajectoryAction)
    rospy.loginfo("Waiting for action server %s", action_server)
    client.wait_for_server()
    rospy.loginfo("Connected to trajectory action server")

    traj = JointTrajectory()
    traj.joint_names = JOINT_NAMES
    traj.header.stamp = rospy.Time.now() + rospy.Duration.from_sec(float(start_delay))

    for positions, velocities, accelerations, t in points:
        pt = JointTrajectoryPoint()
        pt.positions = positions.tolist()
        pt.velocities = []
        pt.accelerations = []
        pt.time_from_start = rospy.Duration.from_sec(float(t))
        traj.points.append(pt)

    goal = FollowJointTrajectoryGoal()
    goal.trajectory = traj

    rospy.loginfo("Sending %d trajectory points", len(traj.points))
    client.send_goal(goal)
    client.wait_for_result()
    result = client.get_result()
    rospy.loginfo("Trajectory finished with error_code=%s", getattr(result, "error_code", None))
    return result


def _read_current_joint_pos7() -> np.ndarray:
    import rospy
    from sensor_msgs.msg import JointState

    msg = rospy.wait_for_message("/joint_states", JointState, timeout=2.0)
    name_to_idx = {name: idx for idx, name in enumerate(msg.name)}
    q7 = np.zeros(7, dtype=float)
    for i, joint_name in enumerate(JOINT_NAMES):
        if joint_name not in name_to_idx:
            raise RuntimeError(f"Joint {joint_name} not found in /joint_states")
        q7[i] = float(msg.position[name_to_idx[joint_name]])
    return q7


def _resolve_args():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--pose_joint7_vel",
        type=float,
        nargs=6,
        help="Target joint7 pose and linear velocity: x y z vx vy vz (world)",
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
    parser.add_argument("--action-server", default=DEFAULT_ACTION_SERVER)
    parser.add_argument("--start-delay", type=float, default=0.2)
    parser.add_argument("--hold-sec", type=float, default=1.0)
    # IK parameters removed (handled inside core.ik_poly5_core)
    # IK tolerances removed (handled inside core.ik_poly5_core)
    parser.add_argument(
        "--plot-dir",
        default=str(_ROOT),
        help="Directory for saved velocity/acceleration plots",
    )
    parser.add_argument("--print-only", action="store_true", help="Build trajectory only; do not send to ROS")

    try:
        import rospy

        argv = rospy.myargv(argv=sys.argv)
        return parser.parse_args(argv[1:])
    except ImportError:
        return parser.parse_args()


def main():
    args = _resolve_args()
    start_q7 = INIT_ARM.copy()

    if not args.print_only:
        import rospy

        rospy.init_node("joint7_pose_sender", anonymous=True)
        start_q7 = _read_current_joint_pos7()

    pin_model = PinKinematics()
    reset_q7 = INIT_ARM.copy()
    if RESET_ARM_POS_WORLD is not None:
        reset_q7 = compute_reset_q7(
            pin_model,
            reset_q7,
            RESET_ARM_POS_WORLD,
            Q_LIMITS_7,
            QDOT_LIMITS_7,
            QDDOT_LIMITS_7,
            control_dt=DT_CONTROL,
            target_frame="panda_link7",
        )
    reset_points, q_reset, _split_idx = build_reset_trajectory(
        start_q7=start_q7,
        reset_q7=reset_q7,
        control_dt=DT_CONTROL,
        waypoint_density=WAYPOINT_DENSITY,
        settle_sec=RESET_ARM_SETTLE_SEC,
        extra_hold_sec=RESET_BALL_WAIT_SEC,
        q_limits=Q_LIMITS_7,
        qdot_limits=QDOT_LIMITS_7,
        limit_scale=LIMIT_SCALE,
        joint_limit_margin=JOINT_LIMIT_MARGIN,
        smoothing_window=SMOOTHING_WINDOW,
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
        target_pose_vel = np.asarray(args.pose_joint7_vel, dtype=float)

    t_exec, q_exec, qdot_exec, qddot_exec, _u_exec = build_execution_trajectory(
        target_pose_vel=target_pose_vel,
        start_q7=q_reset,
        q_limits=Q_LIMITS_7,
        qdot_limits=QDOT_LIMITS_7,
        qddot_limits=QDDOT_LIMITS_7,
        control_dt=DT_CONTROL,
        waypoint_density=WAYPOINT_DENSITY,
        path_plan_waypoints=PATH_PLAN_WAYPOINTS,
        target_frame="panda_link7",
        pin_model=pin_model,
        qdot_start=np.zeros(7, dtype=float),
        joint_limit_margin=JOINT_LIMIT_MARGIN,
    )
    if float(args.hold_sec) > 0.0:
        dt_waypoint = DT_CONTROL / float(WAYPOINT_DENSITY)
        n_hold = max(1, int(np.ceil(float(args.hold_sec) / dt_waypoint)))
        t_hold = t_exec[-1] + np.arange(1, n_hold + 1) * dt_waypoint
        q_hold = np.repeat(q_exec[-1][None, :], n_hold, axis=0)
        qdot_hold = np.zeros_like(q_hold)
        qddot_hold = np.zeros_like(q_hold)
        t_exec = np.concatenate([t_exec, t_hold])
        q_exec = np.concatenate([q_exec, q_hold])
        qdot_exec = np.concatenate([qdot_exec, qdot_hold])
        qddot_exec = np.concatenate([qddot_exec, qddot_hold])
    # Build points list for ROS
    points = reset_points.copy()
    t_offset = float(reset_points[-1][3]) if reset_points else 0.0
    for i in range(len(t_exec)):
        points.append((q_exec[i], qdot_exec[i], qddot_exec[i], float(t_exec[i] + t_offset)))

    print(f"[plan] built {len(points)} trajectory points", flush=True)
    print(f"[plan] final_q_cmd7={q_exec[-1].tolist()}", flush=True)
    _save_csv_and_plot(t_exec, q_exec, qdot_exec, qddot_exec, Path(args.plot_dir))

    if args.print_only:
        tau_path = Path(args.plot_dir) / "trajectory.csv"
        write_trace_with_tau_csv(str(tau_path), t_exec, q_exec, qdot_exec, qddot_exec)
        return

    exec_points = [(q_exec[i], qdot_exec[i], qddot_exec[i], float(t_exec[i])) for i in range(len(t_exec))]
    _send_ros_trajectory(
        points=reset_points,
        action_server=args.action_server,
        start_delay=float(args.start_delay),
    )
    _send_ros_trajectory(
        points=exec_points,
        action_server=args.action_server,
        start_delay=float(args.start_delay),
    )


if __name__ == "__main__":
    main()
