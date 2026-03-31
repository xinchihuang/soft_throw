#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Move the real Panda so joint7 follows a target pose/velocity profile.

This mirrors the motion logic in sim/run_min_throw_sim.py, but targets
the joint7 frame directly instead of a lacrosse attachment.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

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
    QDOT_LIMITS_7,
    QDDOT_LIMITS_7,
    Q_LIMITS_7,
)
from core.kinematics_pin import PinKinematics, pin
from core.ik_poly5_core import solve_ik_for_q_goal, poly5_trajectory
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
EE_SPEED_LIMIT = 0.2
JOINT_LIMIT_MARGIN = 0.2
SMOOTHING_WINDOW = 9


def _effective_q_limits() -> Tuple[np.ndarray, np.ndarray]:
    q_min = Q_LIMITS_7[:, 0] + JOINT_LIMIT_MARGIN
    q_max = Q_LIMITS_7[:, 1] - JOINT_LIMIT_MARGIN
    return q_min, q_max


def _save_csv_and_plot(t, q, qdot, qddot, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "joint_traces.csv"
    header = ["t"] + [f"q{j}" for j in range(7)] + [f"dq{j}" for j in range(7)] + [f"ddq{j}" for j in range(7)]
    data = np.column_stack([t, q, qdot, qddot])
    np.savetxt(csv_path, data, delimiter=",", header=",".join(header), comments="")
    plot_from_csv(str(csv_path), str(output_dir))


def _points_to_arrays(points):
    t = np.asarray([pt[3] for pt in points], dtype=float)
    q = np.asarray([pt[0] for pt in points], dtype=float)
    qdot = np.asarray([pt[1] for pt in points], dtype=float)
    qddot = np.asarray([pt[2] for pt in points], dtype=float)
    return t, q, qdot, qddot


def _append_point(points, positions, velocities, time_from_start: float) -> None:
    points.append(
        (
            np.asarray(positions, dtype=float).copy(),
            np.asarray(velocities, dtype=float).copy(),
            np.zeros(7, dtype=float),
            float(time_from_start),
        )
    )


def _append_hold(points, positions, duration_sec: float, time_from_start: float) -> float:
    dt_waypoint = DT_CONTROL / float(WAYPOINT_DENSITY)
    n_steps = max(1, int(np.ceil(float(duration_sec) / dt_waypoint)))
    for _ in range(n_steps):
        time_from_start += dt_waypoint
        _append_point(points, positions, np.zeros(7, dtype=float), time_from_start)
    return time_from_start


def _postprocess_trajectory(points):
    if not points:
        return points

    q = np.asarray([pt[0] for pt in points], dtype=float)
    t = np.asarray([pt[3] for pt in points], dtype=float)
    n = len(points)

    if n >= SMOOTHING_WINDOW:
        pad = SMOOTHING_WINDOW // 2
        kernel = np.ones(SMOOTHING_WINDOW, dtype=float) / float(SMOOTHING_WINDOW)
        q_smooth = q.copy()
        for j in range(q.shape[1]):
            series = np.pad(q[:, j], (pad, pad), mode="edge")
            q_smooth[:, j] = np.convolve(series, kernel, mode="valid")
        q = q_smooth
        q_min, q_max = _effective_q_limits()
        q = np.clip(q, q_min, q_max)

    qdot = np.zeros_like(q)
    if n >= 2:
        dt_f = np.diff(t)
        dq_f = np.diff(q, axis=0)
        valid_f = dt_f > 1e-9
        qdot[:-1][valid_f] = dq_f[valid_f] / dt_f[valid_f, None]
        qdot[-1] = np.zeros(7, dtype=float)

    qddot = np.zeros_like(q)
    if n >= 2:
        dt_v = np.diff(t)
        dqdot = np.diff(qdot, axis=0)
        valid_v = dt_v > 1e-9
        qddot[:-1][valid_v] = dqdot[valid_v] / dt_v[valid_v, None]
        qddot[-1] = np.zeros(7, dtype=float)

    out = []
    for i in range(n):
        out.append((q[i].copy(), qdot[i].copy(), qddot[i].copy(), float(t[i])))
    return out


def _append_interp(points, q_from: np.ndarray, q_to: np.ndarray, time_from_start: float) -> Tuple[np.ndarray, float]:
    q_from = np.asarray(q_from, dtype=float).copy()
    q_to = np.asarray(q_to, dtype=float).copy()

    qdot_limits = LIMIT_SCALE * QDOT_LIMITS_7
    dt_waypoint = DT_CONTROL / float(WAYPOINT_DENSITY)
    dq = np.abs(q_to - q_from)
    step_counts = np.ceil(dq / np.maximum(1e-9, qdot_limits * dt_waypoint)).astype(int)
    n_steps = max(1, int(np.max(step_counts)))

    for k in range(n_steps):
        u = float(k + 1) / float(n_steps)
        alpha = u * u * (3.0 - 2.0 * u)
        q = (1.0 - alpha) * q_from + alpha * q_to
        q_prev = q_from if k == 0 else points[-1][0]
        qdot = (q - q_prev) / dt_waypoint
        time_from_start += dt_waypoint
        _append_point(points, q, qdot, time_from_start)

    return q_to.copy(), time_from_start


def _safe_step_velocity(q_cmd7: np.ndarray, qdot_des: np.ndarray, qdot_prev: np.ndarray) -> np.ndarray:
    """Apply acceleration, velocity, and one-step position limits."""
    qdot_limits = LIMIT_SCALE * QDOT_LIMITS_7
    qddot_limits = LIMIT_SCALE * QDDOT_LIMITS_7
    q_min, q_max = _effective_q_limits()

    max_dq = qddot_limits * DT_CONTROL
    dq = np.clip(qdot_des - qdot_prev, -max_dq, max_dq)
    qdot7 = qdot_prev + dq
    qdot7 = np.clip(qdot7, -qdot_limits, qdot_limits)

    qdot_min_from_pos = (q_min - q_cmd7) / DT_CONTROL
    qdot_max_from_pos = (q_max - q_cmd7) / DT_CONTROL
    qdot7 = np.clip(qdot7, qdot_min_from_pos, qdot_max_from_pos)
    return qdot7


def _build_jacobian_arm7(pin_model: PinKinematics, q_cmd7: np.ndarray, frame_id: int) -> np.ndarray:
    q_full = pin_model.make_q_full_from_arm7(q_cmd7)
    pin.forwardKinematics(pin_model.model, pin_model.data, q_full)
    pin.updateFramePlacements(pin_model.model, pin_model.data)
    J = pin.computeFrameJacobian(
        pin_model.model, pin_model.data, q_full, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
    )
    J_arm = np.zeros((6, 7), dtype=float)
    for j in range(1, 8):
        jid = pin_model.model.getJointId(f"panda_joint{j}")
        idx_v = int(pin_model.model.joints[jid].idx_v)
        J_arm[:, j - 1] = J[:, idx_v]
    return J_arm


def _clip_cartesian_speed(pin_model: PinKinematics, q_cmd7: np.ndarray, qdot7: np.ndarray) -> np.ndarray:
    ee_speed = pin_model.ee_speed_from_qdot7(q_cmd7, qdot7)
    if ee_speed <= EE_SPEED_LIMIT or ee_speed <= 1e-9:
        return qdot7
    return qdot7 * (EE_SPEED_LIMIT / ee_speed)


def _append_dense_segment(
    points,
    q_start: np.ndarray,
    q_end: np.ndarray,
    qdot_start: np.ndarray,
    qdot_end: np.ndarray,
    time_from_start: float,
) -> float:
    q_start = np.asarray(q_start, dtype=float)
    q_end = np.asarray(q_end, dtype=float)
    qdot_start = np.asarray(qdot_start, dtype=float)
    qdot_end = np.asarray(qdot_end, dtype=float)

    seg_dt = DT_CONTROL
    dt_waypoint = seg_dt / float(WAYPOINT_DENSITY)
    for sub in range(WAYPOINT_DENSITY):
        s = float(sub + 1) / float(WAYPOINT_DENSITY)
        h00 = 2.0 * s**3 - 3.0 * s**2 + 1.0
        h10 = s**3 - 2.0 * s**2 + s
        h01 = -2.0 * s**3 + 3.0 * s**2
        h11 = s**3 - s**2
        dh00 = 6.0 * s**2 - 6.0 * s
        dh10 = 3.0 * s**2 - 4.0 * s + 1.0
        dh01 = -6.0 * s**2 + 6.0 * s
        dh11 = 3.0 * s**2 - 2.0 * s
        q = h00 * q_start + h10 * seg_dt * qdot_start + h01 * q_end + h11 * seg_dt * qdot_end
        qdot = (
            dh00 * q_start / seg_dt
            + dh10 * qdot_start
            + dh01 * q_end / seg_dt
            + dh11 * qdot_end
        )
        time_from_start += dt_waypoint
        _append_point(points, q, qdot, time_from_start)
    return time_from_start


def _build_reset_trajectory(start_q7: np.ndarray) -> Tuple[List[Tuple[np.ndarray, np.ndarray, np.ndarray, float]], np.ndarray]:
    q_min, q_max = _effective_q_limits()
    q_init = np.clip(INIT_ARM.copy(), q_min, q_max)
    q_cmd7 = np.clip(np.asarray(start_q7, dtype=float).reshape(7), q_min, q_max)
    points: List[Tuple[np.ndarray, np.ndarray, np.ndarray, float]] = []
    time_from_start = 0.0
    q_cmd7, time_from_start = _append_interp(points, q_cmd7, q_init, time_from_start)
    time_from_start = _append_hold(points, q_cmd7, RESET_ARM_SETTLE_SEC, time_from_start)
    time_from_start = _append_hold(points, q_cmd7, RESET_BALL_WAIT_SEC, time_from_start)
    return _postprocess_trajectory(points), q_cmd7


def _build_execution_trajectory(
    target_pose_vel: np.ndarray,
    start_q7: np.ndarray,
    hold_sec: float,
    control_dt: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pin_model = PinKinematics()
    target_frame = "panda_link7"
    target_frame_id = pin_model.model.getFrameId(target_frame)
    if target_frame_id == len(pin_model.model.frames):
        raise RuntimeError(f"[pin] target frame not found: {target_frame}")

    pose = np.asarray(target_pose_vel, dtype=float).reshape(6)
    p_j7_des = pose[:3]
    v_j7_des = pose[3:]

    q_min, q_max = _effective_q_limits()
    q_start = np.clip(np.asarray(start_q7, dtype=float).reshape(7), q_min, q_max)

    q_full = pin_model.make_q_full_from_arm7(q_start)
    pin.forwardKinematics(pin_model.model, pin_model.data, q_full)
    pin.updateFramePlacements(pin_model.model, pin_model.data)
    T0 = pin_model.data.oMf[target_frame_id]
    R_j7_des = np.array(T0.rotation, dtype=float)

    q_goal = solve_ik_for_q_goal(
        pin_model,
        target_frame_id,
        q_start,
        p_j7_des,
        R_j7_des,
        Q_LIMITS_7,
        QDOT_LIMITS_7,
        QDDOT_LIMITS_7,
        control_dt=control_dt,
        max_iter=300,
        kp_pos=2.0,
        kp_rot=1.0,
        v_j7_des=v_j7_des,
    )

    t, q, qdot, qddot, u = poly5_trajectory(
        q_start,
        q_goal,
        Q_LIMITS_7,
        QDOT_LIMITS_7,
        QDDOT_LIMITS_7,
        control_dt=control_dt,
    )

    # append hold segment
    if hold_sec > 0:
        n_hold = max(1, int(np.ceil(hold_sec / control_dt)))
        t_hold = t[-1] + np.arange(1, n_hold + 1) * control_dt
        q_hold = np.repeat(q[-1][None, :], n_hold, axis=0)
        qdot_hold = np.zeros_like(q_hold)
        qddot_hold = np.zeros_like(q_hold)
        u_hold = np.zeros_like(q_hold)
        t = np.concatenate([t, t_hold])
        q = np.concatenate([q, q_hold])
        qdot = np.concatenate([qdot, qdot_hold])
        qddot = np.concatenate([qddot, qddot_hold])
        u = np.concatenate([u, u_hold])

    return t, q, qdot, qddot, u


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
    parser.add_argument(
        "--pose_joint7_vel",
        type=float,
        nargs=6,
        required=True,
        help="Target joint7 pose and linear velocity: x y z vx vy vz (world)",
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

    reset_points, q_reset = _build_reset_trajectory(start_q7=start_q7)
    t_exec, q_exec, qdot_exec, qddot_exec, u_exec = _build_execution_trajectory(
        target_pose_vel=np.asarray(args.pose_joint7_vel, dtype=float),
        start_q7=q_reset,
        hold_sec=float(args.hold_sec),
        control_dt=DT_CONTROL,
    )
    # Build points list for ROS
    points = reset_points.copy()
    t_offset = float(reset_points[-1][3]) if reset_points else 0.0
    for i in range(len(t_exec)):
        points.append((q_exec[i], qdot_exec[i], qddot_exec[i], float(t_exec[i] + t_offset)))

    print(f"[plan] built {len(points)} trajectory points", flush=True)
    print(f"[plan] final_q_cmd7={q_exec[-1].tolist()}", flush=True)
    _save_csv_and_plot(t_exec, q_exec, qdot_exec, qddot_exec, Path(args.plot_dir))

    if args.print_only:
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
