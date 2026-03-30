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
    RESET_ARM_SETTLE_SEC,
    RESET_BALL_WAIT_SEC,
    INIT_ARM,
    QDOT_LIMITS_7,
    QDDOT_LIMITS_7,
    Q_LIMITS_7,
)
from core.kinematics_pin import PinKinematics, pin


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
WAYPOINT_DENSITY = 8
JOINT_LIMIT_MARGIN = 0.2
SMOOTHING_WINDOW = 9


def _effective_q_limits() -> Tuple[np.ndarray, np.ndarray]:
    q_min = Q_LIMITS_7[:, 0] + JOINT_LIMIT_MARGIN
    q_max = Q_LIMITS_7[:, 1] - JOINT_LIMIT_MARGIN
    return q_min, q_max


def _plot_joint_commands(points, output_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib not available; skipping plots", flush=True)
        return

    if not points:
        print("[plot] no trajectory points available; skipping plots", flush=True)
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    t = np.asarray([pt[3] for pt in points], dtype=float)
    q = np.asarray([pt[0] for pt in points], dtype=float)
    qdot = np.asarray([pt[1] for pt in points], dtype=float)
    qddot = np.asarray([pt[2] for pt in points], dtype=float)

    fig_pos, axes_pos = plt.subplots(7, 1, figsize=(12, 16), sharex=True)
    for idx, ax in enumerate(np.atleast_1d(axes_pos)):
        ax.plot(t, q[:, idx], linewidth=1.5)
        ax.axhline(Q_LIMITS_7[idx, 0], linestyle="--", linewidth=1.0, color="tab:red")
        ax.axhline(Q_LIMITS_7[idx, 1], linestyle="--", linewidth=1.0, color="tab:red")
        ax.set_ylabel(f"q{idx+1} [rad]")
        ax.set_title(JOINT_NAMES[idx], loc="left")
        ax.grid(True, alpha=0.3)
    axes_pos[-1].set_xlabel("Trajectory time_from_start [s]")
    fig_pos.suptitle("Commanded Joint Positions")
    fig_pos.tight_layout()
    pos_path = output_dir / "real_joint_positions.png"
    fig_pos.savefig(pos_path, dpi=150)

    fig_vel, axes_vel = plt.subplots(7, 1, figsize=(12, 16), sharex=True)
    for idx, ax in enumerate(np.atleast_1d(axes_vel)):
        ax.plot(t, qdot[:, idx], linewidth=1.5)
        ax.axhline(LIMIT_SCALE * QDOT_LIMITS_7[idx], linestyle="--", linewidth=1.0, color="tab:red")
        ax.axhline(-(LIMIT_SCALE * QDOT_LIMITS_7[idx]), linestyle="--", linewidth=1.0, color="tab:red")
        ax.set_ylabel(f"dq{idx+1} [rad/s]")
        ax.set_title(JOINT_NAMES[idx], loc="left")
        ax.grid(True, alpha=0.3)
    axes_vel[-1].set_xlabel("Trajectory time_from_start [s]")
    fig_vel.suptitle("Commanded Joint Velocities")
    fig_vel.tight_layout()
    vel_path = output_dir / "real_joint_velocities.png"
    fig_vel.savefig(vel_path, dpi=150)

    fig_vel_scatter, axes_vel_scatter = plt.subplots(7, 1, figsize=(12, 16), sharex=True)
    for idx, ax in enumerate(np.atleast_1d(axes_vel_scatter)):
        ax.scatter(t, qdot[:, idx], s=8)
        ax.axhline(LIMIT_SCALE * QDOT_LIMITS_7[idx], linestyle="--", linewidth=1.0, color="tab:red")
        ax.axhline(-(LIMIT_SCALE * QDOT_LIMITS_7[idx]), linestyle="--", linewidth=1.0, color="tab:red")
        ax.set_ylabel(f"dq{idx+1} [rad/s]")
        ax.set_title(JOINT_NAMES[idx], loc="left")
        ax.grid(True, alpha=0.3)
    axes_vel_scatter[-1].set_xlabel("Trajectory time_from_start [s]")
    fig_vel_scatter.suptitle("Commanded Joint Velocities (Scatter)")
    fig_vel_scatter.tight_layout()
    vel_scatter_path = output_dir / "real_joint_velocities_scatter.png"
    fig_vel_scatter.savefig(vel_scatter_path, dpi=150)

    fig_acc, axes_acc = plt.subplots(7, 1, figsize=(12, 16), sharex=True)
    for idx, ax in enumerate(np.atleast_1d(axes_acc)):
        ax.plot(t, qddot[:, idx], linewidth=1.5)
        ax.axhline(LIMIT_SCALE * QDDOT_LIMITS_7[idx], linestyle="--", linewidth=1.0, color="tab:red")
        ax.axhline(-(LIMIT_SCALE * QDDOT_LIMITS_7[idx]), linestyle="--", linewidth=1.0, color="tab:red")
        ax.set_ylabel(f"ddq{idx+1} [rad/s^2]")
        ax.set_title(JOINT_NAMES[idx], loc="left")
        ax.grid(True, alpha=0.3)
    axes_acc[-1].set_xlabel("Trajectory time_from_start [s]")
    fig_acc.suptitle("Commanded Joint Accelerations")
    fig_acc.tight_layout()
    acc_path = output_dir / "real_joint_accelerations.png"
    fig_acc.savefig(acc_path, dpi=150)

    plt.close(fig_pos)
    plt.close(fig_vel)
    plt.close(fig_vel_scatter)
    plt.close(fig_acc)

    print(f"[plot] saved {pos_path}", flush=True)
    print(f"[plot] saved {vel_path}", flush=True)
    print(f"[plot] saved {vel_scatter_path}", flush=True)
    print(f"[plot] saved {acc_path}", flush=True)


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
    max_iter: int,
    kp_pos: float,
    kp_rot: float,
    ik_tol_pos: float,
    ik_tol_rot: float,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray, np.ndarray, float]], np.ndarray]:
    pin_model = PinKinematics()
    target_frame = "panda_link7"
    target_frame_id = pin_model.model.getFrameId(target_frame)
    if target_frame_id == len(pin_model.model.frames):
        raise RuntimeError(f"[pin] target frame not found: {target_frame}")

    pose = np.asarray(target_pose_vel, dtype=float).reshape(6)
    p_j7_des = pose[:3]
    v_j7_des = pose[3:]

    q_min, q_max = _effective_q_limits()
    q_cmd7 = np.clip(np.asarray(start_q7, dtype=float).reshape(7), q_min, q_max)
    qdot_prev = np.zeros(7, dtype=float)
    points: List[Tuple[np.ndarray, np.ndarray, np.ndarray, float]] = []
    time_from_start = 0.0

    q_full = pin_model.make_q_full_from_arm7(q_cmd7)
    pin.forwardKinematics(pin_model.model, pin_model.data, q_full)
    pin.updateFramePlacements(pin_model.model, pin_model.data)
    T0 = pin_model.data.oMf[target_frame_id]
    R_j7_des = np.array(T0.rotation, dtype=float)

    for _ in range(max_iter):
        q_full = pin_model.make_q_full_from_arm7(q_cmd7)
        pin.forwardKinematics(pin_model.model, pin_model.data, q_full)
        pin.updateFramePlacements(pin_model.model, pin_model.data)
        T = pin_model.data.oMf[target_frame_id]
        p_cur = np.array(T.translation, dtype=float)
        R_cur = np.array(T.rotation, dtype=float)

        ep = p_j7_des - p_cur
        eR = pin.log3(R_cur.T @ R_j7_des)
        if np.linalg.norm(ep) < ik_tol_pos and np.linalg.norm(eR) < ik_tol_rot:
            break

        J_arm = _build_jacobian_arm7(pin_model, q_cmd7, target_frame_id)
        lam = 0.05
        JJt = J_arm @ J_arm.T
        J_pinv = J_arm.T @ np.linalg.inv(JJt + (lam * lam) * np.eye(6))

        v_cmd = v_j7_des + kp_pos * ep
        w_cmd = kp_rot * eR
        qdot_des = J_pinv @ np.concatenate([v_cmd, w_cmd])
        qdot7 = _safe_step_velocity(q_cmd7, qdot_des, qdot_prev)
        qdot7 = _clip_cartesian_speed(pin_model, q_cmd7, qdot7)
        qdot7 = _safe_step_velocity(q_cmd7, qdot7, qdot_prev)

        q_next = q_cmd7 + qdot7 * DT_CONTROL
        qdot_prev = qdot7.copy()
        time_from_start = _append_dense_segment(
            points,
            q_start=q_cmd7,
            q_end=q_next,
            qdot_start=np.zeros(7, dtype=float) if not points else points[-1][1],
            qdot_end=qdot7,
            time_from_start=time_from_start,
        )
        q_cmd7 = q_next

    time_from_start = _append_hold(points, q_cmd7, hold_sec, time_from_start)
    return _postprocess_trajectory(points), q_cmd7


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
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--kp-pos", type=float, default=2.0)
    parser.add_argument("--kp-rot", type=float, default=1.0)
    parser.add_argument("--ik-tol-pos", type=float, default=1e-3)
    parser.add_argument("--ik-tol-rot", type=float, default=1e-3)
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
    exec_points, q_final = _build_execution_trajectory(
        target_pose_vel=np.asarray(args.pose_joint7_vel, dtype=float),
        start_q7=q_reset,
        hold_sec=float(args.hold_sec),
        max_iter=int(args.max_iter),
        kp_pos=float(args.kp_pos),
        kp_rot=float(args.kp_rot),
        ik_tol_pos=float(args.ik_tol_pos),
        ik_tol_rot=float(args.ik_tol_rot),
    )
    points = reset_points + [
        (q.copy(), qdot.copy(), qddot.copy(), float(t + reset_points[-1][3]))
        for q, qdot, qddot, t in exec_points
    ]

    print(f"[plan] built {len(points)} trajectory points", flush=True)
    print(f"[plan] final_q_cmd7={q_final.tolist()}", flush=True)
    _plot_joint_commands(points, Path(args.plot_dir))

    if args.print_only:
        return

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
