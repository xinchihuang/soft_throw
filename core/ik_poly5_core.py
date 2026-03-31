#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""IK solve + poly5 trajectory generation for joint-space control."""

import numpy as np
import pinocchio as pin


def solve_ik_for_q_goal(
    pin_model,
    target_frame_id,
    q_init,
    p_j7_des,
    R_j7_des,
    q_limits,
    qdot_limits,
    qddot_limits,
    control_dt=0.01,
    max_iter=300,
    kp_pos=2.0,
    kp_rot=1.0,
    v_j7_des=None,
):
    """Solve IK (velocity DLS) to get q_goal without actuating."""
    q_goal = np.asarray(q_init, dtype=float).copy()
    qdot_prev = np.zeros(7, dtype=float)
    if v_j7_des is None:
        v_j7_des = np.zeros(3, dtype=float)

    for _ in range(max_iter):
        q_full = pin_model.make_q_full_from_arm7(q_goal)
        pin.forwardKinematics(pin_model.model, pin_model.data, q_full)
        pin.updateFramePlacements(pin_model.model, pin_model.data)
        T = pin_model.data.oMf[target_frame_id]
        p_cur = np.array(T.translation, dtype=float)
        R_cur = np.array(T.rotation, dtype=float)

        ep = p_j7_des - p_cur
        eR = pin.log3(R_cur.T @ R_j7_des)
        if np.linalg.norm(ep) < 1e-3 and np.linalg.norm(eR) < 1e-3:
            break

        J = pin.computeFrameJacobian(
            pin_model.model, pin_model.data, q_full, target_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        J_arm = np.zeros((6, 7), dtype=float)
        for j in range(1, 8):
            jid = pin_model.model.getJointId(f"panda_joint{j}")
            idx_v = int(pin_model.model.joints[jid].idx_v)
            J_arm[:, j - 1] = J[:, idx_v]

        lam = 0.05
        JJt = J_arm @ J_arm.T
        inv = np.linalg.inv(JJt + (lam * lam) * np.eye(6))
        J_pinv = J_arm.T @ inv
        v_cmd = v_j7_des + kp_pos * ep
        w_cmd = kp_rot * eR
        v6 = np.concatenate([v_cmd, w_cmd])
        qdot_des = J_pinv @ v6

        max_dq = qddot_limits * control_dt
        dq = np.clip(qdot_des - qdot_prev, -max_dq, max_dq)
        qdot7 = qdot_prev + dq
        qdot7 = np.clip(qdot7, -qdot_limits, qdot_limits)

        q_goal = q_goal + qdot7 * control_dt
        q_goal = np.clip(q_goal, q_limits[:, 0], q_limits[:, 1])
        qdot_prev = qdot7.copy()

    return q_goal


def poly5_trajectory(q_start, q_goal, q_limits, qdot_limits, qddot_limits, control_dt=0.01):
    """Generate poly5 joint trajectory from q_start to q_goal.

    Returns t, q, qdot, qddot, u (u=qdot).
    """
    q_start = np.asarray(q_start, dtype=float)
    q_goal = np.asarray(q_goal, dtype=float)
    dq = np.abs(q_goal - q_start)
    v_coeff = 1.875
    a_coeff = 5.7735
    T_v = np.max(v_coeff * dq / np.maximum(qdot_limits, 1e-6))
    T_a = np.max(np.sqrt(a_coeff * dq / np.maximum(qddot_limits, 1e-6)))
    T = float(max(T_v, T_a, 0.5))
    n_steps = max(2, int(np.ceil(T / control_dt)) + 1)

    def poly5(q0, qf, T, t):
        tau = t / T
        tau2 = tau * tau
        tau3 = tau2 * tau
        tau4 = tau3 * tau
        tau5 = tau4 * tau
        s = 10 * tau3 - 15 * tau4 + 6 * tau5
        sd = (30 * tau2 - 60 * tau3 + 30 * tau4) / T
        sdd = (60 * tau - 180 * tau2 + 120 * tau3) / (T * T)
        q = q0 + (qf - q0) * s
        qd = (qf - q0) * sd
        qdd = (qf - q0) * sdd
        return q, qd, qdd

    t_hist = []
    q_hist = []
    qdot_hist = []
    qddot_hist = []
    u_hist = []

    for k in range(n_steps):
        t = min(k * control_dt, T)
        q, qd, qdd = poly5(q_start, q_goal, T, t)
        q = np.clip(q, q_limits[:, 0], q_limits[:, 1])
        qd = np.clip(qd, -qdot_limits, qdot_limits)
        qdd = np.clip(qdd, -qddot_limits, qddot_limits)

        t_hist.append(t)
        q_hist.append(q.copy())
        qdot_hist.append(qd.copy())
        qddot_hist.append(qdd.copy())
        u_hist.append(qd.copy())

    return (
        np.array(t_hist),
        np.array(q_hist),
        np.array(qdot_hist),
        np.array(qddot_hist),
        np.array(u_hist),
    )
