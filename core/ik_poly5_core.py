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

    return q_goal, qdot_prev


def solve_ik_for_reset_pos(
    pin_model,
    q_init,
    p_des_world,
    q_limits,
    qdot_limits,
    qddot_limits,
    control_dt=0.01,
    max_iter=300,
    kp_pos=2.0,
    kp_rot=1.0,
    target_frame_id=None,
):
    """Solve IK to reach a desired world position using current orientation as target."""
    if target_frame_id is None:
        target_frame_id = pin_model.ee_id

    q_full = pin_model.make_q_full_from_arm7(q_init)
    pin.forwardKinematics(pin_model.model, pin_model.data, q_full)
    pin.updateFramePlacements(pin_model.model, pin_model.data)
    T0 = pin_model.data.oMf[target_frame_id]
    R_des = np.array(T0.rotation, dtype=float)

    return solve_ik_for_q_goal(
        pin_model,
        target_frame_id,
        q_init,
        np.asarray(p_des_world, dtype=float).reshape(3),
        R_des,
        q_limits,
        qdot_limits,
        qddot_limits,
        control_dt=control_dt,
        max_iter=max_iter,
        kp_pos=kp_pos,
        kp_rot=kp_rot,
        v_j7_des=None,
    )

def poly5_trajectory(
    q_start,
    q_goal,
    q_limits,
    qdot_limits,
    qddot_limits,
    control_dt=0.01,
    qdot_start=None,
    qdot_goal=None,
    qddot_start=None,
    qddot_goal=None,
    t_search_max_iter=50,
    t_search_grow=1.2,
):
    """Generate poly5 joint trajectory with position/velocity/accel boundary conditions.

    Returns t, q, qdot, qddot, u (u=qdot).
    """
    q_start = np.asarray(q_start, dtype=float)
    q_goal = np.asarray(q_goal, dtype=float)
    if qdot_start is None:
        qdot_start = np.zeros_like(q_start)
    if qdot_goal is None:
        qdot_goal = np.zeros_like(q_start)
    if qddot_start is None:
        qddot_start = np.zeros_like(q_start)
    if qddot_goal is None:
        qddot_goal = np.zeros_like(q_start)
    qdot_start = np.asarray(qdot_start, dtype=float)
    qdot_goal = np.asarray(qdot_goal, dtype=float)
    qddot_start = np.asarray(qddot_start, dtype=float)
    qddot_goal = np.asarray(qddot_goal, dtype=float)

    dq = np.abs(q_goal - q_start)
    v_coeff = 1.875
    a_coeff = 5.7735
    T_v = np.max(v_coeff * dq / np.maximum(qdot_limits, 1e-6))
    T_a = np.max(np.sqrt(a_coeff * dq / np.maximum(qddot_limits, 1e-6)))
    T = float(max(T_v, T_a, 0.5))

    # Solve for polynomial coefficients with boundary velocities/accelerations.
    # q(t) = a0 + a1 t + a2 t^2 + a3 t^3 + a4 t^4 + a5 t^5
    a0 = q_start
    a1 = qdot_start
    a2 = 0.5 * qddot_start

    def _solve_coeffs(T_val: float):
        T2 = T_val * T_val
        T3 = T2 * T_val
        T4 = T3 * T_val
        T5 = T4 * T_val
        rhs0 = q_goal - (a0 + a1 * T_val + a2 * T2)
        rhs1 = qdot_goal - (a1 + 2.0 * a2 * T_val)
        rhs2 = qddot_goal - (2.0 * a2)
        M = np.array(
            [
                [T3, T4, T5],
                [3.0 * T2, 4.0 * T3, 5.0 * T4],
                [6.0 * T_val, 12.0 * T2, 20.0 * T3],
            ],
            dtype=float,
        )
        rhs = np.vstack([rhs0, rhs1, rhs2])
        a3_a4_a5 = np.linalg.solve(M, rhs)
        a3 = a3_a4_a5[0]
        a4 = a3_a4_a5[1]
        a5 = a3_a4_a5[2]
        return a3, a4, a5

    def _within_limits(T_val: float) -> bool:
        a3, a4, a5 = _solve_coeffs(T_val)
        n_steps = max(2, int(np.ceil(T_val / control_dt)) + 1)
        for k in range(n_steps):
            t = min(k * control_dt, T_val)
            t2 = t * t
            t3 = t2 * t
            t4 = t3 * t
            t5 = t4 * t
            q = a0 + a1 * t + a2 * t2 + a3 * t3 + a4 * t4 + a5 * t5
            qd = a1 + 2.0 * a2 * t + 3.0 * a3 * t2 + 4.0 * a4 * t3 + 5.0 * a5 * t4
            qdd = 2.0 * a2 + 6.0 * a3 * t + 12.0 * a4 * t2 + 20.0 * a5 * t3
            if np.any(q < q_limits[:, 0]) or np.any(q > q_limits[:, 1]):
                return False
            if np.any(np.abs(qd) > qdot_limits):
                return False
            if np.any(np.abs(qdd) > qddot_limits):
                return False
        return True

    # Expand T until all limits are satisfied (numerical search).
    for _ in range(int(t_search_max_iter)):
        if _within_limits(T):
            break
        T *= float(t_search_grow)

    n_steps = max(2, int(np.ceil(T / control_dt)) + 1)
    a3, a4, a5 = _solve_coeffs(T)

    t_hist = []
    q_hist = []
    qdot_hist = []
    qddot_hist = []
    u_hist = []

    for k in range(n_steps):
        t = min(k * control_dt, T)
        t2 = t * t
        t3 = t2 * t
        t4 = t3 * t
        t5 = t4 * t
        q = a0 + a1 * t + a2 * t2 + a3 * t3 + a4 * t4 + a5 * t5
        qd = a1 + 2.0 * a2 * t + 3.0 * a3 * t2 + 4.0 * a4 * t3 + 5.0 * a5 * t4
        qdd = 2.0 * a2 + 6.0 * a3 * t + 12.0 * a4 * t2 + 20.0 * a5 * t3
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
