#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Evaluate residual PPO policy on real robot (dry-run prints).

This script mirrors eval_residual_ppo_app.py logic but does NOT use Isaac Sim.
It prints the 14 joint targets (pos/vel) each step via RealRobotIO.
"""

import argparse
import time
import math
import os
import sys
from pathlib import Path
import numpy as np

# Ensure repo root is on sys.path when running from real/
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from stable_baselines3 import PPO

from core.throw_params import (
    DT_CONTROL,
    MAX_EPISODE_SEC,
    QDOT_LIMITS_3,
    JOINT0_YAW_LIMIT,
    THROW_START_Q,
    INIT_ARM,
    TUBE_T0,
    TUBE_HALF_WINDOW,
    RESID_VEL_MAX_3,
    ACTION_EMA_ALPHA,
    EE_VEL_MAX,
    USE_TUBE_ACCELERATION_PLANNER,
)
from core.kinematics_pin import PinKinematics
from core.targets import sample_target_xyz
from core.tube_baseline import solve_ballistic_velocity, tube_velocity_command, project_velocity_to_tube
from core.tube_accel_planner import tube_acceleration_select_release
from real.robot_io import RealRobotIO


def _resolve_model_path(path_str: str) -> str:
    candidates = []
    p0 = os.path.expanduser(path_str)
    candidates.append(p0)
    if not os.path.isabs(p0):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        candidates.append(os.path.join(script_dir, p0))

    for p in candidates:
        if os.path.isfile(p):
            return p

    tried = "\n  - ".join(os.path.abspath(p) for p in candidates)
    raise FileNotFoundError(
        f"Model file not found: {path_str}\n"
        f"Tried:\n  - {tried}\n"
        f"Tip: from repo root, use --model outputs/ppo_spoon_throw_residual.zip"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to trained PPO model zip")
    parser.add_argument("--target", type=float, nargs=3, default=None, help="Fixed target xyz in world (omit for random)")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--steps", type=int, default=None, help="Override steps per episode")
    args = parser.parse_args()

    model_path = _resolve_model_path(args.model)
    model = PPO.load(model_path)

    robot = RealRobotIO(print_only=True)
    pin = PinKinematics()

    for ep in range(1, int(args.episodes) + 1):
        rng = np.random.default_rng(ep)
        target_xyz = np.asarray(args.target, dtype=float).reshape(3) if args.target is not None else sample_target_xyz(rng)

        # initialize joint state
        q_cmd7 = robot.get_joint_pos7()
        if q_cmd7 is None or len(q_cmd7) != 7:
            q_cmd7 = INIT_ARM.copy()

        # yaw align joint0 toward target
        yaw = math.atan2(float(target_xyz[1]), float(target_xyz[0]))
        yaw = float(np.clip(yaw, -JOINT0_YAW_LIMIT, JOINT0_YAW_LIMIT))
        throw_start_q = THROW_START_Q.copy()
        throw_start_q[0] = yaw

        # plan release velocity
        q_full = pin.make_q_full_from_arm7(throw_start_q)
        pin._pin_forward(q_full)
        T = pin.data.oMf[pin.ee_id]
        p_release_nom = np.array(T.translation, dtype=float)

        v_star = None
        if USE_TUBE_ACCELERATION_PLANNER:
            try:
                J_lin = pin.jacobian_linear_sub(throw_start_q)
                cand = tube_acceleration_select_release(
                    target_xyz=target_xyz,
                    yaw=float(throw_start_q[0]),
                    p_release_nom=p_release_nom,
                    J_lin=J_lin,
                    qdot_limits=QDOT_LIMITS_3,
                )
                if cand is not None:
                    p_release_nom = cand["release_pos_w"].copy()
                    v_star = cand["release_vel_w"].copy()
            except Exception as e:
                print(f"[tube_acceleration] planner failed, fallback to ballistic: {e}", flush=True)

        if v_star is None:
            v_star, _T = solve_ballistic_velocity(p_release_nom, target_xyz)

        v_norm = float(np.linalg.norm(v_star))
        if v_norm > EE_VEL_MAX and v_norm > 1e-6:
            v_star = v_star * (EE_VEL_MAX / v_norm)
        v_hat = v_star / float(np.linalg.norm(v_star)) if float(np.linalg.norm(v_star)) > 1e-9 else np.array([1.0, 0.0, 0.0])

        tube_t0 = float(TUBE_T0)
        tube_t1 = float(TUBE_T0 + 2.0 * TUBE_HALF_WINDOW)

        # residual filters
        a_prev3 = np.zeros(3, dtype=float)
        a_filt3 = np.zeros(3, dtype=float)

        # episode loop
        t = 0.0
        max_steps = int(args.steps) if args.steps is not None else int(MAX_EPISODE_SEC / DT_CONTROL)
        for step in range(max_steps):
            # build obs
            ball_pos, ball_vel = robot.get_ball_state()
            ee = pin.get_ee_world_pos(q_cmd7)
            obs = np.concatenate([
                ball_pos, ball_vel, ee,
                a_filt3,
                np.array([0.0], dtype=float),  # released
                np.array([t], dtype=float),
            ]).astype(np.float32)

            action, _ = model.predict(obs, deterministic=True)
            a3 = np.asarray(action, dtype=float).reshape(3)
            a3 = np.clip(a3, -1.0, 1.0)

            # baseline velocity
            v_cmd = tube_velocity_command(t, v_star, v_hat, tube_t0, tube_t1)

            # residual
            a_filt3 = (1.0 - ACTION_EMA_ALPHA) * a_filt3 + ACTION_EMA_ALPHA * a3
            dv = a_filt3 * RESID_VEL_MAX_3
            v_cmd = project_velocity_to_tube(v_cmd + dv, v_star, v_hat)

            # joint velocity command
            qdot3 = pin.compute_qdot3_for_v(q_cmd7, v_cmd)
            qdot3 = np.clip(qdot3, -QDOT_LIMITS_3, QDOT_LIMITS_3)

            qdot7 = np.zeros(7, dtype=float)
            qdot7[[1, 3, 5]] = qdot3

            q_cmd7 = q_cmd7.copy()
            q_cmd7[[1, 3, 5]] = q_cmd7[[1, 3, 5]] + qdot3 * DT_CONTROL
            keep = np.ones(7, dtype=bool)
            keep[[1, 3, 5]] = False
            q_cmd7[keep] = throw_start_q[keep]

            robot.send_joint_targets(q_cmd7, qdot7)

            t += DT_CONTROL
            time.sleep(DT_CONTROL)

        print(f"[EP {ep:03d}] done steps={max_steps}")


if __name__ == "__main__":
    main()
