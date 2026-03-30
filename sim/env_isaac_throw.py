#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import csv
import math
import time
from datetime import datetime
import numpy as np

import gymnasium as gym
from gymnasium import spaces

import omni.timeline

from core.throw_params import (
    ROBOT_USD,
    ROBOT_PRIM,
    FRANKA_ROOT,
    BALL_PATH,
    LACROSSE_PATH,
    CONTROL_HZ,
    DT_CONTROL,
    SIM_UPDATES_PER_STEP,
    LEARN_JIDX,
    QDOT_LIMITS_3,
    ACTION_EMA_ALPHA,
    EE_VEL_MAX,
    USE_TUBE_BASELINE,
    USE_RESIDUAL,
    TARGET_XYZ,
    JOINT0_YAW_LIMIT,
    USE_TUBE_ACCELERATION_PLANNER,
    TUBE_T0,
    TUBE_HALF_WINDOW,
    TUBE_DECEL_SEC,
    RESID_VEL_MAX_3,
    MAX_EPISODE_SEC,
    DROP_TIMEOUT_AFTER_RELEASE_SEC,
    RESET_ARM_SETTLE_SEC,
    BALL_RESET_POS_WORLD,
    RESET_BALL_WAIT_SEC,
    BALL_DETACH_DIST,
    INIT_ARM,
    THROW_START_Q,
    PREP_STEPS,
    JOINT0_ALIGN_STEPS,
    JOINT0_ALIGN_WAIT_SEC,
    W_FORWARD,
    W_LAND,
    W_ACT,
    W_QDOT,
    W_SMOOTH,
    W_DEVIATE,
    SAVE_DEMO_TRAJ,
    SAVE_DEMO_DIR,
    SAVE_ONLY_BEST,
)

from core.kinematics_pin import PinKinematics
from core.rewards import landing_reward
from core.targets import sample_target_xyz
from core.tube_accel_planner import tube_acceleration_select_release
from core.tube_baseline import solve_ballistic_velocity, tube_velocity_command, project_velocity_to_tube

from sim.isaac_scene import (
    get_stage,
    ensure_physics_scene,
    spawn_ground,
    spawn_lights,
    add_robot_reference,
    detect_franka_root,
    tune_rb_collision_recursive,
    apply_no_bounce_material,
)
from sim.isaac_robot_io import (
    find_joint_paths,
    apply_arm_targets,
    reset_ball,
    get_ball_world_pos_vel,
    ball_detached_from_net,
    ball_landed,
)


class SpoonThrowEnvCoreTubeRandomYawAlign(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, simulation_app, seed: int = 0, use_residual=None):
        super().__init__()
        self.simulation_app = simulation_app
        self.np_random = np.random.default_rng(seed)

        self.stage = get_stage()
        self.timeline = omni.timeline.get_timeline_interface()

        # build/ensure stage content (lights, ground, physics, robot ref)
        self.timeline.stop()
        for _ in range(30):
            self.simulation_app.update()

        ensure_physics_scene(self.stage)
        spawn_ground(self.stage)
        spawn_lights(self.stage)
        add_robot_reference(self.stage, ROBOT_PRIM, ROBOT_USD)

        # warmup
        for _ in range(240):
            self.simulation_app.update()

        self.franka_root = detect_franka_root(self.stage, FRANKA_ROOT)
        self.joint_paths = find_joint_paths(self.stage, self.franka_root)

        # tune rigidbodies / collisions and materials
        tune_rb_collision_recursive(self.stage, BALL_PATH, "BALL")
        tune_rb_collision_recursive(self.stage, LACROSSE_PATH, "LACROSSE")
        apply_no_bounce_material(self.stage, BALL_PATH, "BALL")
        apply_no_bounce_material(self.stage, LACROSSE_PATH, "LACROSSE")

        # build pinocchio model
        self.pin = PinKinematics()

        # state
        self.episode_t = 0.0
        self.released = False
        self.release_time_wall = None
        self.release_time_ep = None
        self.release_step = None
        self.step_count = 0

        # action filter / residuals
        self._a_prev3 = np.zeros(3, dtype=float)
        self._a_filt3 = np.zeros(3, dtype=float)
        self.q_resid3 = np.zeros(3, dtype=float)

        # command
        self.q_cmd7 = INIT_ARM.copy()
        self.qdot7 = np.zeros(7, dtype=float)

        # tube baseline plan (3D)
        self.use_tube_baseline = bool(USE_TUBE_BASELINE)
        self.use_residual = bool(USE_RESIDUAL if use_residual is None else use_residual)
        self.target_xyz = TARGET_XYZ.astype(float).copy()
        self.throw_start_q = THROW_START_Q.copy()
        self.tube_t0 = float(TUBE_T0)
        self.tube_t1 = float(TUBE_T0 + 2.0 * TUBE_HALF_WINDOW)
        self.v_release_star = np.zeros(3, dtype=float)
        self.v_release_hat = np.array([1.0, 0.0, 0.0], dtype=float)
        self.p_release_nom = np.zeros(3, dtype=float)
        self.release_plan_method = "none"
        self.release_plan_pos = np.zeros(3, dtype=float)
        self.release_plan_vel = np.zeros(3, dtype=float)

        # ball tracking
        self.ball_prev_x = 0.0

        # recording for demo
        self.demo_ep_counter = 0
        self.best_land_score = -1e9
        self._clear_recording_buffers()

        # gym spaces
        # obs: [ball_pos(3), ball_vel(3), ee_pos(3), resid3(3), released(1), time(1)]
        obs_dim = 3 + 3 + 3 + 3 + 1 + 1
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # start sim
        self.timeline.play()

    # ---------------- Recording & Saving ----------------
    def _clear_recording_buffers(self):
        self.seq_t = []
        self.seq_q_cmd7 = []
        self.seq_action3 = []
        self.seq_ball_pos = []
        self.seq_ee_pos = []

    def _save_qcmd7_demo(self, info: dict):
        if not SAVE_DEMO_TRAJ:
            return
        os.makedirs(SAVE_DEMO_DIR, exist_ok=True)

        land_score = float(info.get("land_score", 0.0))
        if SAVE_ONLY_BEST and (land_score <= self.best_land_score):
            return
        if SAVE_ONLY_BEST:
            self.best_land_score = land_score

        self.demo_ep_counter += 1
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = f"demo_ep{self.demo_ep_counter:04d}_{stamp}"

        csv_path = os.path.join(SAVE_DEMO_DIR, base + ".csv")
        meta_path = os.path.join(SAVE_DEMO_DIR, base + ".json")

        seq_t = np.asarray(info["seq_t"], dtype=np.float32)
        seq_q = np.asarray(info["seq_q_cmd7"], dtype=np.float32)

        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", "q1", "q2", "q3", "q4", "q5", "q6", "q7"])  # radians
            for i in range(len(seq_t)):
                row = [float(seq_t[i])] + [float(x) for x in seq_q[i]]
                w.writerow(row)

        meta = {
            "stop_reason": info.get("stop_reason", ""),
            "released": bool(info.get("released", False)),
            "release_step": info.get("release_step", None),
            "release_time_ep": info.get("release_time_ep", None),
            "landing_xy": (info.get("landing_xy", None).tolist()
                           if hasattr(info.get("landing_xy", None), "tolist") else info.get("landing_xy", None)),
            "land_score": land_score,
            "target_xy": self.target_xyz[:2].tolist(),
            "dt_control": float(DT_CONTROL),
            "control_hz": float(CONTROL_HZ),
            "sim_updates_per_step": int(SIM_UPDATES_PER_STEP),
            "ball_detach_dist": float(BALL_DETACH_DIST),
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        print(f"[demo] saved q_cmd7 -> {csv_path}", flush=True)

    # ---------------- Obs ----------------
    def _get_obs(self):
        bpos, bvel, _ = get_ball_world_pos_vel(self.stage)
        if bpos is None:
            bpos = np.zeros(3, dtype=float)
        if bvel is None:
            bvel = np.zeros(3, dtype=float)
        ee = self.pin.get_ee_world_pos(self.q_cmd7)
        obs = np.concatenate([
            bpos, bvel, ee,
            self.q_resid3,
            np.array([1.0 if self.released else 0.0], dtype=float),
            np.array([self.episode_t], dtype=float),
        ]).astype(np.float32)
        return obs

    # -------- Tube baseline helpers --------
    def _plan_tube_baseline(self, target_xyz: np.ndarray):
        """Plan nominal 3D ballistic release velocity from the EE pose at the (possibly yaw-aligned) throw-start pose."""
        target_xyz = np.asarray(target_xyz, dtype=float).reshape(3)
        self.target_xyz = target_xyz.copy()

        # Nominal release point: EE position at THROW_START_Q
        q_full = self.pin.make_q_full_from_arm7(getattr(self, "throw_start_q", THROW_START_Q))
        self.pin._pin_forward(q_full)
        T = self.pin.data.oMf[self.pin.ee_id]
        self.p_release_nom = np.array(T.translation, dtype=float)

        v_star = None
        method = "ballistic"
        if USE_TUBE_ACCELERATION_PLANNER:
            try:
                J_lin = self.pin.jacobian_linear_sub(self.throw_start_q)
                cand = tube_acceleration_select_release(
                    target_xyz=target_xyz,
                    yaw=float(self.throw_start_q[0]),
                    p_release_nom=self.p_release_nom,
                    J_lin=J_lin,
                    qdot_limits=QDOT_LIMITS_3,
                )
                if cand is not None:
                    self.p_release_nom = cand["release_pos_w"].copy()
                    v_star = cand["release_vel_w"].copy()
                    method = "tube_acceleration"
            except Exception as e:
                print(f"[tube_acceleration] planner failed, fallback to ballistic: {e}", flush=True)

        if v_star is None:
            v_star, _T_star = solve_ballistic_velocity(self.p_release_nom, target_xyz)
            method = "ballistic"

        # clamp to EE speed limit
        v_norm = float(np.linalg.norm(v_star))
        if v_norm > EE_VEL_MAX and v_norm > 1e-6:
            v_star = v_star * (EE_VEL_MAX / v_norm)
        self.v_release_star = v_star
        n = float(np.linalg.norm(v_star))
        self.v_release_hat = (v_star / n) if n > 1e-9 else np.array([1.0, 0.0, 0.0], dtype=float)
        self.release_plan_method = method
        self.release_plan_pos = self.p_release_nom.copy()
        self.release_plan_vel = self.v_release_star.copy()

        self.tube_t0 = float(TUBE_T0)
        self.tube_t1 = float(TUBE_T0 + 2.0 * TUBE_HALF_WINDOW)

        # Always print planned r,z in the yaw-aligned throwing plane
        try:
            e_r = np.array([math.cos(self.throw_start_q[0]), math.sin(self.throw_start_q[0]), 0.0], dtype=float)
            e_z = np.array([0.0, 0.0, 1.0], dtype=float)
            E = self.release_plan_pos
            r = float(np.dot(self.target_xyz - E, e_r))
            z = float(np.dot(E - self.target_xyz, e_z))
            print(
                f"[release_plan_plane] method={self.release_plan_method} r={r:.3f} z={z:.3f} "
                f"release_pos={self.release_plan_pos.tolist()}",
                flush=True,
            )
        except Exception as e:
            print(f"[release_plan_plane] failed to compute r,z: {e}", flush=True)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        self.episode_t = 0.0
        self.step_count = 0
        self.released = False
        self.release_time_wall = None
        self.release_time_ep = None
        self.release_step = None

        self._a_prev3[:] = 0.0
        self._a_filt3[:] = 0.0
        self.q_resid3[:] = 0.0

        self._clear_recording_buffers()

        # --------- REQUIRED RESET ORDER ----------
        # 1) arm -> INIT, hold 2s
        self.q_cmd7 = INIT_ARM.copy()
        self.qdot7[:] = 0.0

        t0 = time.time()
        while time.time() - t0 < RESET_ARM_SETTLE_SEC:
            apply_arm_targets(self.stage, self.joint_paths, self.q_cmd7, self.qdot7)
            self.simulation_app.update()

        # 2) ball reset at arm INIT position
        reset_ball(self.stage, BALL_RESET_POS_WORLD)

        # 3) wait 2s after ball reset, keep arm at INIT
        t1 = time.time()
        while time.time() - t1 < RESET_BALL_WAIT_SEC:
            apply_arm_targets(self.stage, self.joint_paths, self.q_cmd7, self.qdot7)
            self.simulation_app.update()

        # 4) sample or use provided 3D target for this episode
        if options and ("target_xyz" in options):
            t = np.asarray(options["target_xyz"], dtype=float).reshape(3)
            self.target_xyz = t.copy()
        else:
            self.target_xyz = sample_target_xyz(self.np_random)

        # 5) yaw-align joint0 (panda_joint1) towards the target in world XY.
        #    With only joints (2,4,6) controlled for throwing, we must rotate the arm plane towards target.
        yaw = math.atan2(float(self.target_xyz[1]), float(self.target_xyz[0]))
        yaw = float(np.clip(yaw, -JOINT0_YAW_LIMIT, JOINT0_YAW_LIMIT))
        self.throw_start_q = THROW_START_Q.copy()
        self.throw_start_q[0] = yaw  # joint0

        # 6a) move only joint0 first: INIT -> yaw-aligned pose
        q_yaw = INIT_ARM.copy()
        q_yaw[0] = yaw
        for k in range(JOINT0_ALIGN_STEPS):
            alpha = (k + 1) / float(JOINT0_ALIGN_STEPS)
            q = (1 - alpha) * INIT_ARM + alpha * q_yaw
            self.q_cmd7 = q.copy()
            self.qdot7[:] = 0.0
            apply_arm_targets(self.stage, self.joint_paths, self.q_cmd7, self.qdot7)
            for _ in range(SIM_UPDATES_PER_STEP):
                self.simulation_app.update()

        # 6b) wait for joint0 alignment to settle
        t2 = time.time()
        while time.time() - t2 < JOINT0_ALIGN_WAIT_SEC:
            self.q_cmd7 = q_yaw.copy()
            self.qdot7[:] = 0.0
            apply_arm_targets(self.stage, self.joint_paths, self.q_cmd7, self.qdot7)
            self.simulation_app.update()

        # 6c) then move only throwing joints (2,4,6) to THROW_START
        q_prep = q_yaw.copy()
        q_prep[LEARN_JIDX] = self.throw_start_q[LEARN_JIDX]
        for k in range(PREP_STEPS):
            alpha = (k + 1) / float(PREP_STEPS)
            q = (1 - alpha) * q_yaw + alpha * q_prep
            self.q_cmd7 = q.copy()
            self.qdot7[:] = 0.0
            apply_arm_targets(self.stage, self.joint_paths, self.q_cmd7, self.qdot7)
            for _ in range(SIM_UPDATES_PER_STEP):
                self.simulation_app.update()

        # 7) plan tube baseline from the yaw-aligned THROW_START pose
        self._plan_tube_baseline(self.target_xyz)
        print(
            f"[release_plan] method={self.release_plan_method} "
            f"pos={self.release_plan_pos.tolist()} "
            f"vel={self.release_plan_vel.tolist()}",
            flush=True,
        )

        # ball reference for forward progress
        bpos, _, _ = get_ball_world_pos_vel(self.stage)
        self.ball_prev_x = float(bpos[0]) if bpos is not None else 0.0

        return self._get_obs(), {
            "release_plan_method": self.release_plan_method,
            "release_plan_pos": self.release_plan_pos.copy(),
            "release_plan_vel": self.release_plan_vel.copy(),
        }

    def step(self, action):
        self.step_count += 1
        self.episode_t += DT_CONTROL

        # detect release (phase switch) but DO NOT terminate
        if (not self.released) and ball_detached_from_net(self.stage, self.pin.get_ee_world_pos(self.q_cmd7)):
            self.released = True
            self.release_time_wall = time.time()
            self.release_time_ep = float(self.episode_t)
            self.release_step = self.step_count
            # freeze arm at current q_cmd7
            # (keep q_cmd7 as is, set qdot to zero)
            self.qdot7[:] = 0.0

        # ---------------- control update ----------------
        a3 = np.asarray(action, dtype=float).reshape(3)
        a3 = np.clip(a3, -1.0, 1.0)

        act_pen = 0.0
        qdot_pen = 0.0
        smooth_pen = 0.0
        dev_pen = 0.0

        if not self.released:
            # Ensure tube plan exists
            if (self.step_count == 1) and (np.linalg.norm(self.v_release_star) < 1e-6):
                self._plan_tube_baseline(self.target_xyz)

            # Baseline tube command in task space (EE linear velocity)
            v_cmd = tube_velocity_command(self.episode_t, self.v_release_star, self.v_release_hat, self.tube_t0, self.tube_t1)

            # --- Residual interface (disabled by default) ---
            # Future: allow RL to predict a residual EE velocity delta, then project back into tube.
            if self.use_residual:
                self._a_filt3 = (1.0 - ACTION_EMA_ALPHA) * self._a_filt3 + ACTION_EMA_ALPHA * a3
                dv = self._a_filt3 * RESID_VEL_MAX_3
                self.q_resid3[:] = dv
                v_cmd = project_velocity_to_tube(v_cmd + dv, self.v_release_star, self.v_release_hat)

                # penalties (only if residual enabled)
                act_pen = W_ACT * float(np.dot(a3, a3))
                smooth_pen = W_SMOOTH * float(np.dot(a3 - self._a_prev3, a3 - self._a_prev3))
                self._a_prev3 = a3.copy()
            else:
                self.q_resid3[:] = 0.0

            # Map EE velocity to learned joint velocities
            qdot3 = self.pin.compute_qdot3_for_v(self.q_cmd7, v_cmd)

            # joint speed limit
            qdot3 = np.clip(qdot3, -QDOT_LIMITS_3, QDOT_LIMITS_3)

            # Build full qdot7 (only learned joints move during throw)
            self.qdot7[:] = 0.0
            self.qdot7[LEARN_JIDX] = qdot3

            # Integrate to position targets (position drive)
            self.q_cmd7[LEARN_JIDX] = self.q_cmd7[LEARN_JIDX] + qdot3 * DT_CONTROL
            # Keep other joints fixed at THROW_START_Q
            keep = np.ones(7, dtype=bool)
            keep[LEARN_JIDX] = False
            self.q_cmd7[keep] = self.throw_start_q[keep]

            # Penalty on joint velocities (tube baseline energy)
            qdot_pen = W_QDOT * float(np.dot(qdot3, qdot3))
        else:
            # after release: freeze arm, no penalties on actions
            self.qdot7[:] = 0.0
            self.q_resid3[:] = 0.0

        # step sim
        for _ in range(SIM_UPDATES_PER_STEP):
            apply_arm_targets(self.stage, self.joint_paths, self.q_cmd7, self.qdot7)
            self.simulation_app.update()

        # ---------------- reward ----------------
        reward = 0.0

        bpos, _, _ = get_ball_world_pos_vel(self.stage)
        if bpos is not None:
            dx = float(bpos[0] - self.ball_prev_x)
            self.ball_prev_x = float(bpos[0])
            reward += W_FORWARD * dx

        reward -= act_pen + qdot_pen + smooth_pen + dev_pen

        terminated = False
        truncated = False
        info = {
            "released": self.released,
            "release_step": self.release_step,
            "release_plan_method": self.release_plan_method,
            "release_plan_pos": self.release_plan_pos.copy(),
            "release_plan_vel": self.release_plan_vel.copy(),
        }

        # landing reward happens when ball lands (even after release)
        if ball_landed(self.stage) and (bpos is not None):
            landing_xy = bpos[:2].copy()
            land_r = landing_reward(landing_xy, self.target_xyz[:2])
            reward += W_LAND * land_r
            terminated = True
            info["stop_reason"] = "ball_landed"
            info["landing_xy"] = landing_xy.copy()
            info["land_score"] = float(land_r)

        # timeout after release
        if (not terminated) and self.released and (self.release_time_ep is not None):
            if (self.episode_t - self.release_time_ep) >= DROP_TIMEOUT_AFTER_RELEASE_SEC:
                truncated = True
                info["stop_reason"] = "timeout_after_release"
                info["land_score"] = 0.0

        # max episode
        if (not terminated) and (not truncated) and (self.episode_t >= MAX_EPISODE_SEC):
            truncated = True
            info["stop_reason"] = "max_episode"
            info["land_score"] = 0.0

        # ---------------- recording ----------------
        ee_pos = self.pin.get_ee_world_pos(self.q_cmd7)
        self.seq_t.append(float(self.episode_t))
        self.seq_q_cmd7.append(self.q_cmd7.astype(np.float32).copy())
        self.seq_action3.append(a3.astype(np.float32).copy())
        self.seq_ball_pos.append((bpos.astype(np.float32).copy() if bpos is not None else np.zeros(3, np.float32)))
        self.seq_ee_pos.append(ee_pos.astype(np.float32).copy())

        if terminated or truncated:
            info["seq_t"] = np.asarray(self.seq_t, dtype=np.float32)
            info["seq_q_cmd7"] = np.asarray(self.seq_q_cmd7, dtype=np.float32)
            info["seq_action3"] = np.asarray(self.seq_action3, dtype=np.float32)
            info["seq_ball_pos"] = np.asarray(self.seq_ball_pos, dtype=np.float32)
            info["seq_ee_pos"] = np.asarray(self.seq_ee_pos, dtype=np.float32)
            info["release_time_ep"] = (float(self.seq_t[self.release_step - 1])
                                      if self.release_step is not None and self.release_step - 1 < len(self.seq_t)
                                      else None)
            self._save_qcmd7_demo(info)

        return self._get_obs(), float(reward), terminated, truncated, info


# Backward-compatible alias for training / rollout scripts.
SpoonThrowEnvCore = SpoonThrowEnvCoreTubeRandomYawAlign
