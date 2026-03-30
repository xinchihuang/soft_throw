#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_residual_ppo_app.py

目的：
- 在 Tube baseline 的基础上，用 RL 学习 residual（Δv）来修正系统性误差。
- env 仍然负责：随机 3D target、joint0 yaw 对准、prep、tube baseline。
- RL 只输出 action(3) -> Δv (m/s) 的残差（通过 env 内的 project-to-tube 约束安全）。

运行：
  ~/isaac-sim/python.sh scripts/train_residual_ppo_app.py

输出：
  outputs/ppo_spoon_throw_residual.zip
  outputs/tb_spoon_throw_residual/
"""

import os
import time
import argparse

from isaaclab.app import AppLauncher

# -----------------------------------------------------------
# Launch Isaac Sim
# -----------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", help="Run Isaac Sim without UI")
args = parser.parse_args()

# Default to UI mode for interactive debugging/inspection unless --headless is set.
app_launcher = AppLauncher(headless=bool(args.headless))
simulation_app = app_launcher.app

# IMPORTANT: import env AFTER simulation_app is created
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

try:
    # Direct script execution: ~/isaac-sim/python.sh scripts/train_residual_ppo_app.py
    from env_spoon_throw_core import SpoonThrowEnvCore
except ModuleNotFoundError:
    # Package-style execution fallback
    from scripts.env_spoon_throw_core import SpoonThrowEnvCore

OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(OUT_DIR, "ppo_spoon_throw_residual.zip")
TB_DIR = os.path.join(OUT_DIR, "tb_spoon_throw_residual")


def make_env(seed: int = 0):
    def _thunk():
        env = SpoonThrowEnvCore(simulation_app=simulation_app, seed=seed, use_residual=True)
        env = Monitor(env)
        return env
    return _thunk


def main():
    vec_env = DummyVecEnv([make_env(seed=0)])

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        n_steps=2048,
        batch_size=256,
        gae_lambda=0.95,
        gamma=0.99,
        n_epochs=10,
        ent_coef=0.0,
        learning_rate=3e-4,
        clip_range=0.2,
        verbose=1,
        tensorboard_log=TB_DIR,
        device="cuda",  # change to "cpu" if needed
    )

    # 训练步数你可以按算力调整
    total_timesteps = 200_000
    model.learn(total_timesteps=total_timesteps, progress_bar=True)

    model.save(MODEL_PATH)
    print(f"[OK] Saved model -> {MODEL_PATH}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
