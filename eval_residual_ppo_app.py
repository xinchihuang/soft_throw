#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eval_residual_ppo_app.py

使用训练好的 PPO residual 模型 + tube baseline，执行定点投掷。

运行：
  ~/isaac-sim/python.sh scripts/eval_residual_ppo_app.py --model outputs/ppo_spoon_throw_residual.zip --target 2.0 0.0 0.0
"""

import argparse
import os
import time
import math

from isaaclab.app import AppLauncher

# -----------------------------------------------------------
# Launch Isaac Sim
# -----------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, help="Path to trained PPO model zip")
parser.add_argument("--target", type=float, nargs=3, default=None, help="Fixed target xyz in world (omit for random target)")
parser.add_argument("--episodes", type=int, default=10)
parser.add_argument("--headless", action="store_true")
args = parser.parse_args()

def _resolve_model_path(path_str: str) -> str:
    # Support running from repo root: training script saves under scripts/outputs/.
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
        f"Tip: from repo root, use --model scripts/outputs/ppo_spoon_throw_residual.zip"
    )

args.model = _resolve_model_path(args.model)

app_launcher = AppLauncher(headless=bool(args.headless), enable_viewport=not bool(args.headless))
simulation_app = app_launcher.app

# prevent auto-quit
try:
    import carb
    settings = carb.settings.get_settings()
    for k in [
        "/app/quitAfterRunning",
        "/app/quitOnScriptCompletion",
        "/app/quitAfterTests",
        "/app/quitAfterExecute",
        "/app/quitAfterScript",
    ]:
        try:
            settings.set_bool(k, False)
        except Exception:
            pass
except Exception:
    pass

from stable_baselines3 import PPO

try:
    # Direct script execution
    from env_spoon_throw_core import SpoonThrowEnvCore
    from core import throw_params as _throw_params
except ModuleNotFoundError:
    from scripts.env_spoon_throw_core import SpoonThrowEnvCore
    from scripts.core import throw_params as _throw_params


def main():
    # Enable arm target debug prints for eval
    _throw_params.DEBUG_PRINT_ARM_TARGETS = True
    model = PPO.load(args.model)

    env = SpoonThrowEnvCore(simulation_app=simulation_app, seed=0, use_residual=True)

    num_eps = int(args.episodes)

    for ep in range(1, num_eps + 1):
        reset_options = {"target_xyz": args.target} if args.target is not None else None
        if reset_options is None:
            obs, info = env.reset(seed=ep)
        else:
            obs, info = env.reset(seed=ep, options=reset_options)

        expected_xyz = getattr(env, "target_xyz", None)
        print(f"\n[EP {ep:03d}] expected_target_xyz={expected_xyz}")

        done = False
        ep_rew = 0.0
        steps = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rew, terminated, truncated, info = env.step(action)
            ep_rew += float(rew)
            steps += 1
            done = bool(terminated or truncated)

        stop_reason = info.get("stop_reason", "?")
        land_score = info.get("land_score", None)
        landing_xy = info.get("landing_xy", None)
        expected_xy = None if expected_xyz is None else expected_xyz[:2]
        print(f"[EP {ep:03d}] done reason={stop_reason} steps={steps} ep_rew={ep_rew:.3f}")
        landing_error = None
        if (expected_xy is not None) and (landing_xy is not None):
            dx = float(landing_xy[0]) - float(expected_xy[0])
            dy = float(landing_xy[1]) - float(expected_xy[1])
            landing_error = math.hypot(dx, dy)

        print(f"           expected_landing_xy={expected_xy}")
        print(f"           actual_landing_xy={landing_xy}")
        print(f"           landing_error_m={landing_error}")
        if land_score is not None:
            print(f"           land_score={land_score}")
        time.sleep(0.2)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
