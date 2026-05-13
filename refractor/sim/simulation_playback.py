#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Repeatedly reload and execute the default trajectory CSV in Isaac Sim.

Controls:
- Enter/Space: reload trajectory CSV and execute it once.
- q: quit.
"""

from __future__ import annotations

import os
import re
import select
import sys
import termios
import time
import tty
from pathlib import Path

import numpy as np


_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from refractor.config import params
from refractor.sim.lacrosse_collision import ensure_lacrosse_sdf_collider
from refractor.trajectory_afterprocess import assert_valid_trajectory
from refractor.trajectory_csv import read_trajectory_csv, trajectory_list_to_arrays


def _load_env_from_file(path: str | Path) -> None:
    path = Path(path)
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("export "):
            continue
        match = re.match(r"export\s+([A-Za-z_][A-Za-z0-9_]*)=(.*)", line)
        if not match:
            continue
        key, raw_value = match.groups()
        value = raw_value.strip().strip('"').strip("'")
        if key not in os.environ:
            os.environ[key] = value


def _read_key_nonblocking(stdin_is_tty: bool) -> str | None:
    ready, _, _ = select.select([sys.stdin], [], [], 0.0)
    if not ready:
        return None
    if stdin_is_tty:
        return sys.stdin.read(1).lower()
    return sys.stdin.readline()[:1].lower()


def _load_default_trajectory() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    trajectory = read_trajectory_csv(params.DEFAULT_TRAJECTORY_CSV)
    assert_valid_trajectory(trajectory)
    t, q, dq, ddq = trajectory_list_to_arrays(trajectory)
    if t.size < 2:
        raise ValueError("Need at least 2 commands to execute a trajectory")
    return t, q, dq, ddq


def _set_physics_timestep(stage, physics_hz: float) -> None:
    from pxr import PhysxSchema

    physics_hz = float(physics_hz)
    if physics_hz <= 0.0:
        raise ValueError(f"physics_hz must be > 0, got {physics_hz}")

    prim = stage.GetPrimAtPath(params.DEFAULT_SIM_PHYSICS_SCENE_PATH)
    if not (prim and prim.IsValid()):
        raise RuntimeError(f"physics scene not found at {params.DEFAULT_SIM_PHYSICS_SCENE_PATH}")
    scene = PhysxSchema.PhysxSceneAPI.Apply(prim)
    attr = scene.GetTimeStepsPerSecondAttr()
    if not attr:
        attr = scene.CreateTimeStepsPerSecondAttr()
    attr.Set(physics_hz)
    stage.SetTimeCodesPerSecond(physics_hz)
    actual_physics_hz = attr.Get()
    actual_time_codes = stage.GetTimeCodesPerSecond()
    print(
        f"[sim] physics timeStepsPerSecond requested={physics_hz:g} "
        f"actual={float(actual_physics_hz):g} timeCodesPerSecond={float(actual_time_codes):g}",
        flush=True,
    )


def _configure_timeline_timing(timeline, physics_hz: float) -> None:
    import carb.settings

    physics_hz = float(physics_hz)
    if physics_hz <= 0.0:
        raise ValueError(f"physics_hz must be > 0, got {physics_hz}")

    settings = carb.settings.acquire_settings_interface()
    settings.set(params.DEFAULT_SIM_USE_FIXED_TIME_STEPPING_SETTING, True)
    settings.set(params.DEFAULT_SIM_COMPENSATE_PLAY_DELAY_SETTING, float(params.DEFAULT_SIM_COMPENSATE_PLAY_DELAY_SEC))

    if timeline.is_playing():
        timeline.pause()
        timeline.commit()
    timeline.set_time_codes_per_second(physics_hz)
    timeline.set_ticks_per_frame(int(params.DEFAULT_SIM_TIMELINE_TICKS_PER_FRAME))
    timeline.set_play_every_frame(True)
    timeline.set_target_framerate(physics_hz)
    timeline.commit()
    print(
        f"[sim] timeline fixed_step={bool(settings.get(params.DEFAULT_SIM_USE_FIXED_TIME_STEPPING_SETTING))} "
        f"timeCodesPerSecond={float(timeline.get_time_codes_per_seconds()):g} "
        f"ticksPerFrame={int(timeline.get_ticks_per_frame())} "
        f"playEveryFrame={bool(timeline.get_play_every_frame())} "
        f"targetFramerate={float(timeline.get_target_framerate()):g}",
        flush=True,
    )


def _steps_for_seconds(seconds: float, physics_hz: float) -> int:
    return max(1, int(round(max(0.0, float(seconds)) * float(physics_hz))))


def _measure_update_frequency(simulation_app, timeline, *, steps: int, label: str) -> None:
    steps = int(steps)
    if steps <= 0:
        return
    sim_t0 = float(timeline.get_current_time())
    wall_t0 = time.perf_counter()
    for _ in range(steps):
        simulation_app.update()
    wall_elapsed = time.perf_counter() - wall_t0
    sim_elapsed = float(timeline.get_current_time()) - sim_t0
    sim_dt_per_update = sim_elapsed / float(steps) if steps > 0 else 0.0
    sim_update_hz = 1.0 / sim_dt_per_update if sim_dt_per_update > 0.0 else 0.0
    wall_update_hz = float(steps) / wall_elapsed if wall_elapsed > 0.0 else 0.0
    realtime_factor = sim_elapsed / wall_elapsed if wall_elapsed > 0.0 else 0.0
    print(
        f"[sim_freq] {label}: steps={steps} sim_elapsed={sim_elapsed:.9g}s "
        f"wall_elapsed={wall_elapsed:.9g}s sim_dt_per_update={sim_dt_per_update:.9g}s "
        f"sim_update_hz={sim_update_hz:.9g} wall_update_hz={wall_update_hz:.9g} "
        f"realtime_factor={realtime_factor:.9g}",
        flush=True,
    )


def main() -> int:
    if "SOFT_THROW_URDF" not in os.environ:
        _load_env_from_file(params.DEFAULT_ENV_SH)

    from core.throw_params import (
        ROBOT_USD,
        ROBOT_PRIM,
        FRANKA_ROOT,
    )

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=bool(params.DEFAULT_SIM_HEADLESS))
    simulation_app = app_launcher.app

    import omni.timeline
    from sim.isaac_scene import (
        add_robot_reference,
        detect_franka_root,
        ensure_physics_scene,
        get_stage,
        repair_lmm_asset_references,
        set_prim_translate,
        spawn_ground,
        spawn_lights,
    )
    from sim.isaac_robot_io import apply_arm_targets, find_joint_paths
    from sim.show_asset_sim import _find_ball_prim, _find_lacrosse_head_prim, _set_world_translation, _zero_rigid_body_velocity

    stage = get_stage()
    ensure_physics_scene(stage)
    _set_physics_timestep(stage, params.DEFAULT_SIM_PLAYBACK_PHYSICS_HZ)
    if bool(params.DEFAULT_SIM_SPAWN_GROUND):
        spawn_ground(stage, color=params.DEFAULT_SIM_GROUND_COLOR)
    spawn_lights(stage)
    add_robot_reference(stage, ROBOT_PRIM, ROBOT_USD)
    if abs(float(params.DEFAULT_SIM_ASSET_Z_OFFSET)) > 0.0:
        set_prim_translate(stage, ROBOT_PRIM, (0.0, 0.0, float(params.DEFAULT_SIM_ASSET_Z_OFFSET)), label="asset root")
    repair_lmm_asset_references(stage, ROBOT_PRIM)
    if bool(params.DEFAULT_LACROSSE_COLLIDER_ENABLED):
        ensure_lacrosse_sdf_collider(stage)

    timeline = omni.timeline.get_timeline_interface()
    _configure_timeline_timing(timeline, params.DEFAULT_SIM_PLAYBACK_PHYSICS_HZ)
    timeline.play()
    timeline.commit()

    _measure_update_frequency(
        simulation_app,
        timeline,
        steps=int(params.DEFAULT_SIM_FREQUENCY_MEASURE_STEPS),
        label="startup",
    )

    franka_root = detect_franka_root(stage, FRANKA_ROOT)
    joint_paths = find_joint_paths(stage, franka_root)
    physics_hz = float(params.DEFAULT_SIM_PLAYBACK_PHYSICS_HZ)

    def get_lacrosse_world_translation() -> np.ndarray | None:
        from pxr import Usd, UsdGeom, Gf

        head = _find_lacrosse_head_prim(stage, params.DEFAULT_SIM_HEAD_PRIM)
        if head is None:
            return None
        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        matrix = cache.GetLocalToWorldTransform(head)
        point = matrix.Transform(Gf.Vec3d(0.0, 0.0, 0.0))
        return np.array([point[0], point[1], point[2]], dtype=float)

    def reset_ball_above_lacrosse_origin(offset_m: float) -> bool:
        ball = _find_ball_prim(stage, params.DEFAULT_SIM_BALL_PRIM)
        if ball is None:
            print("[reset] ball prim not found", flush=True)
            return False
        pos = get_lacrosse_world_translation()
        if pos is None:
            print("[reset] lacrosse head prim not found", flush=True)
            return False
        target = (float(pos[0]), float(pos[1]), float(pos[2]) + float(offset_m))
        _set_world_translation(ball, target)
        _zero_rigid_body_velocity(ball)
        print(
            f"[reset] moved {ball.GetPath()} to lacrosse origin + z offset: "
            f"head={np.round(pos, 6).tolist()} target={target}",
            flush=True,
        )
        return True

    def hold_initial_pose(q_cmd: np.ndarray, seconds: float, *, label: str) -> None:
        steps = _steps_for_seconds(seconds, physics_hz)
        print(f"[reset] {label}: seconds={float(seconds):.6g} steps={steps}", flush=True)
        for _ in range(steps):
            apply_arm_targets(stage, joint_paths, q_cmd, np.zeros(params.TRAJECTORY_DOF, dtype=float))
            simulation_app.update()

    def command_initial_pose_until_lacrosse_stable(q_cmd: np.ndarray) -> None:
        timeout_sec = float(params.DEFAULT_SIM_RESET_SETTLE_SEC)
        min_sec = float(params.DEFAULT_SIM_RESET_MIN_SETTLE_SEC)
        stable_eps = float(params.DEFAULT_SIM_RESET_STABLE_POS_EPS)
        stable_needed = int(params.DEFAULT_SIM_RESET_STABLE_STEPS)

        t_start = time.perf_counter()
        last_pos = None
        stable_count = 0
        while time.perf_counter() - t_start < timeout_sec:
            apply_arm_targets(stage, joint_paths, q_cmd, np.zeros(params.TRAJECTORY_DOF, dtype=float))
            simulation_app.update()
            pos = get_lacrosse_world_translation()
            elapsed = time.perf_counter() - t_start
            if pos is None:
                continue
            if last_pos is not None and elapsed >= min_sec:
                delta = float(np.linalg.norm(pos - last_pos))
                if delta <= stable_eps:
                    stable_count += 1
                    if stable_count >= stable_needed:
                        print(
                            f"[reset] lacrosse stable after {elapsed:.3f}s pos={np.round(pos, 6).tolist()}",
                            flush=True,
                        )
                        return
                else:
                    stable_count = 0
            last_pos = pos

        pos = get_lacrosse_world_translation()
        if pos is None:
            print(f"[reset] lacrosse pose unavailable after {timeout_sec:.3f}s settle", flush=True)
        else:
            print(
                f"[reset] settle timeout {timeout_sec:.3f}s; using current lacrosse pos={np.round(pos, 6).tolist()}",
                flush=True,
            )

    def execute_once(pass_index: int) -> None:
        print(f"[reload] pass={pass_index} reading {params.DEFAULT_TRAJECTORY_CSV}", flush=True)
        t, q, dq, _ddq = _load_default_trajectory()
        print(
            f"[ready] pass={pass_index} commands={len(t)} duration={float(t[-1] - t[0]):.6g}s "
            f"physics_hz={physics_hz:g}",
            flush=True,
        )

        q0 = np.asarray(q[0], dtype=float).reshape(params.TRAJECTORY_DOF)
        print(f"[reset] commanding first trajectory pose: {np.round(q0, 6).tolist()}", flush=True)
        command_initial_pose_until_lacrosse_stable(q0)
        hold_initial_pose(
            q0,
            float(params.DEFAULT_SIM_WAIT_BEFORE_BALL_RESET_SEC),
            label="waiting at first trajectory pose before ball reset",
        )
        reset_ball_above_lacrosse_origin(float(params.DEFAULT_SIM_BALL_OFFSET))

        print(f"[play] pass={pass_index}", flush=True)
        t0 = float(t[0])
        current_step = 0
        progress_stride = max(1, len(t) // 10)
        for k in range(len(t)):
            if k % progress_stride == 0:
                print(f"[play] pass={pass_index} command={k}/{len(t)} sim_t={float(t[k] - t0):.6g}s", flush=True)
            apply_arm_targets(stage, joint_paths, q[k], dq[k])
            if k + 1 >= len(t):
                break
            next_step = int(round(max(0.0, float(t[k + 1] - t0)) * physics_hz))
            if next_step <= current_step:
                continue
            for _ in range(next_step - current_step):
                simulation_app.update()
            current_step = next_step

        print(f"[hold] pass={pass_index} seconds={float(params.DEFAULT_SIM_HOLD_AFTER_SEC):.6g}", flush=True)
        n_hold = _steps_for_seconds(params.DEFAULT_SIM_HOLD_AFTER_SEC, physics_hz)
        for _ in range(n_hold):
            apply_arm_targets(stage, joint_paths, q[-1], np.zeros(params.TRAJECTORY_DOF, dtype=float))
            simulation_app.update()
        print(f"[done] pass={pass_index}", flush=True)

    print("[sim] Press Enter/Space to reload+execute trajectory, q to quit.", flush=True)
    stdin_is_tty = bool(sys.stdin.isatty())
    old_settings = None
    fd = sys.stdin.fileno()
    if stdin_is_tty:
        old_settings = termios.tcgetattr(fd)
        tty.setcbreak(fd)

    pass_index = 1
    try:
        while True:
            key = _read_key_nonblocking(stdin_is_tty)
            if key in ("\r", "\n", " "):
                try:
                    execute_once(pass_index)
                    pass_index += 1
                except Exception as exc:
                    print(f"[error] pass={pass_index} failed: {exc}", flush=True)
                print("[sim] Press Enter/Space to reload+execute trajectory, q to quit.", flush=True)
            elif key == "q":
                print("[sim] quit", flush=True)
                return 0
            simulation_app.update()
            time.sleep(float(params.DEFAULT_SIM_IDLE_SLEEP_SEC))
    except KeyboardInterrupt:
        return 0
    finally:
        if old_settings is not None:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


if __name__ == "__main__":
    raise SystemExit(main())
