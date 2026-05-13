#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Generate ballistic release candidates from a desired landing point.

This module does not decide the final strategy yet. It builds a configurable
family of parabolas that all pass through the requested landing point. The
release side is biased along the direction from the landing point back toward
the current initial ball/lacrosse-head position, but it does not require the
parabola to pass through that initial point.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import re
import sys
from pathlib import Path
from typing import Callable, Iterable

import numpy as np


_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from refractor.config import params

os.environ.setdefault("MPLCONFIGDIR", params.DEFAULT_MPLCONFIGDIR)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


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


if "SOFT_THROW_URDF" not in os.environ:
    _load_env_from_file(params.DEFAULT_ENV_SH)

from core.ik_poly5_core import solve_ik_for_q_goal
from core.ik_scurve_core import stopping_distance_jerk_limited
from core.kinematic_forward import PinKinematics, pin


@dataclass(frozen=True)
class ReleaseCandidate:
    """One release state and its ballistic path."""

    release_pos: np.ndarray
    release_vel: np.ndarray
    landing_pos: np.ndarray
    flight_time: float
    speed: float
    path: np.ndarray

    def as_dict(self) -> dict:
        return {
            "release_pos": self.release_pos.copy(),
            "release_vel": self.release_vel.copy(),
            "landing_pos": self.landing_pos.copy(),
            "flight_time": float(self.flight_time),
            "speed": float(self.speed),
            "path": self.path.copy(),
        }


@dataclass(frozen=True)
class ReachableReleaseSelection:
    """Selected release candidate plus the joint state found by IK."""

    candidate: ReleaseCandidate
    q_release: np.ndarray
    dq_release: np.ndarray
    pos_error: float
    vel_error: float
    q_distance: float
    score: float


def ballistic_velocity_to_land(
    release_pos: np.ndarray,
    landing_pos: np.ndarray,
    flight_time: float,
    *,
    gravity: float = params.DEFAULT_RELEASE_GRAVITY,
) -> np.ndarray:
    """Return release velocity that reaches ``landing_pos`` after ``flight_time``."""
    p0 = np.asarray(release_pos, dtype=float).reshape(3)
    p1 = np.asarray(landing_pos, dtype=float).reshape(3)
    t = float(flight_time)
    if t <= 0.0:
        raise ValueError(f"flight_time must be positive, got {t}")
    return np.array(
        [
            (p1[0] - p0[0]) / t,
            (p1[1] - p0[1]) / t,
            (p1[2] - p0[2] + 0.5 * float(gravity) * t * t) / t,
        ],
        dtype=float,
    )


def sample_ballistic_path(
    release_pos: np.ndarray,
    release_vel: np.ndarray,
    flight_time: float,
    *,
    gravity: float = params.DEFAULT_RELEASE_GRAVITY,
    sample_count: int = params.DEFAULT_RELEASE_PATH_SAMPLE_COUNT,
) -> np.ndarray:
    """Sample the release-to-landing parabola as Nx3 world positions."""
    p0 = np.asarray(release_pos, dtype=float).reshape(3)
    v0 = np.asarray(release_vel, dtype=float).reshape(3)
    n = max(2, int(sample_count))
    t = np.linspace(0.0, float(flight_time), n)
    path = p0[None, :] + t[:, None] * v0[None, :]
    path[:, 2] -= 0.5 * float(gravity) * t * t
    return path


def _initial_side_direction(landing_pos: np.ndarray, initial_pos: np.ndarray) -> np.ndarray:
    landing = np.asarray(landing_pos, dtype=float).reshape(3)
    initial = np.asarray(initial_pos, dtype=float).reshape(3)
    direction = initial - landing
    direction[2] = 0.0
    norm = float(np.linalg.norm(direction))
    if norm < 1e-9:
        raise ValueError("initial_pos and landing_pos must differ in x/y to define release-side direction")
    return direction / norm


def _grid_values(value_range: tuple[float, float], count: int) -> np.ndarray:
    lo, hi = map(float, value_range)
    if count <= 1:
        return np.array([0.5 * (lo + hi)], dtype=float)
    return np.linspace(lo, hi, int(count), dtype=float)


def generate_release_candidates(
    *,
    landing_pos: np.ndarray | None = None,
    initial_pos: np.ndarray | None = None,
    candidate_count: int | None = None,
    distance_range: tuple[float, float] = params.DEFAULT_RELEASE_DISTANCE_RANGE,
    height_offset_range: tuple[float, float] = params.DEFAULT_RELEASE_HEIGHT_OFFSET_RANGE,
    flight_time_range: tuple[float, float] = params.DEFAULT_RELEASE_FLIGHT_TIME_RANGE,
    max_speed: float = params.DEFAULT_RELEASE_MAX_SPEED,
    min_height: float = params.DEFAULT_RELEASE_MIN_HEIGHT,
    max_height: float = params.DEFAULT_RELEASE_MAX_HEIGHT,
    gravity: float = params.DEFAULT_RELEASE_GRAVITY,
    path_sample_count: int = params.DEFAULT_RELEASE_PATH_SAMPLE_COUNT,
) -> list[ReleaseCandidate]:
    """Generate feasible release position/velocity candidates.

    Every candidate path passes through ``landing_pos``. Release positions are
    placed on the side of the landing point that points back toward
    ``initial_pos``. The caller can later replace ``select_release_candidate``
    with a real scoring function.
    """
    landing = np.asarray(params.DEFAULT_RELEASE_LANDING_POS if landing_pos is None else landing_pos, dtype=float).reshape(3)
    initial = np.asarray(params.DEFAULT_RELEASE_INITIAL_POS if initial_pos is None else initial_pos, dtype=float).reshape(3)
    requested = int(params.DEFAULT_RELEASE_CANDIDATE_COUNT if candidate_count is None else candidate_count)
    if requested <= 0:
        raise ValueError(f"candidate_count must be positive, got {requested}")

    direction = _initial_side_direction(landing, initial)
    candidates: list[ReleaseCandidate] = []
    base_grid_count = max(2, int(np.ceil(requested ** (1.0 / 3.0))) + 1)
    for grid_count in range(base_grid_count, base_grid_count + 20):
        candidates.clear()
        distances = _grid_values(distance_range, grid_count)
        height_offsets = _grid_values(height_offset_range, grid_count)
        flight_times = _grid_values(flight_time_range, grid_count)

        for distance in distances:
            xy = landing[:2] + direction[:2] * float(distance)
            for height_offset in height_offsets:
                z = float(landing[2] + height_offset)
                if z < float(min_height) or z > float(max_height):
                    continue
                release_pos = np.array([xy[0], xy[1], z], dtype=float)
                for flight_time in flight_times:
                    release_vel = ballistic_velocity_to_land(
                        release_pos,
                        landing,
                        float(flight_time),
                        gravity=gravity,
                    )
                    speed = float(np.linalg.norm(release_vel))
                    if speed > float(max_speed):
                        continue
                    path = sample_ballistic_path(
                        release_pos,
                        release_vel,
                        float(flight_time),
                        gravity=gravity,
                        sample_count=path_sample_count,
                    )
                    if not np.allclose(path[-1], landing, atol=1e-9):
                        continue
                    candidates.append(
                        ReleaseCandidate(
                            release_pos=release_pos,
                            release_vel=release_vel,
                            landing_pos=landing.copy(),
                            flight_time=float(flight_time),
                            speed=speed,
                            path=path,
                        )
                    )
        if len(candidates) >= requested:
            break

    candidates.sort(key=lambda c: (c.speed, c.flight_time, float(np.linalg.norm(c.release_pos - initial))))
    return candidates[:requested]


def select_release_candidate(
    candidates: Iterable[ReleaseCandidate],
    *,
    selector: Callable[[list[ReleaseCandidate]], ReleaseCandidate] | None = None,
) -> ReleaseCandidate:
    """Select one candidate.

    ``selector`` is the intended future scoring hook. The current default is a
    deterministic placeholder: pick the middle item from the generated list.
    """
    items = list(candidates)
    if not items:
        raise ValueError("no release candidates available")
    if selector is not None:
        return selector(items)
    return items[len(items) // 2]


def _arm_jacobian_point_world(
    pin_model: PinKinematics,
    q7: np.ndarray,
    frame_name: str,
    object_offset: np.ndarray | None,
) -> np.ndarray:
    q7 = np.asarray(q7, dtype=float).reshape(params.TRAJECTORY_DOF)
    q_full = pin_model.make_q_full_from_arm7(q7)
    pin_model._pin_forward(q_full)
    frame_id = pin_model.model.getFrameId(frame_name)
    if frame_id == len(pin_model.model.frames):
        raise RuntimeError(f"pin frame not found: {frame_name}")
    transform = pin_model.data.oMf[frame_id]
    offset_local = np.zeros(3, dtype=float) if object_offset is None else np.asarray(object_offset, dtype=float).reshape(3)
    offset_world = np.array(transform.rotation, dtype=float) @ offset_local
    J = pin.computeFrameJacobian(pin_model.model, pin_model.data, q_full, frame_id, pin.ReferenceFrame.WORLD)
    J_arm = np.zeros((3, params.TRAJECTORY_DOF), dtype=float)
    for i in range(1, params.TRAJECTORY_DOF + 1):
        jid = pin_model.model.getJointId(params.DEFAULT_PIN_ARM_JOINT_NAME_FORMAT.format(i))
        idx_v = int(pin_model.model.joints[jid].idx_v)
        omega_col = J[:3, idx_v]
        vel_col = J[3:6, idx_v]
        J_arm[:, i - 1] = vel_col + np.cross(omega_col, offset_world)
    return J_arm


def _frame_state_with_object_offset(
    pin_model: PinKinematics,
    q7: np.ndarray,
    dq7: np.ndarray,
    frame_name: str,
    object_offset: np.ndarray | None,
    object_offset_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    state = pin_model.get_frame_world_state(q7, frame_name, dq7)
    if object_offset is None or object_offset_mode == "none":
        return state["pos_w"], state["vel_w"]
    if object_offset_mode == "world":
        return state["pos_w"] + object_offset, state["vel_w"]
    if object_offset_mode in ("local", "local_from_initial"):
        offset_world = state["R_w"] @ object_offset
        return state["pos_w"] + offset_world, state["vel_w"] + np.cross(state["omega_w"], offset_world)
    raise ValueError(f"unknown object_offset_mode: {object_offset_mode!r}")


def _qdot_for_point_velocity(
    pin_model: PinKinematics,
    q7: np.ndarray,
    frame_name: str,
    velocity_world: np.ndarray,
    object_offset: np.ndarray | None,
    object_offset_mode: str,
) -> np.ndarray:
    J_offset = object_offset if object_offset_mode in ("local", "local_from_initial") else None
    J = _arm_jacobian_point_world(pin_model, q7, frame_name, J_offset)
    v = np.asarray(velocity_world, dtype=float).reshape(3)
    lam = float(params.DEFAULT_RELEASE_SELECTOR_DLS_LAMBDA)
    inv = np.linalg.inv(J @ J.T + (lam * lam) * np.eye(3))
    return (J.T @ inv @ v).astype(float).reshape(params.TRAJECTORY_DOF)


def _clamp_qdot_for_decel_feasibility(
    q7: np.ndarray,
    dq7: np.ndarray,
    *,
    q_limits: np.ndarray,
    ddq_limits: np.ndarray,
    jerk_limits: np.ndarray,
    margin: float,
) -> np.ndarray:
    q = np.asarray(q7, dtype=float).reshape(params.TRAJECTORY_DOF)
    dq = np.asarray(dq7, dtype=float).reshape(params.TRAJECTORY_DOF).copy()
    q_limits = np.asarray(q_limits, dtype=float).reshape(params.TRAJECTORY_DOF, 2)
    ddq_limits = np.asarray(ddq_limits, dtype=float).reshape(params.TRAJECTORY_DOF)
    jerk_limits = np.asarray(jerk_limits, dtype=float).reshape(params.TRAJECTORY_DOF)
    margin = max(0.0, float(margin))

    for joint in range(params.TRAJECTORY_DOF):
        v = float(dq[joint])
        if abs(v) <= 1e-12:
            continue
        available = float(q_limits[joint, 1] - q[joint]) if v > 0.0 else float(q[joint] - q_limits[joint, 0])
        available = max(0.0, available - margin)
        if available <= 1e-12:
            dq[joint] = 0.0
            continue
        stop_distance = stopping_distance_jerk_limited(
            abs(v),
            amax=float(ddq_limits[joint]),
            jmax=float(jerk_limits[joint]),
        )
        if stop_distance <= available + 1e-12:
            continue
        lo = 0.0
        hi = abs(v)
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            d_mid = stopping_distance_jerk_limited(
                mid,
                amax=float(ddq_limits[joint]),
                jmax=float(jerk_limits[joint]),
            )
            if d_mid <= available:
                lo = mid
            else:
                hi = mid
        dq[joint] = float(np.sign(v) * lo)
    return dq


def select_reachable_release_candidate(
    candidates: Iterable[ReleaseCandidate],
    *,
    reference_q: np.ndarray | None = None,
    pin_model: PinKinematics | None = None,
    frame_name: str = params.DEFAULT_RELEASE_SELECTOR_FRAME,
    object_offset: np.ndarray | None = params.DEFAULT_RELEASE_SELECTOR_OBJECT_OFFSET,
    object_offset_mode: str = params.DEFAULT_RELEASE_SELECTOR_OBJECT_OFFSET_MODE,
    q_limits: np.ndarray = params.Q_LIMITS_7,
    dq_limits: np.ndarray = params.QDOT_LIMITS_7,
    ddq_limits: np.ndarray = params.QDDOT_LIMITS_7,
    jerk_limits: np.ndarray = params.JERK_LIMITS_7,
    pos_tol: float = params.DEFAULT_RELEASE_SELECTOR_POS_TOL,
    vel_tol: float = params.DEFAULT_RELEASE_SELECTOR_VEL_TOL,
    q_weight: float = params.DEFAULT_RELEASE_SELECTOR_Q_WEIGHT,
    pos_weight: float = params.DEFAULT_RELEASE_SELECTOR_POS_WEIGHT,
    vel_weight: float = params.DEFAULT_RELEASE_SELECTOR_VEL_WEIGHT,
    ik_max_iter: int = params.DEFAULT_RELEASE_SELECTOR_IK_MAX_ITER,
    require_decel_feasible: bool = params.DEFAULT_RELEASE_SELECTOR_REQUIRE_DECEL_FEASIBLE,
    decel_margin: float = params.DEFAULT_RELEASE_SELECTOR_DECEL_MARGIN,
) -> ReachableReleaseSelection:
    """Select a candidate reachable near ``reference_q``.

    This searches over the already generated ballistic candidates. For each
    candidate it solves IK for the requested release position and velocity,
    checks the achieved frame position/velocity, and selects the lowest score.
    The world release position is allowed to differ from the user's preferred
    joint configuration; closeness is measured in joint space.
    """
    items = list(candidates)
    if not items:
        raise ValueError("no release candidates available")

    q_ref = np.asarray(params.DEFAULT_RELEASE_REFERENCE_Q if reference_q is None else reference_q, dtype=float).reshape(params.TRAJECTORY_DOF)
    object_offset_mode = str(object_offset_mode)
    object_offset = None if object_offset is None else np.asarray(object_offset, dtype=float).reshape(3)
    q_limits = np.asarray(q_limits, dtype=float).reshape(params.TRAJECTORY_DOF, 2)
    dq_limits = np.asarray(dq_limits, dtype=float).reshape(params.TRAJECTORY_DOF)
    ddq_limits = np.asarray(ddq_limits, dtype=float).reshape(params.TRAJECTORY_DOF)
    jerk_limits = np.asarray(jerk_limits, dtype=float).reshape(params.TRAJECTORY_DOF)
    pin_model = PinKinematics() if pin_model is None else pin_model
    frame_id = pin_model.model.getFrameId(frame_name)
    if frame_id == len(pin_model.model.frames):
        raise RuntimeError(f"pin frame not found: {frame_name}")

    pos_ref, r_des = pin_model.get_frame_world_pose(q_ref, frame_name)
    if object_offset is None and object_offset_mode == "world":
        object_offset = np.asarray(params.DEFAULT_RELEASE_INITIAL_POS, dtype=float).reshape(3) - pos_ref
    elif object_offset is None and object_offset_mode == "local_from_initial":
        object_offset = r_des.T @ (np.asarray(params.DEFAULT_RELEASE_INITIAL_POS, dtype=float).reshape(3) - pos_ref)
    best: ReachableReleaseSelection | None = None
    fallback: ReachableReleaseSelection | None = None

    for candidate in items:
        try:
            target_frame_pos = candidate.release_pos.copy()
            if object_offset is not None and object_offset_mode == "world":
                target_frame_pos = candidate.release_pos - object_offset
            elif object_offset is not None and object_offset_mode in ("local", "local_from_initial"):
                target_frame_pos = candidate.release_pos - r_des @ object_offset
            q_release, _dq_ik = solve_ik_for_q_goal(
                pin_model,
                frame_id,
                q_ref,
                target_frame_pos,
                r_des,
                q_limits,
                dq_limits,
                ddq_limits,
                control_dt=float(params.DEFAULT_SOFT_THROW_CONTROL_DT),
                max_iter=int(ik_max_iter),
                kp_pos=float(params.DEFAULT_SOFT_THROW_IK_KP_POS),
                kp_rot=float(params.DEFAULT_SOFT_THROW_IK_KP_ROT),
                v_j7_des=candidate.release_vel,
            )
            dq_release = _qdot_for_point_velocity(
                pin_model,
                q_release,
                frame_name,
                candidate.release_vel,
                object_offset,
                object_offset_mode,
            )
            if bool(require_decel_feasible):
                dq_release = _clamp_qdot_for_decel_feasibility(
                    q_release,
                    dq_release,
                    q_limits=q_limits,
                    ddq_limits=ddq_limits,
                    jerk_limits=jerk_limits,
                    margin=float(decel_margin),
                )
        except Exception:
            continue

        if np.any(q_release < q_limits[:, 0]) or np.any(q_release > q_limits[:, 1]):
            continue
        if np.any(np.abs(dq_release) > dq_limits):
            continue

        pos_actual, vel_actual = _frame_state_with_object_offset(
            pin_model,
            q_release,
            dq_release,
            frame_name,
            object_offset,
            object_offset_mode,
        )
        pos_error = float(np.linalg.norm(pos_actual - candidate.release_pos))
        vel_error = float(np.linalg.norm(vel_actual - candidate.release_vel))
        q_distance = float(np.linalg.norm(q_release - q_ref))
        score = float(q_weight) * q_distance + float(pos_weight) * pos_error + float(vel_weight) * vel_error
        result = ReachableReleaseSelection(
            candidate=candidate,
            q_release=q_release.astype(float),
            dq_release=dq_release.astype(float),
            pos_error=pos_error,
            vel_error=vel_error,
            q_distance=q_distance,
            score=score,
        )
        if fallback is None or result.score < fallback.score:
            fallback = result
        if pos_error > float(pos_tol) or vel_error > float(vel_tol):
            continue
        if best is None or result.score < best.score:
            best = result

    if best is not None:
        return best
    if fallback is not None:
        return fallback
    raise ValueError("no candidate passed IK/joint-limit checks")


def make_reachable_selector(
    *,
    reference_q: np.ndarray | None = None,
    pin_model: PinKinematics | None = None,
    frame_name: str = params.DEFAULT_RELEASE_SELECTOR_FRAME,
    object_offset: np.ndarray | None = params.DEFAULT_RELEASE_SELECTOR_OBJECT_OFFSET,
    object_offset_mode: str = params.DEFAULT_RELEASE_SELECTOR_OBJECT_OFFSET_MODE,
    require_decel_feasible: bool = params.DEFAULT_RELEASE_SELECTOR_REQUIRE_DECEL_FEASIBLE,
) -> Callable[[list[ReleaseCandidate]], ReleaseCandidate]:
    """Return a selector hook compatible with ``select_release_candidate``."""

    def _selector(candidates: list[ReleaseCandidate]) -> ReleaseCandidate:
        return select_reachable_release_candidate(
            candidates,
            reference_q=reference_q,
            pin_model=pin_model,
            frame_name=frame_name,
            object_offset=object_offset,
            object_offset_mode=object_offset_mode,
            require_decel_feasible=require_decel_feasible,
        ).candidate

    return _selector


def compute_release_candidate_set(
    *,
    landing_pos: np.ndarray | None = None,
    initial_pos: np.ndarray | None = None,
    candidate_count: int | None = None,
    selector: Callable[[list[ReleaseCandidate]], ReleaseCandidate] | None = None,
) -> tuple[list[ReleaseCandidate], ReleaseCandidate]:
    """Generate candidates and return ``(all_candidates, selected_candidate)``."""
    candidates = generate_release_candidates(
        landing_pos=landing_pos,
        initial_pos=initial_pos,
        candidate_count=candidate_count,
    )
    selected = select_release_candidate(candidates, selector=selector)
    return candidates, selected


def plot_release_candidates(
    candidates: Iterable[ReleaseCandidate],
    selected: ReleaseCandidate | None = None,
    *,
    initial_pos: np.ndarray | None = None,
    output_path: str | Path | None = None,
    max_paths: int = params.DEFAULT_RELEASE_PLOT_MAX_PATHS,
) -> Path:
    """Plot candidate ballistic paths and save a 3D preview image."""
    items = list(candidates)
    if not items:
        raise ValueError("no release candidates to plot")

    initial = np.asarray(params.DEFAULT_RELEASE_INITIAL_POS if initial_pos is None else initial_pos, dtype=float).reshape(3)
    selected = selected if selected is not None else select_release_candidate(items)
    output = Path(params.DEFAULT_RELEASE_PLOT_PATH if output_path is None else output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    max_paths = max(1, int(max_paths))
    if len(items) <= max_paths:
        plotted = items
    else:
        indices = np.linspace(0, len(items) - 1, max_paths).round().astype(int)
        plotted = [items[i] for i in indices]

    fig = plt.figure(figsize=params.DEFAULT_RELEASE_PLOT_FIGSIZE)
    ax = fig.add_subplot(111, projection="3d")

    speeds = np.array([c.speed for c in plotted], dtype=float)
    speed_min = float(np.min(speeds))
    speed_span = float(np.max(speeds) - speed_min)
    cmap = plt.get_cmap(params.DEFAULT_RELEASE_PLOT_SPEED_CMAP)

    for candidate in plotted:
        path = np.asarray(candidate.path, dtype=float)
        color_scale = 0.0 if speed_span < 1e-12 else (candidate.speed - speed_min) / speed_span
        ax.plot(
            path[:, 0],
            path[:, 1],
            path[:, 2],
            color=cmap(color_scale),
            linewidth=float(params.DEFAULT_RELEASE_PLOT_PATH_LINEWIDTH),
            alpha=float(params.DEFAULT_RELEASE_PLOT_PATH_ALPHA),
        )

    release_points = np.array([c.release_pos for c in items], dtype=float)
    ax.scatter(
        release_points[:, 0],
        release_points[:, 1],
        release_points[:, 2],
        s=float(params.DEFAULT_RELEASE_PLOT_CANDIDATE_SIZE),
        c="tab:blue",
        alpha=float(params.DEFAULT_RELEASE_PLOT_CANDIDATE_ALPHA),
        label="release candidates",
    )
    ax.scatter(
        selected.release_pos[0],
        selected.release_pos[1],
        selected.release_pos[2],
        s=float(params.DEFAULT_RELEASE_PLOT_SELECTED_SIZE),
        c="tab:red",
        marker="o",
        label="selected release",
    )
    landing = selected.landing_pos
    ax.scatter(landing[0], landing[1], landing[2], s=float(params.DEFAULT_RELEASE_PLOT_LANDING_SIZE), c="black", marker="x", label="landing")
    ax.scatter(
        initial[0],
        initial[1],
        initial[2],
        s=float(params.DEFAULT_RELEASE_PLOT_INITIAL_SIZE),
        c="tab:orange",
        marker="^",
        label="initial reference",
    )
    ax.quiver(
        selected.release_pos[0],
        selected.release_pos[1],
        selected.release_pos[2],
        selected.release_vel[0],
        selected.release_vel[1],
        selected.release_vel[2],
        length=float(params.DEFAULT_RELEASE_PLOT_ARROW_LENGTH),
        normalize=True,
        color="tab:red",
    )

    all_points = np.vstack([release_points, landing.reshape(1, 3), initial.reshape(1, 3)])
    for candidate in plotted:
        all_points = np.vstack([all_points, candidate.path])
    mins = np.min(all_points, axis=0)
    maxs = np.max(all_points, axis=0)
    center = 0.5 * (mins + maxs)
    radius = float(params.DEFAULT_RELEASE_PLOT_RADIUS_SCALE) * float(np.max(maxs - mins))
    radius = max(radius, float(params.DEFAULT_RELEASE_PLOT_MIN_RADIUS))
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(max(0.0, center[2] - radius), center[2] + radius)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title("Ballistic release candidates")
    ax.legend(loc="upper right")
    ax.view_init(elev=float(params.DEFAULT_RELEASE_PLOT_VIEW_ELEV), azim=float(params.DEFAULT_RELEASE_PLOT_VIEW_AZIM))
    fig.tight_layout()
    fig.savefig(output, dpi=int(params.DEFAULT_RELEASE_PLOT_DPI))
    plt.close(fig)
    return output


def main() -> int:
    candidates = generate_release_candidates()
    reachable_selection = None
    if bool(params.DEFAULT_RELEASE_SELECTOR_USE_REACHABLE):
        reachable_selection = select_reachable_release_candidate(candidates)
        selected = reachable_selection.candidate
    else:
        selected = select_release_candidate(candidates)
    print(f"[release] candidates={len(candidates)}")
    print(f"[release] selected_pos={np.round(selected.release_pos, 6).tolist()}")
    print(f"[release] selected_vel={np.round(selected.release_vel, 6).tolist()}")
    print(f"[release] selected_flight_time={selected.flight_time:.6g}s speed={selected.speed:.6g}m/s")
    if reachable_selection is not None:
        print(f"[release] selected_q={np.round(reachable_selection.q_release, 6).tolist()}")
        print(f"[release] selected_dq={np.round(reachable_selection.dq_release, 6).tolist()}")
        print(
            f"[release] selector_score={reachable_selection.score:.6g} "
            f"pos_error={reachable_selection.pos_error:.6g} "
            f"vel_error={reachable_selection.vel_error:.6g} "
            f"q_distance={reachable_selection.q_distance:.6g}"
        )
    if bool(params.DEFAULT_RELEASE_PLOT):
        plot_path = plot_release_candidates(candidates, selected)
        print(f"[release] plot={plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
