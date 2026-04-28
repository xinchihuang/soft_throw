#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Tube-acceleration helper methods for release-state generation."""

from __future__ import annotations

import numpy as np

from .throw_params import (
    BALLISTIC_T_MIN,
    BALLISTIC_T_MAX,
    BALLISTIC_T_STEP,
    EE_VEL_MAX,
)

G = 9.81


def landing_position_from_release_state(release_point, release_velocity, target_z=0.0, g=G):
    p0 = np.asarray(release_point, dtype=float)
    v0 = np.asarray(release_velocity, dtype=float)

    a = -0.5 * g
    b = v0[2]
    c = p0[2] - target_z

    roots = np.roots([a, b, c])
    roots = roots[np.isreal(roots)].real
    roots = roots[roots > 1e-8]
    if len(roots) == 0:
        return None, None

    t_hit = np.max(roots)
    p_land = np.array([
        p0[0] + v0[0] * t_hit,
        p0[1] + v0[1] * t_hit,
        target_z,
    ])
    return p_land, t_hit


def simulate_ballistic_trajectory(release_point, release_velocity, target_z=0.0, g=G, n_pts=120):
    p_land, t_hit = landing_position_from_release_state(
        release_point, release_velocity, target_z=target_z, g=g
    )
    if p_land is None:
        return None

    p0 = np.asarray(release_point, dtype=float)
    v0 = np.asarray(release_velocity, dtype=float)

    ts = np.linspace(0.0, t_hit, n_pts)
    x = p0[0] + v0[0] * ts
    y = p0[1] + v0[1] * ts
    z = p0[2] + v0[2] * ts - 0.5 * g * ts**2

    return np.vstack([x, y, z]).T


def velocity_from_target_release_and_time(target_point, release_point, flight_time, g=G):
    target_point = np.asarray(target_point, dtype=float)
    release_point = np.asarray(release_point, dtype=float)
    T = float(flight_time)

    vx = (target_point[0] - release_point[0]) / T
    vy = (target_point[1] - release_point[1]) / T
    vz = (target_point[2] - release_point[2] + 0.5 * g * T**2) / T
    return np.array([vx, vy, vz], dtype=float)


def solve_ballistic_velocity(
    release_point: np.ndarray,
    target_point: np.ndarray,
    *,
    t_min: float = BALLISTIC_T_MIN,
    t_max: float = BALLISTIC_T_MAX,
    t_step: float = BALLISTIC_T_STEP,
    ee_vel_max: float = EE_VEL_MAX,
    g: float = G,
) -> tuple[np.ndarray, float]:
    """Solve a feasible release velocity for a ballistic flight from release_point to target_point.

    Uses a simple grid search over flight time and selects the lowest-cost candidate.
    """
    p_r = np.asarray(release_point, dtype=float).reshape(3)
    p_t = np.asarray(target_point, dtype=float).reshape(3)
    t_min = float(t_min)
    t_max = float(t_max)
    t_step = float(t_step)
    ee_vel_max = float(ee_vel_max)

    if not (t_min > 0.0 and t_max >= t_min and t_step > 0.0):
        raise ValueError(f"Invalid time range: t_min={t_min} t_max={t_max} t_step={t_step}")

    best: tuple[float, np.ndarray, float] | None = None  # (cost, v, T)
    for T in np.arange(t_min, t_max + 1e-9, t_step):
        v = velocity_from_target_release_and_time(p_t, p_r, float(T), g=g)
        speed = float(np.linalg.norm(v))
        if speed < 1e-9:
            continue
        if speed > ee_vel_max * 0.98:
            continue
        cost = speed + 0.2 * float(T)
        if best is None or cost < best[0]:
            best = (cost, v, float(T))

    if best is None:
        # Fallback: pick a nominal time and clamp to speed limit.
        T = 0.45
        v = velocity_from_target_release_and_time(p_t, p_r, T, g=g)
        speed = float(np.linalg.norm(v))
        if speed > ee_vel_max and speed > 1e-9:
            v = v * (ee_vel_max / speed)
        return v.astype(float), float(T)

    return best[1].astype(float), float(best[2])


def generate_middle_trajectory_method(
    target_point,
    release_point,
    flight_time_range=(0.45, 0.85),
    n_family=21,
    g=G,
    n_pts=120,
):
    T_min, T_max = flight_time_range
    T_list = np.linspace(T_min, T_max, n_family)

    valid_family = []
    valid_velocities = []

    for T in T_list:
        v = velocity_from_target_release_and_time(target_point, release_point, T, g=g)
        traj = simulate_ballistic_trajectory(
            release_point, v, target_z=target_point[2], g=g, n_pts=n_pts
        )
        valid_family.append(traj)
        valid_velocities.append(v)

    idx = len(T_list) // 2
    selected_T = T_list[idx]
    selected_v = valid_velocities[idx]
    selected_traj = valid_family[idx]

    return {
        "method_name": "Middle-time method",
        "family_times": T_list,
        "family_trajs": valid_family,
        "family_velocities": np.array(valid_velocities),
        "selected_index": idx,
        "selected_time": selected_T,
        "selected_velocity": selected_v,
        "selected_traj": selected_traj,
    }


def generate_paper_like_trajectory_method(
    target_point,
    release_point,
    nominal_flight_time=0.65,
    release_window=0.12,
    g=G,
    n_window_samples=13,
    n_ball_pts=120,
):
    target_point = np.asarray(target_point, dtype=float)
    release_point = np.asarray(release_point, dtype=float)

    v0_nom = velocity_from_target_release_and_time(
        target_point=target_point,
        release_point=release_point,
        flight_time=nominal_flight_time,
        g=g,
    )

    T = float(release_window)
    tau = nominal_flight_time
    coeff = 0.5 * T**2 + T * tau

    gravity_term = np.array([0.0, 0.0, -0.5 * g * tau**2])
    rhs = target_point - release_point - v0_nom * (T + tau) - gravity_term
    a_tube = rhs / coeff

    t_list = np.linspace(0.0, T, n_window_samples)

    family_release_points = []
    family_release_velocities = []
    family_ball_trajs = []

    for t in t_list:
        p_t = release_point + v0_nom * t + 0.5 * a_tube * t**2
        v_t = v0_nom + a_tube * t

        family_release_points.append(p_t)
        family_release_velocities.append(v_t)

        ball_traj = simulate_ballistic_trajectory(
            p_t, v_t, target_z=target_point[2], g=g, n_pts=n_ball_pts
        )
        family_ball_trajs.append(ball_traj)

    idx = len(t_list) // 2
    selected_release_point = family_release_points[idx]
    selected_release_velocity = family_release_velocities[idx]
    selected_ball_traj = family_ball_trajs[idx]

    return {
        "method_name": "Paper-like tube-acceleration method",
        "nominal_velocity": v0_nom,
        "tube_acceleration": a_tube,
        "release_window": T,
        "family_times_in_window": t_list,
        "family_release_points": np.array(family_release_points),
        "family_release_velocities": np.array(family_release_velocities),
        "family_ball_trajs": family_ball_trajs,
        "selected_index": idx,
        "selected_release_point": selected_release_point,
        "selected_release_velocity": selected_release_velocity,
        "selected_ball_traj": selected_ball_traj,
    }


def tube_window_states_from_nominal(
    target_point,
    release_point,
    nominal_flight_time,
    release_window,
    n_window_samples,
    g=G,
):
    target_point = np.asarray(target_point, dtype=float)
    release_point = np.asarray(release_point, dtype=float)
    v0_nom = velocity_from_target_release_and_time(
        target_point=target_point,
        release_point=release_point,
        flight_time=nominal_flight_time,
        g=g,
    )

    T = float(release_window)
    tau = float(nominal_flight_time)
    coeff = 0.5 * T**2 + T * tau
    gravity_term = np.array([0.0, 0.0, -0.5 * g * tau**2])
    rhs = target_point - release_point - v0_nom * (T + tau) - gravity_term
    a_tube = rhs / coeff

    t_list = np.linspace(0.0, T, int(n_window_samples))
    p_list = []
    v_list = []
    for t in t_list:
        p_t = release_point + v0_nom * t + 0.5 * a_tube * t**2
        v_t = v0_nom + a_tube * t
        p_list.append(p_t)
        v_list.append(v_t)

    return {
        "times": t_list,
        "release_points": np.asarray(p_list),
        "release_velocities": np.asarray(v_list),
        "tube_acceleration": a_tube,
        "nominal_velocity": v0_nom,
    }


def _resolve_window_samples(hold_sec: float, n_window_samples: int | None) -> int:
    if n_window_samples is not None:
        return max(2, int(n_window_samples))
    base_hold_sec = 0.5
    base_samples = 10
    base_step = base_hold_sec / float(base_samples - 1)
    n_samples = int(round(float(hold_sec) / base_step)) + 1
    return max(2, n_samples)


def generate_tube_window_trajectory(
    target_point,
    release_point,
    hold_sec=0.5,
    n_window_samples=None,
    method="middle",
    flight_time_range=(0.45, 0.85),
    n_family=21,
    nominal_flight_time=0.65,
    g=G,
):
    target_point = np.asarray(target_point, dtype=float)
    release_point = np.asarray(release_point, dtype=float)

    if method == "middle":
        middle = generate_middle_trajectory_method(
            target_point=target_point,
            release_point=release_point,
            flight_time_range=flight_time_range,
            n_family=n_family,
            g=g,
            n_pts=120,
        )
        nominal_flight_time = float(middle["selected_time"])
        meta = {
            "method": "middle",
            "selected_time": float(middle["selected_time"]),
            "selected_velocity": np.asarray(middle["selected_velocity"], dtype=float),
        }
    elif method == "paper_like":
        meta = {"method": "paper_like", "selected_time": float(nominal_flight_time)}
    else:
        raise ValueError(f"Unknown tube method: {method}")

    n_window_samples = _resolve_window_samples(hold_sec, n_window_samples)
    window = tube_window_states_from_nominal(
        target_point=target_point,
        release_point=release_point,
        nominal_flight_time=nominal_flight_time,
        release_window=float(hold_sec),
        n_window_samples=int(n_window_samples),
        g=g,
    )
    selected_index = len(window["times"]) // 2
    meta.update({
        "release_window": float(hold_sec),
        "tube_acceleration": window["tube_acceleration"],
        "nominal_velocity": window["nominal_velocity"],
        "selected_index": int(selected_index),
        "selected_release_point": window["release_points"][selected_index].copy(),
        "selected_release_velocity": window["release_velocities"][selected_index].copy(),
    })
    return {
        "times": window["times"],
        "release_points": window["release_points"],
        "release_velocities": window["release_velocities"],
        "meta": meta,
    }
