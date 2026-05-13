#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Trajectory validation/post-processing dispatch for sim and real modes."""

from __future__ import annotations

from dataclasses import dataclass
import sys
from pathlib import Path
from typing import Callable, Literal, Sequence

import numpy as np

try:
    from .config import params
    from .trajectory_csv import TrajectoryList, read_trajectory_csv, trajectory_list_to_arrays, write_trajectory_csv
    from .trajectory_plot import StageProvider, plot_trajectory
except ImportError:
    _ROOT = Path(__file__).resolve().parents[1]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    from refractor.config import params
    from refractor.trajectory_csv import TrajectoryList, read_trajectory_csv, trajectory_list_to_arrays, write_trajectory_csv
    from refractor.trajectory_plot import StageProvider, plot_trajectory


ValidationFn = Callable[[TrajectoryList], bool | None]
PostprocessFn = Callable[[TrajectoryList], TrajectoryList]
OutputMode = Literal["simulation", "real"]


@dataclass(frozen=True)
class JointLimitViolation:
    command_index: int
    time: float
    joint_index: int
    command: str
    value: float
    lower: float | None
    upper: float | None
    limit_name: str

    def message(self) -> str:
        joint_name = f"joint{self.joint_index + 1}"
        if self.lower is not None and self.upper is not None:
            limit_text = f"[{self.lower:.12g}, {self.upper:.12g}]"
        elif self.upper is not None:
            limit_text = f"<= {self.upper:.12g}"
        else:
            limit_text = "undefined"
        return (
            f"command_index={self.command_index} t={self.time:.12g} {joint_name} "
            f"command={self.command} value={self.value:.12g} violates {self.limit_name} {limit_text}"
        )


def validate_trajectory_joint_limits(
    trajectory: TrajectoryList,
    *,
    eps: float = params.DEFAULT_VALIDATION_EPS,
) -> list[JointLimitViolation]:
    """Return all q/dq/ddq joint-limit violations in a trajectory list."""
    t, q, dq, ddq = trajectory_list_to_arrays(trajectory)
    eps = float(eps)
    violations: list[JointLimitViolation] = []

    q_limits = np.asarray(params.Q_LIMITS_7, dtype=float).reshape(params.TRAJECTORY_DOF, 2)
    dq_limits = np.asarray(params.QDOT_LIMITS_7, dtype=float).reshape(params.TRAJECTORY_DOF)
    ddq_limits = np.asarray(params.QDDOT_LIMITS_7, dtype=float).reshape(params.TRAJECTORY_DOF)

    for i in range(t.size):
        for j in range(params.TRAJECTORY_DOF):
            q_lo = float(q_limits[j, 0])
            q_hi = float(q_limits[j, 1])
            q_val = float(q[i, j])
            if q_val < q_lo - eps or q_val > q_hi + eps:
                violations.append(
                    JointLimitViolation(i, float(t[i]), j, "q", q_val, q_lo, q_hi, "Q_LIMITS_7")
                )

            dq_lim = float(dq_limits[j])
            dq_val = float(dq[i, j])
            if abs(dq_val) > dq_lim + eps:
                violations.append(
                    JointLimitViolation(i, float(t[i]), j, "dq", dq_val, -dq_lim, dq_lim, "QDOT_LIMITS_7")
                )

            ddq_lim = float(ddq_limits[j])
            ddq_val = float(ddq[i, j])
            if abs(ddq_val) > ddq_lim + eps:
                violations.append(
                    JointLimitViolation(i, float(t[i]), j, "ddq", ddq_val, -ddq_lim, ddq_lim, "QDDOT_LIMITS_7")
                )
    return violations


def assert_valid_trajectory(trajectory: TrajectoryList) -> bool:
    """Validate trajectory shape, time ordering, finite values, and joint limits."""
    if not trajectory:
        raise ValueError("Trajectory list is empty")

    t, q, dq, ddq = trajectory_list_to_arrays(trajectory)
    for name, arr in (("t", t), ("q", q), ("dq", dq), ("ddq", ddq)):
        if not np.all(np.isfinite(arr)):
            bad = np.argwhere(~np.isfinite(arr))
            first = bad[0].tolist() if bad.size else []
            raise ValueError(f"Trajectory contains non-finite {name} at index={first}")

    if t.size >= 2:
        nonmono = np.where(np.diff(t) <= 0.0)[0]
        if nonmono.size:
            i = int(nonmono[0])
            raise ValueError(
                f"Trajectory time is not strictly increasing at command_index={i}: "
                f"t[i]={float(t[i]):.12g}, t[i+1]={float(t[i + 1]):.12g}"
            )

    violations = validate_trajectory_joint_limits(trajectory)
    if violations:
        print("[trajectory validation] joint limit violations:", flush=True)
        for violation in violations:
            print(f"  [VIOL] {violation.message()}", flush=True)
        raise ValueError(f"Trajectory validation failed with {len(violations)} joint-limit violation(s)")

    print(f"[trajectory validation] ok: samples={len(trajectory)}", flush=True)
    return True


def _default_postprocess(trajectory: TrajectoryList) -> TrajectoryList:
    """Default post-processing pipeline."""
    return time_scale_trajectory(trajectory, scale=params.DEFAULT_AFTERPROCESS_TIME_SCALE)


def time_scale_trajectory(trajectory: TrajectoryList, *, scale: float) -> TrajectoryList:
    """Slow down a trajectory by increasing command count and duration.

    For ``scale=S`` this implements ``q_slow(t)=q_original(t/S)``.
    Therefore:
    - command count becomes approximately ``N*S``
    - duration becomes ``duration*S``
    - ``dq`` is divided by ``S``
    - ``ddq`` is divided by ``S^2``
    """
    scale = float(scale)
    if scale <= 0.0:
        raise ValueError(f"time scale must be > 0, got {scale}")
    if not trajectory:
        raise ValueError("Trajectory list is empty")
    if abs(scale - 1.0) <= 1e-12:
        return [
            {
                "t": float(point["t"]),
                "q": np.asarray(point["q"], dtype=float).copy(),
                "dq": np.asarray(point.get("dq", np.zeros(params.TRAJECTORY_DOF)), dtype=float).copy(),
                "ddq": np.asarray(point.get("ddq", np.zeros(params.TRAJECTORY_DOF)), dtype=float).copy(),
            }
            for point in trajectory
        ]

    t, q, dq, ddq = trajectory_list_to_arrays(trajectory)
    n_old = int(t.size)
    if n_old == 1:
        return [
            {
                "t": float(t[0]),
                "q": q[0].copy(),
                "dq": (dq[0] / scale).copy(),
                "ddq": (ddq[0] / (scale * scale)).copy(),
            }
        ]

    duration = float(t[-1] - t[0])
    if duration <= 0.0:
        raise ValueError(f"Trajectory duration must be > 0 for time scaling, got {duration}")

    n_new = max(2, int(np.ceil(n_old * scale)))
    new_duration = duration * scale
    t_new = t[0] + np.linspace(0.0, new_duration, n_new, dtype=float)
    t_query = t[0] + (t_new - t_new[0]) / scale
    t_query = np.clip(t_query, t[0], t[-1])

    q_new = np.column_stack([np.interp(t_query, t, q[:, j]) for j in range(params.TRAJECTORY_DOF)])
    dq_interp = np.column_stack([np.interp(t_query, t, dq[:, j]) for j in range(params.TRAJECTORY_DOF)])
    ddq_interp = np.column_stack([np.interp(t_query, t, ddq[:, j]) for j in range(params.TRAJECTORY_DOF)])
    dq_new = dq_interp / scale
    ddq_new = ddq_interp / (scale * scale)

    scaled: TrajectoryList = []
    for i in range(n_new):
        scaled.append(
            {
                "t": float(t_new[i]),
                "q": q_new[i].copy(),
                "dq": dq_new[i].copy(),
                "ddq": ddq_new[i].copy(),
            }
        )
    print(
        f"[time_scale] scale={scale:g} samples={n_old}->{n_new} duration={duration:.12g}->{new_duration:.12g}",
        flush=True,
    )
    return scaled


def process_trajectory(
    trajectory: TrajectoryList,
    *,
    mode: OutputMode,
    output_csv: str | Path | None = None,
    validation_fn: ValidationFn | None = None,
    postprocess_fn: PostprocessFn | None = None,
    plot: bool = False,
    plot_dir: str | Path | None = None,
    stages: Sequence[dict] | None = None,
    stage_provider: StageProvider | None = None,
    phase_csv_path: str | Path | None = None,
) -> TrajectoryList | Path:
    """Validate, post-process, and dispatch a trajectory.

    In ``simulation`` mode this returns the processed trajectory list.
    In ``real`` mode this writes a canonical trajectory CSV and returns its path.
    """
    validator = validation_fn or assert_valid_trajectory
    postprocessor = postprocess_fn or _default_postprocess

    processed = postprocessor(trajectory)

    validation_result = validator(processed)
    if validation_result is False:
        raise ValueError("Trajectory validation failed")

    if plot:
        out_dir = params.DEFAULT_AFTERPROCESS_PLOT_DIR if plot_dir is None else plot_dir
        written = plot_trajectory(
            processed,
            out_dir=out_dir,
            stages=stages,
            stage_provider=stage_provider,
            phase_csv_path=params.DEFAULT_TRAJECTORY_CSV if phase_csv_path is None else phase_csv_path,
        )
        print(f"[plot] wrote {len(written)} joint trace plot(s) to {Path(out_dir)}", flush=True)

    if mode == "simulation":
        return processed
    if mode == "real":
        if output_csv is None:
            raise ValueError("output_csv is required when mode='real'")
        out_path = Path(output_csv).expanduser()
        write_trajectory_csv(out_path, processed)
        return out_path

    raise ValueError(f"Unsupported mode: {mode!r} (expected 'simulation' or 'real')")


def main() -> int:
    csv_path = params.DEFAULT_TRAJECTORY_CSV
    mode = params.DEFAULT_AFTERPROCESS_MODE
    output_csv = params.DEFAULT_AFTERPROCESS_OUTPUT_CSV

    trajectory = read_trajectory_csv(csv_path)
    result = process_trajectory(
        trajectory,
        mode=mode,
        output_csv=output_csv,
        plot=bool(params.DEFAULT_AFTERPROCESS_PLOT),
        plot_dir=params.DEFAULT_AFTERPROCESS_PLOT_DIR,
        phase_csv_path=csv_path,
    )
    if mode == "simulation":
        print(f"[ok] processed trajectory samples={len(result)}")
    else:
        print(f"[ok] wrote {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
