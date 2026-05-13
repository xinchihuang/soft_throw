#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Read/write joint trajectory CSV files.

The canonical CSV format matches ``trajectory.csv``:
``t,q0..q6,dq0..dq6,ddq0..ddq6``.
The reader also accepts the older 1-based column names ``q1..q7`` etc.
Legacy input CSVs with ``tau*`` columns can still be read, but those columns are ignored.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import numpy as np

try:
    from refractor.config import params
except ImportError:
    _ROOT = Path(__file__).resolve().parents[1]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    from refractor.config import params


TrajectoryPoint = dict[str, Any]
TrajectoryList = list[TrajectoryPoint]


def _resolve_joint_cols(names: list[str], prefix: str) -> list[str] | None:
    zero_based = [f"{prefix}{i}" for i in range(params.TRAJECTORY_DOF)]
    if all(name in names for name in zero_based):
        return zero_based

    one_based = [f"{prefix}{i}" for i in range(1, params.TRAJECTORY_DOF + 1)]
    if all(name in names for name in one_based):
        return one_based

    return None


def read_trajectory_csv(csv_path: str | Path) -> TrajectoryList:
    """Read a trajectory CSV and return a list of trajectory point dictionaries."""
    path = Path(csv_path).expanduser()
    arr = np.genfromtxt(str(path), delimiter=",", names=True, dtype=None, encoding="utf-8")
    if arr is None or len(arr) == 0:
        raise ValueError(f"Empty trajectory CSV: {path}")

    names = list(arr.dtype.names or [])
    if params.TRAJECTORY_TIME_COLUMN not in names:
        raise ValueError(f"Trajectory CSV missing required column {params.TRAJECTORY_TIME_COLUMN!r}. Found: {names}")

    q_cols = _resolve_joint_cols(names, params.TRAJECTORY_Q_PREFIX)
    dq_cols = _resolve_joint_cols(names, params.TRAJECTORY_DQ_PREFIX)
    ddq_cols = _resolve_joint_cols(names, params.TRAJECTORY_DDQ_PREFIX)
    if q_cols is None:
        raise ValueError(f"Trajectory CSV missing q columns. Found: {names}")

    t = np.asarray(arr[params.TRAJECTORY_TIME_COLUMN], dtype=float).reshape(-1)
    q = np.column_stack([arr[col] for col in q_cols]).astype(float)
    dq = np.column_stack([arr[col] for col in dq_cols]).astype(float) if dq_cols else np.zeros_like(q)
    ddq = np.column_stack([arr[col] for col in ddq_cols]).astype(float) if ddq_cols else np.zeros_like(q)

    trajectory: TrajectoryList = []
    for i in range(t.size):
        trajectory.append(
            {
                "t": float(t[i]),
                "q": q[i].copy(),
                "dq": dq[i].copy(),
                "ddq": ddq[i].copy(),
            }
        )
    return trajectory


def trajectory_list_to_arrays(trajectory: TrajectoryList) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert a trajectory list into ``t, q, dq, ddq`` arrays."""
    if not trajectory:
        raise ValueError("Trajectory list is empty")

    t = np.asarray([point["t"] for point in trajectory], dtype=float).reshape(-1)
    q = np.asarray([point["q"] for point in trajectory], dtype=float).reshape(-1, params.TRAJECTORY_DOF)
    dq = np.asarray([point.get("dq", np.zeros(params.TRAJECTORY_DOF)) for point in trajectory], dtype=float).reshape(-1, params.TRAJECTORY_DOF)
    ddq = np.asarray([point.get("ddq", np.zeros(params.TRAJECTORY_DOF)) for point in trajectory], dtype=float).reshape(-1, params.TRAJECTORY_DOF)
    return t, q, dq, ddq


def write_trajectory_csv(csv_path: str | Path, trajectory: TrajectoryList) -> None:
    """Write a trajectory list using the canonical ``trajectory.csv`` format."""
    path = Path(csv_path).expanduser()
    t, q, dq, ddq = trajectory_list_to_arrays(trajectory)

    header = params.TRAJECTORY_CANONICAL_COLUMNS
    data = np.column_stack([t, q, dq, ddq])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(str(path), data, delimiter=",", header=",".join(header), comments="")
