#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Plot refactor-stage trajectory traces."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

try:
    from .config import params
    from .trajectory_csv import TrajectoryList, trajectory_list_to_arrays
except ImportError:
    _ROOT = Path(__file__).resolve().parents[1]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    from refractor.config import params
    from refractor.trajectory_csv import TrajectoryList, trajectory_list_to_arrays


StageProvider = Callable[[TrajectoryList], Sequence[dict]]


def _normalize_stage_ranges(trajectory: TrajectoryList, stages: Sequence[dict] | None) -> list[dict]:
    """Normalize optional stage ranges.

    Expected stage dict format:
    ``{"name": str, "start_command": int, "end_command": int}``.
    A later stage provider can compute these ranges from external parameters.
    """
    if not stages:
        return []
    n = len(trajectory)
    out: list[dict] = []
    for stage in stages:
        name = str(stage.get("name", "stage"))
        start = int(stage.get("start_command", 0))
        end = int(stage.get("end_command", n - 1))
        start = max(0, min(start, n - 1))
        end = max(start, min(end, n - 1))
        out.append({"name": name, "start_command": start, "end_command": end})
    return out


def _load_phase_sidecar(csv_path: str | Path | None) -> list[dict] | None:
    if csv_path is None:
        return None
    path = Path(csv_path).expanduser()
    candidates = [
        Path(str(path) + params.DEFAULT_PHASE_SIDECAR_SUFFIX),
        path.with_suffix(path.suffix + params.DEFAULT_PHASE_SIDECAR_SUFFIX) if path.suffix else Path(str(path) + params.DEFAULT_PHASE_SIDECAR_SUFFIX),
        path.with_suffix(params.DEFAULT_PHASE_SIDECAR_SUFFIX),
    ]
    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.expanduser()
        if candidate in seen:
            continue
        seen.add(candidate)
        if not candidate.is_file():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            return None
        phases = payload.get("phases", None)
        if isinstance(phases, list) and phases:
            return phases
    return None


def _normalize_phase_ranges(phases: Sequence[dict] | None) -> list[dict]:
    if not phases:
        return []
    out: list[dict] = []
    for phase in phases:
        try:
            name = str(phase["name"])
            t_start = float(phase["t_start"])
            t_end = float(phase["t_end"])
        except Exception:
            continue
        if t_end < t_start:
            continue
        out.append({"name": name, "t_start": t_start, "t_end": t_end})
    return out


def _annotate_stage_ranges(axes, t: np.ndarray, stages: Sequence[dict]) -> None:
    if not stages:
        return
    for stage in stages:
        start = int(stage["start_command"])
        end = int(stage["end_command"])
        t0 = float(t[start])
        t1 = float(t[end])
        for ax in axes:
            ax.axvline(
                t0,
                color=params.DEFAULT_TRAJECTORY_PLOT_MARKER_COLOR,
                linestyle=params.DEFAULT_TRAJECTORY_PLOT_MARKER_LINESTYLE,
                linewidth=float(params.DEFAULT_TRAJECTORY_PLOT_MARKER_LINEWIDTH),
                alpha=float(params.DEFAULT_TRAJECTORY_PLOT_STAGE_ALPHA),
            )
            ax.axvline(
                t1,
                color=params.DEFAULT_TRAJECTORY_PLOT_MARKER_COLOR,
                linestyle=params.DEFAULT_TRAJECTORY_PLOT_MARKER_LINESTYLE,
                linewidth=float(params.DEFAULT_TRAJECTORY_PLOT_MARKER_LINEWIDTH),
                alpha=float(params.DEFAULT_TRAJECTORY_PLOT_STAGE_ALPHA),
            )
        if t1 >= t0:
            axes[0].text(
                0.5 * (t0 + t1),
                0.98,
                str(stage["name"]),
                ha="center",
                va="top",
                transform=axes[0].get_xaxis_transform(),
                fontsize=int(params.DEFAULT_TRAJECTORY_PLOT_LABEL_FONTSIZE),
                bbox=dict(params.DEFAULT_TRAJECTORY_PLOT_LABEL_BBOX),
            )


def _annotate_phase_ranges(axes, phases: Sequence[dict]) -> None:
    if not phases:
        return
    boundaries = sorted(set(float(phase["t_end"]) for phase in phases))
    for ax in axes:
        for x in boundaries[:-1]:
            ax.axvline(
                x,
                color=params.DEFAULT_TRAJECTORY_PLOT_MARKER_COLOR,
                linestyle=params.DEFAULT_TRAJECTORY_PLOT_MARKER_LINESTYLE,
                linewidth=float(params.DEFAULT_TRAJECTORY_PLOT_MARKER_LINEWIDTH),
                alpha=float(params.DEFAULT_TRAJECTORY_PLOT_PHASE_ALPHA),
            )

    for phase in phases:
        t0 = float(phase["t_start"])
        t1 = float(phase["t_end"])
        if t1 <= t0:
            continue
        axes[0].text(
            0.5 * (t0 + t1),
            0.98,
            str(phase["name"]),
            ha="center",
            va="top",
            transform=axes[0].get_xaxis_transform(),
            fontsize=int(params.DEFAULT_TRAJECTORY_PLOT_LABEL_FONTSIZE),
            bbox=dict(params.DEFAULT_TRAJECTORY_PLOT_LABEL_BBOX),
        )


def plot_trajectory(
    trajectory: TrajectoryList,
    *,
    out_dir: str | Path,
    stages: Sequence[dict] | None = None,
    stage_provider: StageProvider | None = None,
    phases: Sequence[dict] | None = None,
    phase_csv_path: str | Path | None = None,
) -> list[Path]:
    """Plot q/dq/ddq traces with joint limits.

    ``stage_provider`` is the reserved interface for future stage assignment.
    It should return dicts with ``name``, ``start_command``, and ``end_command``.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t, q, dq, ddq = trajectory_list_to_arrays(trajectory)
    stage_ranges = _normalize_stage_ranges(trajectory, stage_provider(trajectory) if stage_provider else stages)
    phase_ranges = _normalize_phase_ranges(phases) or _normalize_phase_ranges(_load_phase_sidecar(phase_csv_path))
    out_path = Path(out_dir).expanduser()
    out_path.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for j in range(params.TRAJECTORY_DOF):
        fig, axes = plt.subplots(3, 1, figsize=params.DEFAULT_TRAJECTORY_PLOT_FIGSIZE, sharex=True)
        axes[0].scatter(t, q[:, j], s=float(params.DEFAULT_TRAJECTORY_PLOT_POINT_SIZE), label=f"q{j + 1}")
        axes[1].scatter(t, dq[:, j], s=float(params.DEFAULT_TRAJECTORY_PLOT_POINT_SIZE), label=f"dq{j + 1}")
        axes[2].scatter(t, ddq[:, j], s=float(params.DEFAULT_TRAJECTORY_PLOT_POINT_SIZE), label=f"ddq{j + 1}")

        q_min, q_max = params.Q_LIMITS_7[j]
        dq_max = float(params.QDOT_LIMITS_7[j])
        ddq_max = float(params.QDDOT_LIMITS_7[j])
        axes[0].axhline(q_min, color=params.DEFAULT_TRAJECTORY_PLOT_LIMIT_COLOR, linestyle=params.DEFAULT_TRAJECTORY_PLOT_LIMIT_LINESTYLE, linewidth=float(params.DEFAULT_TRAJECTORY_PLOT_LIMIT_LINEWIDTH), label="q_min")
        axes[0].axhline(q_max, color=params.DEFAULT_TRAJECTORY_PLOT_LIMIT_COLOR, linestyle=params.DEFAULT_TRAJECTORY_PLOT_LIMIT_LINESTYLE, linewidth=float(params.DEFAULT_TRAJECTORY_PLOT_LIMIT_LINEWIDTH), label="q_max")
        axes[1].axhline(-dq_max, color=params.DEFAULT_TRAJECTORY_PLOT_LIMIT_COLOR, linestyle=params.DEFAULT_TRAJECTORY_PLOT_LIMIT_LINESTYLE, linewidth=float(params.DEFAULT_TRAJECTORY_PLOT_LIMIT_LINEWIDTH), label="-dq_max")
        axes[1].axhline(dq_max, color=params.DEFAULT_TRAJECTORY_PLOT_LIMIT_COLOR, linestyle=params.DEFAULT_TRAJECTORY_PLOT_LIMIT_LINESTYLE, linewidth=float(params.DEFAULT_TRAJECTORY_PLOT_LIMIT_LINEWIDTH), label="dq_max")
        axes[2].axhline(-ddq_max, color=params.DEFAULT_TRAJECTORY_PLOT_LIMIT_COLOR, linestyle=params.DEFAULT_TRAJECTORY_PLOT_LIMIT_LINESTYLE, linewidth=float(params.DEFAULT_TRAJECTORY_PLOT_LIMIT_LINEWIDTH), label="-ddq_max")
        axes[2].axhline(ddq_max, color=params.DEFAULT_TRAJECTORY_PLOT_LIMIT_COLOR, linestyle=params.DEFAULT_TRAJECTORY_PLOT_LIMIT_LINESTYLE, linewidth=float(params.DEFAULT_TRAJECTORY_PLOT_LIMIT_LINEWIDTH), label="ddq_max")

        axes[0].set_ylabel("q (rad)")
        axes[1].set_ylabel("dq (rad/s)")
        axes[2].set_ylabel("ddq (rad/s^2)")
        axes[2].set_xlabel("time (s)")
        for ax in axes:
            ax.grid(True, alpha=float(params.DEFAULT_TRAJECTORY_PLOT_GRID_ALPHA))
            ax.legend(ncol=int(params.DEFAULT_TRAJECTORY_PLOT_LEGEND_COLUMNS), fontsize=int(params.DEFAULT_TRAJECTORY_PLOT_LEGEND_FONTSIZE))
        _annotate_phase_ranges(axes, phase_ranges)
        _annotate_stage_ranges(axes, t, stage_ranges)

        file_path = out_path / f"joint{j + 1}_traces.png"
        fig.tight_layout()
        fig.savefig(file_path, dpi=int(params.DEFAULT_TRAJECTORY_PLOT_DPI))
        plt.close(fig)
        written.append(file_path)
    return written
