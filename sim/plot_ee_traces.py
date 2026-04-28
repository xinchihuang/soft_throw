#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Plot end-effector traces from CSV exported by tube_throw.py."""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np


def _load_phases(phases_path: str | None) -> list[dict[str, Any]] | None:
    if not phases_path:
        return None
    if not os.path.isfile(phases_path):
        return None
    try:
        with open(phases_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        phases = payload.get("phases", None)
        if isinstance(phases, list) and phases:
            return phases
    except Exception:
        return None
    return None


def _annotate_phases(ax, phases):
    if not phases:
        return
    boundaries = []
    for ph in phases:
        try:
            boundaries.append(float(ph["t_end"]))
        except Exception:
            continue
    boundaries = sorted(set(boundaries))
    for x in boundaries[:-1]:
        ax.axvline(x, color="k", linestyle=":", linewidth=1, alpha=0.6)
    for ph in phases:
        try:
            name = str(ph["name"])
            t0 = float(ph["t_start"])
            t1 = float(ph["t_end"])
        except Exception:
            continue
        if t1 <= t0:
            continue
        xm = 0.5 * (t0 + t1)
        ax.text(
            xm,
            0.98,
            name,
            ha="center",
            va="top",
            transform=ax.get_xaxis_transform(),
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.6),
        )


def plot_ee_from_csv(
    ee_csv_path: str,
    out_dir: str,
    *,
    phases_path: str | None = None,
    time_unit: str = "s",
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = np.loadtxt(ee_csv_path, delimiter=",", skiprows=1)
    t = data[:, 0]
    v = data[:, 4:7]
    w = data[:, 7:10]
    speed = np.linalg.norm(v, axis=1)

    phases = _load_phases(phases_path)
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    axes[0].plot(t, v[:, 0], label="vx")
    axes[0].plot(t, v[:, 1], label="vy")
    axes[0].plot(t, v[:, 2], label="vz")
    axes[1].plot(t, speed, label="|v|")
    axes[2].plot(t, w[:, 0], label="wx")
    axes[2].plot(t, w[:, 1], label="wy")
    axes[2].plot(t, w[:, 2], label="wz")

    axes[0].set_ylabel("EE lin vel (m/s)")
    axes[1].set_ylabel("EE speed (m/s)")
    axes[2].set_ylabel("EE ang vel (rad/s)")
    axes[2].set_xlabel(f"time ({time_unit})")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=3, fontsize=9)
    _annotate_phases(axes[0], phases)

    out_path = os.path.join(out_dir, "ee_velocity_traces.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ee-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--phases", default=None, help="Optional phases JSON sidecar path")
    args = parser.parse_args()
    plot_ee_from_csv(args.ee_csv, args.out_dir, phases_path=args.phases)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

