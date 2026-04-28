#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Plot joint traces from CSV produced by the sim/real runners."""

import os
import json
import numpy as np

from core.throw_params import Q_LIMITS_7, QDOT_LIMITS_7, QDDOT_LIMITS_7


def _load_phase_sidecar(csv_path: str):
    candidates = [
        f"{csv_path}.phases.json",
        os.path.splitext(csv_path)[0] + ".phases.json",
    ]
    for path in candidates:
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                phases = payload.get("phases", None)
                if isinstance(phases, list) and phases:
                    return phases
            except Exception:
                return None
    return None


def _annotate_phases(axes, phases):
    if not phases:
        return
    # Draw boundary lines on all subplots.
    boundaries = []
    for ph in phases:
        try:
            boundaries.append(float(ph["t_end"]))
        except Exception:
            continue
    boundaries = sorted(set(boundaries))
    for ax in axes:
        for x in boundaries[:-1]:
            ax.axvline(x, color="k", linestyle=":", linewidth=1, alpha=0.6)

    # Label phases on the top subplot.
    top = axes[0]
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
        top.text(
            xm,
            0.98,
            name,
            ha="center",
            va="top",
            transform=top.get_xaxis_transform(),
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.6),
        )


def plot_from_csv(csv_path, out_dir, time_unit: str = "s"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    t = data[:, 0]
    q = data[:, 1:8]
    qdot = data[:, 8:15]
    qddot = data[:, 15:22]

    os.makedirs(out_dir, exist_ok=True)
    phases = _load_phase_sidecar(csv_path)

    for j in range(7):
        fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
        axes[0].scatter(t, q[:, j], s=6, label=f"q{j+1}")
        axes[1].scatter(t, qdot[:, j], s=6, label=f"qdot{j+1}")
        axes[2].scatter(t, qddot[:, j], s=6, label=f"qddot{j+1}")

        q_min, q_max = Q_LIMITS_7[j]
        qd_max = QDOT_LIMITS_7[j]
        qdd_max = QDDOT_LIMITS_7[j]
        axes[0].axhline(q_min, color="r", linestyle="--", linewidth=1, label="q_min")
        axes[0].axhline(q_max, color="r", linestyle="--", linewidth=1, label="q_max")
        axes[1].axhline(-qd_max, color="r", linestyle="--", linewidth=1, label="-qdot_max")
        axes[1].axhline(qd_max, color="r", linestyle="--", linewidth=1, label="qdot_max")
        axes[2].axhline(-qdd_max, color="r", linestyle="--", linewidth=1, label="-qddot_max")
        axes[2].axhline(qdd_max, color="r", linestyle="--", linewidth=1, label="qddot_max")

        axes[0].set_ylabel("q (rad)")
        axes[1].set_ylabel("qdot (rad/s)")
        axes[2].set_ylabel("qddot (rad/s^2)")
        axes[2].set_xlabel(f"time ({time_unit})")
        for ax in axes:
            ax.grid(True, alpha=0.3)
            ax.legend(ncol=3, fontsize=8)
        _annotate_phases(axes, phases)

        out_path = os.path.join(out_dir, f"joint{j+1}_traces.png")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)


def plot_compare_from_csv(csv_a, csv_b, out_dir, label_a="tube", label_b="no_tube", time_unit: str = "s"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data_a = np.loadtxt(csv_a, delimiter=",", skiprows=1)
    data_b = np.loadtxt(csv_b, delimiter=",", skiprows=1)
    t_a = data_a[:, 0]
    t_b = data_b[:, 0]
    q_a = data_a[:, 1:8]
    q_b = data_b[:, 1:8]
    qdot_a = data_a[:, 8:15]
    qdot_b = data_b[:, 8:15]
    qddot_a = data_a[:, 15:22]
    qddot_b = data_b[:, 15:22]

    if t_b[-1] > 1e-9 and t_a[-1] > 1e-9:
        t_b = t_b * (t_a[-1] / t_b[-1])
    q_b = np.vstack([np.interp(t_a, t_b, q_b[:, j]) for j in range(7)]).T
    qdot_b = np.vstack([np.interp(t_a, t_b, qdot_b[:, j]) for j in range(7)]).T
    qddot_b = np.vstack([np.interp(t_a, t_b, qddot_b[:, j]) for j in range(7)]).T

    os.makedirs(out_dir, exist_ok=True)
    phases = _load_phase_sidecar(csv_a)

    for j in range(7):
        fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
        axes[0].plot(t_a, q_a[:, j], color="tab:blue", label=f"{label_a} q{j+1}")
        axes[0].plot(t_a, q_b[:, j], color="tab:orange", label=f"{label_b} q{j+1}")
        axes[1].plot(t_a, qdot_a[:, j], color="tab:blue", label=f"{label_a} qdot{j+1}")
        axes[1].plot(t_a, qdot_b[:, j], color="tab:orange", label=f"{label_b} qdot{j+1}")
        axes[2].plot(t_a, qddot_a[:, j], color="tab:blue", label=f"{label_a} qddot{j+1}")
        axes[2].plot(t_a, qddot_b[:, j], color="tab:orange", label=f"{label_b} qddot{j+1}")

        q_min, q_max = Q_LIMITS_7[j]
        qd_max = QDOT_LIMITS_7[j]
        qdd_max = QDDOT_LIMITS_7[j]
        axes[0].axhline(q_min, color="r", linestyle="--", linewidth=1, label="q_min")
        axes[0].axhline(q_max, color="r", linestyle="--", linewidth=1, label="q_max")
        axes[1].axhline(-qd_max, color="r", linestyle="--", linewidth=1, label="-qdot_max")
        axes[1].axhline(qd_max, color="r", linestyle="--", linewidth=1, label="qdot_max")
        axes[2].axhline(-qdd_max, color="r", linestyle="--", linewidth=1, label="-qddot_max")
        axes[2].axhline(qdd_max, color="r", linestyle="--", linewidth=1, label="qddot_max")

        axes[0].set_ylabel("q (rad)")
        axes[1].set_ylabel("qdot (rad/s)")
        axes[2].set_ylabel("qddot (rad/s^2)")
        axes[2].set_xlabel(f"time ({time_unit})")
        for ax in axes:
            ax.grid(True, alpha=0.3)
            ax.legend(ncol=2, fontsize=8)
        _annotate_phases(axes, phases)

        out_path = os.path.join(out_dir, f"joint{j+1}_compare.png")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to joint_traces.csv")
    parser.add_argument("--compare-csv", default=None, help="Optional path to baseline CSV")
    parser.add_argument("--out_dir", required=True, help="Output directory for plots")
    args = parser.parse_args()
    if args.compare_csv:
        plot_compare_from_csv(args.csv, args.compare_csv, args.out_dir)
    else:
        plot_from_csv(args.csv, args.out_dir)


if __name__ == "__main__":
    main()
