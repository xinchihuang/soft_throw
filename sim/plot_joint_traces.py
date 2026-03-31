#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Plot joint traces from CSV produced by run_min_throw_sim.py."""

import os
import numpy as np

from core.throw_params import Q_LIMITS_7, QDOT_LIMITS_7, QDDOT_LIMITS_7


def plot_from_csv(csv_path, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    t = data[:, 0]
    q = data[:, 1:8]
    qdot = data[:, 8:15]
    u = data[:, 15:22]

    os.makedirs(out_dir, exist_ok=True)

    for j in [1, 3, 5]:
        fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
        axes[0].scatter(t, q[:, j], s=6, label=f"q{j+1}")
        axes[1].scatter(t, qdot[:, j], s=6, label=f"qdot{j+1}")
        axes[2].scatter(t, u[:, j], s=6, label=f"u{j+1}")

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
        axes[2].set_ylabel("u (rad/s)")
        axes[2].set_xlabel("time (s)")
        for ax in axes:
            ax.grid(True, alpha=0.3)
            ax.legend(ncol=3, fontsize=8)

        out_path = os.path.join(out_dir, f"joint{j+1}_traces.png")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to joint_traces.csv")
    parser.add_argument("--out_dir", required=True, help="Output directory for plots")
    args = parser.parse_args()
    plot_from_csv(args.csv, args.out_dir)


if __name__ == "__main__":
    main()
