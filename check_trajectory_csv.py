#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Check a trajectory CSV for limit violations and discontinuities.

Supports both column naming schemes:
- 0-based: q0..q6, dq0..dq6, ddq0..ddq6
- 1-based: q1..q7, dq1..dq7, ddq1..ddq7

Checks:
- time monotonicity and dt stats
- joint position/velocity/acceleration limits vs core.throw_params
- max per-step jump in dq and ddq (simple discontinuity proxy)
- finite-difference velocity from q and t (to catch inconsistencies)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from core.throw_params import Q_LIMITS_7, QDOT_LIMITS_7, QDDOT_LIMITS_7


DEFAULT_CSV_PATHS = [
    "demo_trajectory.csv",
]


def _resolve_joint_cols(names: list[str]) -> tuple[list[str], list[str], list[str]]:
    if all(k in names for k in [f"q{i}" for i in range(7)]):
        q_cols = [f"q{i}" for i in range(7)]
        dq_cols = [f"dq{i}" for i in range(7)]
        ddq_cols = [f"ddq{i}" for i in range(7)]
        return q_cols, dq_cols, ddq_cols
    if all(k in names for k in [f"q{i}" for i in range(1, 8)]):
        q_cols = [f"q{i}" for i in range(1, 8)]
        dq_cols = [f"dq{i}" for i in range(1, 8)]
        ddq_cols = [f"ddq{i}" for i in range(1, 8)]
        return q_cols, dq_cols, ddq_cols
    raise ValueError(f"Unrecognized q column scheme. Found: {names}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check trajectory CSV limits/discontinuities.")
    parser.add_argument("--csv", type=str, default=None, help="CSV path (optional; uses DEFAULT_CSV_PATHS if omitted)")
    parser.add_argument("--eps", type=float, default=1e-12, help="Numerical tolerance for limit checks")
    args = parser.parse_args()

    if args.csv is None:
        for p in DEFAULT_CSV_PATHS:
            candidate = Path(p)
            if candidate.is_file():
                path = candidate
                break
        else:
            raise SystemExit(f"No --csv provided and none of DEFAULT_CSV_PATHS exist: {DEFAULT_CSV_PATHS}")
    else:
        path = Path(args.csv)
    arr = np.genfromtxt(str(path), delimiter=",", names=True)
    if arr is None or len(arr) == 0:
        raise ValueError(f"Empty CSV: {path}")
    names = list(arr.dtype.names or [])
    if "t" not in names:
        raise ValueError(f"CSV missing 't'. Found: {names}")

    q_cols, dq_cols, ddq_cols = _resolve_joint_cols(names)
    missing = [c for c in [*q_cols, *dq_cols] if c not in names]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    has_ddq = all(c in names for c in ddq_cols)

    t = np.asarray(arr["t"], dtype=float).reshape(-1)
    q = np.column_stack([arr[c] for c in q_cols]).astype(float)
    dq = np.column_stack([arr[c] for c in dq_cols]).astype(float)
    ddq = np.column_stack([arr[c] for c in ddq_cols]).astype(float) if has_ddq else None

    print(f"file: {path.resolve()}")
    print(f"scheme: {q_cols[0]}..{q_cols[-1]}")
    print(f"samples: {len(t)} duration_s: {float(t[-1]) if len(t) else 0.0:.6g}")

    if len(t) >= 2:
        dt = np.diff(t)
        mono = bool(np.all(dt > 0))
        print(f"t monotonic increasing: {mono}")
        print(f"dt min/med/max: {float(dt.min()):.6g} {float(np.median(dt)):.6g} {float(dt.max()):.6g}")
        if not mono:
            bad = np.where(dt <= 0)[0]
            k = int(bad[0])
            print(f"[VIOL] non-increasing t at idx={k}: t={t[k]:.6g} -> {t[k+1]:.6g}")

    q_lo = np.asarray(Q_LIMITS_7[:, 0], dtype=float)
    q_hi = np.asarray(Q_LIMITS_7[:, 1], dtype=float)
    dq_lim = np.asarray(QDOT_LIMITS_7, dtype=float)
    ddq_lim = np.asarray(QDDOT_LIMITS_7, dtype=float)
    eps = float(args.eps)

    viol_q = (q < q_lo[None, :] - eps) | (q > q_hi[None, :] + eps)
    viol_dq = np.abs(dq) > dq_lim[None, :] + eps
    viol_ddq = (np.abs(ddq) > ddq_lim[None, :] + eps) if ddq is not None else None

    def _report(mask: np.ndarray, label: str):
        if not mask.any():
            print(f"[ok] {label}: no violations")
            return
        idx = np.argwhere(mask)
        idx = idx[idx[:, 0].argsort()]
        joints = sorted(set(int(j) for _, j in idx))
        print(f"[VIOL] {label}: count={idx.shape[0]} joints={[j + 1 for j in joints]}")
        for j in joints:
            k = int(idx[idx[:, 1] == j][0, 0])
            if label == "q":
                val = float(q[k, j])
                print(f"  j{j+1}: t={t[k]:.6g} q={val:.6g} lim=[{q_lo[j]:.6g},{q_hi[j]:.6g}]")
            elif label == "|dq|":
                val = float(abs(dq[k, j]))
                print(f"  j{j+1}: t={t[k]:.6g} |dq|={val:.6g} lim={dq_lim[j]:.6g}")
            elif label == "|ddq|":
                val = float(abs(ddq[k, j]))
                print(f"  j{j+1}: t={t[k]:.6g} |ddq|={val:.6g} lim={ddq_lim[j]:.6g}")

    _report(viol_q, "q")
    _report(viol_dq, "|dq|")
    if viol_ddq is not None:
        _report(viol_ddq, "|ddq|")
    else:
        print("[skip] |ddq|: ddq columns not present")

    max_dq = np.max(np.abs(dq), axis=0)
    max_ddq = np.max(np.abs(ddq), axis=0) if ddq is not None else None
    print("max|dq|:", [float(x) for x in max_dq])
    print("dq limits:", [float(x) for x in dq_lim])
    if max_ddq is not None:
        print("max|ddq|:", [float(x) for x in max_ddq])

    if len(t) >= 3:
        d_dq = np.diff(dq, axis=0)
        d_ddq = np.diff(ddq, axis=0) if ddq is not None else None
        max_jump_dq = np.max(np.abs(d_dq), axis=0)
        print("max step jump |Δdq|:", [float(x) for x in max_jump_dq])
        if d_ddq is not None:
            max_jump_ddq = np.max(np.abs(d_ddq), axis=0)
            print("max step jump |Δddq|:", [float(x) for x in max_jump_ddq])
        else:
            print("[skip] max step jump |Δddq|: ddq columns not present")

        # Finite-difference velocity from q and t
        dt = np.diff(t)
        good = dt > 1e-12
        dq_fd = np.zeros_like(dq)
        dq_fd[:-1][good] = (q[1:][good] - q[:-1][good]) / dt[good, None]
        dq_fd[-1] = dq_fd[-2] if len(dq_fd) >= 2 else 0.0
        err = dq - dq_fd
        max_err = np.max(np.abs(err), axis=0)
        print("max |dq - dq_fd|:", [float(x) for x in max_err])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
