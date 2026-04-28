#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Analyze an executor-produced actual trajectory CSV.

This script is meant for quick post-mortem checks on files like:
- trajectory_actual.csv

It reports:
- time monotonicity + dt stats
- q / dq / tau_cmd violations vs JOINT_LIMITS.md (authoritative runtime limits)
- basic discontinuity proxies (step jumps)
- per-phase summaries (if `phase` column exists)

By default it uses the in-code path DEFAULT_CSV_PATH (no CLI required).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np


DEFAULT_CSV_PATH = "trajectory_actual.csv"
DEFAULT_LIMITS_MD = "JOINT_LIMITS.md"


def _resolve_joint_cols(names: list[str]) -> tuple[list[str], list[str], list[str] | None]:
    if all(k in names for k in [f"q{i}" for i in range(7)]):
        q_cols = [f"q{i}" for i in range(7)]
        dq_cols = [f"dq{i}" for i in range(7)]
        ddq_cols = [f"ddq{i}" for i in range(7)]
        return q_cols, dq_cols, ddq_cols if all(k in names for k in ddq_cols) else None
    if all(k in names for k in [f"q{i}" for i in range(1, 8)]):
        q_cols = [f"q{i}" for i in range(1, 8)]
        dq_cols = [f"dq{i}" for i in range(1, 8)]
        ddq_cols = [f"ddq{i}" for i in range(1, 8)]
        return q_cols, dq_cols, ddq_cols if all(k in names for k in ddq_cols) else None
    raise ValueError(f"Unrecognized q column scheme. Found: {names}")


def _parse_joint_limits_md(md_path: Path) -> dict[str, np.ndarray]:
    """Parse JOINT_LIMITS.md into numpy arrays.

    Returns keys:
      - q_soft: (7,2)
      - q_hard: (7,2) or None if missing
      - qdot_hard: (7,)
      - tau_hard: (7,) or None if missing
    """

    text = md_path.read_text(encoding="utf-8")

    def _extract_table(section_start_pat: str, section_end_pat: str | None = None) -> list[list[str]]:
        start = re.search(section_start_pat, text, flags=re.MULTILINE)
        if not start:
            return []
        sub = text[start.end() :]
        if section_end_pat:
            end = re.search(section_end_pat, sub, flags=re.MULTILINE)
            if end:
                sub = sub[: end.start()]
        rows: list[list[str]] = []
        for line in sub.splitlines():
            line = line.strip()
            if not line.startswith("|"):
                continue
            if line.startswith("| Joint") or re.match(r"^\|\s*---", line):
                continue
            if not re.match(r"^\|\s*j[1-7]\b", line):
                continue
            parts = [c.strip() for c in line.strip("|").split("|")]
            rows.append(parts)
        return rows

    # Position table
    pos_rows = _extract_table(r"### Position limits \(`q`\)")
    q_soft = np.zeros((7, 2), dtype=float)
    q_hard = np.zeros((7, 2), dtype=float)
    have_q_hard = False
    for parts in pos_rows:
        # expected: j, q_soft lower, q_soft upper, q_hard lower, q_hard upper
        if len(parts) < 5:
            continue
        j = int(parts[0][1:])
        q_soft[j - 1] = [float(parts[1]), float(parts[2])]
        q_hard[j - 1] = [float(parts[3]), float(parts[4])]
        have_q_hard = True

    # Velocity table
    vel_rows = _extract_table(r"### Velocity limits \(`qdot`\)", r"### Acceleration")
    qdot_hard = np.zeros(7, dtype=float)
    for parts in vel_rows:
        if len(parts) < 2:
            continue
        j = int(parts[0][1:])
        qdot_hard[j - 1] = float(parts[1])

    # Effort table
    effort_rows = _extract_table(r"### Effort / torque limits \(`tau`\)", r"## Gripper")
    tau_hard = np.zeros(7, dtype=float)
    have_tau = False
    for parts in effort_rows:
        # expected: j, tau_hard
        if len(parts) < 2:
            continue
        j = int(parts[0][1:])
        tau_hard[j - 1] = float(parts[1])
        have_tau = True

    out: dict[str, np.ndarray] = {
        "q_soft": q_soft,
        "qdot_hard": qdot_hard,
    }
    if have_q_hard:
        out["q_hard"] = q_hard
    if have_tau:
        out["tau_hard"] = tau_hard
    return out


def _print_dt_stats(t: np.ndarray) -> None:
    if len(t) < 2:
        print("t: <2 samples")
        return
    dt = np.diff(t)
    mono = bool(np.all(dt > 0))
    print(f"t monotonic increasing: {mono}")
    print(f"dt min/med/max: {float(dt.min()):.6g} {float(np.median(dt)):.6g} {float(dt.max()):.6g}")
    if not mono:
        bad = np.where(dt <= 0)[0]
        k = int(bad[0])
        print(f"[VIOL] non-increasing t at idx={k}: t={t[k]:.6g} -> {t[k+1]:.6g}")


def _top_k_violations(mask: np.ndarray, t: np.ndarray, phase: np.ndarray | None, values: np.ndarray, limits: np.ndarray, *, k: int = 10):
    idx = np.argwhere(mask)
    if idx.size == 0:
        return []
    idx = idx[idx[:, 0].argsort()]
    out = []
    for row in idx[:k]:
        i, j = int(row[0]), int(row[1])
        out.append(
            {
                "i": i,
                "t": float(t[i]),
                "phase": None if phase is None else str(phase[i]),
                "joint": j + 1,
                "value": float(values[i, j]),
                "limit": float(limits[j]),
            }
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze an actual trajectory CSV vs JOINT_LIMITS.md.")
    parser.add_argument("--csv", type=str, default=None, help="Optional override path (default uses DEFAULT_CSV_PATH)")
    parser.add_argument("--limits-md", type=str, default=DEFAULT_LIMITS_MD)
    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument("--topk", type=int, default=8)
    args = parser.parse_args()

    csv_path = Path(DEFAULT_CSV_PATH if args.csv is None else args.csv)
    if not csv_path.is_file():
        raise SystemExit(f"CSV not found: {csv_path}")
    md_path = Path(args.limits_md)
    if not md_path.is_file():
        raise SystemExit(f"Limits markdown not found: {md_path}")

    limits = _parse_joint_limits_md(md_path)
    q_soft = np.asarray(limits["q_soft"], dtype=float)
    qdot_hard = np.asarray(limits["qdot_hard"], dtype=float)
    q_hard = np.asarray(limits["q_hard"], dtype=float) if "q_hard" in limits else None
    tau_hard = np.asarray(limits["tau_hard"], dtype=float) if "tau_hard" in limits else None

    arr = np.genfromtxt(str(csv_path), delimiter=",", names=True, dtype=None, encoding="utf-8")
    if arr is None or len(arr) == 0:
        raise ValueError(f"Empty CSV: {csv_path}")
    names = list(arr.dtype.names or [])
    if "t" not in names:
        raise ValueError(f"CSV missing 't'. Found: {names}")

    q_cols, dq_cols, ddq_cols = _resolve_joint_cols(names)
    has_phase = "phase" in names
    phase = np.asarray(arr["phase"]) if has_phase else None

    t = np.asarray(arr["t"], dtype=float).reshape(-1)
    q = np.column_stack([arr[c] for c in q_cols]).astype(float)
    dq = np.column_stack([arr[c] for c in dq_cols]).astype(float)
    ddq = np.column_stack([arr[c] for c in ddq_cols]).astype(float) if ddq_cols is not None else None

    tau_cmd_cols = None
    if all(f"tau_cmd{i}" in names for i in range(7)):
        tau_cmd_cols = [f"tau_cmd{i}" for i in range(7)]
    if all(f"tau_cmd{i}" in names for i in range(1, 8)):
        tau_cmd_cols = [f"tau_cmd{i}" for i in range(1, 8)]
    tau_cmd = np.column_stack([arr[c] for c in tau_cmd_cols]).astype(float) if tau_cmd_cols else None

    print(f"file: {csv_path.resolve()}")
    print(f"samples: {len(t)} duration_s: {float(t[-1]) if len(t) else 0.0:.6g}")
    print(f"scheme: {q_cols[0]}..{q_cols[-1]}")
    if phase is not None:
        uniq = list(dict.fromkeys([str(x) for x in phase.tolist()]))
        print(f"phases: {uniq}")
    _print_dt_stats(t)

    eps = float(args.eps)

    # q violations (soft/hard)
    q_lo_soft = q_soft[:, 0]
    q_hi_soft = q_soft[:, 1]
    viol_q_soft = (q < q_lo_soft[None, :] - eps) | (q > q_hi_soft[None, :] + eps)

    viol_q_hard = None
    if q_hard is not None:
        q_lo_hard = q_hard[:, 0]
        q_hi_hard = q_hard[:, 1]
        viol_q_hard = (q < q_lo_hard[None, :] - eps) | (q > q_hi_hard[None, :] + eps)

    viol_dq = np.abs(dq) > qdot_hard[None, :] + eps

    # tau_cmd violations (if present and md has tau limits)
    viol_tau = None
    if tau_cmd is not None and tau_hard is not None:
        viol_tau = np.abs(tau_cmd) > tau_hard[None, :] + eps

    def _count(mask: np.ndarray | None) -> int:
        return 0 if mask is None else int(mask.sum())

    print("")
    print(f"[q_soft] violations: {_count(viol_q_soft)}")
    if viol_q_hard is not None:
        print(f"[q_hard] violations: {_count(viol_q_hard)}")
    print(f"[qdot_hard] violations: {_count(viol_dq)}")
    if viol_tau is not None:
        print(f"[tau_hard] violations: {_count(viol_tau)}")

    if viol_q_soft.any():
        print("")
        print("[VIOL] q_soft examples:")
        ex = _top_k_violations(viol_q_soft, t, phase, q, np.maximum(np.abs(q_lo_soft), np.abs(q_hi_soft)), k=int(args.topk))
        for e in ex:
            print(f"  t={e['t']:.6g} phase={e['phase']} j{e['joint']} q={e['value']:.6g} soft=[{q_lo_soft[e['joint']-1]:.6g},{q_hi_soft[e['joint']-1]:.6g}]")

    if viol_q_hard is not None and viol_q_hard.any():
        print("")
        print("[VIOL] q_hard examples:")
        idx = np.argwhere(viol_q_hard)[: int(args.topk)]
        for i, j in idx:
            i = int(i); j = int(j)
            ph = None if phase is None else str(phase[i])
            print(f"  t={t[i]:.6g} phase={ph} j{j+1} q={q[i,j]:.6g} hard=[{q_lo_hard[j]:.6g},{q_hi_hard[j]:.6g}]")

    if viol_dq.any():
        print("")
        print("[VIOL] qdot_hard examples:")
        idx = np.argwhere(viol_dq)[: int(args.topk)]
        for i, j in idx:
            i = int(i); j = int(j)
            ph = None if phase is None else str(phase[i])
            print(f"  t={t[i]:.6g} phase={ph} j{j+1} |dq|={abs(dq[i,j]):.6g} lim={qdot_hard[j]:.6g} dq={dq[i,j]:.6g}")

    if viol_tau is not None and viol_tau.any():
        print("")
        print("[VIOL] tau_hard examples:")
        idx = np.argwhere(viol_tau)[: int(args.topk)]
        for i, j in idx:
            i = int(i); j = int(j)
            ph = None if phase is None else str(phase[i])
            print(f"  t={t[i]:.6g} phase={ph} j{j+1} |tau_cmd|={abs(tau_cmd[i,j]):.6g} lim={tau_hard[j]:.6g} tau_cmd={tau_cmd[i,j]:.6g}")

    # Step-jump / discontinuity proxies
    if len(t) >= 3:
        dt = np.diff(t)
        good = dt > 1e-12
        dq_jump = np.diff(dq, axis=0)
        print("")
        print("max step jump |Δdq|:", [float(x) for x in np.max(np.abs(dq_jump), axis=0)])

        if ddq is not None:
            ddq_jump = np.diff(ddq, axis=0)
            print("max step jump |Δddq|:", [float(x) for x in np.max(np.abs(ddq_jump), axis=0)])

        # Finite-difference dq from q
        dq_fd = np.zeros_like(dq)
        dq_fd[:-1][good] = (q[1:][good] - q[:-1][good]) / dt[good, None]
        dq_fd[-1] = dq_fd[-2] if len(dq_fd) >= 2 else 0.0
        max_err = np.max(np.abs(dq - dq_fd), axis=0)
        print("max |dq - dq_fd|:", [float(x) for x in max_err])

        if ddq is not None:
            # Finite-difference ddq from dq (informational; no "robot limit" implied).
            ddq_fd = np.zeros_like(ddq)
            ddq_fd[:-1][good] = (dq[1:][good] - dq[:-1][good]) / dt[good, None]
            ddq_fd[-1] = ddq_fd[-2] if len(ddq_fd) >= 2 else 0.0
            max_err_ddq = np.max(np.abs(ddq - ddq_fd), axis=0)
            print("max |ddq - ddq_fd|:", [float(x) for x in max_err_ddq])

        if tau_cmd is not None:
            tau_jump = np.diff(tau_cmd, axis=0)
            print("max step jump |Δtau_cmd|:", [float(x) for x in np.max(np.abs(tau_jump), axis=0)])

    # Per-phase summary
    if phase is not None:
        print("")
        print("per-phase max|dq|:")
        for ph in list(dict.fromkeys([str(x) for x in phase.tolist()])):
            mask = np.array([str(x) == ph for x in phase], dtype=bool)
            if not mask.any():
                continue
            max_dq = np.max(np.abs(dq[mask]), axis=0)
            print(f"  {ph}: duration={float(t[mask][-1]-t[mask][0]):.6g}s max|dq|={[float(x) for x in max_dq]}")

    any_viol = bool(viol_q_soft.any() or viol_dq.any() or (viol_q_hard is not None and viol_q_hard.any()) or (viol_tau is not None and viol_tau.any()))
    return 2 if any_viol else 0


if __name__ == "__main__":
    raise SystemExit(main())
