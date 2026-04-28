#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Build simple reachability conditions from sampled EE positions (no collision).

Input: CSV with at least columns x,y,z (as written by reachability.py).
Output: JSON with:
- AABB bounds (quantile-trimmed)
- radius bounds (quantile-trimmed)
- z-binned XY convex hull polygons (for a piecewise workspace envelope)

This lets you classify reachability based only on EE world coordinates.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class WorkspaceConditions:
    aabb_min: np.ndarray  # (3,)
    aabb_max: np.ndarray  # (3,)
    r_min: float
    r_max: float
    z_edges: np.ndarray  # (nbins+1,)
    hulls_xy: list[np.ndarray | None]  # len=nbins, each (m,2) polygon CCW or None


def _load_xyz_csv(path: Path) -> np.ndarray:
    raw = np.genfromtxt(str(path), delimiter=",", names=True)
    if raw is None or len(raw) == 0:
        return np.zeros((0, 3), dtype=float)
    names = list(raw.dtype.names or [])
    for needed in ("x", "y", "z"):
        if needed not in names:
            raise ValueError(f"CSV missing column {needed!r}. Found columns: {names}")
    xyz = np.column_stack([raw["x"], raw["y"], raw["z"]]).astype(float)
    return xyz


def _quantile_bounds(x: np.ndarray, q_low: float, q_high: float) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return np.zeros(x.shape[1], dtype=float), np.zeros(x.shape[1], dtype=float)
    lo = np.quantile(x, float(q_low), axis=0)
    hi = np.quantile(x, float(q_high), axis=0)
    return np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)


def _convex_hull_2d(points_xy: np.ndarray) -> np.ndarray:
    """Monotonic chain convex hull. Returns CCW hull without repeating first point."""
    pts = np.asarray(points_xy, dtype=float).reshape(-1, 2)
    if pts.shape[0] <= 1:
        return pts.copy()

    # sort by x then y
    order = np.lexsort((pts[:, 1], pts[:, 0]))
    pts = pts[order]

    def cross(o, a, b) -> float:
        return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))

    lower: list[np.ndarray] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0.0:
            lower.pop()
        lower.append(p)

    upper: list[np.ndarray] = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0.0:
            upper.pop()
        upper.append(p)

    hull = np.vstack(lower[:-1] + upper[:-1])
    return hull.astype(float)


def build_conditions(
    xyz: np.ndarray,
    *,
    trim_q: float,
    z_bins: int,
    min_bin_points: int,
) -> WorkspaceConditions:
    xyz = np.asarray(xyz, dtype=float).reshape(-1, 3)
    if xyz.shape[0] == 0:
        raise ValueError("No points provided")

    trim_q = float(trim_q)
    if not (0.0 <= trim_q < 0.5):
        raise ValueError(f"--trim-q must be in [0,0.5), got {trim_q}")

    aabb_min, aabb_max = _quantile_bounds(xyz, trim_q, 1.0 - trim_q)
    r = np.linalg.norm(xyz, axis=1)
    r_min = float(np.quantile(r, trim_q))
    r_max = float(np.quantile(r, 1.0 - trim_q))

    z_min = float(aabb_min[2])
    z_max = float(aabb_max[2])
    z_edges = np.linspace(z_min, z_max, int(z_bins) + 1)

    hulls: list[np.ndarray | None] = []
    for i in range(int(z_bins)):
        z0 = float(z_edges[i])
        z1 = float(z_edges[i + 1])
        mask = (xyz[:, 2] >= z0) & (xyz[:, 2] < z1 if i < int(z_bins) - 1 else xyz[:, 2] <= z1)
        pts = xyz[mask]
        if pts.shape[0] < int(min_bin_points):
            hulls.append(None)
            continue
        hull = _convex_hull_2d(pts[:, :2])
        if hull.shape[0] < 3:
            hulls.append(None)
            continue
        hulls.append(hull)

    return WorkspaceConditions(
        aabb_min=aabb_min,
        aabb_max=aabb_max,
        r_min=r_min,
        r_max=r_max,
        z_edges=z_edges,
        hulls_xy=hulls,
    )


def _dump_json(conds: WorkspaceConditions) -> dict:
    hulls = []
    for hull in conds.hulls_xy:
        if hull is None:
            hulls.append(None)
        else:
            hulls.append(np.asarray(hull, dtype=float).tolist())
    return {
        "aabb_min": conds.aabb_min.tolist(),
        "aabb_max": conds.aabb_max.tolist(),
        "r_min": float(conds.r_min),
        "r_max": float(conds.r_max),
        "z_edges": conds.z_edges.tolist(),
        "hulls_xy": hulls,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Export reachability conditions from sampled EE points.")
    parser.add_argument("--csv", type=str, required=True, help="CSV with x,y,z columns")
    parser.add_argument("--out", type=str, default="workspace_conditions.json")
    parser.add_argument("--trim-q", type=float, default=0.01, help="Quantile trim for bounds (robustness)")
    parser.add_argument("--z-bins", type=int, default=30)
    parser.add_argument("--min-bin-points", type=int, default=200)
    args = parser.parse_args()

    xyz = _load_xyz_csv(Path(args.csv))
    conds = build_conditions(
        xyz,
        trim_q=float(args.trim_q),
        z_bins=int(args.z_bins),
        min_bin_points=int(args.min_bin_points),
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_dump_json(conds), indent=2), encoding="utf-8")
    print(f"[ok] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

