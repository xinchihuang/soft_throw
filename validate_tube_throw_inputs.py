#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Standalone input validator for `tube_throw.py` (no collision).

Examples:
  python validate_tube_throw_inputs.py --release 0.4,0,0.9
  python validate_tube_throw_inputs.py --release 0.4,0,0.9 --vel 2.0,0,0
"""

from __future__ import annotations

import argparse
import numpy as np

from core.input_checks import validate_ee_linear_velocity, validate_release_pose


def _parse_vec3(text: str) -> np.ndarray:
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Expected 3 comma-separated values, e.g. 0.4,0.0,0.9")
    return np.array([float(p) for p in parts], dtype=float)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate tube_throw inputs (reachability/safety checks).")
    parser.add_argument("--release", type=_parse_vec3, required=True, help="Release XYZ in world: x,y,z")
    parser.add_argument("--vel", type=_parse_vec3, default=None, help="Optional EE linear velocity XYZ in world: vx,vy,vz")
    args = parser.parse_args()

    validate_release_pose(args.release)
    if args.vel is not None:
        validate_ee_linear_velocity(args.vel)

    print("[ok] inputs pass conservative checks", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

