#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from .throw_params import RANDOMIZE_TARGET, TARGET_XYZ, TARGET_X_RANGE, TARGET_Y_RANGE, TARGET_Z


def sample_target_xyz(rng: np.random.Generator) -> np.ndarray:
    """Sample a 3D target on the landing plane (episode-wise)."""
    if not RANDOMIZE_TARGET:
        return TARGET_XYZ.astype(float).copy()
    x = float(rng.uniform(TARGET_X_RANGE[0], TARGET_X_RANGE[1]))
    y = float(rng.uniform(TARGET_Y_RANGE[0], TARGET_Y_RANGE[1]))
    z = float(TARGET_Z)
    return np.array([x, y, z], dtype=float)
