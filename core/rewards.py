#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np

from .throw_params import LAND_SIGMA, TARGET_XY


def landing_reward(landing_xy: np.ndarray, target_xy: np.ndarray = TARGET_XY) -> float:
    d = landing_xy - np.asarray(target_xy, dtype=float).reshape(2)
    dd = float(np.dot(d, d))
    return math.exp(-dd / (2.0 * LAND_SIGMA * LAND_SIGMA))
