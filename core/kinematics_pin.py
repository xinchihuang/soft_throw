#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Backward-compatible shim.

`core/kinematics_pin.py` was renamed to `core/kinematic_forward.py`.
Prefer importing from `core.kinematic_forward`.
"""

from .kinematic_forward import PinKinematics, pin, RobotWrapper  # noqa: F401

__all__ = ["PinKinematics", "pin", "RobotWrapper"]

