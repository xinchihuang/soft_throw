#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Compatibility shim.

This file keeps the original entrypoint name while the implementation
is refactored into simulation/core and simulation/sim modules.
The legacy implementation is preserved as env_spoon_throw_core_legacy.py.
"""

# Re-export constants for backward compatibility
from core.throw_params import *  # noqa: F403

# Re-export env class
from sim.env_isaac_throw import SpoonThrowEnvCoreTubeRandomYawAlign, SpoonThrowEnvCore  # noqa: F401

__all__ = [
    "SpoonThrowEnvCoreTubeRandomYawAlign",
    "SpoonThrowEnvCore",
]
