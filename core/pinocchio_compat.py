#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import importlib
import os
import sys
from typing import List


def _candidate_site_packages() -> List[str]:
    candidates = []

    env_path = os.environ.get("PINOCCHIO_SITE_PACKAGES")
    if env_path:
        candidates.append(env_path)

    candidates.append("/opt/openrobots/lib/python3.8/site-packages")
    return [path for path in candidates if os.path.isdir(path)]


def _clear_pinocchio_modules() -> None:
    for name in list(sys.modules):
        if name == "pinocchio" or name.startswith("pinocchio."):
            del sys.modules[name]


def load_pinocchio():
    try:
        pin = importlib.import_module("pinocchio")
        robot_wrapper = importlib.import_module("pinocchio.robot_wrapper")
        return pin, robot_wrapper.RobotWrapper
    except Exception:
        pass

    for site_packages in _candidate_site_packages():
        if site_packages in sys.path:
            sys.path.remove(site_packages)
        sys.path.insert(0, site_packages)
        _clear_pinocchio_modules()
        try:
            pin = importlib.import_module("pinocchio")
            robot_wrapper = importlib.import_module("pinocchio.robot_wrapper")
            return pin, robot_wrapper.RobotWrapper
        except Exception:
            continue

    raise ModuleNotFoundError(
        "Unable to import the robotics Pinocchio bindings. "
        "A different 'pinocchio' package is installed earlier on PYTHONPATH. "
        "Set PINOCCHIO_SITE_PACKAGES to the correct site-packages directory if needed."
    )
