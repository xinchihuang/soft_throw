#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Launch asset viewer through Isaac Sim Python."""

from __future__ import annotations

import sys

from isaac_python_launcher import exec_with_isaac_python


def main() -> None:
    exec_with_isaac_python("sim/show_asset_sim.py", list(sys.argv[1:]))


if __name__ == "__main__":
    main()
