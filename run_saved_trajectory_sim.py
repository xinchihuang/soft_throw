#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Launch saved trajectory playback through Isaac Sim Python."""

from __future__ import annotations

import sys

from isaac_python_launcher import exec_with_isaac_python


def main() -> None:
    argv = list(sys.argv[1:])
    if argv and not argv[0].startswith("-"):
        csv_path = argv.pop(0)
        argv = ["--csv", csv_path, *argv]
    exec_with_isaac_python("sim/play_csv_trajectory_sim.py", argv)


if __name__ == "__main__":
    main()
