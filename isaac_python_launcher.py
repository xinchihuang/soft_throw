#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Helpers for launching this project through Isaac Sim's python.sh."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def load_env_file(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("export "):
            continue
        match = re.match(r"export\s+([A-Za-z_][A-Za-z0-9_]*)=(.*)", line)
        if not match:
            continue
        key, raw_value = match.groups()
        value = raw_value.strip().strip('"').strip("'")
        if "$" in value or "`" in value:
            continue
        os.environ.setdefault(key, value)


def source_env_auto() -> None:
    env_auto = ROOT / "env.auto.sh"
    if not env_auto.is_file():
        load_env_file(ROOT / "env.sh")
        return
    command = f"source {env_auto}; env"
    result = subprocess.run(
        ["bash", "-lc", command],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    for line in result.stdout.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ[key] = value


def find_isaac_python() -> str:
    explicit = os.environ.get("ISAAC_PYTHON")
    candidates = [
        explicit,
        str(Path.home() / "isaac-sim" / "python.sh"),
        str(Path.home() / "IsaacLab" / "_isaac_sim" / "python.sh"),
    ]
    for candidate in candidates:
        if candidate and os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    raise SystemExit(
        "Isaac Sim python.sh not found. Set ISAAC_PYTHON, for example:\n"
        '  export ISAAC_PYTHON="/path/to/isaac-sim/python.sh"'
    )


def exec_with_isaac_python(script: str, argv: list[str]) -> None:
    source_env_auto()
    isaac_python = find_isaac_python()
    script_path = str(ROOT / script)
    os.execvpe(isaac_python, [isaac_python, script_path, *argv], os.environ)
