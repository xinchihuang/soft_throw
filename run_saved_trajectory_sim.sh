#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

source "${SCRIPT_DIR}/env.auto.sh"

ISAAC_PYTHON="${ISAAC_PYTHON:-}"
if [[ -z "${ISAAC_PYTHON}" ]]; then
  for candidate in \
    "${HOME}/isaac-sim/python.sh" \
    "${HOME}/IsaacLab/_isaac_sim/python.sh"
  do
    if [[ -x "${candidate}" ]]; then
      ISAAC_PYTHON="${candidate}"
      break
    fi
  done
fi

if [[ -z "${ISAAC_PYTHON}" || ! -x "${ISAAC_PYTHON}" ]]; then
  echo "Isaac Sim python.sh not found." >&2
  echo "Set ISAAC_PYTHON manually, for example:" >&2
  echo '  export ISAAC_PYTHON="/path/to/isaac-sim/python.sh"' >&2
  exit 1
fi

CSV_PATH="${1:-trajectory.csv}"
shift || true

exec "${ISAAC_PYTHON}" sim/play_csv_trajectory_sim.py --csv "${CSV_PATH}" "$@"
