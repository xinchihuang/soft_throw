#!/usr/bin/env bash

set -euo pipefail

# Portable source script for interactive shells:
#   source env.auto.sh
#
# It discovers the Franka Panda URDF across common Isaac Sim installs. If you
# already set SOFT_THROW_URDF, that value is used and only validated.

_soft_throw_find_urdf() {
  local candidates=(
    "${HOME}/isaac-sim/exts/isaacsim.asset.importer.urdf/data/urdf/robots/franka_description/robots/panda_arm_hand.urdf"
    "${HOME}/.local/share/ov/data/exts/v2/isaacsim.asset.importer.urdf-"*"/data/urdf/robots/franka_description/robots/panda_arm_hand.urdf"
  )

  local path
  for path in "${candidates[@]}"; do
    if [[ -f "${path}" ]]; then
      printf '%s\n' "${path}"
      return 0
    fi
  done

  if command -v find >/dev/null 2>&1; then
    find "${HOME}" -path '*/franka_description/robots/panda_arm_hand.urdf' -print -quit 2>/dev/null
  fi
}

_soft_throw_find_robot_usd() {
  local candidates=(
    "${HOME}/Desktop/lacosse_large_mobie_manipulator_new.usd"
    "${HOME}/Desktop/lacosse_large_mobie_manipulator.usd"
    "${HOME}/Desktop/Collected_lacosse_large_mobie_manipulator/lacosse_large_mobie_manipulator.usd"
    "${HOME}/Desktop/Collected_World0/SubUSDs/lacosse_large_mobie_manipulator.usd"
    "${HOME}/isaac-sim/Assets/Isaac/5.1/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
    "${HOME}/Desktop/Collected_lacosse_large_mobie_manipulator/omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
    "${HOME}/Desktop/Collected_World0/SubUSDs/franka.usd"
  )

  local path
  for path in "${candidates[@]}"; do
    if [[ -f "${path}" ]]; then
      printf '%s\n' "${path}"
      return 0
    fi
  done

  if command -v find >/dev/null 2>&1; then
    find "${HOME}" \( -name 'lacosse_large_mobie_manipulator_new.usd' -o -name 'lacosse_large_mobie_manipulator.usd' -o -path '*/FrankaPanda/franka.usd' \) -print -quit 2>/dev/null
  fi
}

if [[ -z "${SOFT_THROW_URDF:-}" ]]; then
  SOFT_THROW_URDF="$(_soft_throw_find_urdf)"
fi

if [[ -z "${SOFT_THROW_URDF:-}" || ! -f "${SOFT_THROW_URDF}" ]]; then
  echo "SOFT_THROW_URDF not found." >&2
  echo "Set it manually, for example:" >&2
  echo '  export SOFT_THROW_URDF="/path/to/franka_description/robots/panda_arm_hand.urdf"' >&2
  return 1 2>/dev/null || exit 1
fi

export SOFT_THROW_URDF
export SOFT_THROW_PKG="${SOFT_THROW_PKG:-$(cd "$(dirname "${SOFT_THROW_URDF}")/../.." && pwd)}"

if [[ -z "${SOFT_THROW_ROBOT_USD:-}" ]]; then
  SOFT_THROW_ROBOT_USD="$(_soft_throw_find_robot_usd)"
fi

if [[ -z "${SOFT_THROW_ROBOT_USD:-}" || ! -f "${SOFT_THROW_ROBOT_USD}" ]]; then
  echo "SOFT_THROW_ROBOT_USD not found." >&2
  echo "Set it manually, for example:" >&2
  echo '  export SOFT_THROW_ROBOT_USD="/path/to/FrankaPanda/franka.usd"' >&2
  return 1 2>/dev/null || exit 1
fi

export SOFT_THROW_ROBOT_USD
export SOFT_THROW_FRANKA_ROOT="${SOFT_THROW_FRANKA_ROOT:-/World/LMM/Franka}"
export SOFT_THROW_LOCAL_FRANKA_USD="${SOFT_THROW_LOCAL_FRANKA_USD:-${HOME}/Desktop/Collected_lacosse_large_mobie_manipulator/omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd}"

echo "SOFT_THROW_URDF=${SOFT_THROW_URDF}"
echo "SOFT_THROW_PKG=${SOFT_THROW_PKG}"
echo "SOFT_THROW_ROBOT_USD=${SOFT_THROW_ROBOT_USD}"
echo "SOFT_THROW_FRANKA_ROOT=${SOFT_THROW_FRANKA_ROOT}"
echo "SOFT_THROW_LOCAL_FRANKA_USD=${SOFT_THROW_LOCAL_FRANKA_USD}"
