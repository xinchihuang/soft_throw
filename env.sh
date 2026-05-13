#!/usr/bin/env bash

set -euo pipefail

# Local machine config.
#
# Keep this file as simple `export KEY="value"` assignments: several Python
# entrypoints parse it directly instead of evaluating it as a shell script.
# For a portable auto-detecting setup, source `env.auto.sh` instead.
export SOFT_THROW_URDF="/home/xinchi/isaac-sim/exts/isaacsim.asset.importer.urdf/data/urdf/robots/franka_description/robots/panda_arm_hand.urdf"
export SOFT_THROW_PKG="/home/xinchi/isaac-sim/exts/isaacsim.asset.importer.urdf/data/urdf/robots"
export SOFT_THROW_ROBOT_USD="/home/xinchi/Desktop/lacosse_large_mobie_manipulator.usd"
export SOFT_THROW_FRANKA_ROOT="/World/LMM/Franka"
export SOFT_THROW_LOCAL_FRANKA_USD="/home/xinchi/Desktop/Collected_lacosse_large_mobie_manipulator/omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"

if [[ ! -f "${SOFT_THROW_URDF}" ]]; then
  echo "SOFT_THROW_URDF not found: ${SOFT_THROW_URDF}" >&2
  exit 1
fi

if [[ ! -f "${SOFT_THROW_ROBOT_USD}" ]]; then
  echo "SOFT_THROW_ROBOT_USD not found: ${SOFT_THROW_ROBOT_USD}" >&2
  exit 1
fi

if [[ ! -f "${SOFT_THROW_LOCAL_FRANKA_USD}" ]]; then
  echo "SOFT_THROW_LOCAL_FRANKA_USD not found: ${SOFT_THROW_LOCAL_FRANKA_USD}" >&2
  exit 1
fi
