#!/usr/bin/env bash

set -euo pipefail

export SOFT_THROW_URDF="/home/yifan/.local/share/ov/data/exts/v2/isaacsim.asset.importer.urdf-2.4.31+107.3.3.lx64.r.cp311/data/urdf/robots/franka_description/robots/panda_arm_hand.urdf"
export SOFT_THROW_PKG="/home/yifan/.local/share/ov/data/exts/v2/isaacsim.asset.importer.urdf-2.4.31+107.3.3.lx64.r.cp311/data/urdf/robots"

if [[ ! -f "${SOFT_THROW_URDF}" ]]; then
  echo "SOFT_THROW_URDF not found: ${SOFT_THROW_URDF}" >&2
  exit 1
fi
