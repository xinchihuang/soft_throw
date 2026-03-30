#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
import subprocess
from pathlib import Path
from typing import Tuple

import numpy as np

from .throw_params import URDF, PKG, EE_FRAME, LEARN_JIDX, J_DLS_LAMBDA
from .pinocchio_compat import load_pinocchio


pin, RobotWrapper = load_pinocchio()


def _resolve_urdf_and_pkg(urdf: str, pkg: str) -> Tuple[str, str]:
    urdf = os.path.expanduser(urdf)
    pkg = os.path.expanduser(pkg)

    if os.path.isfile(urdf):
        return urdf, pkg

    env_urdf = os.environ.get("SOFT_THROW_URDF")
    env_pkg = os.environ.get("SOFT_THROW_PKG")
    if env_urdf and os.path.isfile(os.path.expanduser(env_urdf)):
        return os.path.expanduser(env_urdf), os.path.expanduser(env_pkg) if env_pkg else pkg

    xacro = shutil.which("xacro")
    ros_share = "/opt/ros/noetic/share"
    panda_xacro = "/opt/ros/noetic/share/franka_description/robots/panda/panda.urdf.xacro"
    if xacro and os.path.isfile(panda_xacro):
        out_path = Path("/tmp/soft_throw_panda.urdf")
        result = subprocess.run([xacro, panda_xacro], check=True, capture_output=True, text=True)
        out_path.write_text(result.stdout)
        return str(out_path), ros_share

    raise FileNotFoundError(
        "Could not resolve a valid Panda URDF. "
        f"Configured path does not exist: {urdf}. "
        "Set SOFT_THROW_URDF and optionally SOFT_THROW_PKG, or install franka_description with xacro."
    )


class PinKinematics:
    def __init__(self, urdf: str = URDF, pkg: str = PKG, ee_frame: str = EE_FRAME):
        urdf, pkg = _resolve_urdf_and_pkg(urdf, pkg)
        self.robot = RobotWrapper.BuildFromURDF(urdf, package_dirs=[pkg])
        self.model = self.robot.model
        self.data = self.robot.data
        self.ee_id = self.model.getFrameId(ee_frame)
        if self.ee_id == len(self.model.frames):
            raise RuntimeError(f"[pin] EE frame not found: {ee_frame}")
        print(f"[pin] urdf={urdf} pkg={pkg}", flush=True)
        print(f"[pin] model.nq={self.model.nq} nv={self.model.nv} ee={ee_frame} id={self.ee_id}", flush=True)

    def _pin_forward(self, q_full):
        pin.forwardKinematics(self.model, self.data, q_full)
        pin.updateFramePlacements(self.model, self.data)

    def make_q_full_from_arm7(self, q_arm7: np.ndarray) -> np.ndarray:
        q = pin.neutral(self.model).copy()
        # panda_arm_hand includes fingers; keep them zero
        for i in range(1, 8):
            jid = self.model.getJointId(f"panda_joint{i}")
            idx_q = int(self.model.joints[jid].idx_q)
            q[idx_q] = float(q_arm7[i - 1])
        for nm in ["panda_finger_joint1", "panda_finger_joint2"]:
            if not self.model.existJointName(nm):
                continue
            jid = self.model.getJointId(nm)
            if 0 < jid < len(self.model.joints):
                idx_q = int(self.model.joints[jid].idx_q)
                q[idx_q] = 0.0
        return q

    def get_ee_world_pos(self, q_cmd7: np.ndarray) -> np.ndarray:
        q_full = self.make_q_full_from_arm7(q_cmd7)
        self._pin_forward(q_full)
        T = self.data.oMf[self.ee_id]
        return np.array(T.translation, dtype=float)

    def ee_speed_from_qdot7(self, q_cmd7: np.ndarray, qdot7: np.ndarray) -> float:
        q_full = self.make_q_full_from_arm7(q_cmd7)
        self._pin_forward(q_full)
        J = pin.computeFrameJacobian(self.model, self.data, q_full, self.ee_id, pin.ReferenceFrame.WORLD)  # 6 x nv
        # build full qdot (nv) from 7 joints only (others zero)
        v_full = np.zeros(self.model.nv, dtype=float)
        for i in range(1, 8):
            jid = self.model.getJointId(f"panda_joint{i}")
            idx_v = int(self.model.joints[jid].idx_v)
            v_full[idx_v] = float(qdot7[i - 1])
        twist = J @ v_full  # 6
        v_lin = twist[3:6]
        return float(np.linalg.norm(v_lin))

    def jacobian_linear_sub(self, q7: np.ndarray) -> np.ndarray:
        """3x3 linear Jacobian for learned joints (2,4,6) in LOCAL_WORLD_ALIGNED."""
        q7 = np.asarray(q7, dtype=float).reshape(7)
        q_full = self.make_q_full_from_arm7(q7)
        self._pin_forward(q_full)
        J = pin.computeFrameJacobian(
            self.model, self.data, q_full, self.ee_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        J_lin = J[:3, :]
        return J_lin[:, LEARN_JIDX].copy()

    def compute_qdot3_for_v(self, q7: np.ndarray, v_cmd: np.ndarray) -> np.ndarray:
        J = self.jacobian_linear_sub(q7)  # 3x3
        v_cmd = np.asarray(v_cmd, dtype=float).reshape(3)
        lam = float(J_DLS_LAMBDA)
        JJt = J @ J.T
        inv = np.linalg.inv(JJt + (lam * lam) * np.eye(3))
        J_pinv = J.T @ inv
        qdot3 = J_pinv @ v_cmd
        return qdot3.astype(float)

    def compute_qdot7_for_v(self, q7: np.ndarray, v_cmd: np.ndarray) -> np.ndarray:
        """DLS solve for all 7 joints using linear Jacobian (3x7)."""
        q7 = np.asarray(q7, dtype=float).reshape(7)
        q_full = self.make_q_full_from_arm7(q7)
        self._pin_forward(q_full)
        J = pin.computeFrameJacobian(
            self.model, self.data, q_full, self.ee_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        J_lin = J[:3, :]  # 3 x nv
        # build 3x7 by selecting arm joints in order
        J_arm = np.zeros((3, 7), dtype=float)
        for i in range(1, 8):
            jid = self.model.getJointId(f"panda_joint{i}")
            idx_v = int(self.model.joints[jid].idx_v)
            J_arm[:, i - 1] = J_lin[:, idx_v]

        v_cmd = np.asarray(v_cmd, dtype=float).reshape(3)
        lam = float(J_DLS_LAMBDA)
        JJt = J_arm @ J_arm.T
        inv = np.linalg.inv(JJt + (lam * lam) * np.eye(3))
        J_pinv = J_arm.T @ inv
        qdot7 = J_pinv @ v_cmd
        return qdot7.astype(float)
