#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Core parameters for spoon throw task.

This module is Isaac-Sim-agnostic (no Omniverse imports).
"""

import os
import math
import numpy as np

# ============================================================
# USER CONFIG (paths)
# ============================================================
ROBOT_USD = "/home/xinchi/Desktop/lacosse_large_mobie_manipulator_new.usd"
ROBOT_PRIM = "/World/LMM"
FRANKA_ROOT = "/World/LMM/Franka"

BALL_PATH = "/World/LMM/Ball"
LACROSSE_PATH = "/World/LMM/Franka/lacrosse"

# Pinocchio URDF
URDF = "/home/xinchi/isaac-sim/exts/isaacsim.asset.importer.urdf/data/urdf/robots/franka_description/robots/panda_arm_hand.urdf"
PKG = "/home/xinchi/isaac-sim/exts/isaacsim.asset.importer.urdf/data/urdf/robots"
EE_FRAME = "panda_link8"   # end-effector frame


# ============================================================
# TRAIN / CONTROL CONFIG
# ============================================================
CONTROL_HZ = 100.0
DT_CONTROL = 1.0 / CONTROL_HZ
WAYPOINT_DENSITY = 10

# For speed: keep this low. If you set this high, raising CONTROL_HZ will slow wall-clock.
SIM_UPDATES_PER_STEP = 1

# You learn only joints 2,4,6
LEARN_JIDX = np.array([1, 3, 5], dtype=int)   # 0-based indices into 7 joints

# Action -> per-step residual delta (rad/step) (tune for faster motion)
DQ_MAX_RESID_3 = np.array([0.6, 0.6, 1.0], dtype=float)

# Joint speed limit in joint space (rad/s) for learned joints
QDOT_LIMITS_3 = np.array([3.5, 3.5, 4.0], dtype=float)

# Joint position limits for all 7 joints (rad)
Q_LIMITS_7 = np.array([
    [-2.897,  2.897],
    [-1.762,  1.762],
    [-2.897,  2.897],
    [-3.071, -0.070],
    [-2.897,  2.897],
    [-0.017,  3.752],
    [-2.897,  2.897],
], dtype=float)

# Joint speed limits for all 7 joints (rad/s) (from FCI)
QDOT_LIMITS_7 = np.array([2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100], dtype=float)

# Joint acceleration limits for all 7 joints (rad/s^2) (from FCI)
QDDOT_LIMITS_7 = np.array([15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0], dtype=float)

# Optional action EMA smoothing (set higher => less sluggish)
ACTION_EMA_ALPHA = 0.50

# End-effector linear speed limit (m/s) (user constraint)
EE_VEL_MAX = 2.0

# ============================================================
# TUBE BASELINE (robust release window) CONFIG
# ============================================================
# Enable tube-style baseline controller (no RL residual by default)
USE_TUBE_BASELINE = True

# Keep residual interface for future RL (default off)
USE_RESIDUAL = False
BALL_GROUND_Z = 0
# Target landing (world XYZ). z is usually ground height.
# Default target (will be overridden each episode if RANDOMIZE_TARGET=True)
TARGET_XYZ = np.array([2.0, 0.0, BALL_GROUND_Z], dtype=float)

# Random target sampling (episode-wise)
RANDOMIZE_TARGET = True
TARGET_X_RANGE = (1.2, 2.0)   # meters, world X
TARGET_Y_RANGE = (-1.0, 1.0)  # meters, world Y
TARGET_Z = BALL_GROUND_Z      # meters, world Z (landing plane)
JOINT0_YAW_LIMIT = math.radians(80.0)  # joint0 yaw clamp for target alignment


# Ballistic solve: search flight time [s]
BALLISTIC_T_MIN = 0.25
BALLISTIC_T_MAX = 0.70
BALLISTIC_T_STEP = 0.01

# Tube timing (seconds since episode start AFTER prep)
TUBE_T0 = 0.20           # end of ramp-up, start of robust-release window
TUBE_HALF_WINDOW = 0.06  # +/- seconds => window length 2*half
TUBE_DECEL_SEC = 0.25    # decelerate after tube window

# Tube velocity constraints (task-space)
TUBE_CONE_DEG = 12.0     # allowed angular deviation from nominal release velocity
TUBE_V_MIN_RATIO = 0.85  # min speed ratio vs |v*|
TUBE_V_MAX_RATIO = 1.15  # max speed ratio vs |v*|

# Damped least-squares for task-space -> joint-space
J_DLS_LAMBDA = 0.05

# Residual interface (disabled by default; keep for later RL)
RESID_VEL_MAX_3 = np.array([0.5, 0.5, 0.5], dtype=float)  # m/s residual delta-v per axis (when enabled)


# ============================================================
# TUBE ACCELERATION RELEASE PLANNER (landing -> release pos/vel)
# ============================================================
# Enable tube-acceleration planner to pick release point/velocity from landing target.
# Falls back to original ballistic solver if no candidate found.
USE_TUBE_ACCELERATION_PLANNER = True

# BRT settings (throwing plane: r-z)
TUBE_ACCELERATION_T_BACK = 1.0
TUBE_ACCELERATION_DT = 0.01
TUBE_ACCELERATION_N_LANDING = 1200
TUBE_ACCELERATION_RDOT_MIN = 0.2
TUBE_ACCELERATION_RDOT_MAX = 4.0
TUBE_ACCELERATION_ZDOT_MIN = -8.0
TUBE_ACCELERATION_ZDOT_MAX = -2.0
TUBE_ACCELERATION_EPS_R = 0.0
TUBE_ACCELERATION_EPS_Z = 0.0
TUBE_ACCELERATION_MAX_ABS_RDOT = 5.0
TUBE_ACCELERATION_MAX_ABS_ZDOT = 5.0
TUBE_ACCELERATION_SEED = 0

# Candidate filtering/scoring
TUBE_ACCELERATION_MAX_CANDIDATES = 2000
TUBE_ACCELERATION_Z_MIN = -0.2
TUBE_ACCELERATION_Z_MAX = 2.0
TUBE_ACCELERATION_Z_PREF = 1.0
TUBE_ACCELERATION_MAX_RELEASE_OFFSET = 1.5
TUBE_ACCELERATION_W_RANGE = 1.0
TUBE_ACCELERATION_W_MARGIN = 0.5
TUBE_ACCELERATION_W_HEIGHT = 0.1
TUBE_ACCELERATION_W_OFFSET = 0.6
TUBE_ACCELERATION_DEBUG = True


# Episode timing
MAX_EPISODE_SEC = 4.0
DROP_TIMEOUT_AFTER_RELEASE_SEC = 2.0  # after detach

# Reset logic (user requirement)
RESET_ARM_SETTLE_SEC = 2.0   # arm -> INIT hold
BALL_RESET_POS_WORLD = (-0.75, 0.0, 0.6)  # user requirement
RESET_BALL_WAIT_SEC = 3.0    # after ball reset wait 2s then execute

# Arm poses
INIT_ARM = np.array([0.0, -1., 0.0, +1.57, 0.0, 3.14, 0.0], dtype=float)


# Prep trajectory from INIT -> THROW_START
PREP_SEC = 1.0
PREP_STEPS = max(2, int(PREP_SEC * CONTROL_HZ))
JOINT0_ALIGN_SEC = 0.8
JOINT0_ALIGN_STEPS = max(2, int(JOINT0_ALIGN_SEC * CONTROL_HZ))
JOINT0_ALIGN_WAIT_SEC = 0.2

# Detach condition (ball far from EE/net)
BALL_DETACH_DIST = 0.20  # meters

# Ball landed condition
BALL_GROUND_Z = 0.0
BALL_LANDED_V = 0.25
BALL_LANDED_Z_THRESH = 0.03  # account for ball collider radius/contact offset

# Reward weights (tune)
W_FORWARD = 2.0
W_LAND = 6.0
W_ACT = 0.02
W_QDOT = 0.01
W_SMOOTH = 0.02
W_DEVIATE = 0.02

# Landing target (world XY)
TARGET_XY = np.array([2.0, 0.0], dtype=float)
LAND_SIGMA = 0.35

# Drive gains (tune for speed without jitter)
STIFF = 4.0e5
DAMP = 7.0e3
MAXF = 3.0e6

# Physics / contact robustness
CCD_ENABLE = True
RB_SOLVER_POS_ITERS = 24
RB_SOLVER_VEL_ITERS = 6
CONTACT_OFFSET = 0.02
REST_OFFSET = 0.0

# Ball non-elastic material
NO_BOUNCE_RESTITUTION = 0.0
NO_BOUNCE_STATIC_FRICTION = 1.2
NO_BOUNCE_DYNAMIC_FRICTION = 1.0

# Demo save
SAVE_DEMO_TRAJ = False
SAVE_DEMO_DIR = os.path.expanduser("~/IsaacLab/runs_qcmd7")
SAVE_ONLY_BEST = False

# Debug
DEBUG_PRINT_ARM_TARGETS = False
