#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import csv
import math
import time
from datetime import datetime
import numpy as np
from typing import Optional

import gymnasium as gym
from gymnasium import spaces

# IMPORTANT: omni imports must happen after SimulationApp/AppLauncher is created (in train script)
import omni.usd
import omni.timeline
from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, Gf

try:
    from pxr import UsdLux
except Exception:
    UsdLux = None

import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper


# ============================================================
# USER CONFIG (paths)
# ============================================================
ROBOT_USD = "/home/xinchi/Desktop/lacosse_large_mobie_manipulator.usd"
ROBOT_PRIM = "/World/LMM"
FRANKA_ROOT = "/World/LMM/Franka"

BALL_PATH = "/World/LMM/Ball"
LACROSSE_PATH = "/World/LMM/Franka/lacrosse__1_"

# Pinocchio URDF
URDF = "/home/xinchi/isaac-sim/exts/isaacsim.asset.importer.urdf/data/urdf/robots/franka_description/robots/panda_arm_hand.urdf"
PKG  = "/home/xinchi/isaac-sim/exts/isaacsim.asset.importer.urdf/data/urdf/robots"
EE_FRAME = "panda_link8"   # end-effector frame


# ============================================================
# TRAIN / CONTROL CONFIG
# ============================================================
CONTROL_HZ = 30.0
DT_CONTROL = 1.0 / CONTROL_HZ

# For speed: keep this low. If you set this high, raising CONTROL_HZ will slow wall-clock.
SIM_UPDATES_PER_STEP = 1

# You learn only joints 2,4,6
LEARN_JIDX = np.array([1, 3, 5], dtype=int)   # 0-based indices into 7 joints

# Action -> per-step residual delta (rad/step) (tune for faster motion)
DQ_MAX_RESID_3 = np.array([0.6, 0.6, 1.0], dtype=float)

# Joint speed limit in joint space (rad/s) for learned joints
QDOT_LIMITS_3 = np.array([3.5, 3.5, 4.0], dtype=float)

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
BALL_GROUND_Z =0
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
BALL_RESET_POS_WORLD = (-0.75, 0.0, 1.0)  # user requirement
RESET_BALL_WAIT_SEC = 3.0    # after ball reset wait 2s then execute

# Arm poses
INIT_ARM = np.array([0.0, -1.57, 0.0,  1.57, 0.0, 3.14, 0.0], dtype=float)

# Throw start target (user gave)
THROW_START_Q = np.array([0.0, 0, 0.0, 0, 0.0, 1.8, 0.0], dtype=float)

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
DAMP  = 7.0e3
MAXF  = 3.0e6

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


# ============================================================
# Helpers
# ============================================================
def _get_stage():
    return omni.usd.get_context().get_stage()

def _valid(stage, path: str) -> bool:
    prim = stage.GetPrimAtPath(path)
    return bool(prim and prim.IsValid())

def _find_child_prim_by_name(root_prim: Usd.Prim, name: str):
    if not (root_prim and root_prim.IsValid()):
        return None
    stack = [root_prim]
    while stack:
        p = stack.pop()
        if p.GetName() == name:
            return p
        for c in p.GetChildren():
            stack.append(c)
    return None

def _get_or_add_translate_op(xf: UsdGeom.Xformable) -> UsdGeom.XformOp:
    for op in xf.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            return op
    return xf.AddTranslateOp()

def _spawn_lights(stage):
    if UsdLux is None:
        print("[scene] UsdLux not available, skip lights", flush=True)
        return
    root = "/World/Lights"
    if not _valid(stage, root):
        stage.DefinePrim(root, "Xform")

    sun = f"{root}/Sun"
    if not _valid(stage, sun):
        p = stage.DefinePrim(sun, "DistantLight")
        UsdLux.DistantLight(p).CreateIntensityAttr(2500.0)
        UsdGeom.Xformable(p).AddRotateXYZOp().Set(Gf.Vec3f(-45, 45, 0))

    dome = f"{root}/Dome"
    if not _valid(stage, dome):
        p = stage.DefinePrim(dome, "DomeLight")
        UsdLux.DomeLight(p).CreateIntensityAttr(800.0)

    print("[scene] lights ready", flush=True)

def _ensure_physics_scene(stage):
    path = "/World/physicsScene"
    if _valid(stage, path):
        return
    prim = stage.DefinePrim(path, "PhysicsScene")
    scene = UsdPhysics.Scene(prim)
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr(9.81)
    print("[scene] physicsScene created", flush=True)

def _spawn_ground(stage):
    if _valid(stage, "/World/Ground/Mesh"):
        return
    stage.DefinePrim("/World/Ground", "Xform")
    mesh_prim = stage.DefinePrim("/World/Ground/Mesh", "Mesh")
    mesh = UsdGeom.Mesh(mesh_prim)

    size = 50.0
    mesh.CreatePointsAttr([
        Gf.Vec3f(-size, -size, 0.0),
        Gf.Vec3f(+size, -size, 0.0),
        Gf.Vec3f(+size, +size, 0.0),
        Gf.Vec3f(-size, +size, 0.0),
    ])
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateSubdivisionSchemeAttr("none")

    UsdPhysics.CollisionAPI.Apply(mesh_prim)
    PhysxSchema.PhysxCollisionAPI.Apply(mesh_prim)
    UsdGeom.Imageable(mesh_prim).MakeVisible()
    print("[scene] ground ready", flush=True)

def _add_robot_reference(stage, prim_path, usd_path):
    if usd_path.startswith("/"):
        usd_path = "file://" + usd_path
    if not _valid(stage, prim_path):
        stage.DefinePrim(prim_path, "Xform")
    stage.GetPrimAtPath(prim_path).GetReferences().AddReference(usd_path)
    print(f"[scene] robot referenced: {prim_path}", flush=True)

def _detect_franka_root(stage, want_root=FRANKA_ROOT):
    if _valid(stage, want_root):
        return want_root
    for cand in ["/World/LMM/Franka", "/World/Franka", "/World/LMM/franka", "/World/LMM/panda"]:
        if _valid(stage, cand):
            return cand
    raise RuntimeError(f"[usd] Franka root not found under {want_root} and fallbacks")

def _set_drive_target_rad(joint_prim, q_rad: float, qd_rad_s: float):
    if not UsdPhysics.RevoluteJoint(joint_prim):
        return
    drv = UsdPhysics.DriveAPI.Apply(joint_prim, "angular")

    if not drv.GetTargetPositionAttr():
        drv.CreateTargetPositionAttr(0.0)
    if not drv.GetTargetVelocityAttr():
        drv.CreateTargetVelocityAttr(0.0)
    if not drv.GetStiffnessAttr():
        drv.CreateStiffnessAttr(float(STIFF))
    if not drv.GetDampingAttr():
        drv.CreateDampingAttr(float(DAMP))
    if not drv.GetMaxForceAttr():
        drv.CreateMaxForceAttr(float(MAXF))

    drv.GetTargetPositionAttr().Set(float(math.degrees(q_rad)))
    drv.GetTargetVelocityAttr().Set(float(math.degrees(qd_rad_s)))
    drv.GetStiffnessAttr().Set(float(STIFF))
    drv.GetDampingAttr().Set(float(DAMP))
    drv.GetMaxForceAttr().Set(float(MAXF))

def _tune_rb_collision_recursive(stage, root_path: str, label: str):
    if not _valid(stage, root_path):
        print(f"[tune] {label}: prim not found: {root_path}", flush=True)
        return
    root = stage.GetPrimAtPath(root_path)
    stack = [root]
    n_rb = 0
    n_col = 0
    while stack:
        p = stack.pop()

        # rigidbody params + CCD
        try:
            rb = PhysxSchema.PhysxRigidBodyAPI.Apply(p)
            rb = PhysxSchema.PhysxRigidBodyAPI(p)
            if rb:
                n_rb += 1
                if rb.GetSolverPositionIterationCountAttr():
                    rb.GetSolverPositionIterationCountAttr().Set(int(RB_SOLVER_POS_ITERS))
                else:
                    rb.CreateSolverPositionIterationCountAttr(int(RB_SOLVER_POS_ITERS))

                if rb.GetSolverVelocityIterationCountAttr():
                    rb.GetSolverVelocityIterationCountAttr().Set(int(RB_SOLVER_VEL_ITERS))
                else:
                    rb.CreateSolverVelocityIterationCountAttr(int(RB_SOLVER_VEL_ITERS))

                if CCD_ENABLE:
                    if rb.GetEnableCCDAttr():
                        rb.GetEnableCCDAttr().Set(True)
                    else:
                        rb.CreateEnableCCDAttr(True)
        except Exception:
            pass

        # collision offsets
        try:
            pc = PhysxSchema.PhysxCollisionAPI.Apply(p)
            pc = PhysxSchema.PhysxCollisionAPI(p)
            if pc:
                n_col += 1
                if pc.GetContactOffsetAttr():
                    pc.GetContactOffsetAttr().Set(float(CONTACT_OFFSET))
                else:
                    pc.CreateContactOffsetAttr(float(CONTACT_OFFSET))
                if pc.GetRestOffsetAttr():
                    pc.GetRestOffsetAttr().Set(float(REST_OFFSET))
                else:
                    pc.CreateRestOffsetAttr(float(REST_OFFSET))
        except Exception:
            pass

        for c in p.GetChildren():
            stack.append(c)

    print(f"[tune] {label}: RB={n_rb}, COL={n_col}", flush=True)

def _apply_no_bounce_material(stage, prim_path: str, label: str):
    """
    Apply (or create) a physics material with restitution=0 and high friction.
    Bind it to the prim (and its subtree collisions).
    """
    if not _valid(stage, prim_path):
        print(f"[mat] {label}: prim not found: {prim_path}", flush=True)
        return

    mat_path = "/World/Materials/NoBounce"
    if not _valid(stage, "/World/Materials"):
        stage.DefinePrim("/World/Materials", "Scope")

    if not _valid(stage, mat_path):
        mprim = stage.DefinePrim(mat_path, "Material")
        mapi = UsdPhysics.MaterialAPI.Apply(mprim)
        mapi.CreateRestitutionAttr(NO_BOUNCE_RESTITUTION)
        mapi.CreateStaticFrictionAttr(NO_BOUNCE_STATIC_FRICTION)
        mapi.CreateDynamicFrictionAttr(NO_BOUNCE_DYNAMIC_FRICTION)
        # PhysX extras (combine modes) best effort
        try:
            pm = PhysxSchema.PhysxMaterialAPI.Apply(mprim)
            if pm.GetRestitutionCombineModeAttr():
                pm.GetRestitutionCombineModeAttr().Set("min")
            else:
                pm.CreateRestitutionCombineModeAttr("min")
            if pm.GetFrictionCombineModeAttr():
                pm.GetFrictionCombineModeAttr().Set("max")
            else:
                pm.CreateFrictionCombineModeAttr("max")
        except Exception:
            pass
        print("[mat] created /World/Materials/NoBounce", flush=True)

    mat_prim = stage.GetPrimAtPath(mat_path)
    target = stage.GetPrimAtPath(prim_path)
    stack = [target]
    n_bind = 0
    while stack:
        p = stack.pop()
        try:
            # bind only if collision exists or can exist
            UsdPhysics.MaterialBindingAPI.Apply(p).Bind(mat_prim)
            n_bind += 1
        except Exception:
            pass
        for c in p.GetChildren():
            stack.append(c)

    print(f"[mat] {label}: bound NoBounce to {n_bind} prims under {prim_path}", flush=True)

def embed_resid3_to_resid7(resid3: np.ndarray) -> np.ndarray:
    resid7 = np.zeros(7, dtype=float)
    resid7[LEARN_JIDX[0]] = float(resid3[0])
    resid7[LEARN_JIDX[1]] = float(resid3[1])
    resid7[LEARN_JIDX[2]] = float(resid3[2])
    return resid7

def landing_reward(landing_xy: np.ndarray, target_xy: np.ndarray = TARGET_XY) -> float:
    d = landing_xy - np.asarray(target_xy, dtype=float).reshape(2)
    dd = float(np.dot(d, d))
    return math.exp(-dd / (2.0 * LAND_SIGMA * LAND_SIGMA))


# ============================================================
# Tube-acceleration release planner (function-only, no classes)
# ============================================================
_TUBE_ACCELERATION_BRT_CACHE = None

def _unit(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        return v * 0.0
    return v / n

def _tube_acceleration_sample_landing(n: int, rng: np.random.Generator) -> np.ndarray:
    r = rng.uniform(-TUBE_ACCELERATION_EPS_R, TUBE_ACCELERATION_EPS_R, size=n)
    z = rng.uniform(-TUBE_ACCELERATION_EPS_Z, TUBE_ACCELERATION_EPS_Z, size=n)
    rdot = rng.uniform(TUBE_ACCELERATION_RDOT_MIN, TUBE_ACCELERATION_RDOT_MAX, size=n)
    zdot = rng.uniform(TUBE_ACCELERATION_ZDOT_MIN, TUBE_ACCELERATION_ZDOT_MAX, size=n)
    return np.stack([r, z, rdot, zdot], axis=1).astype(float)

def _tube_acceleration_generate_brt_states() -> np.ndarray:
    global _TUBE_ACCELERATION_BRT_CACHE
    if _TUBE_ACCELERATION_BRT_CACHE is not None:
        return _TUBE_ACCELERATION_BRT_CACHE

    rng = np.random.default_rng(TUBE_ACCELERATION_SEED)
    landing = _tube_acceleration_sample_landing(TUBE_ACCELERATION_N_LANDING, rng)
    t_steps = int(max(1, round(TUBE_ACCELERATION_T_BACK / TUBE_ACCELERATION_DT)))
    g = 9.81
    trajs = []
    for x0 in landing:
        r, z, rdot, zdot = [float(v) for v in x0]
        for _ in range(t_steps + 1):
            trajs.append([r, z, rdot, zdot])
            # backward integrate (projectile): reverse time for position, gravity still increases zdot
            r += -rdot * TUBE_ACCELERATION_DT
            z += -zdot * TUBE_ACCELERATION_DT
            zdot += g * TUBE_ACCELERATION_DT

    X = np.array(trajs, dtype=float)
    mask = (np.abs(X[:, 2]) <= TUBE_ACCELERATION_MAX_ABS_RDOT) & (np.abs(X[:, 3]) <= TUBE_ACCELERATION_MAX_ABS_ZDOT)
    X = X[mask]
    _TUBE_ACCELERATION_BRT_CACHE = X
    return X

def _tube_acceleration_vmax_for_direction(J_lin: np.ndarray, qdot_limits: np.ndarray, d_world: np.ndarray) -> float:
    # J_lin: 3x3 linear Jacobian for learned joints in world-aligned frame
    d_world = _unit(np.asarray(d_world, dtype=float).reshape(3))
    if float(np.linalg.norm(d_world)) < 1e-9:
        return 0.0
    J_pinv = np.linalg.pinv(J_lin, rcond=1e-6)
    qdot_dir = J_pinv @ d_world
    s_hi = float("inf")
    for i in range(3):
        a = float(qdot_dir[i])
        if abs(a) < 1e-9:
            continue
        limit = float(qdot_limits[i])
        s_hi = min(s_hi, limit / abs(a))
    if not math.isfinite(s_hi) or s_hi <= 0.0:
        return 0.0
    return float(s_hi)

def _tube_acceleration_select_release(
    target_xyz: np.ndarray,
    yaw: float,
    p_release_nom: np.ndarray,
    J_lin: np.ndarray,
    qdot_limits: np.ndarray,
) -> Optional[dict]:
    """
    Tube-acceleration candidate selection in the yaw-aligned throwing plane.
    Returns dict with release_pos_w, release_vel_w, brt_state, score, vmax, speed, margin.
    """
    X = _tube_acceleration_generate_brt_states()
    if X.size == 0:
        return None

    # fixed plane by joint0 yaw
    e_r = np.array([math.cos(yaw), math.sin(yaw), 0.0], dtype=float)
    e_z = np.array([0.0, 0.0, 1.0], dtype=float)
    target_xyz = np.asarray(target_xyz, dtype=float).reshape(3)
    p_release_nom = np.asarray(p_release_nom, dtype=float).reshape(3)

    if X.shape[0] > TUBE_ACCELERATION_MAX_CANDIDATES:
        stride = max(1, X.shape[0] // TUBE_ACCELERATION_MAX_CANDIDATES)
        X = X[::stride]

    best = None
    cnt_total = 0
    cnt_z = 0
    cnt_vmax0 = 0
    cnt_speed = 0
    cnt_offset = 0
    z_pass = []
    for x in X:
        cnt_total += 1
        r, z, rdot, zdot = [float(v) for v in x]
        if z < TUBE_ACCELERATION_Z_MIN or z > TUBE_ACCELERATION_Z_MAX:
            cnt_z += 1
            continue
        z_pass.append(z)

        speed = math.hypot(rdot, zdot)
        if speed < 1e-6:
            continue

        # direction in plane
        gamma = math.atan2(zdot, rdot)
        d = math.cos(gamma) * e_r + math.sin(gamma) * e_z
        vmax = _tube_acceleration_vmax_for_direction(J_lin, qdot_limits, d)
        if vmax <= 1e-6:
            cnt_vmax0 += 1
            continue
        if speed > vmax:
            cnt_speed += 1
            continue

        # map to world release
        E = target_xyz - r * e_r + z * e_z
        v = rdot * e_r + zdot * e_z
        offset = float(np.linalg.norm(E - p_release_nom))
        if offset > TUBE_ACCELERATION_MAX_RELEASE_OFFSET:
            cnt_offset += 1
            continue

        margin = vmax - speed
        score = (
            TUBE_ACCELERATION_W_RANGE * abs(r) +
            TUBE_ACCELERATION_W_MARGIN * (1.0 / (1e-6 + margin)) +
            TUBE_ACCELERATION_W_HEIGHT * abs(z - TUBE_ACCELERATION_Z_PREF) +
            TUBE_ACCELERATION_W_OFFSET * offset
        )
        cand = {
            "release_pos_w": E,
            "release_vel_w": v,
            "brt_state": np.array([r, z, rdot, zdot], dtype=float),
            "score": float(score),
            "vmax": float(vmax),
            "speed": float(speed),
            "margin": float(margin),
        }
        if (best is None) or (cand["score"] < best["score"]):
            best = cand

    if (best is None) and TUBE_ACCELERATION_DEBUG:
        z_all = X[:, 1] if X.size > 0 else np.zeros((0,), dtype=float)
        z_min = float(np.min(z_all)) if z_all.size > 0 else 0.0
        z_max = float(np.max(z_all)) if z_all.size > 0 else 0.0
        if z_all.size > 0 and z_max > z_min:
            bins = np.linspace(z_min, z_max, 11)
            hist_all, _ = np.histogram(z_all, bins=bins)
            hist_pass, _ = np.histogram(np.asarray(z_pass, dtype=float), bins=bins)
            bins_list = [float(b) for b in bins]
            hist_all_list = [int(v) for v in hist_all]
            hist_pass_list = [int(v) for v in hist_pass]
        else:
            bins_list = []
            hist_all_list = []
            hist_pass_list = []
        print(
            f"[tube_acceleration] no candidate: total={cnt_total} "
            f"z_out={cnt_z} vmax0={cnt_vmax0} speed_gt_vmax={cnt_speed} offset_out={cnt_offset}",
            flush=True,
        )
        print(
            f"[tube_acceleration] z_range_all=[{z_min:.3f}, {z_max:.3f}] "
            f"z_bins={bins_list} z_hist_all={hist_all_list} z_hist_pass={hist_pass_list}",
            flush=True,
        )
    return best


# ============================================================
# Env
# ============================================================
class SpoonThrowEnvCoreTubeRandomYawAlign(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, simulation_app, seed: int = 0, use_residual=None):
        super().__init__()
        self.simulation_app = simulation_app
        self.np_random = np.random.default_rng(seed)

        self.stage = _get_stage()
        self.timeline = omni.timeline.get_timeline_interface()

        # build/ensure stage content (lights, ground, physics, robot ref)
        self.timeline.stop()
        for _ in range(30):
            self.simulation_app.update()

        _ensure_physics_scene(self.stage)
        _spawn_ground(self.stage)
        _spawn_lights(self.stage)
        _add_robot_reference(self.stage, ROBOT_PRIM, ROBOT_USD)

        # warmup
        for _ in range(240):
            self.simulation_app.update()

        self.franka_root = _detect_franka_root(self.stage, FRANKA_ROOT)
        self.joint_paths = self._find_joint_paths(self.franka_root)

        # tune rigidbodies / collisions and materials
        _tune_rb_collision_recursive(self.stage, BALL_PATH, "BALL")
        _tune_rb_collision_recursive(self.stage, LACROSSE_PATH, "LACROSSE")
        _apply_no_bounce_material(self.stage, BALL_PATH, "BALL")
        _apply_no_bounce_material(self.stage, LACROSSE_PATH, "LACROSSE")

        # build pinocchio model
        self.robot, self.ee_id = self._build_pin_robot()
        self.pin_model = self.robot.model
        self.pin_data = self.robot.data

        # state
        self.episode_t = 0.0
        self.released = False
        self.release_time_wall = None
        self.release_time_ep = None
        self.release_step = None
        self.step_count = 0

        # action filter / residuals
        self._a_prev3 = np.zeros(3, dtype=float)
        self._a_filt3 = np.zeros(3, dtype=float)
        self.q_resid3 = np.zeros(3, dtype=float)

        # command
        self.q_cmd7 = INIT_ARM.copy()
        self.qdot7 = np.zeros(7, dtype=float)

        # tube baseline plan (3D)
        self.use_tube_baseline = bool(USE_TUBE_BASELINE)
        self.use_residual = bool(USE_RESIDUAL if use_residual is None else use_residual)
        self.target_xyz = TARGET_XYZ.astype(float).copy()
        self.throw_start_q = THROW_START_Q.copy()
        self.tube_t0 = float(TUBE_T0)
        self.tube_t1 = float(TUBE_T0 + 2.0 * TUBE_HALF_WINDOW)
        self.v_release_star = np.zeros(3, dtype=float)
        self.v_release_hat = np.array([1.0, 0.0, 0.0], dtype=float)
        self.p_release_nom = np.zeros(3, dtype=float)
        self.release_plan_method = "none"
        self.release_plan_pos = np.zeros(3, dtype=float)
        self.release_plan_vel = np.zeros(3, dtype=float)

        # ball tracking
        self.ball_prev_x = 0.0

        # recording for demo
        self.demo_ep_counter = 0
        self.best_land_score = -1e9
        self._clear_recording_buffers()

        # gym spaces
        # obs: [ball_pos(3), ball_vel(3), ee_pos(3), resid3(3), released(1), time(1)]
        obs_dim = 3 + 3 + 3 + 3 + 1 + 1
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # start sim
        self.timeline.play()

    # ---------------- pinocchio ----------------
    def _build_pin_robot(self):
        robot = RobotWrapper.BuildFromURDF(URDF, package_dirs=[PKG])
        m = robot.model
        ee_id = m.getFrameId(EE_FRAME)
        if ee_id == len(m.frames):
            raise RuntimeError(f"[pin] EE frame not found: {EE_FRAME}")
        print(f"[pin] model.nq={m.nq} nv={m.nv} ee={EE_FRAME} id={ee_id}", flush=True)
        return robot, ee_id

    def _pin_forward(self, q_full):
        pin.forwardKinematics(self.pin_model, self.pin_data, q_full)
        pin.updateFramePlacements(self.pin_model, self.pin_data)

    def _make_q_full_from_arm7(self, q_arm7):
        q = pin.neutral(self.pin_model).copy()
        # panda_arm_hand includes fingers; keep them zero
        for i in range(1, 8):
            jid = self.pin_model.getJointId(f"panda_joint{i}")
            idx_q = int(self.pin_model.joints[jid].idx_q)
            q[idx_q] = float(q_arm7[i - 1])
        for nm in ["panda_finger_joint1", "panda_finger_joint2"]:
            jid = self.pin_model.getJointId(nm)
            if jid != 0:
                idx_q = int(self.pin_model.joints[jid].idx_q)
                q[idx_q] = 0.0
        return q

    def _get_ee_world_pos(self):
        q_full = self._make_q_full_from_arm7(self.q_cmd7)
        self._pin_forward(q_full)
        T = self.pin_data.oMf[self.ee_id]
        return np.array(T.translation, dtype=float)

    def _ee_speed_from_qdot7(self, q_cmd7, qdot7):
        q_full = self._make_q_full_from_arm7(q_cmd7)
        self._pin_forward(q_full)
        J = pin.computeFrameJacobian(self.pin_model, self.pin_data, q_full, self.ee_id, pin.ReferenceFrame.WORLD)  # 6 x nv
        # build full qdot (nv) from 7 joints only (others zero)
        v_full = np.zeros(self.pin_model.nv, dtype=float)
        for i in range(1, 8):
            jid = self.pin_model.getJointId(f"panda_joint{i}")
            idx_v = int(self.pin_model.joints[jid].idx_v)
            v_full[idx_v] = float(qdot7[i - 1])
        twist = J @ v_full  # 6
        v_lin = twist[3:6]
        return float(np.linalg.norm(v_lin))

    # ---------------- USD joints ----------------
    def _find_joint_paths(self, franka_root):
        root = self.stage.GetPrimAtPath(franka_root)
        out = {}
        for i in range(1, 8):
            nm = f"panda_joint{i}"
            p = _find_child_prim_by_name(root, nm)
            if p is None:
                raise RuntimeError(f"[usd] cannot find {nm} under {franka_root}")
            out[nm] = p.GetPath().pathString
        print("[usd] joints ready", flush=True)
        return out

    def _apply_arm_targets(self, q_cmd7, qdot7):
        for i in range(1, 8):
            prim = self.stage.GetPrimAtPath(self.joint_paths[f"panda_joint{i}"])
            _set_drive_target_rad(prim, float(q_cmd7[i - 1]), float(qdot7[i - 1]))

    # ---------------- Ball ----------------
    def _reset_ball(self, pos_world):
        if not _valid(self.stage, BALL_PATH):
            print(f"[ball] not found: {BALL_PATH}", flush=True)
            return
        prim = self.stage.GetPrimAtPath(BALL_PATH)
        xf = UsdGeom.Xformable(prim)
        op = _get_or_add_translate_op(xf)
        op.Set(Gf.Vec3d(*pos_world))
        # best-effort zero velocity
        try:
            rb = UsdPhysics.RigidBodyAPI(prim)
            v = rb.GetVelocityAttr()
            w = rb.GetAngularVelocityAttr()
            if v:
                v.Set(Gf.Vec3f(0.0, 0.0, 0.0))
            if w:
                w.Set(Gf.Vec3f(0.0, 0.0, 0.0))
        except Exception:
            pass

    def _get_ball_world_pos_vel(self):
        if not _valid(self.stage, BALL_PATH):
            return None, None, None
        prim = self.stage.GetPrimAtPath(BALL_PATH)
        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        M = cache.GetLocalToWorldTransform(prim)
        p = M.Transform(Gf.Vec3d(0.0, 0.0, 0.0))
        pos = np.array([p[0], p[1], p[2]], dtype=float)

        vel = None
        speed = None
        try:
            rb = UsdPhysics.RigidBodyAPI(prim)
            v_attr = rb.GetVelocityAttr()
            if v_attr:
                v = v_attr.Get()
                vel = np.array([v[0], v[1], v[2]], dtype=float)
                speed = float(np.linalg.norm(vel))
        except Exception:
            pass
        return pos, vel, speed

    def _ball_landed(self) -> bool:
        pos, vel, spd = self._get_ball_world_pos_vel()
        if pos is None:
            return False
        if pos[2] > BALL_LANDED_Z_THRESH:
            return False
        return True

    def _ball_detached_from_net(self) -> bool:
        bpos, _, _ = self._get_ball_world_pos_vel()
        if bpos is None:
            return False
        ee = self._get_ee_world_pos()
        return float(np.linalg.norm(bpos - ee)) > BALL_DETACH_DIST

    # ---------------- Recording & Saving ----------------
    def _clear_recording_buffers(self):
        self.seq_t = []
        self.seq_q_cmd7 = []
        self.seq_action3 = []
        self.seq_ball_pos = []
        self.seq_ee_pos = []

    def _save_qcmd7_demo(self, info: dict):
        if not SAVE_DEMO_TRAJ:
            return
        os.makedirs(SAVE_DEMO_DIR, exist_ok=True)

        land_score = float(info.get("land_score", 0.0))
        if SAVE_ONLY_BEST and (land_score <= self.best_land_score):
            return
        if SAVE_ONLY_BEST:
            self.best_land_score = land_score

        self.demo_ep_counter += 1
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = f"demo_ep{self.demo_ep_counter:04d}_{stamp}"

        csv_path = os.path.join(SAVE_DEMO_DIR, base + ".csv")
        meta_path = os.path.join(SAVE_DEMO_DIR, base + ".json")

        seq_t = np.asarray(info["seq_t"], dtype=np.float32)
        seq_q = np.asarray(info["seq_q_cmd7"], dtype=np.float32)

        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", "q1","q2","q3","q4","q5","q6","q7"])  # radians
            for i in range(len(seq_t)):
                row = [float(seq_t[i])] + [float(x) for x in seq_q[i]]
                w.writerow(row)

        meta = {
            "stop_reason": info.get("stop_reason", ""),
            "released": bool(info.get("released", False)),
            "release_step": info.get("release_step", None),
            "release_time_ep": info.get("release_time_ep", None),
            "landing_xy": (info.get("landing_xy", None).tolist()
                           if hasattr(info.get("landing_xy", None), "tolist") else info.get("landing_xy", None)),
            "land_score": land_score,
            "target_xy": self.target_xyz[:2].tolist(),
            "dt_control": float(DT_CONTROL),
            "control_hz": float(CONTROL_HZ),
            "sim_updates_per_step": int(SIM_UPDATES_PER_STEP),
            "ball_detach_dist": float(BALL_DETACH_DIST),
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        print(f"[demo] saved q_cmd7 -> {csv_path}", flush=True)

    # ---------------- Obs ----------------
    def _get_obs(self):
        bpos, bvel, _ = self._get_ball_world_pos_vel()
        if bpos is None:
            bpos = np.zeros(3, dtype=float)
        if bvel is None:
            bvel = np.zeros(3, dtype=float)
        ee = self._get_ee_world_pos()
        obs = np.concatenate([
            bpos, bvel, ee,
            self.q_resid3,
            np.array([1.0 if self.released else 0.0], dtype=float),
            np.array([self.episode_t], dtype=float),
        ]).astype(np.float32)
        return obs


    def _sample_target_xyz(self) -> np.ndarray:
        """Sample a 3D target on the landing plane (episode-wise)."""
        if not RANDOMIZE_TARGET:
            return TARGET_XYZ.astype(float).copy()
        x = float(self.np_random.uniform(TARGET_X_RANGE[0], TARGET_X_RANGE[1]))
        y = float(self.np_random.uniform(TARGET_Y_RANGE[0], TARGET_Y_RANGE[1]))
        z = float(TARGET_Z)
        return np.array([x, y, z], dtype=float)

    # ============================================================
    # Gym API
    # ============================================================

    # -------- Tube baseline helpers --------
    def _plan_tube_baseline(self, target_xyz: np.ndarray):
        """Plan nominal 3D ballistic release velocity from the EE pose at the (possibly yaw-aligned) throw-start pose."""
        target_xyz = np.asarray(target_xyz, dtype=float).reshape(3)
        self.target_xyz = target_xyz.copy()

        # Nominal release point: EE position at THROW_START_Q
        q_full = self._make_q_full_from_arm7(getattr(self, 'throw_start_q', THROW_START_Q))
        self._pin_forward(q_full)
        T = self.pin_data.oMf[self.ee_id]
        self.p_release_nom = np.array(T.translation, dtype=float)

        v_star = None
        method = "ballistic"
        print("USE",USE_TUBE_ACCELERATION_PLANNER)
        if USE_TUBE_ACCELERATION_PLANNER:
            try:
                J_lin = self._pin_jacobian_linear_sub(self.throw_start_q)
                cand = _tube_acceleration_select_release(
                    target_xyz=target_xyz,
                    yaw=float(self.throw_start_q[0]),
                    p_release_nom=self.p_release_nom,
                    J_lin=J_lin,
                    qdot_limits=QDOT_LIMITS_3,
                )
                print(cand)
                if cand is not None:
                    self.p_release_nom = cand["release_pos_w"].copy()
                    v_star = cand["release_vel_w"].copy()
                    method = "tube_acceleration"
            except Exception as e:
                print(f"[tube_acceleration] planner failed, fallback to ballistic: {e}", flush=True)

        if v_star is None:
            v_star, _T_star = self._solve_ballistic_velocity(self.p_release_nom, target_xyz)
            method = "ballistic"

        # clamp to EE speed limit
        v_norm = float(np.linalg.norm(v_star))
        if v_norm > EE_VEL_MAX and v_norm > 1e-6:
            v_star = v_star * (EE_VEL_MAX / v_norm)
        self.v_release_star = v_star
        n = float(np.linalg.norm(v_star))
        self.v_release_hat = (v_star / n) if n > 1e-9 else np.array([1.0, 0.0, 0.0], dtype=float)
        self.release_plan_method = method
        self.release_plan_pos = self.p_release_nom.copy()
        self.release_plan_vel = self.v_release_star.copy()

        self.tube_t0 = float(TUBE_T0)
        self.tube_t1 = float(TUBE_T0 + 2.0 * TUBE_HALF_WINDOW)

    def _solve_ballistic_velocity(self, p_r: np.ndarray, p_t: np.ndarray):
        """Solve ballistic release velocity v given release point p_r and target point p_t in 3D."""
        p_r = np.asarray(p_r, dtype=float).reshape(3)
        p_t = np.asarray(p_t, dtype=float).reshape(3)
        g = 9.81
        best = None  # (cost, v, T)
        for T in np.arange(BALLISTIC_T_MIN, BALLISTIC_T_MAX + 1e-9, BALLISTIC_T_STEP):
            d = p_t - p_r
            vx, vy = d[0] / T, d[1] / T
            vz = (d[2] + 0.5 * g * T * T) / T
            v = np.array([vx, vy, vz], dtype=float)
            speed = float(np.linalg.norm(v))
            if speed < 1e-6:
                continue
            if speed > EE_VEL_MAX * 0.98:
                continue
            cost = speed + 0.2 * T
            if (best is None) or (cost < best[0]):
                best = (cost, v, float(T))
        if best is None:
            T = 0.45
            d = p_t - p_r
            vx, vy = d[0] / T, d[1] / T
            vz = (d[2] + 0.5 * g * T * T) / T
            v = np.array([vx, vy, vz], dtype=float)
            speed = float(np.linalg.norm(v))
            if speed > EE_VEL_MAX and speed > 1e-6:
                v = v * (EE_VEL_MAX / speed)
            return v, float(T)
        return best[1], best[2]

    def _smoothstep01(self, x: float) -> float:
        x = float(np.clip(x, 0.0, 1.0))
        return x * x * (3.0 - 2.0 * x)

    def _project_velocity_to_tube(self, v: np.ndarray) -> np.ndarray:
        """Project/clip a velocity into tube constraints (cone + speed bounds)."""
        v = np.asarray(v, dtype=float).reshape(3)
        v_star = self.v_release_star
        v_hat = self.v_release_hat
        v_star_norm = float(np.linalg.norm(v_star))

        vmin = TUBE_V_MIN_RATIO * v_star_norm
        vmax = TUBE_V_MAX_RATIO * v_star_norm
        vmax = min(vmax, EE_VEL_MAX)

        speed = float(np.linalg.norm(v))
        if speed < 1e-9:
            return v_hat * max(0.0, min(vmin, EE_VEL_MAX))

        # cone clamp
        cos_alpha = math.cos(math.radians(TUBE_CONE_DEG))
        dir_v = v / speed
        cosang = float(np.dot(dir_v, v_hat))
        if cosang < cos_alpha:
            parallel = cosang * v_hat
            perp = dir_v - parallel
            perp_norm = float(np.linalg.norm(perp))
            if perp_norm < 1e-9:
                dir_clamped = v_hat
            else:
                perp_unit = perp / perp_norm
                # boundary direction
                sin_alpha = math.sin(math.radians(TUBE_CONE_DEG))
                dir_clamped = cos_alpha * v_hat + sin_alpha * perp_unit
                dir_clamped = dir_clamped / float(np.linalg.norm(dir_clamped))
            v = dir_clamped * speed

        # speed clamp
        speed = float(np.linalg.norm(v))
        speed_clamped = float(np.clip(speed, vmin, vmax))
        if speed > 1e-9:
            v = v * (speed_clamped / speed)
        return v

    def _pin_jacobian_linear_sub(self, q7: np.ndarray) -> np.ndarray:
        """3x3 linear Jacobian for learned joints (2,4,6) in LOCAL_WORLD_ALIGNED."""
        q7 = np.asarray(q7, dtype=float).reshape(7)
        q_full = self._make_q_full_from_arm7(q7)
        self._pin_forward(q_full)
        J = pin.computeFrameJacobian(
            self.pin_model, self.pin_data, q_full, self.ee_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        J_lin = J[:3, :]
        return J_lin[:, LEARN_JIDX].copy()

    def _compute_qdot3_for_v(self, q7: np.ndarray, v_cmd: np.ndarray) -> np.ndarray:
        J = self._pin_jacobian_linear_sub(q7)  # 3x3
        v_cmd = np.asarray(v_cmd, dtype=float).reshape(3)
        lam = float(J_DLS_LAMBDA)
        JJt = J @ J.T
        inv = np.linalg.inv(JJt + (lam * lam) * np.eye(3))
        J_pinv = J.T @ inv
        qdot3 = J_pinv @ v_cmd
        return qdot3.astype(float)

    def _tube_velocity_command(self, t: float) -> np.ndarray:
        """Baseline desired EE velocity as a function of episode time."""
        t = float(t)
        v_star = self.v_release_star
        if t <= self.tube_t0:
            s = self._smoothstep01(t / max(1e-6, self.tube_t0))
            v = s * v_star
        elif t <= self.tube_t1:
            v = v_star.copy()
        else:
            td = t - self.tube_t1
            if td >= TUBE_DECEL_SEC:
                v = np.zeros(3, dtype=float)
            else:
                s = 1.0 - self._smoothstep01(td / max(1e-6, TUBE_DECEL_SEC))
                v = s * v_star
        return self._project_velocity_to_tube(v)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        self.episode_t = 0.0
        self.step_count = 0
        self.released = False
        self.release_time_wall = None
        self.release_time_ep = None
        self.release_step = None

        self._a_prev3[:] = 0.0
        self._a_filt3[:] = 0.0
        self.q_resid3[:] = 0.0

        self._clear_recording_buffers()

        # --------- REQUIRED RESET ORDER ----------
        # 1) arm -> INIT, hold 2s
        self.q_cmd7 = INIT_ARM.copy()
        self.qdot7[:] = 0.0

        t0 = time.time()
        while time.time() - t0 < RESET_ARM_SETTLE_SEC:
            self._apply_arm_targets(self.q_cmd7, self.qdot7)
            self.simulation_app.update()

        # 2) ball reset at arm INIT position
        self._reset_ball(BALL_RESET_POS_WORLD)

        # 3) wait 2s after ball reset, keep arm at INIT
        t1 = time.time()
        while time.time() - t1 < RESET_BALL_WAIT_SEC:
            self._apply_arm_targets(self.q_cmd7, self.qdot7)
            self.simulation_app.update()

        # 4) sample or use provided 3D target for this episode
        if options and ("target_xyz" in options):
            t = np.asarray(options["target_xyz"], dtype=float).reshape(3)
            self.target_xyz = t.copy()
        else:
            self.target_xyz = self._sample_target_xyz()

        # 5) yaw-align joint0 (panda_joint1) towards the target in world XY.
        #    With only joints (2,4,6) controlled for throwing, we must rotate the arm plane towards target.
        yaw = math.atan2(float(self.target_xyz[1]), float(self.target_xyz[0]))
        yaw = float(np.clip(yaw, -JOINT0_YAW_LIMIT, JOINT0_YAW_LIMIT))
        self.throw_start_q = THROW_START_Q.copy()
        self.throw_start_q[0] = yaw  # joint0

        # 6a) move only joint0 first: INIT -> yaw-aligned pose
        q_yaw = INIT_ARM.copy()
        q_yaw[0] = yaw
        for k in range(JOINT0_ALIGN_STEPS):
            alpha = (k + 1) / float(JOINT0_ALIGN_STEPS)
            q = (1 - alpha) * INIT_ARM + alpha * q_yaw
            self.q_cmd7 = q.copy()
            self.qdot7[:] = 0.0
            self._apply_arm_targets(self.q_cmd7, self.qdot7)
            for _ in range(SIM_UPDATES_PER_STEP):
                self.simulation_app.update()

        # 6b) wait for joint0 alignment to settle
        t2 = time.time()
        while time.time() - t2 < JOINT0_ALIGN_WAIT_SEC:
            self.q_cmd7 = q_yaw.copy()
            self.qdot7[:] = 0.0
            self._apply_arm_targets(self.q_cmd7, self.qdot7)
            self.simulation_app.update()

        # 6c) then move only throwing joints (2,4,6) to THROW_START
        q_prep = q_yaw.copy()
        q_prep[LEARN_JIDX] = self.throw_start_q[LEARN_JIDX]
        for k in range(PREP_STEPS):
            alpha = (k + 1) / float(PREP_STEPS)
            q = (1 - alpha) * q_yaw + alpha * q_prep
            self.q_cmd7 = q.copy()
            self.qdot7[:] = 0.0
            self._apply_arm_targets(self.q_cmd7, self.qdot7)
            for _ in range(SIM_UPDATES_PER_STEP):
                self.simulation_app.update()

        # 7) plan tube baseline from the yaw-aligned THROW_START pose
        self._plan_tube_baseline(self.target_xyz)
        print(
            f"[release_plan] method={self.release_plan_method} "
            f"pos={self.release_plan_pos.tolist()} "
            f"vel={self.release_plan_vel.tolist()}",
            flush=True,
        )

        # ball reference for forward progress
        bpos, _, _ = self._get_ball_world_pos_vel()
        self.ball_prev_x = float(bpos[0]) if bpos is not None else 0.0

        return self._get_obs(), {
            "release_plan_method": self.release_plan_method,
            "release_plan_pos": self.release_plan_pos.copy(),
            "release_plan_vel": self.release_plan_vel.copy(),
        }

    def step(self, action):
        self.step_count += 1
        self.episode_t += DT_CONTROL

        # detect release (phase switch) but DO NOT terminate
        if (not self.released) and self._ball_detached_from_net():
            self.released = True
            self.release_time_wall = time.time()
            self.release_time_ep = float(self.episode_t)
            self.release_step = self.step_count
            # freeze arm at current q_cmd7
            # (keep q_cmd7 as is, set qdot to zero)
            self.qdot7[:] = 0.0

        # ---------------- control update ----------------
        a3 = np.asarray(action, dtype=float).reshape(3)
        a3 = np.clip(a3, -1.0, 1.0)

        act_pen = 0.0
        qdot_pen = 0.0
        smooth_pen = 0.0
        dev_pen = 0.0

        if not self.released:
            # Ensure tube plan exists
            if (self.step_count == 1) and (np.linalg.norm(self.v_release_star) < 1e-6):
                self._plan_tube_baseline(self.target_xyz)

            # Baseline tube command in task space (EE linear velocity)
            v_cmd = self._tube_velocity_command(self.episode_t)

            # --- Residual interface (disabled by default) ---
            # Future: allow RL to predict a residual EE velocity delta, then project back into tube.
            if self.use_residual:
                self._a_filt3 = (1.0 - ACTION_EMA_ALPHA) * self._a_filt3 + ACTION_EMA_ALPHA * a3
                dv = self._a_filt3 * RESID_VEL_MAX_3
                self.q_resid3[:] = dv
                v_cmd = self._project_velocity_to_tube(v_cmd + dv)

                # penalties (only if residual enabled)
                act_pen = W_ACT * float(np.dot(a3, a3))
                smooth_pen = W_SMOOTH * float(np.dot(a3 - self._a_prev3, a3 - self._a_prev3))
                self._a_prev3 = a3.copy()
            else:
                self.q_resid3[:] = 0.0

            # Map EE velocity to learned joint velocities
            qdot3 = self._compute_qdot3_for_v(self.q_cmd7, v_cmd)

            # joint speed limit
            qdot3 = np.clip(qdot3, -QDOT_LIMITS_3, QDOT_LIMITS_3)

            # Build full qdot7 (only learned joints move during throw)
            self.qdot7[:] = 0.0
            self.qdot7[LEARN_JIDX] = qdot3

            # Integrate to position targets (position drive)
            self.q_cmd7[LEARN_JIDX] = self.q_cmd7[LEARN_JIDX] + qdot3 * DT_CONTROL
            # Keep other joints fixed at THROW_START_Q
            keep = np.ones(7, dtype=bool)
            keep[LEARN_JIDX] = False
            self.q_cmd7[keep] = self.throw_start_q[keep]

            # Penalty on joint velocities (tube baseline energy)
            qdot_pen = W_QDOT * float(np.dot(qdot3, qdot3))
        else:
            # after release: freeze arm, no penalties on actions
            self.qdot7[:] = 0.0
            self.q_resid3[:] = 0.0

        # step sim

        for _ in range(SIM_UPDATES_PER_STEP):
            self._apply_arm_targets(self.q_cmd7, self.qdot7)
            self.simulation_app.update()

        # ---------------- reward ----------------
        reward = 0.0

        bpos, _, _ = self._get_ball_world_pos_vel()
        if bpos is not None:
            dx = float(bpos[0] - self.ball_prev_x)
            self.ball_prev_x = float(bpos[0])
            reward += W_FORWARD * dx

        reward -= act_pen + qdot_pen + smooth_pen + dev_pen

        terminated = False
        truncated = False
        info = {
            "released": self.released,
            "release_step": self.release_step,
            "release_plan_method": self.release_plan_method,
            "release_plan_pos": self.release_plan_pos.copy(),
            "release_plan_vel": self.release_plan_vel.copy(),
        }

        # landing reward happens when ball lands (even after release)
        if self._ball_landed() and (bpos is not None):
            landing_xy = bpos[:2].copy()
            land_r = landing_reward(landing_xy, self.target_xyz[:2])
            reward += W_LAND * land_r
            terminated = True
            info["stop_reason"] = "ball_landed"
            info["landing_xy"] = landing_xy.copy()
            info["land_score"] = float(land_r)

        # timeout after release
        if (not terminated) and self.released and (self.release_time_ep is not None):
            if (self.episode_t - self.release_time_ep) >= DROP_TIMEOUT_AFTER_RELEASE_SEC:
                truncated = True
                info["stop_reason"] = "timeout_after_release"
                info["land_score"] = 0.0

        # max episode
        if (not terminated) and (not truncated) and (self.episode_t >= MAX_EPISODE_SEC):
            truncated = True
            info["stop_reason"] = "max_episode"
            info["land_score"] = 0.0

        # ---------------- recording ----------------
        ee_pos = self._get_ee_world_pos()
        self.seq_t.append(float(self.episode_t))
        self.seq_q_cmd7.append(self.q_cmd7.astype(np.float32).copy())
        self.seq_action3.append(a3.astype(np.float32).copy())
        self.seq_ball_pos.append((bpos.astype(np.float32).copy() if bpos is not None else np.zeros(3, np.float32)))
        self.seq_ee_pos.append(ee_pos.astype(np.float32).copy())

        if terminated or truncated:
            info["seq_t"] = np.asarray(self.seq_t, dtype=np.float32)
            info["seq_q_cmd7"] = np.asarray(self.seq_q_cmd7, dtype=np.float32)
            info["seq_action3"] = np.asarray(self.seq_action3, dtype=np.float32)
            info["seq_ball_pos"] = np.asarray(self.seq_ball_pos, dtype=np.float32)
            info["seq_ee_pos"] = np.asarray(self.seq_ee_pos, dtype=np.float32)
            info["release_time_ep"] = (float(self.seq_t[self.release_step - 1])
                                      if self.release_step is not None and self.release_step - 1 < len(self.seq_t)
                                      else None)
            self._save_qcmd7_demo(info)

        return self._get_obs(), float(reward), terminated, truncated, info


# Backward-compatible alias for training / rollout scripts.
SpoonThrowEnvCore = SpoonThrowEnvCoreTubeRandomYawAlign
