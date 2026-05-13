#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import os
from pathlib import Path

from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, Gf, Sdf

try:
    from pxr import UsdLux
except Exception:
    UsdLux = None

try:
    from pxr import UsdShade
except Exception:
    UsdShade = None

from core.throw_params import (
    ROBOT_USD,
    ROBOT_PRIM,
    FRANKA_ROOT,
    STIFF,
    DAMP,
    MAXF,
    CCD_ENABLE,
    RB_SOLVER_POS_ITERS,
    RB_SOLVER_VEL_ITERS,
    CONTACT_OFFSET,
    REST_OFFSET,
    NO_BOUNCE_RESTITUTION,
    NO_BOUNCE_STATIC_FRICTION,
    NO_BOUNCE_DYNAMIC_FRICTION,
)


def get_stage():
    import omni.usd

    return omni.usd.get_context().get_stage()


def valid(stage, path: str) -> bool:
    prim = stage.GetPrimAtPath(path)
    return bool(prim and prim.IsValid())


def find_child_prim_by_name(root_prim: Usd.Prim, name: str):
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


def get_or_add_translate_op(xf: UsdGeom.Xformable) -> UsdGeom.XformOp:
    for op in xf.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            return op
    return xf.AddTranslateOp()


def set_prim_translate(stage, prim_path: str, xyz, label: str = "prim"):
    if not valid(stage, prim_path):
        print(f"[scene] {label} not found for translate: {prim_path}", flush=True)
        return False
    prim = stage.GetPrimAtPath(prim_path)
    op = get_or_add_translate_op(UsdGeom.Xformable(prim))
    op.Set(Gf.Vec3d(float(xyz[0]), float(xyz[1]), float(xyz[2])))
    print(f"[scene] {label} translate set: {prim_path} -> {tuple(float(v) for v in xyz)}", flush=True)
    return True


def spawn_lights(stage):
    if UsdLux is None:
        print("[scene] UsdLux not available, skip lights", flush=True)
        return
    root = "/World/Lights"
    if not valid(stage, root):
        stage.DefinePrim(root, "Xform")

    sun = f"{root}/Sun"
    if not valid(stage, sun):
        p = stage.DefinePrim(sun, "DistantLight")
        UsdLux.DistantLight(p).CreateIntensityAttr(2500.0)
        UsdGeom.Xformable(p).AddRotateXYZOp().Set(Gf.Vec3f(-45, 45, 0))

    dome = f"{root}/Dome"
    if not valid(stage, dome):
        p = stage.DefinePrim(dome, "DomeLight")
        UsdLux.DomeLight(p).CreateIntensityAttr(800.0)

    print("[scene] lights ready", flush=True)


def ensure_physics_scene(stage):
    path = "/World/physicsScene"
    if valid(stage, path):
        return
    prim = stage.DefinePrim(path, "PhysicsScene")
    scene = UsdPhysics.Scene(prim)
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr(9.81)
    print("[scene] physicsScene created", flush=True)


def _bind_preview_surface_material(stage, prim, *, material_path: str, color):
    if UsdShade is None:
        return
    if not valid(stage, "/World/Materials"):
        stage.DefinePrim("/World/Materials", "Scope")
    mat = UsdShade.Material.Define(stage, material_path)
    shader = UsdShade.Shader.Define(stage, f"{material_path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat)


def spawn_ground(stage, color=(0.42, 0.52, 0.45)):
    if valid(stage, "/World/Ground/Mesh"):
        _bind_preview_surface_material(
            stage,
            stage.GetPrimAtPath("/World/Ground/Mesh"),
            material_path="/World/Materials/Ground",
            color=color,
        )
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
    _bind_preview_surface_material(stage, mesh_prim, material_path="/World/Materials/Ground", color=color)
    UsdGeom.Imageable(mesh_prim).MakeVisible()
    print("[scene] ground ready", flush=True)


def add_robot_reference(stage, prim_path: str = ROBOT_PRIM, usd_path: str = ROBOT_USD):
    # Accept:
    # - filesystem paths (absolute or relative) -> convert to file:// when the file exists
    # - Nucleus / mounted assets paths like "/Isaac/Robots/..." -> keep as-is
    # - already-qualified URLs like "omniverse://", "file://", "http(s)://"
    if "://" not in usd_path:
        p = Path(usd_path)
        if p.is_absolute() or p.exists():
            try:
                p_abs = p.resolve() if not p.is_absolute() else p
            except Exception:
                p_abs = p
            if os.path.isfile(p_abs):
                usd_path = "file://" + str(p_abs)
    if not valid(stage, prim_path):
        stage.DefinePrim(prim_path, "Xform")
    stage.GetPrimAtPath(prim_path).GetReferences().AddReference(usd_path)
    print(f"[scene] robot referenced: {prim_path}", flush=True)


def has_panda_joints(stage, root_path: str) -> bool:
    root = stage.GetPrimAtPath(root_path)
    if not (root and root.IsValid()):
        return False
    stack = [root]
    while stack:
        prim = stack.pop()
        if prim.GetName().startswith("panda_joint"):
            return True
        for child in prim.GetChildren():
            stack.append(child)
    return False


def repair_lmm_asset_references(stage, root_path: str = ROBOT_PRIM):
    """Best-effort local fallbacks for exported LMM USDs with remote/broken references."""
    franka_root = os.environ.get("SOFT_THROW_FRANKA_ROOT", f"{root_path}/Franka")
    local_franka = os.environ.get("SOFT_THROW_LOCAL_FRANKA_USD", "")
    if local_franka and os.path.isfile(local_franka) and not has_panda_joints(stage, franka_root):
        if not valid(stage, franka_root):
            stage.DefinePrim(franka_root, "Xform")
        stage.GetPrimAtPath(franka_root).GetReferences().AddReference("file://" + str(Path(local_franka).resolve()))
        print(f"[scene] local Franka fallback referenced: {franka_root}", flush=True)


def detect_franka_root(stage, want_root: str = FRANKA_ROOT) -> str:
    if valid(stage, want_root):
        return want_root
    for cand in ["/World/LMM/Franka", "/World/Franka", "/World/LMM/franka", "/World/LMM/panda", "/World/LMM"]:
        if valid(stage, cand):
            return cand
    raise RuntimeError(f"[usd] Franka root not found under {want_root} and fallbacks")


def set_drive_target_rad(joint_prim, q_rad: float, qd_rad_s: float):
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


def tune_rb_collision_recursive(stage, root_path: str, label: str):
    if not valid(stage, root_path):
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


def apply_no_bounce_material(stage, prim_path: str, label: str):
    """
    Apply (or create) a physics material with restitution=0 and high friction.
    Bind it to the prim (and its subtree collisions).
    """
    if not valid(stage, prim_path):
        print(f"[mat] {label}: prim not found: {prim_path}", flush=True)
        return

    mat_path = "/World/Materials/NoBounce"
    if not valid(stage, "/World/Materials"):
        stage.DefinePrim("/World/Materials", "Scope")

    if not valid(stage, mat_path):
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
