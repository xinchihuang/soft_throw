#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Runtime lacrosse collider repair helpers."""

from __future__ import annotations

from refractor.config import params


def _valid_prim(prim) -> bool:
    return bool(prim and prim.IsValid())


def _find_lacrosse_source_mesh(stage, lacrosse_root: str):
    from pxr import UsdGeom, UsdPhysics

    root = stage.GetPrimAtPath(lacrosse_root)
    if not _valid_prim(root):
        raise RuntimeError(f"lacrosse root not found: {lacrosse_root}")

    fallback = None
    stack = [root]
    while stack:
        prim = stack.pop()
        if prim.IsA(UsdGeom.Mesh):
            fallback = fallback or prim
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                return prim
        for child in prim.GetChildren():
            stack.append(child)
    if fallback is not None:
        return fallback
    raise RuntimeError(f"cannot find a Mesh under {lacrosse_root}")


def _copy_mesh_geometry(src_prim, dst_prim) -> None:
    from pxr import UsdGeom

    src = UsdGeom.Mesh(src_prim)
    dst = UsdGeom.Mesh(dst_prim)

    attr_pairs = [
        (src.GetPointsAttr(), dst.CreatePointsAttr()),
        (src.GetFaceVertexCountsAttr(), dst.CreateFaceVertexCountsAttr()),
        (src.GetFaceVertexIndicesAttr(), dst.CreateFaceVertexIndicesAttr()),
        (src.GetExtentAttr(), dst.CreateExtentAttr()),
        (src.GetNormalsAttr(), dst.CreateNormalsAttr()),
        (src.GetSubdivisionSchemeAttr(), dst.CreateSubdivisionSchemeAttr()),
    ]
    for src_attr, dst_attr in attr_pairs:
        if src_attr and src_attr.HasAuthoredValueOpinion():
            value = src_attr.Get()
            if value is not None:
                dst_attr.Set(value)

    src_orientation = src.GetOrientationAttr()
    if src_orientation and src_orientation.HasAuthoredValueOpinion():
        dst.CreateOrientationAttr(src_orientation.Get())
    src_normal_interp = src.GetNormalsInterpolation()
    if src_normal_interp:
        dst.SetNormalsInterpolation(src_normal_interp)


def _copy_world_alignment_to_child(stage, *, source_path: str, parent_path: str, dst_prim) -> None:
    from pxr import Usd, UsdGeom

    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    source_world = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(source_path))
    parent_world = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(parent_path))
    dst_local = parent_world.GetInverse() * source_world

    xformable = UsdGeom.Xformable(dst_prim)
    xformable.ClearXformOpOrder()
    xformable.AddTransformOp().Set(dst_local)


def _apply_sdf_collision(dst_prim, *, sdf_resolution: int, approximation: str) -> None:
    from pxr import PhysxSchema, UsdPhysics

    collision = UsdPhysics.CollisionAPI.Apply(dst_prim)
    collision.CreateCollisionEnabledAttr(True)

    physx_collision = PhysxSchema.PhysxCollisionAPI.Apply(dst_prim)
    mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(dst_prim)
    mesh_collision.CreateApproximationAttr().Set(str(approximation))

    sdf_collision = PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(dst_prim)
    sdf_collision.CreateSdfResolutionAttr().Set(int(sdf_resolution))

    try:
        from core.throw_params import CONTACT_OFFSET, REST_OFFSET

        physx_collision.CreateContactOffsetAttr(float(CONTACT_OFFSET))
        physx_collision.CreateRestOffsetAttr(float(REST_OFFSET))
    except Exception:
        pass


def ensure_lacrosse_sdf_collider(
    stage,
    *,
    lacrosse_root: str = params.DEFAULT_SIM_HEAD_PRIM,
    collider_path: str = params.DEFAULT_LACROSSE_COLLIDER_PRIM,
    sdf_resolution: int = params.DEFAULT_LACROSSE_COLLIDER_SDF_RESOLUTION,
    approximation: str = params.DEFAULT_LACROSSE_COLLIDER_APPROXIMATION,
    hide_collider_visual: bool = params.DEFAULT_LACROSSE_COLLIDER_HIDE_VISUAL,
) -> str:
    """Create a real SDF mesh collider under the lacrosse rigid body.

    The source asset uses an instanced prototype. PhysX can miss that shape as
    the runtime rigid-body collider. This helper copies the prototype mesh into
    a normal child Mesh prim under ``lacrosse_root`` and applies SDF collision.
    SDF keeps the concave pocket shape, unlike convex hull collision.
    """
    from pxr import UsdGeom

    root = stage.GetPrimAtPath(lacrosse_root)
    if not _valid_prim(root):
        raise RuntimeError(f"lacrosse root not found: {lacrosse_root}")

    source_mesh = _find_lacrosse_source_mesh(stage, lacrosse_root)
    dst_mesh = UsdGeom.Mesh.Define(stage, collider_path)
    dst_prim = dst_mesh.GetPrim()

    _copy_mesh_geometry(source_mesh, dst_prim)
    _copy_world_alignment_to_child(
        stage,
        source_path=source_mesh.GetPath().pathString,
        parent_path=lacrosse_root,
        dst_prim=dst_prim,
    )
    _apply_sdf_collision(dst_prim, sdf_resolution=int(sdf_resolution), approximation=str(approximation))

    if hide_collider_visual:
        UsdGeom.Imageable(dst_prim).MakeInvisible()

    print(
        f"[lacrosse_collision] source={source_mesh.GetPath()} collider={collider_path} "
        f"approximation={approximation} sdf_resolution={int(sdf_resolution)}",
        flush=True,
    )
    return collider_path
