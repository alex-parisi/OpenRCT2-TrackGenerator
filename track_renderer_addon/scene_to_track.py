"""Read the Blender scene into the track generator's config + meshes.

The ``bpy -> Track`` adapter: pull geometry straight from scene objects (rather
than OBJ files) and hand the core ``build_track`` an in-memory config dict, a
``Mesh`` list, and a ``special_models`` map — the same shape the YAML loader
produces.

Objects carry a role (``tg_object.role``): the TRACK tile is deformed along the
section curves, the MASK plate drives occlusion carving, and SPECIAL meshes are
the named brake/booster/support models. Every mesh is rotated ``rotate_y(-90°)``
on the way in, exactly as the CLI's ``load_track_meshes`` does, so meshes authored
with the rail along Blender +X end up +Z along-track for the deform.
"""

from __future__ import annotations

import os

import bpy
import numpy as np
from openrct2_object_common.blender.bake import bake_materials
from openrct2_object_common.blender.mesh_extract import (
    SceneError,
    extract_mesh,
    load_preview,
    material_base,
)
from openrct2_x7_renderer.constants import MaterialFlag
from openrct2_x7_renderer.geometry import rotate_y
from openrct2_x7_renderer.mesh import Material, Mesh, load_texture

_REGION_MAP = {
    "NONE": (0, 0),
    "REMAP1": (MaterialFlag.IS_REMAPPABLE, 1),
    "REMAP2": (MaterialFlag.IS_REMAPPABLE, 2),
    "REMAP3": (MaterialFlag.IS_REMAPPABLE, 3),
    "GREYSCALE": (0, 4),
    "PEEP": (0, 5),
    "CHAIN": (0, 6),
}

# maketrack's load rotation: meshes are authored +X along-track; this makes Z the
# (centred) along-track axis the deform expects. Mirrors loader._MESH_LOAD_TRANSFORM.
_LOAD_ROT = rotate_y(-0.5 * np.pi)


# Material -> baked Texture map for the current build, populated by
# build_config_and_meshes before extraction (see bake.bake_materials).
_baked_textures: dict = {}


def _material_from_bpy(bmat) -> Material:
    m, s = material_base(bmat, prop_attr="tg_material", region_map=_REGION_MAP)
    if s is None:
        return m
    # Visible mask (rendered into the silhouette) overrides a plain mask.
    if s.is_visible_mask:
        m.flags &= ~MaterialFlag.IS_MASK
        m.flags |= MaterialFlag.IS_VISIBLE_MASK
    if s.flat_shaded:
        m.flags |= MaterialFlag.IS_FLAT_SHADED
    # Texture sources, in priority order: explicit image > baked procedural nodes.
    if s.texture is not None:
        path = bpy.path.abspath(s.texture.filepath_from_user() or s.texture.filepath)
        if path and os.path.exists(path):
            m.texture = load_texture(path)
            m.flags |= MaterialFlag.HAS_TEXTURE
    if not (m.flags & MaterialFlag.HAS_TEXTURE) and bmat in _baked_textures:
        m.texture = _baked_textures[bmat]
        m.flags |= MaterialFlag.HAS_TEXTURE
    return m


def _rotated(mesh: Mesh) -> Mesh:
    """Apply the maketrack load rotation to a freshly-extracted (OBJ-space) mesh."""
    rt = _LOAD_ROT.T
    return Mesh(
        vertices=(mesh.vertices.astype(np.float64) @ rt).astype(np.float32),
        normals=(mesh.normals.astype(np.float64) @ rt).astype(np.float32),
        uvs=mesh.uvs,
        faces=mesh.faces,
        face_materials=mesh.face_materials,
        materials=mesh.materials,
    )


def _extract(obj, depsgraph) -> Mesh | None:
    mesh = extract_mesh(obj, depsgraph, _material_from_bpy)
    return None if mesh is None else _rotated(mesh)


def build_config_and_meshes(context):
    """Return ``(config, meshes, special_models, preview)`` read from the active scene.

    Raises :class:`SceneError` with a user-facing message on invalid scenes.
    """
    scene = context.scene
    tt = scene.tg_track
    depsgraph = context.evaluated_depsgraph_get()

    # Bake any procedural-node materials to textures up front (main thread, Cycles),
    # then feed them into extraction via _material_from_bpy. Re-assigned each call.
    bake_objs = [
        obj
        for obj in scene.objects
        if obj.type == "MESH" and obj.tg_object.role != "IGNORE"
    ]
    global _baked_textures
    _baked_textures = bake_materials(context, bake_objs, prop_attr="tg_material")

    track_mesh: Mesh | None = None
    mask_mesh: Mesh | None = None
    special_models: dict[str, Mesh] = {}

    for obj in scene.objects:
        if obj.type != "MESH":
            continue
        role = obj.tg_object.role
        if role == "IGNORE":
            continue
        mesh = _extract(obj, depsgraph)
        if mesh is None:
            continue
        if role == "TRACK":
            if track_mesh is not None:
                raise SceneError(
                    "Multiple objects have the 'Track' role; exactly one is allowed."
                )
            track_mesh = mesh
        elif role == "MASK":
            mask_mesh = mesh
        elif role == "SPECIAL":
            special_models[obj.tg_object.special_model] = mesh

    if track_mesh is None:
        raise SceneError(
            "No track mesh found. Add a mesh and set its role to 'Track' in the "
            "OpenRCT2 Track panel."
        )
    if not tt.sections:
        raise SceneError("No sections selected. Add at least one in the Sections list.")

    meshes: list[Mesh] = [track_mesh]
    mask_index = -1
    if mask_mesh is not None:
        meshes.append(mask_mesh)
        mask_index = 1

    flags: list[str] = []
    if tt.has_lift:
        flags.append("has_lift")
    if tt.has_supports:
        flags.append("has_supports")

    authors = [a.strip() for a in tt.authors.split(",") if a.strip()]

    config: dict = {
        "id": tt.id,
        "name": tt.name,
        "description": tt.description,
        "authors": authors,
        "version": tt.version,
        "ride_type": tt.ride_type,
        "units_per_tile": float(tt.units_per_tile),
        "flat_shaded": bool(tt.flat_shaded),
        "z_offset": float(tt.z_offset),
        "flags": flags,
        "support_spacing": float(tt.support_spacing),
        "pivot": float(tt.pivot),
        "brake_length": float(tt.brake_length),
        "track_mesh_index": 0,
        "mask_mesh_index": mask_index,
        "sections": [s.section for s in tt.sections],
    }
    return config, meshes, special_models, load_preview(tt.preview)
