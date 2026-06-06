"""
Deform a flat track-tile mesh along a section curve.

This is the genuinely-new capability versus the vehicle/scenery generators, which
only apply rigid transforms. Ported from RCTGen ``track.cpp`` (``track_transform`` /
``get_track_point``) and vectorized over the whole vertex array: each vertex's local
``z`` is read as an arc-distance along the section's curve, and its ``x``/``y`` are
placed in that point's cross-section frame (``position + normal·y + binormal·x``).
The result is a brand-new :class:`~openrct2_x7_renderer.mesh.Mesh` with identical
topology/UVs/materials, so it is a drop-in for ``SceneBuilder.add_model``.
"""

import numpy as np
from numpy.typing import NDArray
from openrct2_x7_renderer.mesh import Mesh

from .constants import CLEARANCE_HEIGHT, TILE_SIZE, TrackFlag
from .types import CurveFn, TrackPointArray, TrackSection

__all__ = ["deform_mesh", "get_track_point_array"]


def _to_render_space(arr: NDArray[np.float64]) -> NDArray[np.float64]:
    """Map curve-space ``(x, y, z)`` to the renderer's axes.

    This is RCTGen ``track.cpp``'s ``change_coordinates`` swap ``(x,y,z) -> (z,y,x)``,
    kept because X7 descends from the same isometric renderer the legacy track code
    targeted. It is the highest-risk integration point: if a ``--test`` preview comes
    out rotated or mirrored, this single mapping is what to adjust.
    """
    return arr[:, ::-1].copy()


def get_track_point_array(
    curve: CurveFn,
    flags: TrackFlag,
    z_offset: float,
    length: float,
    start_offset: NDArray[np.float64],
    end_offset: NDArray[np.float64],
    u: NDArray[np.float64],
) -> TrackPointArray:
    """Evaluate the curve frame at arc-distances ``u`` with the legacy positioning.

    Distances outside ``[0, length]`` extend straight along the end tangent; the
    diagonal/vertical/height shifts and the Hermite start/end smoothing blend match
    ``track.cpp``'s ``get_track_point``. Only ``position`` is shifted — the
    tangent/normal/binormal frame is returned unchanged.
    """
    u_clamped = np.clip(u, 0.0, length)
    tp = curve(u_clamped)
    pos = tp.position.copy()

    below = u < 0.0
    above = u > length
    if below.any():
        pos[below] += tp.tangent[below] * u[below, None]
    if above.any():
        pos[above] += tp.tangent[above] * (u[above] - length)[:, None]

    if flags & TrackFlag.DIAGONAL:
        pos[:, 0] += 0.5 * TILE_SIZE
    if flags & TrackFlag.DIAGONAL_2:
        pos[:, 2] += 0.5 * TILE_SIZE
    pos[:, 1] += z_offset - 2.0 * CLEARANCE_HEIGHT
    if not (flags & TrackFlag.VERTICAL):
        pos[:, 2] -= 0.5 * TILE_SIZE

    v = np.clip(u / length, 0.0, 1.0)
    w_start = 2.0 * v**3 - 3.0 * v**2 + 1.0
    w_end = -2.0 * v**3 + 3.0 * v**2
    pos = pos + start_offset[None, :] * w_start[:, None] + end_offset[None, :] * w_end[:, None]

    return TrackPointArray(pos, tp.tangent, tp.normal, tp.binormal)


def deform_mesh(
    mesh: Mesh,
    section: TrackSection,
    *,
    scale: float | None = None,
    offset: float = 0.0,
    track_length: float | None = None,
    z_offset: float | None = None,
    flat_shaded: bool = False,
    start_offset: NDArray[np.float64] | None = None,
    end_offset: NDArray[np.float64] | None = None,
) -> Mesh:
    """Bend ``mesh`` along ``section``'s curve, returning a new deformed Mesh.

    The mesh is authored straight, with its local ``z`` running along the track. By
    default the mesh's ``z`` extent is auto-fit to ``[0, section.length]`` (so a
    one-tile mesh fills exactly one section); pass an explicit ``scale``/``offset`` to
    override. ``z_offset`` defaults to the section's own. With ``flat_shaded`` every
    normal uses the section's central frame (matching the legacy flat-shaded path).
    """
    if mesh.faces.shape[0] == 0 or mesh.vertices.shape[0] == 0:
        return mesh

    verts = mesh.vertices.astype(np.float64)
    norms = mesh.normals.astype(np.float64)

    if scale is None:
        z = verts[:, 2]
        span = float(z.max() - z.min())
        if span <= 0.0:
            span = TILE_SIZE
        scale = section.length / span
        offset = -float(z.min()) * scale

    length = section.length
    if track_length is None:
        track_length = length
    if z_offset is None:
        z_offset = section.z_offset
    so = np.zeros(3) if start_offset is None else start_offset
    eo = np.zeros(3) if end_offset is None else end_offset

    # track.cpp: vertex.z = scale*vertex.z + offset, read as arc distance.
    u = scale * verts[:, 2] + offset
    tp = get_track_point_array(section.curve, section.flags, z_offset, length, so, eo, u)

    x = verts[:, 0:1]
    y = verts[:, 1:2]
    world = tp.position + tp.normal * y + tp.binormal * x
    out_verts = _to_render_space(world)

    nx = norms[:, 0:1]
    ny = norms[:, 1:2]
    nz = norms[:, 2:3]
    if flat_shaded:
        central = np.array([np.clip(offset + track_length / 2.0, 0.0, length)])
        c = section.curve(central)
        frame_n = c.tangent[0] * nz + c.normal[0] * ny + c.binormal[0] * nx
    else:
        frame_n = tp.tangent * nz + tp.normal * ny + tp.binormal * nx
    out_norms = _to_render_space(frame_n)
    nl = np.linalg.norm(out_norms, axis=1, keepdims=True)
    nl[nl == 0.0] = 1.0
    out_norms = out_norms / nl

    return Mesh(
        vertices=out_verts.astype(np.float32),
        normals=out_norms.astype(np.float32),
        uvs=mesh.uvs,
        faces=mesh.faces,
        face_materials=mesh.face_materials,
        materials=mesh.materials,
    )
