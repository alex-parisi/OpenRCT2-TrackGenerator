"""Deformation correctness on small known meshes."""

import numpy as np
from openrct2_track_generator import curves
from openrct2_track_generator.constants import CLEARANCE_HEIGHT, TILE_SIZE, TrackFlag
from openrct2_track_generator.deform import deform_mesh
from openrct2_track_generator.sections import SECTION_REGISTRY
from openrct2_track_generator.types import TrackSection
from openrct2_x7_renderer.mesh import Material, Mesh


def _mesh(vertices):
    v = np.asarray(vertices, dtype=np.float32)
    return Mesh(
        vertices=v,
        normals=np.tile(np.array([0.0, 1.0, 0.0], dtype=np.float32), (v.shape[0], 1)),
        uvs=np.zeros((v.shape[0], 2), dtype=np.float32),
        faces=np.array([[0, 1, 2]], dtype=np.uint32),
        face_materials=np.zeros((1,), dtype=np.uint32),
        materials=[Material()],
    )


def test_flat_maps_known_vertices():
    # Mesh spans z in [0, TILE_SIZE] -> auto-fit scale=1, offset=0 for the flat section.
    mesh = _mesh([[0.0, 0.0, 0.0], [0.0, 0.0, TILE_SIZE], [1.0, 0.0, 0.0]])
    out = deform_mesh(mesh, SECTION_REGISTRY["flat"])
    verts = out.vertices.astype(np.float64)
    # Curve-space y shift is z_offset - 2*CH; z shift is -0.5*TILE_SIZE; then the
    # (x,y,z)->(z,y,x) render swap. Vertex (0,0,0):
    np.testing.assert_allclose(verts[0], [-0.5 * TILE_SIZE, -2 * CLEARANCE_HEIGHT, 0.0], atol=1e-5)
    # Vertex at z=length lands at the section end (+0.5*TILE_SIZE in render z).
    np.testing.assert_allclose(verts[1], [0.5 * TILE_SIZE, -2 * CLEARANCE_HEIGHT, 0.0], atol=1e-5)
    # Vertex with x=1 is pushed along the binormal (-1,0,0) -> render z = -1.
    np.testing.assert_allclose(verts[2], [-0.5 * TILE_SIZE, -2 * CLEARANCE_HEIGHT, -1.0], atol=1e-5)


def test_topology_passes_through():
    mesh = _mesh([[0.0, 0.0, 0.0], [0.0, 0.0, TILE_SIZE], [1.0, 0.0, 0.0]])
    out = deform_mesh(mesh, SECTION_REGISTRY["flat"])
    assert out.faces is mesh.faces
    assert out.uvs is mesh.uvs
    assert out.face_materials is mesh.face_materials
    assert out.materials is mesh.materials


def test_empty_mesh_returns_unchanged():
    out = deform_mesh(Mesh.empty(), SECTION_REGISTRY["flat"])
    assert out.vertices.shape[0] == 0


def test_curved_section_bends_geometry():
    # The same mesh deformed along a turn must differ from the flat result.
    mesh = _mesh([[0.0, 0.0, 0.0], [0.0, 0.0, TILE_SIZE], [1.0, 0.0, 0.0]])
    flat = deform_mesh(mesh, SECTION_REGISTRY["flat"]).vertices
    turn = deform_mesh(mesh, SECTION_REGISTRY["small_turn_left"]).vertices
    assert not np.allclose(flat, turn)


def test_out_of_range_distance_and_diagonal_flags():
    # An explicit scale/offset pushes some vertices' arc-distance outside
    # [0, length]; those extend straight along the end tangent. The DIAGONAL
    # flags add the half-tile shifts.
    section = TrackSection(
        "diag", curves.flat_curve, curves.FLAT_LENGTH,
        flags=TrackFlag.DIAGONAL | TrackFlag.DIAGONAL_2,
    )
    mesh = _mesh([[0.0, 0.0, 0.0], [0.0, 0.0, TILE_SIZE], [1.0, 0.0, 0.0]])
    below = deform_mesh(mesh, section, scale=1.0, offset=-1.0)  # u < 0 rows
    above = deform_mesh(mesh, section, scale=1.0, offset=section.length)  # u > length rows
    assert below.vertices.shape == mesh.vertices.shape
    assert above.vertices.shape == mesh.vertices.shape


def test_degenerate_span_falls_back_to_tile_size():
    # All vertices share one z -> auto-fit span is zero -> TILE_SIZE fallback.
    mesh = _mesh([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    out = deform_mesh(mesh, SECTION_REGISTRY["flat"])
    assert out.vertices.shape == mesh.vertices.shape


def test_flat_shaded_uses_single_frame():
    # On flat_to_gentle the frame's normal rotates along the piece. Per-vertex
    # shading varies; flat shading collapses every normal to the central frame.
    mesh = _mesh(
        [[0.0, 0.0, 0.0], [0.0, 0.0, TILE_SIZE / 2], [0.0, 0.0, TILE_SIZE]]
    )
    section = SECTION_REGISTRY["flat_to_gentle"]
    smooth = deform_mesh(mesh, section, flat_shaded=False).normals.astype(np.float64)
    flat = deform_mesh(mesh, section, flat_shaded=True).normals.astype(np.float64)
    assert not np.allclose(smooth[0], smooth[-1])  # smooth normals vary
    np.testing.assert_allclose(flat, np.tile(flat[0], (3, 1)), atol=1e-6)  # flat are uniform
