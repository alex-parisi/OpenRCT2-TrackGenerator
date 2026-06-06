"""Section tiling layout + the ghost/extrude scene building."""

import numpy as np
from openrct2_object_common.testing import FakeContext
from openrct2_track_generator.constants import TILE_SIZE
from openrct2_track_generator.masks import load_section_masks
from openrct2_track_generator.section_renderer import _section_layout, render_section
from openrct2_track_generator.sections import SECTION_REGISTRY
from openrct2_track_generator.types import Track
from openrct2_x7_renderer.constants import MeshFlag
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


def test_layout_tiles_turn_into_two_copies():
    # A one-tile mesh; the small turn (~2.36 tiles long) tiles into 2 copies.
    mesh = _mesh([[0, 0, 0], [0, 0, TILE_SIZE], [1, 0, 0]])
    num, scale, length = _section_layout(mesh, SECTION_REGISTRY["small_turn_left"])
    assert num == 2
    assert np.isclose(num * length, SECTION_REGISTRY["small_turn_left"].length)


def test_layout_empty_mesh_falls_back_to_one():
    assert _section_layout(Mesh.empty(), SECTION_REGISTRY["flat"])[0] == 1


def test_layout_degenerate_z_uses_tile_size():
    mesh = _mesh([[0, 0, 2], [1, 0, 2], [0, 1, 2]])  # all z equal -> zero extent
    num, _scale, _length = _section_layout(mesh, SECTION_REGISTRY["flat"])
    assert num == 1


def test_render_emits_ghost_end_caps():
    # Every full render adds ghost end-cap copies (occlude neighbours, not drawn).
    mesh = _mesh([[0, 0, 0], [0, 0, TILE_SIZE], [1, 0, 0]])
    track = Track(meshes=[mesh], track_mesh_index=0)
    ctx = FakeContext()
    render_section(track, SECTION_REGISTRY["flat"], ctx, load_section_masks("flat"))
    assert ("add", int(MeshFlag.GHOST)) in ctx.events
