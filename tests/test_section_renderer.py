"""Section tiling layout + the ghost/extrude scene building."""

import numpy as np
from openrct2_object_common.testing import FakeContext
from openrct2_track_generator.constants import TILE_SIZE
from openrct2_track_generator.masks import load_section_masks
from openrct2_track_generator.section_renderer import (
    _section_layout,
    angle_plan,
    render_section,
)
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


def test_angle_plan_lift_expands_flat_but_not_turn():
    mesh = _mesh([[0, 0, 0], [0, 0, TILE_SIZE], [1, 0, 0]])
    flat, turn = SECTION_REGISTRY["flat"], SECTION_REGISTRY["small_turn_left"]
    flat_masks = load_section_masks("flat")  # 2 views
    turn_masks = load_section_masks("small_turn_left")  # 4 views, no chain

    no_lift = Track(meshes=[mesh])
    assert len(angle_plan(no_lift, flat, flat_masks)) == 2

    lift = Track(meshes=[mesh], has_lift=True)
    flat_plan = angle_plan(lift, flat, flat_masks)
    assert len(flat_plan) == 4  # 2-view flat expands to 4 chain directions
    assert all(chain is not None for _a, _vm, chain in flat_plan)
    # The turn has no chain, so lift doesn't add a stamp (and it's already 4 views).
    turn_plan = angle_plan(lift, turn, turn_masks)
    assert len(turn_plan) == 4 and all(chain is None for _a, _vm, chain in turn_plan)


def test_angle_plan_lift_expands_chainless_two_view_section():
    # A lift expands ANY 2-view section to 4 (e.g. an S-bend), with no chain stamp.
    mesh = _mesh([[0, 0, 0], [0, 0, TILE_SIZE], [1, 0, 0]])
    sbend = SECTION_REGISTRY["s_bend_left"]
    masks = load_section_masks("s_bend_left")
    assert len(masks) == 2 and sbend.chain is None
    assert len(angle_plan(Track(meshes=[mesh]), sbend, masks)) == 2  # no lift
    plan = angle_plan(Track(meshes=[mesh], has_lift=True), sbend, masks)
    assert len(plan) == 4 and all(chain is None for _a, _vm, chain in plan)


def test_render_section_with_lift_expands_and_stamps():
    mesh = _mesh([[0, 0, 0], [0, 0, TILE_SIZE], [1, 0, 0]])
    track = Track(meshes=[mesh], track_mesh_index=0, has_lift=True)
    masks = load_section_masks("flat")
    imgs = render_section(track, SECTION_REGISTRY["flat"], FakeContext(), masks)
    assert len(imgs) == 4  # was 2 without lift
