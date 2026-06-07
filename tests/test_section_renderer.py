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


def _draw_count(ctx):
    """Number of drawn (flag 0) models added across the render."""
    return sum(1 for e in ctx.events if e == ("add", 0))


def test_render_section_draws_tiled_special_mesh():
    # A brake section with its special mesh present draws extra (tiled) mechanism models
    # on top of the track; without the model it falls back to plain track.
    mesh = _mesh([[0, 0, 0], [0, 0, TILE_SIZE], [1, 0, 0]])
    special = _mesh([[0, 0.1, 0], [0, 0.1, TILE_SIZE], [1, 0.1, 0]])
    masks = load_section_masks("brake")

    plain = FakeContext()
    render_section(
        Track(meshes=[mesh], track_mesh_index=0), SECTION_REGISTRY["brake"], plain, masks
    )

    withspecial = FakeContext()
    render_section(
        Track(meshes=[mesh], track_mesh_index=0, special_models={"brake": special}),
        SECTION_REGISTRY["brake"], withspecial, masks,
    )
    assert _draw_count(withspecial) > _draw_count(plain)


def test_render_section_block_brake_tiles_by_one_tile():
    # block_brake resolves the BLOCK_BRAKE branch (tiles by one TILE_SIZE, not brake_length).
    mesh = _mesh([[0, 0, 0], [0, 0, TILE_SIZE], [1, 0, 0]])
    special = _mesh([[0, 0.1, 0], [0, 0.1, TILE_SIZE], [1, 0.1, 0]])
    ctx = FakeContext()
    track = Track(
        meshes=[mesh], track_mesh_index=0,
        special_models={"block_brake": special}, brake_length=TILE_SIZE / 2,
    )
    render_section(track, SECTION_REGISTRY["block_brake"], ctx, load_section_masks("block_brake"))
    assert _draw_count(ctx) > 0


def test_render_section_missing_special_model_falls_back_to_plain():
    # A special section whose model isn't supplied renders as plain track (no crash).
    mesh = _mesh([[0, 0, 0], [0, 0, TILE_SIZE], [1, 0, 0]])
    ctx = FakeContext()
    render_section(
        Track(meshes=[mesh], track_mesh_index=0),  # no special_models
        SECTION_REGISTRY["booster"], ctx, load_section_masks("booster"),
    )
    assert _draw_count(ctx) > 0


def test_render_section_places_rigid_support_model():
    # A rigid special (barrel_roll, flips the view matrix, non-vertical -> x-centred) places
    # its support model once, on top of the track.
    mesh = _mesh([[0, 0, 0], [0, 0, TILE_SIZE], [1, 0, 0]])
    support = _mesh([[0, -0.5, 0], [0, -0.5, TILE_SIZE], [1, -0.5, 0]])
    masks = load_section_masks("barrel_roll_left")

    plain = FakeContext()
    render_section(Track(meshes=[mesh], track_mesh_index=0),
                   SECTION_REGISTRY["barrel_roll_left"], plain, masks)
    withsupport = FakeContext()
    render_section(
        Track(meshes=[mesh], track_mesh_index=0, special_models={"support_barrel_roll": support}),
        SECTION_REGISTRY["barrel_roll_left"], withsupport, masks,
    )
    assert _draw_count(withsupport) > _draw_count(plain)


def test_render_section_rigid_right_variant_and_vertical():
    # Covers the *_RIGHT no-flip branch (barrel_roll_right) and the VERTICAL x=0 branch
    # (vertical) — both place their support model without error.
    mesh = _mesh([[0, 0, 0], [0, 0, TILE_SIZE], [1, 0, 0]])
    support = _mesh([[0, -0.5, 0], [0, -0.5, TILE_SIZE], [1, -0.5, 0]])
    cases = [("barrel_roll_right", "support_barrel_roll"), ("vertical", "support_vertical")]
    for sec, key in cases:
        ctx = FakeContext()
        render_section(
            Track(meshes=[mesh], track_mesh_index=0, special_models={key: support}),
            SECTION_REGISTRY[sec], ctx, load_section_masks(sec),
        )
        assert _draw_count(ctx) > 0


def test_render_section_rigid_missing_model_falls_back_to_plain():
    # A rigid special with no support model supplied renders as plain track (no crash).
    mesh = _mesh([[0, 0, 0], [0, 0, TILE_SIZE], [1, 0, 0]])
    ctx = FakeContext()
    render_section(
        Track(meshes=[mesh], track_mesh_index=0),  # no special_models
        SECTION_REGISTRY["vertical"], ctx, load_section_masks("vertical"),
    )
    assert _draw_count(ctx) > 0


def test_render_section_supports_add_base_and_posts():
    # With has_supports + the base/post models, a flat section gains a per-tile base and
    # support posts on top of the track.
    mesh = _mesh([[0, 0, 0], [0, 0, TILE_SIZE], [1, 0, 0]])
    base = _mesh([[0, -1, 0], [0, -1, TILE_SIZE], [1, -1, 0]])
    post = _mesh([[0, -1, 0], [0, 0, 0], [0.2, -1, 0]])
    masks = load_section_masks("flat")

    plain = FakeContext()
    render_section(Track(meshes=[mesh], track_mesh_index=0), SECTION_REGISTRY["flat"], plain, masks)
    supported = FakeContext()
    render_section(
        Track(meshes=[mesh], track_mesh_index=0, has_supports=True,
              special_models={"support_base": base, "support_flat": post}),
        SECTION_REGISTRY["flat"], supported, masks,
    )
    assert _draw_count(supported) > _draw_count(plain)


def test_render_section_no_supports_section_skips_supports():
    # A section flagged NO_SUPPORTS gets no base/posts even when the track has supports.
    mesh = _mesh([[0, 0, 0], [0, 0, TILE_SIZE], [1, 0, 0]])
    base = _mesh([[0, -1, 0], [0, -1, TILE_SIZE], [1, -1, 0]])
    masks = load_section_masks("vertical")
    off = FakeContext()
    render_section(
        Track(meshes=[mesh], track_mesh_index=0), SECTION_REGISTRY["vertical"], off, masks
    )
    on = FakeContext()
    render_section(
        Track(meshes=[mesh], track_mesh_index=0, has_supports=True,
              special_models={"support_base": base}),
        SECTION_REGISTRY["vertical"], on, masks,
    )
    assert _draw_count(on) == _draw_count(off)


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
