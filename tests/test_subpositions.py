"""Subposition encoding port (``subpositions.py`` / ``subposition.cpp``)."""

import numpy as np
from openrct2_track_generator._sprite_rotations import SPRITE_GROUPS
from openrct2_track_generator.constants import TILE_SIZE
from openrct2_track_generator.sections import resolve_section
from openrct2_track_generator.subpositions import (
    SPRITE_GROUP_BASE,
    SPRITE_GROUP_CORKSCREW,
    SPRITE_GROUP_DIVE_LOOP,
    SPRITE_GROUP_INLINE_TWIST,
    SPRITE_GROUP_ORTHOGONAL,
    SPRITE_GROUP_ZERO_G_ROLLS_ORTHOGONAL,
    Subposition,
    _groups_for_section,
    _rotate_x,
    _rotate_y,
    _rotate_z,
    build_subposition_data,
    calc_differing_coords,
    generate_view_subposition_data,
    get_closest_rotation,
    get_subposition,
    track_point_get_rotation,
)
from openrct2_track_generator.types import Track, TrackPointArray, TrackSection

# --- get_subposition: the 8 view+4*diag coordinate transforms ----------------------------


def test_get_subposition_all_cases_from_origin():
    o = np.zeros(3)
    assert get_subposition(o, 0, 0) == (32, 16, 0)
    assert get_subposition(o, 1, 0) == (16, 0, 0)
    assert get_subposition(o, 2, 0) == (0, 16, 0)
    assert get_subposition(o, 3, 0) == (16, 32, 0)
    assert get_subposition(o, 0, 1) == (16, 15, 0)
    assert get_subposition(o, 1, 1) == (16, 16, 0)
    assert get_subposition(o, 2, 1) == (16, 15, 0)
    assert get_subposition(o, 3, 1) == (16, 16, 0)


def test_get_subposition_scales_and_swaps():
    # x grid = round(32*z/TILE) = 4, y grid = round(32*x/TILE) = 2; case 1 -> (16-y, x).
    pos = np.array([2 * TILE_SIZE / 32.0, 0.0, 4 * TILE_SIZE / 32.0])
    assert get_subposition(pos, 1, 0) == (14, 4, 0)


def test_calc_differing_coords():
    assert calc_differing_coords((0, 0, 0), (0, 0, 0)) == 0
    assert calc_differing_coords((0, 0, 0), (1, 0, 0)) == 1
    assert calc_differing_coords((0, 0, 0), (1, 1, 1)) == 3
    assert calc_differing_coords((0, 0, 0), (2, 0, 0)) == -1


# --- rotation snap (get_closest_rotation) ------------------------------------------------


def test_get_closest_rotation_roundtrips_table_rows():
    # A matrix built from a table row's (yaw,pitch,roll) must snap back to that row's sprites.
    for name, bit in [
        ("orthogonal", 1), ("diagonal", 2), ("turn", 4),
        ("inline_twist", 8), ("corkscrew", 16), ("dive_loop", 256),
    ]:
        for ys, ps, bs, yaw, pitch, roll in SPRITE_GROUPS[name][::9]:
            mat = _rotate_y(yaw) @ _rotate_z(pitch) @ _rotate_x(roll)
            assert get_closest_rotation(mat, bit) == (ys, ps, bs)


def test_get_closest_rotation_respects_group_mask():
    # An orthogonal row is found under the orthogonal bit but a disjoint group can't return it.
    ys, ps, bs, yaw, pitch, roll = SPRITE_GROUPS["orthogonal"][5]
    mat = _rotate_y(yaw) @ _rotate_z(pitch) @ _rotate_x(roll)
    assert get_closest_rotation(mat, SPRITE_GROUP_ORTHOGONAL) == (ys, ps, bs)
    # Searching only the inline-twist group yields some inline-twist row instead.
    assert get_closest_rotation(mat, SPRITE_GROUP_INLINE_TWIST) != (ys, ps, bs)


def test_track_point_get_rotation_columns():
    t = np.array([0.0, 0.0, 1.0])
    n = np.array([0.0, 1.0, 0.0])
    b = np.array([1.0, 0.0, 0.0])
    m = track_point_get_rotation(t, n, b)
    # Columns are cc(t), cc(n), -cc(b).
    assert np.allclose(m[:, 0], [t[2], t[1], t[0]])
    assert np.allclose(m[:, 1], [n[2], n[1], n[0]])
    assert np.allclose(m[:, 2], [-b[2], -b[1], -b[0]])


# --- whole-section walks (generate_view_subposition_data) --------------------------------


def test_flat_yaw_progression_per_view():
    flat = resolve_section("flat")
    for view in range(4):
        pts = generate_view_subposition_data(flat, SPRITE_GROUP_BASE, view)
        assert pts, "flat produced no subpositions"
        # Flat: unbanked, level; yaw sprite is 8*view + 0.
        assert {p.yaw_sprite for p in pts} == {8 * view}
        assert {p.pitch_sprite for p in pts} == {0}
        assert {p.bank_sprite for p in pts} == {0}


def test_gentle_is_pitched_up():
    pts = generate_view_subposition_data(resolve_section("gentle"), SPRITE_GROUP_BASE, 0)
    assert {p.pitch_sprite for p in pts} == {2}  # up25


def test_diagonal_section_walks():
    # flat_diag starts on a diagonal heading -> diag=1 path of get_subposition.
    pts = generate_view_subposition_data(resolve_section("flat_diag"), SPRITE_GROUP_BASE, 0)
    assert pts


def test_turn_section_walks():
    # A turn whose exit heads into a negative atan2 quadrant exercises the finish-angle wrap.
    pts = generate_view_subposition_data(resolve_section("small_turn_left"), SPRITE_GROUP_BASE, 0)
    assert pts


def test_start_angle_wrap_for_plus_x_heading():
    # No real section starts heading +x, so a synthetic straight curve covers the start wrap.
    def _straight_x(d: np.ndarray) -> TrackPointArray:
        d = np.atleast_1d(np.asarray(d, dtype=np.float64))
        n = d.shape[0]
        pos = np.zeros((n, 3))
        pos[:, 0] = d  # heading +x
        tan = np.tile([1.0, 0.0, 0.0], (n, 1))
        nor = np.tile([0.0, 1.0, 0.0], (n, 1))
        bn = np.tile([0.0, 0.0, 1.0], (n, 1))
        return TrackPointArray(pos, tan, nor, bn)

    sec = TrackSection("fake_plus_x", _straight_x, TILE_SIZE)
    assert generate_view_subposition_data(sec, SPRITE_GROUP_BASE, 0)


def test_reverse_pass_runs():
    # The reverse path (descending backwards) exercises the transform + the skip_start pop.
    fwd = generate_view_subposition_data(resolve_section("gentle"), SPRITE_GROUP_BASE, 0)
    rev = generate_view_subposition_data(resolve_section("gentle"), SPRITE_GROUP_BASE, 0, reverse=6)
    assert fwd and rev


# --- group selection + sidecar -----------------------------------------------------------


def test_groups_for_section_heuristics():
    assert _groups_for_section(resolve_section("flat")) == SPRITE_GROUP_BASE
    assert _groups_for_section(resolve_section("corkscrew_left")) == (
        SPRITE_GROUP_CORKSCREW | SPRITE_GROUP_ORTHOGONAL
    )
    assert _groups_for_section(resolve_section("zero_g_roll_left")) == (
        SPRITE_GROUP_ZERO_G_ROLLS_ORTHOGONAL | SPRITE_GROUP_INLINE_TWIST | SPRITE_GROUP_ORTHOGONAL
    )
    assert _groups_for_section(resolve_section("inline_twist_left")) == (
        SPRITE_GROUP_INLINE_TWIST | SPRITE_GROUP_BASE
    )
    assert _groups_for_section(resolve_section("barrel_roll_left")) == (
        SPRITE_GROUP_INLINE_TWIST | SPRITE_GROUP_BASE
    )
    assert _groups_for_section(resolve_section("dive_loop_45_left")) == (
        SPRITE_GROUP_DIVE_LOOP | SPRITE_GROUP_CORKSCREW | SPRITE_GROUP_ORTHOGONAL
    )


def test_build_subposition_data_shape():
    track = Track(id="t.x", sections=[resolve_section("flat"), resolve_section("gentle")])
    data = build_subposition_data(track)
    assert data["id"] == "t.x"
    assert data["encoding"] == "openrct2_sprite_indices"
    assert [s["section"] for s in data["sections"]] == ["flat", "gentle"]
    flat_views = data["sections"][0]["views"]
    assert len(flat_views) == 4
    pt = flat_views[0][0]
    assert set(pt) == {"x", "y", "z", "yaw_sprite", "pitch_sprite", "bank_sprite"}


def test_subposition_dataclass_fields():
    s = Subposition(1, 2, 3, 8, 2, 0)
    assert (s.x, s.y, s.z, s.yaw_sprite, s.pitch_sprite, s.bank_sprite) == (1, 2, 3, 8, 2, 0)
