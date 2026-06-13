"""Offset-table classification, lookup, blend, and loading (``offsets.py`` / loader)."""

import numpy as np
import pytest
from openrct2_object_common.config import LoadError
from openrct2_track_generator.constants import CLEARANCE_HEIGHT, TILE_SIZE
from openrct2_track_generator.deform import get_track_point_array
from openrct2_track_generator.loader import build_track, load_offset_table
from openrct2_track_generator.offsets import (
    get_offset,
    offset_table_index,
    set_offset,
)
from openrct2_track_generator.sections import resolve_section


def _index_of(section_name: str, distance: float) -> int:
    section = resolve_section(section_name)
    d = distance if distance >= 0 else section.length
    tp = section.curve(np.array([d], dtype=np.float64))
    return offset_table_index(tp.tangent[0], tp.normal[0], tp.binormal[0])


# --- Classification (offset_table_index) -------------------------------------------------


def test_flat_endpoints_classify_as_flat() -> None:
    assert _index_of("flat", 0.0) == 0  # OFFSET_FLAT
    assert _index_of("flat", -1.0) == 0  # -1 sentinel -> length


def test_gentle_classifies_as_gentle() -> None:
    assert _index_of("gentle", 0.0) == 1  # OFFSET_GENTLE


def test_steep_classifies_as_steep() -> None:
    assert _index_of("steep", 0.0) == 2  # OFFSET_STEEP


def test_left_bank_end_is_banked() -> None:
    # left_bank's exit is a full 45 bank -> low nibble is OFFSET_BANK (3).
    code = _index_of("left_bank", -1.0)
    assert (code & 0xF) == 3


# Synthetic frames exercising every classification branch (the classifier is a pure
# function of tangent/normal/binormal, so crafted frames hit branches real sections rarely do).
_CH = CLEARANCE_HEIGHT
_T = TILE_SIZE
_UP = np.array([0.0, 1.0, 0.0])
# A 45 bank: normal tilted so sqrt(nx^2+nz^2) = sin(45). binormal y-sign picks left/right.
_BANK_NORMAL = np.array([np.sqrt(0.5), np.sqrt(0.5), 0.0])
_BANK_BINORMAL_LEFT = np.array([0.0, 0.5, 0.0])   # binormal.y > 0 -> not "right"
_BANK_BINORMAL_RIGHT = np.array([0.0, -0.5, 0.0])  # binormal.y < 0 -> right bit


def _unit(v: list[float]) -> np.ndarray:
    a = np.array(v, dtype=np.float64)
    return a / np.linalg.norm(a)


def test_classify_inverted() -> None:
    assert offset_table_index(_unit([0, 0, _T]), np.array([0.0, -1.0, 0.0]), _UP) == 5


def test_classify_flat_banked_left_and_right() -> None:
    flat = _unit([0, 0, _T])
    assert offset_table_index(flat, _BANK_NORMAL, _BANK_BINORMAL_LEFT) == 3  # OFFSET_BANK
    assert offset_table_index(flat, _BANK_NORMAL, _BANK_BINORMAL_RIGHT) == (0x10 | 3)


def test_classify_gentle_banked() -> None:
    gentle = _unit([0, 2 * _CH, _T])
    assert offset_table_index(gentle, _BANK_NORMAL, _BANK_BINORMAL_LEFT) == 4  # GENTLE_BANK


def test_classify_diagonal_variants() -> None:
    assert offset_table_index(_unit([-_T, 0, _T]), _UP, _UP) == 6           # DIAGONAL
    assert offset_table_index(_unit([-_T, 0, _T]), _BANK_NORMAL, _BANK_BINORMAL_LEFT) == 7
    assert offset_table_index(_unit([-_T, 2 * _CH, _T]), _UP, _UP) == 8     # DIAGONAL_GENTLE
    assert offset_table_index(_unit([-_T, 8 * _CH, _T]), _UP, _UP) == 9     # DIAGONAL_STEEP


def test_classify_no_match() -> None:
    # Straight up (not a recognized track tangent) matches nothing at any rotation.
    assert offset_table_index(_unit([0, 1, 0]), _UP, _UP) == 0xFF


def test_classify_rotated_left_and_right() -> None:
    # A flat tangent pointing +x matches "flat" only after a left rotation (0x60 tag); -x right.
    assert offset_table_index(_unit([_T, 0, 0]), _UP, _UP) == (0x60 | 0)
    assert offset_table_index(_unit([-_T, 0, 0]), _UP, _UP) == (0x20 | 0)


# --- Lookup math (get_offset) ------------------------------------------------------------


def test_get_offset_no_match_is_zero() -> None:
    table = np.ones((10, 8))
    assert np.array_equal(get_offset(0xFF, 0, table), np.zeros(3))


def test_get_offset_flat_scales_z_and_y() -> None:
    table = np.zeros((10, 8))
    table[0, 0] = 32.0  # z column for view 0
    table[0, 1] = 8.0   # y column for view 0
    off = get_offset(0, 0, table)  # index 0, end_angle 0, right 0, view 0
    assert off[0] == 0.0
    assert off[2] == pytest.approx(TILE_SIZE)        # 32 * TILE_SIZE/32
    assert off[1] == pytest.approx(CLEARANCE_HEIGHT)  # 8 * CLEARANCE_HEIGHT/8


def test_get_offset_right_bank_negates_z() -> None:
    table = np.zeros((10, 8))
    # code: right bit | OFFSET_BANK(3); right -> rotated_view_angle = (0+0+2)%4 = 2 -> col 4.
    table[3, 4] = 32.0
    off = get_offset(0x10 | 3, 0, table)
    assert off[2] == pytest.approx(-TILE_SIZE)


def test_get_offset_diagonal_splits_into_x_and_z() -> None:
    table = np.zeros((10, 8))
    table[6, 0] = 32.0  # OFFSET_DIAGONAL
    off = get_offset(6, 0, table)
    assert off[2] == pytest.approx(TILE_SIZE * np.sqrt(0.5))
    assert off[0] == pytest.approx(off[2])


def test_get_offset_end_rotation_applied() -> None:
    table = np.zeros((10, 8))
    table[0, 0] = 32.0
    # code 0x60 -> end_angle 3: rotated_view_angle = (0+3)%4 = 3 (col 6), then rotate_y(-3*-90).
    table[0, 6] = 32.0
    off = get_offset(0x60, 0, table)
    # rotate_y(-1.5*pi) maps +z to -x (roughly); just assert it moved off the z axis.
    assert not np.allclose(off, [0.0, 0.0, TILE_SIZE])


def test_set_offset_zero_table_is_zero() -> None:
    section = resolve_section("gentle")
    start, end = set_offset(0, section, np.zeros((10, 8)))
    assert np.array_equal(start, np.zeros(3))
    assert np.array_equal(end, np.zeros(3))


# --- Hermite blend in the deform ---------------------------------------------------------


def test_hermite_blend_weights_endpoints() -> None:
    section = resolve_section("flat")
    length = section.length
    u = np.array([0.0, length / 2.0, length])
    z = np.zeros(3)
    base = get_track_point_array(section.curve, section.flags, 0.0, length, z, z, u)
    shifted = get_track_point_array(
        section.curve, section.flags, 0.0, length, np.array([0.0, 1.0, 0.0]), z, u
    )
    diff = shifted.position - base.position
    # w_start(v) = 2v^3 - 3v^2 + 1: 1 at v=0, 0.5 at v=0.5, 0 at v=1.
    assert diff[0] == pytest.approx([0.0, 1.0, 0.0])
    assert diff[1] == pytest.approx([0.0, 0.5, 0.0])
    assert diff[2] == pytest.approx([0.0, 0.0, 0.0])


# --- Loading (load_offset_table) ---------------------------------------------------------


def test_load_offset_table_absent_is_zeros() -> None:
    table = load_offset_table({})
    assert table.shape == (10, 8)
    assert np.array_equal(table, np.zeros((10, 8)))


def test_load_offset_table_parses_named_rows() -> None:
    table = load_offset_table(
        {"offsets": {"flat": [0, 0.5, 0, 0, 0, 0.5, 0, 0], "steep": [-2.25, 0, -2, 0, 0, 0, 0, 0]}}
    )
    assert table[0, 1] == pytest.approx(0.5)
    assert table[2, 0] == pytest.approx(-2.25)
    assert np.array_equal(table[1], np.zeros(8))  # gentle untouched


def test_load_offset_table_not_object_raises() -> None:
    with pytest.raises(LoadError, match="offsets"):
        load_offset_table({"offsets": [1, 2, 3]})


def test_load_offset_table_wrong_length_raises() -> None:
    with pytest.raises(LoadError, match="length 8"):
        load_offset_table({"offsets": {"flat": [0, 1, 2]}})


def test_load_offset_table_non_numeric_raises() -> None:
    with pytest.raises(LoadError, match="non-numeric"):
        load_offset_table({"offsets": {"flat": [0, 1, 2, 3, 4, 5, 6, "x"]}})


def test_build_track_sets_special_end_offsets_flag_and_table() -> None:
    from openrct2_x7_renderer.mesh import Mesh

    cfg = {
        "id": "t", "name": "T", "ride_type": "steel_roller_coaster",
        "meshes": ["unused.obj"], "sections": ["flat"],
        "flags": ["special_end_offsets"],
        "offsets": {"flat": [0, 0.5, 0, 0, 0, 0.5, 0, 0]},
    }
    track = build_track(cfg, [Mesh.empty()])
    assert track.special_end_offsets is True
    assert track.offset_table[0, 1] == pytest.approx(0.5)


def test_build_track_defaults_no_special_offsets() -> None:
    from openrct2_x7_renderer.mesh import Mesh

    cfg = {
        "id": "t", "name": "T", "ride_type": "steel_roller_coaster",
        "meshes": ["unused.obj"], "sections": ["flat"],
    }
    track = build_track(cfg, [Mesh.empty()])
    assert track.special_end_offsets is False
    assert np.array_equal(track.offset_table, np.zeros((10, 8)))
