"""Section registry: membership, arc lengths, and unknown-name handling."""

import math

import pytest
from openrct2_track_generator import curves
from openrct2_track_generator.constants import TrackFlag
from openrct2_track_generator.sections import SECTION_REGISTRY, resolve_section

_EXPECTED_LENGTHS = {
    "flat": curves.FLAT_LENGTH,
    "flat_to_gentle": curves.FLAT_TO_GENTLE_LENGTH,
    "gentle": curves.GENTLE_LENGTH,
    "gentle_to_steep": curves.GENTLE_TO_STEEP_LENGTH,
    "steep": curves.STEEP_LENGTH,
    "small_turn_left": curves.SMALL_TURN_LENGTH,
    # Batch A
    "brake": curves.FLAT_LENGTH,
    "magnetic_brake": curves.FLAT_LENGTH,
    "block_brake": curves.FLAT_LENGTH,
    "booster": curves.FLAT_LENGTH,
    "gentle_to_flat": curves.FLAT_TO_GENTLE_LENGTH,
    "steep_to_gentle": curves.GENTLE_TO_STEEP_LENGTH,
    "flat_to_steep": curves.FLAT_TO_STEEP_LENGTH,
    "steep_to_flat": curves.FLAT_TO_STEEP_LENGTH,
    "vertical": curves.VERTICAL_LENGTH,
    "steep_to_vertical": curves.STEEP_TO_VERTICAL_LENGTH,
    "vertical_to_steep": curves.STEEP_TO_VERTICAL_LENGTH,
    # Batch B
    "medium_turn_left": curves.MEDIUM_TURN_LENGTH,
    "large_turn_left_to_diag": curves.LARGE_TURN_LENGTH,
    "large_turn_right_to_diag": curves.LARGE_TURN_LENGTH,
    # Batch C (diagonals)
    "flat_diag": curves.FLAT_DIAG_LENGTH,
    "flat_to_gentle_diag": curves.FLAT_TO_GENTLE_DIAG_LENGTH,
    "gentle_to_flat_diag": curves.FLAT_TO_GENTLE_DIAG_LENGTH,
    "gentle_diag": curves.GENTLE_DIAG_LENGTH,
    "gentle_to_steep_diag": curves.GENTLE_TO_STEEP_DIAG_LENGTH,
    "steep_to_gentle_diag": curves.GENTLE_TO_STEEP_DIAG_LENGTH,
    "steep_diag": curves.STEEP_DIAG_LENGTH,
    # Batch D (banking, banked turns, S-bends)
    "flat_to_left_bank": curves.FLAT_LENGTH,
    "flat_to_right_bank": curves.FLAT_LENGTH,
    "left_bank": curves.FLAT_LENGTH,
    "left_bank_to_gentle": curves.FLAT_TO_GENTLE_LENGTH,
    "right_bank_to_gentle": curves.FLAT_TO_GENTLE_LENGTH,
    "gentle_to_left_bank": curves.FLAT_TO_GENTLE_LENGTH,
    "gentle_to_right_bank": curves.FLAT_TO_GENTLE_LENGTH,
    "small_turn_left_bank": curves.SMALL_TURN_LENGTH,
    "medium_turn_left_bank": curves.MEDIUM_TURN_LENGTH,
    "large_turn_left_to_diag_bank": curves.LARGE_TURN_LENGTH,
    "large_turn_right_to_diag_bank": curves.LARGE_TURN_LENGTH,
    "s_bend_left": curves.S_BEND_LENGTH,
    "s_bend_right": curves.S_BEND_LENGTH,
    # Batch D follow-up (diagonal banking, banked S-bends, helices)
    "flat_to_left_bank_diag": curves.FLAT_DIAG_LENGTH,
    "flat_to_right_bank_diag": curves.FLAT_DIAG_LENGTH,
    "left_bank_diag": curves.FLAT_DIAG_LENGTH,
    "left_bank_to_gentle_diag": curves.FLAT_TO_GENTLE_DIAG_LENGTH,
    "right_bank_to_gentle_diag": curves.FLAT_TO_GENTLE_DIAG_LENGTH,
    "gentle_to_left_bank_diag": curves.FLAT_TO_GENTLE_DIAG_LENGTH,
    "gentle_to_right_bank_diag": curves.FLAT_TO_GENTLE_DIAG_LENGTH,
    "s_bend_left_bank": curves.S_BEND_LENGTH,
    "s_bend_right_bank": curves.S_BEND_LENGTH,
    "small_helix_left": curves.SMALL_HELIX_LENGTH,
    "small_helix_right": curves.SMALL_HELIX_LENGTH,
    "medium_helix_left": curves.MEDIUM_HELIX_LENGTH,
    "medium_helix_right": curves.MEDIUM_HELIX_LENGTH,
    # Batch E (inversions)
    "barrel_roll_left": curves.BARREL_ROLL_LENGTH,
    "barrel_roll_right": curves.BARREL_ROLL_LENGTH,
    "corkscrew_left": curves.CORKSCREW_LENGTH,
    "corkscrew_right": curves.CORKSCREW_LENGTH,
    "inline_twist_left": curves.INLINE_TWIST_LENGTH,
    "inline_twist_right": curves.INLINE_TWIST_LENGTH,
    "large_corkscrew_left": curves.LARGE_CORKSCREW_LENGTH,
    "large_corkscrew_right": curves.LARGE_CORKSCREW_LENGTH,
    "quarter_loop": curves.QUARTER_LOOP_LENGTH,
    "half_loop": curves.HALF_LOOP_LENGTH,
    "medium_half_loop_left": curves.MEDIUM_HALF_LOOP_LENGTH,
    "medium_half_loop_right": curves.MEDIUM_HALF_LOOP_LENGTH,
    "large_half_loop_left": curves.LARGE_HALF_LOOP_LENGTH,
    "large_half_loop_right": curves.LARGE_HALF_LOOP_LENGTH,
    "zero_g_roll_left": curves.ZERO_G_ROLL_LENGTH,
    "zero_g_roll_right": curves.ZERO_G_ROLL_LENGTH,
    "large_zero_g_roll_left": curves.LARGE_ZERO_G_ROLL_LENGTH,
    "large_zero_g_roll_right": curves.LARGE_ZERO_G_ROLL_LENGTH,
    "dive_loop_45_left": curves.DIVE_LOOP_45_LENGTH,
    "dive_loop_45_right": curves.DIVE_LOOP_45_LENGTH,
    "vertical_twist_left": curves.VERTICAL_TWIST_LENGTH,
    "vertical_twist_right": curves.VERTICAL_TWIST_LENGTH,
    "vertical_twist_left_to_diag": curves.VERTICAL_TWIST_45_LENGTH,
    "vertical_twist_right_to_diag": curves.VERTICAL_TWIST_45_LENGTH,
    "vertical_twist_left_to_orthogonal": curves.VERTICAL_TWIST_45_LENGTH,
    "vertical_twist_right_to_orthogonal": curves.VERTICAL_TWIST_45_LENGTH,
    "vertical_loop_left": curves.VERTICAL_LOOP_LENGTH,
    "vertical_loop_right": curves.VERTICAL_LOOP_LENGTH,
    # Batch F: gentle<->bank combo transitions
    "gentle_left_bank": curves.GENTLE_LENGTH,
    "gentle_right_bank": curves.GENTLE_LENGTH,
    "gentle_to_gentle_left_bank": curves.GENTLE_LENGTH,
    "gentle_to_gentle_right_bank": curves.GENTLE_LENGTH,
    "gentle_left_bank_to_gentle": curves.GENTLE_LENGTH,
    "gentle_right_bank_to_gentle": curves.GENTLE_LENGTH,
    "flat_to_gentle_left_bank": curves.FLAT_TO_GENTLE_LENGTH,
    "flat_to_gentle_right_bank": curves.FLAT_TO_GENTLE_LENGTH,
    "gentle_left_bank_to_flat": curves.FLAT_TO_GENTLE_LENGTH,
    "gentle_right_bank_to_flat": curves.FLAT_TO_GENTLE_LENGTH,
    "left_bank_to_gentle_left_bank": curves.FLAT_TO_GENTLE_LENGTH,
    "right_bank_to_gentle_right_bank": curves.FLAT_TO_GENTLE_LENGTH,
    "gentle_left_bank_to_left_bank": curves.FLAT_TO_GENTLE_LENGTH,
    "gentle_right_bank_to_right_bank": curves.FLAT_TO_GENTLE_LENGTH,
    # Batch G: gentle<->bank combo transitions (diagonal)
    "gentle_left_bank_diag": curves.GENTLE_DIAG_LENGTH,
    "gentle_right_bank_diag": curves.GENTLE_DIAG_LENGTH,
    "gentle_to_gentle_left_bank_diag": curves.GENTLE_DIAG_LENGTH,
    "gentle_to_gentle_right_bank_diag": curves.GENTLE_DIAG_LENGTH,
    "gentle_left_bank_to_gentle_diag": curves.GENTLE_DIAG_LENGTH,
    "gentle_right_bank_to_gentle_diag": curves.GENTLE_DIAG_LENGTH,
    "flat_to_gentle_left_bank_diag": curves.FLAT_TO_GENTLE_DIAG_LENGTH,
    "flat_to_gentle_right_bank_diag": curves.FLAT_TO_GENTLE_DIAG_LENGTH,
    "gentle_left_bank_to_flat_diag": curves.FLAT_TO_GENTLE_DIAG_LENGTH,
    "gentle_right_bank_to_flat_diag": curves.FLAT_TO_GENTLE_DIAG_LENGTH,
    "left_bank_to_gentle_left_bank_diag": curves.FLAT_TO_GENTLE_DIAG_LENGTH,
    "right_bank_to_gentle_right_bank_diag": curves.FLAT_TO_GENTLE_DIAG_LENGTH,
    "gentle_left_bank_to_left_bank_diag": curves.FLAT_TO_GENTLE_DIAG_LENGTH,
    "gentle_right_bank_to_right_bank_diag": curves.FLAT_TO_GENTLE_DIAG_LENGTH,
    # Batch H: gentle turns + banked variants
    "small_turn_left_gentle": curves.SMALL_TURN_GENTLE_LENGTH,
    "small_turn_right_gentle": curves.SMALL_TURN_GENTLE_LENGTH,
    "medium_turn_left_gentle": curves.MEDIUM_TURN_GENTLE_LENGTH,
    "medium_turn_right_gentle": curves.MEDIUM_TURN_GENTLE_LENGTH,
    "large_turn_left_to_diag_gentle": curves.LARGE_TURN_GENTLE_LENGTH,
    "large_turn_right_to_diag_gentle": curves.LARGE_TURN_GENTLE_LENGTH,
    "large_turn_left_to_orthogonal_gentle": curves.LARGE_TURN_GENTLE_LENGTH,
    "large_turn_right_to_orthogonal_gentle": curves.LARGE_TURN_GENTLE_LENGTH,
    "small_turn_left_bank_gentle": curves.SMALL_TURN_GENTLE_LENGTH,
    "small_turn_right_bank_gentle": curves.SMALL_TURN_GENTLE_LENGTH,
    "medium_turn_left_bank_gentle": curves.MEDIUM_TURN_GENTLE_LENGTH,
    "medium_turn_right_bank_gentle": curves.MEDIUM_TURN_GENTLE_LENGTH,
    "large_turn_left_bank_to_diag_gentle": curves.LARGE_TURN_GENTLE_LENGTH,
    "large_turn_right_bank_to_diag_gentle": curves.LARGE_TURN_GENTLE_LENGTH,
    "large_turn_left_bank_to_orthogonal_gentle": curves.LARGE_TURN_GENTLE_LENGTH,
    "large_turn_right_bank_to_orthogonal_gentle": curves.LARGE_TURN_GENTLE_LENGTH,
    "small_turn_left_bank_to_gentle": curves.TURN_BANK_TRANSITION_LENGTH,
    "small_turn_right_bank_to_gentle": curves.TURN_BANK_TRANSITION_LENGTH,
    # Batch I: steep turns
    "very_small_turn_left_steep": curves.VERY_SMALL_TURN_STEEP_LENGTH,
    "very_small_turn_right_steep": curves.VERY_SMALL_TURN_STEEP_LENGTH,
    "small_turn_left_steep": curves.SMALL_TURN_STEEP_LENGTH,
    "small_turn_right_steep": curves.SMALL_TURN_STEEP_LENGTH,
    "large_turn_left_to_diag_steep": curves.LARGE_TURN_STEEP_LENGTH,
    "large_turn_right_to_diag_steep": curves.LARGE_TURN_STEEP_LENGTH,
    "large_turn_left_to_orthogonal_steep": curves.LARGE_TURN_STEEP_LENGTH,
    "large_turn_right_to_orthogonal_steep": curves.LARGE_TURN_STEEP_LENGTH,
    # Batch J: diagonal / small slope transitions
    "small_flat_to_steep": curves.SMALL_FLAT_TO_STEEP_LENGTH,
    "small_steep_to_flat": curves.SMALL_FLAT_TO_STEEP_LENGTH,
    "small_flat_to_steep_diag": curves.SMALL_FLAT_TO_STEEP_DIAG_LENGTH,
    "small_steep_to_flat_diag": curves.SMALL_FLAT_TO_STEEP_DIAG_LENGTH,
    "flat_to_steep_diag": curves.FLAT_TO_STEEP_DIAG_LENGTH,
    "steep_to_flat_diag": curves.FLAT_TO_STEEP_DIAG_LENGTH,
    "steep_to_vertical_diag": curves.STEEP_TO_VERTICAL_DIAG_LENGTH,
    "vertical_to_steep_diag": curves.STEEP_TO_VERTICAL_DIAG_LENGTH,
    "vertical_diag": curves.VERTICAL_LENGTH,
    # Batch K: steep<->bank transitions
    "gentle_left_bank_to_steep": curves.GENTLE_TO_STEEP_LENGTH,
    "gentle_right_bank_to_steep": curves.GENTLE_TO_STEEP_LENGTH,
    "steep_to_gentle_left_bank": curves.GENTLE_TO_STEEP_LENGTH,
    "steep_to_gentle_right_bank": curves.GENTLE_TO_STEEP_LENGTH,
    "gentle_left_bank_to_steep_diag": curves.GENTLE_TO_STEEP_DIAG_LENGTH,
    "gentle_right_bank_to_steep_diag": curves.GENTLE_TO_STEEP_DIAG_LENGTH,
    "steep_to_gentle_left_bank_diag": curves.GENTLE_TO_STEEP_DIAG_LENGTH,
    "steep_to_gentle_right_bank_diag": curves.GENTLE_TO_STEEP_DIAG_LENGTH,
    # Batch L: banked inversions, dive_loop_90, brakes/boosters/misc
    "banked_barrel_roll_left": curves.BARREL_ROLL_LENGTH,
    "banked_barrel_roll_right": curves.BARREL_ROLL_LENGTH,
    "banked_inline_twist_left": curves.INLINE_TWIST_LENGTH,
    "banked_inline_twist_right": curves.INLINE_TWIST_LENGTH,
    "banked_zero_g_roll_left": curves.ZERO_G_ROLL_LENGTH,
    "banked_zero_g_roll_right": curves.ZERO_G_ROLL_LENGTH,
    "dive_loop_90_left": curves.DIVE_LOOP_90_LENGTH,
    "dive_loop_90_right": curves.DIVE_LOOP_90_LENGTH,
    "flat_asymmetric": curves.FLAT_LENGTH,
    "brake_gentle": curves.GENTLE_LENGTH,
    "magnetic_brake_gentle": curves.GENTLE_LENGTH,
    "launched_lift": curves.GENTLE_LENGTH,
    "vertical_booster": curves.VERTICAL_LENGTH,
    "brake_diag": curves.FLAT_DIAG_LENGTH,
    "block_brake_diag": curves.FLAT_DIAG_LENGTH,
    "magnetic_brake_diag": curves.FLAT_DIAG_LENGTH,
    "brake_gentle_diag": curves.GENTLE_DIAG_LENGTH,
    "magnetic_brake_gentle_diag": curves.GENTLE_DIAG_LENGTH,
}


def test_registry_matches_expected_sections():
    assert set(SECTION_REGISTRY) == set(_EXPECTED_LENGTHS)


def test_section_lengths_match_constants():
    for name, length in _EXPECTED_LENGTHS.items():
        assert math.isclose(SECTION_REGISTRY[name].length, length)


def test_section_flags():
    assert SECTION_REGISTRY["gentle_to_steep"].flags == TrackFlag.ALT_PREFER_ODD
    assert SECTION_REGISTRY["steep"].flags == TrackFlag.ALT_INVERT
    assert SECTION_REGISTRY["small_turn_left"].flags == TrackFlag.EXIT_90_DEG_LEFT
    assert SECTION_REGISTRY["flat"].flags == TrackFlag.NONE
    assert SECTION_REGISTRY["vertical"].flags == TrackFlag.VERTICAL | TrackFlag.NO_SUPPORTS
    assert TrackFlag.OFFSET_SPRITE_MASK in SECTION_REGISTRY["flat_to_steep"].flags


def test_section_chains():
    assert SECTION_REGISTRY["brake"].chain == "flat"
    assert SECTION_REGISTRY["steep_to_gentle"].chain == "gentle"
    assert SECTION_REGISTRY["vertical"].chain is None
    assert SECTION_REGISTRY["flat_diag"].chain == "flat_diag"


def test_diagonal_sections_have_diagonal_flag():
    for name in ["flat_diag", "gentle_diag", "steep_diag", "gentle_to_steep_diag"]:
        assert TrackFlag.DIAGONAL in SECTION_REGISTRY[name].flags


def test_resolve_known_section():
    assert resolve_section("flat").name == "flat"


def test_resolve_unknown_section_raises():
    with pytest.raises(KeyError, match="Unknown track section"):
        resolve_section("loop_de_loop")
