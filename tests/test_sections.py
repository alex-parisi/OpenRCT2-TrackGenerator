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
}


def test_registry_has_slice_sections():
    assert set(SECTION_REGISTRY) == set(_EXPECTED_LENGTHS)


def test_section_lengths_match_constants():
    for name, length in _EXPECTED_LENGTHS.items():
        assert math.isclose(SECTION_REGISTRY[name].length, length)


def test_section_flags():
    assert SECTION_REGISTRY["gentle_to_steep"].flags == TrackFlag.ALT_PREFER_ODD
    assert SECTION_REGISTRY["steep"].flags == TrackFlag.ALT_INVERT
    assert SECTION_REGISTRY["small_turn_left"].flags == TrackFlag.EXIT_90_DEG_LEFT
    assert SECTION_REGISTRY["flat"].flags == TrackFlag.NONE


def test_resolve_known_section():
    assert resolve_section("flat").name == "flat"


def test_resolve_unknown_section_raises():
    with pytest.raises(KeyError, match="Unknown track section"):
        resolve_section("loop_de_loop")
