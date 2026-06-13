"""
Tests for section-group expansion (``groups.py``).

The expected section sets are the ground truth from RCTGen's ``maketrack``: the unique
section names in the golden PNGs rendered for the three example projects
(``tests/regression/golden/<project>/tracks/*.png`` in ``~/code/RCTGen``). alpine = 42,
hybrid = 109, single_rail = 125 distinct sections.
"""

import pytest
from openrct2_track_generator.groups import (
    GROUP_FLAGS,
    RULES,
    expand_groups,
    is_group,
)
from openrct2_track_generator.loader import _build_sections
from openrct2_track_generator.sections import SECTION_REGISTRY

# The three RCTGen example projects' ``sections`` arrays (group names).
ALPINE_GROUPS = [
    "flat", "turns", "gentle_slopes", "diagonals", "gentle_sloped_turns",
    "banked_turns", "s_bends", "helices", "turn_bank_transitions",
]
HYBRID_GROUPS = [
    "flat", "turns", "gentle_slopes", "steep_slopes", "vertical_slopes",
    "large_slope_transitions", "diagonals", "sloped_turns", "banked_turns",
    "banked_sloped_turns", "s_bends", "helices", "barrel_rolls", "quarter_loops",
    "brakes", "boosters", "launched_lifts", "turn_bank_transitions", "zero_g_rolls",
    "large_sloped_turns", "large_banked_sloped_turns",
]
SINGLE_RAIL_GROUPS = [
    "flat", "turns", "gentle_slopes", "steep_slopes", "vertical_slopes",
    "large_slope_transitions", "diagonals", "sloped_turns", "banked_turns",
    "banked_sloped_turns", "s_bends", "helices", "barrel_rolls", "quarter_loops",
    "half_loops", "large_half_loops", "corkscrews", "brakes", "turn_bank_transitions",
    "small_slope_transitions", "large_corkscrews", "medium_half_loops", "zero_g_rolls",
    "large_sloped_turns", "large_banked_sloped_turns", "diagonal_brakes",
    "sloped_brakes", "dive_loops",
]

# The full alpine expansion (42 sections), from the alpine goldens.
ALPINE_SECTIONS = {
    "flat", "flat_diag", "flat_to_gentle", "flat_to_gentle_diag", "flat_to_left_bank",
    "flat_to_left_bank_diag", "flat_to_right_bank", "flat_to_right_bank_diag", "gentle",
    "gentle_diag", "gentle_to_flat", "gentle_to_flat_diag", "gentle_to_left_bank",
    "gentle_to_left_bank_diag", "gentle_to_right_bank", "gentle_to_right_bank_diag",
    "large_turn_left_to_diag", "large_turn_left_to_diag_bank", "large_turn_right_to_diag",
    "large_turn_right_to_diag_bank", "left_bank", "left_bank_diag", "left_bank_to_gentle",
    "left_bank_to_gentle_diag", "medium_helix_left", "medium_helix_right", "medium_turn_left",
    "medium_turn_left_bank", "medium_turn_left_gentle", "medium_turn_right_gentle",
    "right_bank_to_gentle", "right_bank_to_gentle_diag", "s_bend_left", "s_bend_right",
    "small_helix_left", "small_helix_right", "small_turn_left", "small_turn_left_bank",
    "small_turn_left_bank_to_gentle", "small_turn_left_gentle",
    "small_turn_right_bank_to_gentle", "small_turn_right_gentle",
}


def test_every_rule_section_is_in_registry() -> None:
    for _required, sections in RULES:
        for name in sections:
            assert name in SECTION_REGISTRY, f"{name} missing from SECTION_REGISTRY"


def test_every_group_flag_token_is_used_by_a_rule() -> None:
    rule_tokens = {token for required, _ in RULES for token in required}
    group_tokens = {token for tokens in GROUP_FLAGS.values() for token in tokens}
    assert group_tokens == rule_tokens


def test_is_group() -> None:
    assert is_group("turns")
    assert is_group("flat")  # both a group and a section name; group wins
    assert not is_group("small_turn_left")
    assert not is_group("nonsense")


def test_alpine_expansion_matches_golden() -> None:
    sections = expand_groups(ALPINE_GROUPS)
    assert len(sections) == 42
    assert set(sections) == ALPINE_SECTIONS
    assert len(set(sections)) == len(sections)  # no duplicates


def test_hybrid_and_single_rail_counts_match_golden() -> None:
    assert len(expand_groups(HYBRID_GROUPS)) == 109
    assert len(expand_groups(SINGLE_RAIL_GROUPS)) == 125


def test_expansion_is_canonical_order_regardless_of_input_order() -> None:
    # "flat" emits first in write_track_type order even if listed last.
    assert expand_groups(["gentle_slopes", "flat"])[0] == "flat"
    assert expand_groups(["flat", "gentle_slopes"]) == expand_groups(["gentle_slopes", "flat"])


def test_diagonals_alone_yields_only_flat_diag() -> None:
    # The diagonal-gentle / diagonal-banked pieces require their slope/bank group too.
    assert expand_groups(["diagonals"]) == ["flat_diag"]


def test_diagonals_with_gentle_slopes_adds_diagonal_gentle() -> None:
    sections = set(expand_groups(["diagonals", "gentle_slopes"]))
    assert {"gentle_diag", "flat_to_gentle_diag", "gentle_to_flat_diag"} <= sections


def test_gentle_sloped_turns_does_not_trigger_steep_sloped_turns() -> None:
    # gentle_sloped_turns sets only SLOPED_TURNS, so very_small_turn_*_steep must NOT appear.
    sections = set(expand_groups(["gentle_sloped_turns", "steep_slopes"]))
    assert "very_small_turn_left_steep" not in sections


def test_sloped_turns_triggers_steep_sloped_turns() -> None:
    # sloped_turns sets SLOPED_TURNS|STEEP_SLOPED_TURNS.
    sections = set(expand_groups(["sloped_turns", "steep_slopes"]))
    assert {"very_small_turn_left_steep", "very_small_turn_right_steep"} <= sections


def test_diagonal_brakes_requires_a_brake_family() -> None:
    assert expand_groups(["diagonal_brakes"]) == []
    assert "brake_diag" in expand_groups(["diagonal_brakes", "brakes"])
    assert "block_brake_diag" in expand_groups(["diagonal_brakes", "block_brakes"])


def test_banked_inline_twist_maps_to_banked_registry_key() -> None:
    # enum BARREL_ROLL_LEFT_BANK -> variable banked_barrel_roll_left, etc.
    assert expand_groups(["banked_barrel_rolls"]) == [
        "banked_barrel_roll_left", "banked_barrel_roll_right"
    ]
    assert expand_groups(["banked_inline_twists"]) == [
        "banked_inline_twist_left", "banked_inline_twist_right"
    ]
    assert expand_groups(["banked_zero_g_rolls"]) == [
        "banked_zero_g_roll_left", "banked_zero_g_roll_right"
    ]


def test_unknown_group_raises_keyerror() -> None:
    with pytest.raises(KeyError, match="Unknown section group"):
        expand_groups(["not_a_group"])


def test_loader_build_sections_expands_groups() -> None:
    sections = _build_sections({"sections": ALPINE_GROUPS})
    assert [s.name for s in sections] == expand_groups(ALPINE_GROUPS)


def test_loader_build_sections_backcompat_explicit_sections() -> None:
    # A pure list of section names keeps listed order (existing config behaviour).
    names = ["flat_to_gentle", "gentle", "small_turn_left"]
    sections = _build_sections({"sections": names})
    assert [s.name for s in sections] == names


def test_loader_build_sections_groups_then_explicit() -> None:
    # Groups expand first (canonical), then explicit sections, deduped.
    sections = _build_sections({"sections": ["flat", "gentle", "small_turn_left"]})
    # "flat" is a group -> [flat]; then explicit gentle, small_turn_left appended.
    assert [s.name for s in sections] == ["flat", "gentle", "small_turn_left"]
