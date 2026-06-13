"""
Track *section groups* — the authoring shorthand maketrack uses instead of listing
all 170 sections by hand.

A maketrack ``sections`` array lists ~43 *group* names (``turns``, ``gentle_slopes``,
``banked_sloped_turns``, …). ``write_track_type`` (``track.cpp``) expands them into an
ordered list of concrete sections, with several *compound* conditions — e.g. the
diagonal-gentle pieces are emitted only when **both** ``diagonals`` and ``gentle_slopes``
are present, and ``diagonal_brakes`` emits a piece per co-present brake family.

This module is a faithful port of that expansion. :data:`GROUP_FLAGS` mirrors
``main.cpp``'s ``load_groups`` (group name -> the flag token(s) it sets); :data:`RULES`
mirrors ``write_track_type``'s emission order (a sequence of ``(required flags, sections)``
evaluated against the union of all requested groups' flags). :func:`expand_groups` runs the
table once over the collected flag set so compound conditions fire correctly.

Section *names* here are the registry keys in :mod:`~openrct2_track_generator.sections`,
which equal the C++ ``track_section_t`` variable names. Three enum IDs differ from their
variable name — ``INLINE_TWIST_LEFT_BANK`` / ``BARREL_ROLL_LEFT_BANK`` /
``ZERO_G_ROLL_LEFT_BANK`` map to the ``banked_*`` variables — and are spelled with the
``banked_*`` registry key below.
"""

from __future__ import annotations

from collections.abc import Iterable

__all__ = ["GROUP_FLAGS", "GROUP_NAMES", "RULES", "expand_groups", "is_group"]

# Group name -> flag token(s) it sets. Mirrors ``load_groups`` (``main.cpp:56-97``). Most
# groups set one eponymous flag; ``sloped_turns`` additionally sets ``steep_sloped_turns``
# (the C++ ``TRACK_GROUP_SLOPED_TURNS|TRACK_GROUP_STEEP_SLOPED_TURNS``), while
# ``gentle_sloped_turns`` sets only ``sloped_turns``.
GROUP_FLAGS: dict[str, frozenset[str]] = {
    "flat": frozenset({"flat"}),
    "brakes": frozenset({"brakes"}),
    "block_brakes": frozenset({"block_brakes"}),
    "diagonal_brakes": frozenset({"diagonal_brakes"}),
    "sloped_brakes": frozenset({"sloped_brakes"}),
    "magnetic_brakes": frozenset({"magnetic_brakes"}),
    "turns": frozenset({"turns"}),
    "gentle_slopes": frozenset({"gentle_slopes"}),
    "steep_slopes": frozenset({"steep_slopes"}),
    "vertical_slopes": frozenset({"vertical_slopes"}),
    "diagonals": frozenset({"diagonals"}),
    "sloped_turns": frozenset({"sloped_turns", "steep_sloped_turns"}),
    "gentle_sloped_turns": frozenset({"sloped_turns"}),
    "banked_turns": frozenset({"banked_turns"}),
    "banked_sloped_turns": frozenset({"banked_sloped_turns"}),
    "large_sloped_turns": frozenset({"large_sloped_turns"}),
    "large_banked_sloped_turns": frozenset({"large_banked_sloped_turns"}),
    "s_bends": frozenset({"s_bends"}),
    "banked_s_bends": frozenset({"banked_s_bends"}),
    "helices": frozenset({"helices"}),
    "small_slope_transitions": frozenset({"small_slope_transitions"}),
    "large_slope_transitions": frozenset({"large_slope_transitions"}),
    "barrel_rolls": frozenset({"barrel_rolls"}),
    "inline_twists": frozenset({"inline_twists"}),
    "quarter_loops": frozenset({"quarter_loops"}),
    "corkscrews": frozenset({"corkscrews"}),
    "large_corkscrews": frozenset({"large_corkscrews"}),
    "half_loops": frozenset({"half_loops"}),
    "vertical_loops": frozenset({"vertical_loops"}),
    "medium_half_loops": frozenset({"medium_half_loops"}),
    "large_half_loops": frozenset({"large_half_loops"}),
    "zero_g_rolls": frozenset({"zero_g_rolls"}),
    "dive_loops": frozenset({"dive_loops"}),
    "boosters": frozenset({"boosters"}),
    "launched_lifts": frozenset({"launched_lifts"}),
    "turn_bank_transitions": frozenset({"turn_bank_transitions"}),
    "steep_bank_transitions": frozenset({"steep_bank_transitions"}),
    "large_steep_sloped_turns": frozenset({"large_steep_sloped_turns"}),
    "banked_barrel_rolls": frozenset({"banked_barrel_rolls"}),
    "banked_inline_twists": frozenset({"banked_inline_twists"}),
    "banked_zero_g_rolls": frozenset({"banked_zero_g_rolls"}),
    "vertical_boosters": frozenset({"vertical_boosters"}),
}

GROUP_NAMES: frozenset[str] = frozenset(GROUP_FLAGS)

# Emission table: ``(required flag tokens, sections)`` in ``write_track_type`` order. A rule
# fires when every token in its key is present in the collected flag set; the sections are
# appended in order. Compound conditions (``A`` ∧ ``B``) list both tokens; the nested
# ``diagonal_brakes`` sub-conditions are split into one rule per co-present brake family.
RULES: tuple[tuple[frozenset[str], tuple[str, ...]], ...] = (
    (frozenset({"flat"}), ("flat",)),
    (frozenset({"brakes"}), ("brake",)),
    (frozenset({"block_brakes"}), ("block_brake",)),
    (frozenset({"sloped_brakes"}), ("brake_gentle",)),
    (frozenset({"magnetic_brakes"}), ("magnetic_brake",)),
    (frozenset({"boosters"}), ("booster",)),
    (frozenset({"launched_lifts"}), ("launched_lift",)),
    (frozenset({"vertical_boosters"}), ("vertical_booster",)),
    (frozenset({"gentle_slopes"}), ("flat_to_gentle", "gentle_to_flat", "gentle")),
    (frozenset({"magnetic_brakes"}), ("magnetic_brake_gentle",)),
    (frozenset({"steep_slopes"}), ("gentle_to_steep", "steep_to_gentle", "steep")),
    (frozenset({"vertical_slopes"}), ("steep_to_vertical", "vertical_to_steep", "vertical")),
    (
        frozenset({"turns"}),
        (
            "small_turn_left",
            "medium_turn_left",
            "large_turn_left_to_diag",
            "large_turn_right_to_diag",
        ),
    ),
    (frozenset({"diagonals"}), ("flat_diag",)),
    (frozenset({"diagonal_brakes", "brakes"}), ("brake_diag",)),
    (frozenset({"diagonal_brakes", "block_brakes"}), ("block_brake_diag",)),
    (frozenset({"diagonal_brakes", "magnetic_brakes"}), ("magnetic_brake_diag",)),
    (
        frozenset({"diagonals", "gentle_slopes"}),
        ("flat_to_gentle_diag", "gentle_to_flat_diag", "gentle_diag"),
    ),
    (frozenset({"diagonal_brakes", "sloped_brakes"}), ("brake_gentle_diag",)),
    (frozenset({"diagonal_brakes", "magnetic_brakes"}), ("magnetic_brake_gentle_diag",)),
    (
        frozenset({"diagonals", "steep_slopes"}),
        ("gentle_to_steep_diag", "steep_to_gentle_diag", "steep_diag"),
    ),
    (
        frozenset({"banked_turns"}),
        (
            "flat_to_left_bank",
            "flat_to_right_bank",
            "left_bank_to_gentle",
            "right_bank_to_gentle",
            "gentle_to_left_bank",
            "gentle_to_right_bank",
            "left_bank",
        ),
    ),
    (
        frozenset({"banked_turns", "diagonals"}),
        (
            "flat_to_left_bank_diag",
            "flat_to_right_bank_diag",
            "left_bank_to_gentle_diag",
            "right_bank_to_gentle_diag",
            "gentle_to_left_bank_diag",
            "gentle_to_right_bank_diag",
            "left_bank_diag",
        ),
    ),
    (
        frozenset({"banked_turns"}),
        (
            "small_turn_left_bank",
            "medium_turn_left_bank",
            "large_turn_left_to_diag_bank",
            "large_turn_right_to_diag_bank",
        ),
    ),
    (
        frozenset({"sloped_turns", "gentle_slopes"}),
        (
            "small_turn_left_gentle",
            "small_turn_right_gentle",
            "medium_turn_left_gentle",
            "medium_turn_right_gentle",
        ),
    ),
    (
        frozenset({"steep_sloped_turns", "steep_slopes"}),
        ("very_small_turn_left_steep", "very_small_turn_right_steep"),
    ),
    (
        frozenset({"sloped_turns", "vertical_slopes"}),
        ("vertical_twist_left", "vertical_twist_right"),
    ),
    (
        frozenset({"banked_sloped_turns"}),
        (
            "gentle_to_gentle_left_bank",
            "gentle_to_gentle_right_bank",
            "gentle_left_bank_to_gentle",
            "gentle_right_bank_to_gentle",
            "left_bank_to_gentle_left_bank",
            "right_bank_to_gentle_right_bank",
            "gentle_left_bank_to_left_bank",
            "gentle_right_bank_to_right_bank",
            "gentle_left_bank",
            "gentle_right_bank",
            "flat_to_gentle_left_bank",
            "flat_to_gentle_right_bank",
            "gentle_left_bank_to_flat",
            "gentle_right_bank_to_flat",
            "small_turn_left_bank_gentle",
            "small_turn_right_bank_gentle",
            "medium_turn_left_bank_gentle",
            "medium_turn_right_bank_gentle",
        ),
    ),
    (frozenset({"s_bends"}), ("s_bend_left", "s_bend_right")),
    (frozenset({"banked_s_bends"}), ("s_bend_left_bank", "s_bend_right_bank")),
    (
        frozenset({"helices"}),
        ("small_helix_left", "small_helix_right", "medium_helix_left", "medium_helix_right"),
    ),
    (
        frozenset({"steep_bank_transitions"}),
        (
            "gentle_left_bank_to_steep",
            "gentle_right_bank_to_steep",
            "steep_to_gentle_left_bank",
            "steep_to_gentle_right_bank",
            "gentle_left_bank_to_steep_diag",
            "gentle_right_bank_to_steep_diag",
            "steep_to_gentle_left_bank_diag",
            "steep_to_gentle_right_bank_diag",
        ),
    ),
    (
        frozenset({"large_steep_sloped_turns"}),
        (
            "small_turn_left_steep",
            "small_turn_right_steep",
            "large_turn_left_to_diag_steep",
            "large_turn_right_to_diag_steep",
            "large_turn_left_to_orthogonal_steep",
            "large_turn_right_to_orthogonal_steep",
        ),
    ),
    (frozenset({"inline_twists"}), ("inline_twist_left", "inline_twist_right")),
    (
        frozenset({"banked_inline_twists"}),
        ("banked_inline_twist_left", "banked_inline_twist_right"),
    ),
    (frozenset({"barrel_rolls"}), ("barrel_roll_left", "barrel_roll_right")),
    (
        frozenset({"banked_barrel_rolls"}),
        ("banked_barrel_roll_left", "banked_barrel_roll_right"),
    ),
    (frozenset({"half_loops"}), ("half_loop",)),
    (frozenset({"vertical_loops"}), ("vertical_loop_left", "vertical_loop_right")),
    (
        frozenset({"large_slope_transitions"}),
        ("flat_to_steep", "steep_to_flat", "flat_to_steep_diag", "steep_to_flat_diag"),
    ),
    (frozenset({"quarter_loops"}), ("quarter_loop",)),
    (frozenset({"corkscrews"}), ("corkscrew_left", "corkscrew_right")),
    (frozenset({"large_corkscrews"}), ("large_corkscrew_left", "large_corkscrew_right")),
    (
        frozenset({"turn_bank_transitions"}),
        ("small_turn_left_bank_to_gentle", "small_turn_right_bank_to_gentle"),
    ),
    (frozenset({"medium_half_loops"}), ("medium_half_loop_left", "medium_half_loop_right")),
    (frozenset({"large_half_loops"}), ("large_half_loop_left", "large_half_loop_right")),
    (
        frozenset({"zero_g_rolls"}),
        (
            "zero_g_roll_left",
            "zero_g_roll_right",
            "large_zero_g_roll_left",
            "large_zero_g_roll_right",
        ),
    ),
    (
        frozenset({"banked_zero_g_rolls"}),
        ("banked_zero_g_roll_left", "banked_zero_g_roll_right"),
    ),
    (frozenset({"dive_loops"}), ("dive_loop_45_left", "dive_loop_45_right")),
    (
        frozenset({"small_slope_transitions"}),
        (
            "small_flat_to_steep",
            "small_steep_to_flat",
            "small_flat_to_steep_diag",
            "small_steep_to_flat_diag",
        ),
    ),
    (
        frozenset({"large_sloped_turns"}),
        (
            "large_turn_left_to_diag_gentle",
            "large_turn_right_to_diag_gentle",
            "large_turn_left_to_orthogonal_gentle",
            "large_turn_right_to_orthogonal_gentle",
        ),
    ),
    (
        frozenset({"large_banked_sloped_turns"}),
        (
            "gentle_to_gentle_left_bank_diag",
            "gentle_to_gentle_right_bank_diag",
            "gentle_left_bank_to_gentle_diag",
            "gentle_right_bank_to_gentle_diag",
            "left_bank_to_gentle_left_bank_diag",
            "right_bank_to_gentle_right_bank_diag",
            "gentle_left_bank_to_left_bank_diag",
            "gentle_right_bank_to_right_bank_diag",
            "gentle_left_bank_diag",
            "gentle_right_bank_diag",
            "flat_to_gentle_left_bank_diag",
            "flat_to_gentle_right_bank_diag",
            "gentle_left_bank_to_flat_diag",
            "gentle_right_bank_to_flat_diag",
            "large_turn_left_bank_to_diag_gentle",
            "large_turn_right_bank_to_diag_gentle",
            "large_turn_left_bank_to_orthogonal_gentle",
            "large_turn_right_bank_to_orthogonal_gentle",
        ),
    ),
)


def is_group(name: str) -> bool:
    """True if ``name`` is a known section-group name."""
    return name in GROUP_FLAGS


def expand_groups(names: Iterable[str]) -> list[str]:
    """Expand group names into an ordered, de-duplicated list of section names.

    Collects the union of every named group's flag tokens, then walks :data:`RULES` once so
    compound conditions (e.g. ``diagonals`` ∧ ``gentle_slopes``) fire — exactly mirroring
    ``write_track_type``'s sequential emission. Raises :class:`KeyError` for an unknown
    group name (message lists the known groups).
    """
    flags: set[str] = set()
    for name in names:
        try:
            flags |= GROUP_FLAGS[name]
        except KeyError:
            known = ", ".join(sorted(GROUP_FLAGS))
            raise KeyError(f'Unknown section group "{name}" (known: {known})') from None

    out: list[str] = []
    seen: set[str] = set()
    for required, sections in RULES:
        if required <= flags:
            for section in sections:
                if section not in seen:
                    seen.add(section)
                    out.append(section)
    return out
