"""
The registry binding section names to their geometry.

This is the seam that scales the generator to the full RCTGen catalogue: adding a
section is a one-line :class:`TrackSection` entry referencing a curve from
:mod:`~openrct2_track_generator.curves`. The vertical slice registers six. Flags and
arc lengths are copied from ``track_sections.cpp``'s ``const track_section_t`` table.
"""

from . import curves
from .constants import TrackFlag
from .types import TrackSection

__all__ = ["SECTION_REGISTRY", "resolve_section"]

# `chain` names the lift-hill chain stamp (sprites.cpp): flat-ish pieces use "flat",
# sloped pieces use "gentle"; turns have none.
SECTION_REGISTRY: dict[str, TrackSection] = {
    "flat": TrackSection("flat", curves.flat_curve, curves.FLAT_LENGTH, chain="flat"),
    "flat_to_gentle": TrackSection(
        "flat_to_gentle", curves.flat_to_gentle_curve, curves.FLAT_TO_GENTLE_LENGTH, chain="flat"
    ),
    "gentle": TrackSection("gentle", curves.gentle_curve, curves.GENTLE_LENGTH, chain="gentle"),
    "gentle_to_steep": TrackSection(
        "gentle_to_steep",
        curves.gentle_to_steep_curve,
        curves.GENTLE_TO_STEEP_LENGTH,
        flags=TrackFlag.ALT_PREFER_ODD,
        chain="gentle",
    ),
    "steep": TrackSection(
        "steep", curves.steep_curve, curves.STEEP_LENGTH, flags=TrackFlag.ALT_INVERT, chain="gentle"
    ),
    "small_turn_left": TrackSection(
        "small_turn_left",
        curves.small_turn_left_curve,
        curves.SMALL_TURN_LENGTH,
        flags=TrackFlag.EXIT_90_DEG_LEFT,
    ),
    # --- Batch A: brakes/booster (flat geometry; the brake mechanism mesh is deferred),
    #     reverse + large slope transitions, and vertical. ---
    "brake": TrackSection("brake", curves.flat_curve, curves.FLAT_LENGTH, chain="flat"),
    "magnetic_brake": TrackSection(
        "magnetic_brake", curves.flat_curve, curves.FLAT_LENGTH, chain="flat"
    ),
    "block_brake": TrackSection(
        "block_brake", curves.flat_curve, curves.FLAT_LENGTH, chain="flat"
    ),
    "booster": TrackSection("booster", curves.flat_curve, curves.FLAT_LENGTH, chain="flat"),
    "gentle_to_flat": TrackSection(
        "gentle_to_flat", curves.gentle_to_flat_curve, curves.FLAT_TO_GENTLE_LENGTH, chain="flat"
    ),
    "steep_to_gentle": TrackSection(
        "steep_to_gentle",
        curves.steep_to_gentle_curve,
        curves.GENTLE_TO_STEEP_LENGTH,
        flags=TrackFlag.ALT_INVERT | TrackFlag.ALT_PREFER_ODD,
        chain="gentle",
    ),
    "flat_to_steep": TrackSection(
        "flat_to_steep",
        curves.flat_to_steep_curve,
        curves.FLAT_TO_STEEP_LENGTH,
        flags=TrackFlag.OFFSET_SPRITE_MASK | TrackFlag.ALT_PREFER_ODD,
    ),
    "steep_to_flat": TrackSection(
        "steep_to_flat",
        curves.steep_to_flat_curve,
        curves.FLAT_TO_STEEP_LENGTH,
        flags=TrackFlag.OFFSET_SPRITE_MASK | TrackFlag.ALT_INVERT | TrackFlag.ALT_PREFER_ODD,
    ),
    "vertical": TrackSection(
        "vertical",
        curves.vertical_curve,
        curves.VERTICAL_LENGTH,
        flags=TrackFlag.VERTICAL | TrackFlag.NO_SUPPORTS,
    ),
    "steep_to_vertical": TrackSection(
        "steep_to_vertical",
        curves.steep_to_vertical_curve,
        curves.STEEP_TO_VERTICAL_LENGTH,
        flags=TrackFlag.VERTICAL | TrackFlag.NO_SUPPORTS | TrackFlag.ALT_INVERT,
    ),
    "vertical_to_steep": TrackSection(
        "vertical_to_steep",
        curves.vertical_to_steep_curve,
        curves.STEEP_TO_VERTICAL_LENGTH,
        flags=TrackFlag.VERTICAL | TrackFlag.NO_SUPPORTS,
    ),
    # --- Batch B: flat turns (right turns of small/medium are game-mirrored, so only
    #     left exists; the large orthogonal<->diagonal turns have both). ---
    "medium_turn_left": TrackSection(
        "medium_turn_left",
        curves.medium_turn_left_curve,
        curves.MEDIUM_TURN_LENGTH,
        flags=TrackFlag.EXIT_90_DEG_LEFT,
    ),
    "large_turn_left_to_diag": TrackSection(
        "large_turn_left_to_diag",
        curves.large_turn_left_to_diag_curve,
        curves.LARGE_TURN_LENGTH,
        flags=TrackFlag.EXIT_45_DEG_LEFT,
    ),
    "large_turn_right_to_diag": TrackSection(
        "large_turn_right_to_diag",
        curves.large_turn_right_to_diag_curve,
        curves.LARGE_TURN_LENGTH,
        flags=TrackFlag.EXIT_45_DEG_RIGHT,
    ),
    # --- Batch C: diagonals (DIAGONAL flag shifts the piece onto the 45° axis). ---
    "flat_diag": TrackSection(
        "flat_diag", curves.flat_diag_curve, curves.FLAT_DIAG_LENGTH,
        flags=TrackFlag.DIAGONAL, chain="flat_diag",
    ),
    "flat_to_gentle_diag": TrackSection(
        "flat_to_gentle_diag", curves.flat_to_gentle_diag_curve, curves.FLAT_TO_GENTLE_DIAG_LENGTH,
        flags=TrackFlag.DIAGONAL | TrackFlag.SUPPORT_BASE, chain="flat_diag",
    ),
    "gentle_to_flat_diag": TrackSection(
        "gentle_to_flat_diag", curves.gentle_to_flat_diag_curve, curves.FLAT_TO_GENTLE_DIAG_LENGTH,
        flags=TrackFlag.DIAGONAL | TrackFlag.SUPPORT_BASE, chain="flat_diag",
    ),
    "gentle_diag": TrackSection(
        "gentle_diag", curves.gentle_diag_curve, curves.GENTLE_DIAG_LENGTH,
        flags=TrackFlag.DIAGONAL | TrackFlag.SUPPORT_BASE, chain="flat_diag",
    ),
    "gentle_to_steep_diag": TrackSection(
        "gentle_to_steep_diag",
        curves.gentle_to_steep_diag_curve,
        curves.GENTLE_TO_STEEP_DIAG_LENGTH,
        flags=TrackFlag.DIAGONAL | TrackFlag.SUPPORT_BASE,
    ),
    "steep_to_gentle_diag": TrackSection(
        "steep_to_gentle_diag",
        curves.steep_to_gentle_diag_curve,
        curves.GENTLE_TO_STEEP_DIAG_LENGTH,
        flags=TrackFlag.DIAGONAL | TrackFlag.SUPPORT_BASE,
    ),
    "steep_diag": TrackSection(
        "steep_diag", curves.steep_diag_curve, curves.STEEP_DIAG_LENGTH,
        flags=TrackFlag.DIAGONAL | TrackFlag.SUPPORT_BASE,
    ),
    # --- Batch D: banking, banked turns, S-bends (all via banked_curve). ---
    "flat_to_left_bank": TrackSection(
        "flat_to_left_bank", curves.flat_to_left_bank_curve, curves.FLAT_LENGTH,
        flags=TrackFlag.EXIT_BANK_LEFT,
    ),
    "flat_to_right_bank": TrackSection(
        "flat_to_right_bank", curves.flat_to_right_bank_curve, curves.FLAT_LENGTH,
        flags=TrackFlag.EXIT_BANK_RIGHT,
    ),
    "left_bank": TrackSection(
        "left_bank", curves.left_bank_curve, curves.FLAT_LENGTH,
        flags=TrackFlag.ENTRY_BANK_LEFT | TrackFlag.EXIT_BANK_LEFT,
    ),
    "left_bank_to_gentle": TrackSection(
        "left_bank_to_gentle", curves.left_bank_to_gentle_curve, curves.FLAT_TO_GENTLE_LENGTH,
        flags=TrackFlag.ENTRY_BANK_LEFT,
    ),
    "right_bank_to_gentle": TrackSection(
        "right_bank_to_gentle", curves.right_bank_to_gentle_curve, curves.FLAT_TO_GENTLE_LENGTH,
        flags=TrackFlag.ENTRY_BANK_RIGHT,
    ),
    "gentle_to_left_bank": TrackSection(
        "gentle_to_left_bank", curves.gentle_to_left_bank_curve, curves.FLAT_TO_GENTLE_LENGTH,
        flags=TrackFlag.EXIT_BANK_LEFT,
    ),
    "gentle_to_right_bank": TrackSection(
        "gentle_to_right_bank", curves.gentle_to_right_bank_curve, curves.FLAT_TO_GENTLE_LENGTH,
        flags=TrackFlag.EXIT_BANK_RIGHT,
    ),
    "small_turn_left_bank": TrackSection(
        "small_turn_left_bank", curves.small_turn_left_bank_curve, curves.SMALL_TURN_LENGTH,
        flags=TrackFlag.ENTRY_BANK_LEFT | TrackFlag.EXIT_BANK_LEFT | TrackFlag.EXIT_90_DEG_LEFT,
    ),
    "medium_turn_left_bank": TrackSection(
        "medium_turn_left_bank", curves.medium_turn_left_bank_curve, curves.MEDIUM_TURN_LENGTH,
        flags=TrackFlag.ENTRY_BANK_LEFT | TrackFlag.EXIT_BANK_LEFT | TrackFlag.EXIT_90_DEG_LEFT,
    ),
    "large_turn_left_to_diag_bank": TrackSection(
        "large_turn_left_to_diag_bank",
        curves.large_turn_left_to_diag_bank_curve,
        curves.LARGE_TURN_LENGTH,
        flags=TrackFlag.ENTRY_BANK_LEFT | TrackFlag.EXIT_BANK_LEFT | TrackFlag.EXIT_45_DEG_LEFT,
    ),
    "large_turn_right_to_diag_bank": TrackSection(
        "large_turn_right_to_diag_bank",
        curves.large_turn_right_to_diag_bank_curve,
        curves.LARGE_TURN_LENGTH,
        flags=TrackFlag.ENTRY_BANK_RIGHT | TrackFlag.EXIT_BANK_RIGHT | TrackFlag.EXIT_45_DEG_RIGHT,
    ),
    "s_bend_left": TrackSection("s_bend_left", curves.s_bend_left_curve, curves.S_BEND_LENGTH),
    "s_bend_right": TrackSection("s_bend_right", curves.s_bend_right_curve, curves.S_BEND_LENGTH),
    # --- Batch D follow-up: diagonal banking, banked S-bends, helices. ---
    "flat_to_left_bank_diag": TrackSection(
        "flat_to_left_bank_diag", curves.flat_to_left_bank_diag_curve, curves.FLAT_DIAG_LENGTH,
        flags=TrackFlag.DIAGONAL | TrackFlag.EXIT_BANK_LEFT,
    ),
    "flat_to_right_bank_diag": TrackSection(
        "flat_to_right_bank_diag", curves.flat_to_right_bank_diag_curve, curves.FLAT_DIAG_LENGTH,
        flags=TrackFlag.DIAGONAL | TrackFlag.EXIT_BANK_RIGHT,
    ),
    "left_bank_diag": TrackSection(
        "left_bank_diag", curves.left_bank_diag_curve, curves.FLAT_DIAG_LENGTH,
        flags=TrackFlag.DIAGONAL | TrackFlag.ENTRY_BANK_LEFT | TrackFlag.EXIT_BANK_LEFT,
    ),
    "left_bank_to_gentle_diag": TrackSection(
        "left_bank_to_gentle_diag",
        curves.left_bank_to_gentle_diag_curve,
        curves.FLAT_TO_GENTLE_DIAG_LENGTH,
        flags=TrackFlag.DIAGONAL | TrackFlag.ENTRY_BANK_LEFT | TrackFlag.SUPPORT_BASE,
    ),
    "right_bank_to_gentle_diag": TrackSection(
        "right_bank_to_gentle_diag",
        curves.right_bank_to_gentle_diag_curve,
        curves.FLAT_TO_GENTLE_DIAG_LENGTH,
        flags=TrackFlag.DIAGONAL | TrackFlag.ENTRY_BANK_RIGHT | TrackFlag.SUPPORT_BASE,
    ),
    "gentle_to_left_bank_diag": TrackSection(
        "gentle_to_left_bank_diag",
        curves.gentle_to_left_bank_diag_curve,
        curves.FLAT_TO_GENTLE_DIAG_LENGTH,
        flags=TrackFlag.DIAGONAL | TrackFlag.EXIT_BANK_LEFT | TrackFlag.SUPPORT_BASE,
    ),
    "gentle_to_right_bank_diag": TrackSection(
        "gentle_to_right_bank_diag",
        curves.gentle_to_right_bank_diag_curve,
        curves.FLAT_TO_GENTLE_DIAG_LENGTH,
        flags=TrackFlag.DIAGONAL | TrackFlag.EXIT_BANK_RIGHT | TrackFlag.SUPPORT_BASE,
    ),
    "s_bend_left_bank": TrackSection(
        "s_bend_left_bank", curves.s_bend_left_bank_curve, curves.S_BEND_LENGTH
    ),
    "s_bend_right_bank": TrackSection(
        "s_bend_right_bank", curves.s_bend_right_bank_curve, curves.S_BEND_LENGTH
    ),
    "small_helix_left": TrackSection(
        "small_helix_left", curves.small_helix_left_curve, curves.SMALL_HELIX_LENGTH,
        flags=TrackFlag.ENTRY_BANK_LEFT | TrackFlag.EXIT_BANK_LEFT
        | TrackFlag.SUPPORT_BASE | TrackFlag.EXIT_180_DEG,
    ),
    "small_helix_right": TrackSection(
        "small_helix_right", curves.small_helix_right_curve, curves.SMALL_HELIX_LENGTH,
        flags=TrackFlag.ENTRY_BANK_RIGHT | TrackFlag.EXIT_BANK_RIGHT
        | TrackFlag.SUPPORT_BASE | TrackFlag.EXIT_180_DEG,
    ),
    "medium_helix_left": TrackSection(
        "medium_helix_left", curves.medium_helix_left_curve, curves.MEDIUM_HELIX_LENGTH,
        flags=TrackFlag.ENTRY_BANK_LEFT | TrackFlag.EXIT_BANK_LEFT
        | TrackFlag.SUPPORT_BASE | TrackFlag.EXIT_180_DEG,
    ),
    "medium_helix_right": TrackSection(
        "medium_helix_right", curves.medium_helix_right_curve, curves.MEDIUM_HELIX_LENGTH,
        flags=TrackFlag.ENTRY_BANK_RIGHT | TrackFlag.EXIT_BANK_RIGHT
        | TrackFlag.SUPPORT_BASE | TrackFlag.EXIT_180_DEG,
    ),
    # --- Batch E: inversions. The TRACK_SPECIAL_* model selector is a deferred special
    #     mesh; geometry is the curve. ---
    "barrel_roll_left": TrackSection(
        "barrel_roll_left", curves.barrel_roll_left_curve, curves.BARREL_ROLL_LENGTH,
        flags=TrackFlag.NO_SUPPORTS | TrackFlag.OFFSET_SPRITE_MASK,
    ),
    "barrel_roll_right": TrackSection(
        "barrel_roll_right", curves.barrel_roll_right_curve, curves.BARREL_ROLL_LENGTH,
        flags=TrackFlag.NO_SUPPORTS | TrackFlag.OFFSET_SPRITE_MASK,
    ),
    "corkscrew_left": TrackSection(
        "corkscrew_left", curves.corkscrew_left_curve, curves.CORKSCREW_LENGTH,
        flags=TrackFlag.NO_SUPPORTS | TrackFlag.OFFSET_SPRITE_MASK | TrackFlag.EXIT_90_DEG_LEFT,
    ),
    "corkscrew_right": TrackSection(
        "corkscrew_right", curves.corkscrew_right_curve, curves.CORKSCREW_LENGTH,
        flags=TrackFlag.NO_SUPPORTS | TrackFlag.OFFSET_SPRITE_MASK | TrackFlag.EXIT_90_DEG_RIGHT,
    ),
    "inline_twist_left": TrackSection(
        "inline_twist_left", curves.inline_twist_left_curve, curves.INLINE_TWIST_LENGTH,
        flags=TrackFlag.NO_SUPPORTS | TrackFlag.OFFSET_SPRITE_MASK,
    ),
    "inline_twist_right": TrackSection(
        "inline_twist_right", curves.inline_twist_right_curve, curves.INLINE_TWIST_LENGTH,
        flags=TrackFlag.NO_SUPPORTS | TrackFlag.OFFSET_SPRITE_MASK,
    ),
    "large_corkscrew_left": TrackSection(
        "large_corkscrew_left", curves.large_corkscrew_left_curve, curves.LARGE_CORKSCREW_LENGTH,
        flags=TrackFlag.NO_SUPPORTS | TrackFlag.OFFSET_SPRITE_MASK | TrackFlag.EXIT_90_DEG_LEFT,
    ),
    "large_corkscrew_right": TrackSection(
        "large_corkscrew_right", curves.large_corkscrew_right_curve, curves.LARGE_CORKSCREW_LENGTH,
        flags=TrackFlag.NO_SUPPORTS | TrackFlag.OFFSET_SPRITE_MASK | TrackFlag.EXIT_90_DEG_RIGHT,
    ),
    "quarter_loop": TrackSection(
        "quarter_loop", curves.quarter_loop_curve, curves.QUARTER_LOOP_LENGTH,
        flags=TrackFlag.VERTICAL | TrackFlag.NO_SUPPORTS | TrackFlag.EXIT_180_DEG,
    ),
    "half_loop": TrackSection(
        "half_loop", curves.half_loop_curve, curves.HALF_LOOP_LENGTH,
        flags=TrackFlag.NO_SUPPORTS | TrackFlag.EXIT_180_DEG,
    ),
    "medium_half_loop_left": TrackSection(
        "medium_half_loop_left", curves.medium_half_loop_left_curve,
        curves.MEDIUM_HALF_LOOP_LENGTH,
        flags=TrackFlag.OFFSET_SPRITE_MASK | TrackFlag.EXIT_180_DEG,
    ),
    "medium_half_loop_right": TrackSection(
        "medium_half_loop_right", curves.medium_half_loop_right_curve,
        curves.MEDIUM_HALF_LOOP_LENGTH,
        flags=TrackFlag.OFFSET_SPRITE_MASK | TrackFlag.EXIT_180_DEG,
    ),
    "large_half_loop_left": TrackSection(
        "large_half_loop_left", curves.large_half_loop_left_curve,
        curves.LARGE_HALF_LOOP_LENGTH,
        flags=TrackFlag.OFFSET_SPRITE_MASK | TrackFlag.EXIT_180_DEG,
    ),
    "large_half_loop_right": TrackSection(
        "large_half_loop_right", curves.large_half_loop_right_curve,
        curves.LARGE_HALF_LOOP_LENGTH,
        flags=TrackFlag.OFFSET_SPRITE_MASK | TrackFlag.EXIT_180_DEG,
    ),
    "zero_g_roll_left": TrackSection(
        "zero_g_roll_left", curves.zero_g_roll_left_curve, curves.ZERO_G_ROLL_LENGTH,
        flags=TrackFlag.NO_SUPPORTS | TrackFlag.OFFSET_SPRITE_MASK,
    ),
    "zero_g_roll_right": TrackSection(
        "zero_g_roll_right", curves.zero_g_roll_right_curve, curves.ZERO_G_ROLL_LENGTH,
        flags=TrackFlag.NO_SUPPORTS | TrackFlag.OFFSET_SPRITE_MASK,
    ),
    "large_zero_g_roll_left": TrackSection(
        "large_zero_g_roll_left", curves.large_zero_g_roll_left_curve,
        curves.LARGE_ZERO_G_ROLL_LENGTH,
        flags=TrackFlag.NO_SUPPORTS | TrackFlag.OFFSET_SPRITE_MASK
        | TrackFlag.ALT_INVERT | TrackFlag.ALT_PREFER_ODD,
    ),
    "large_zero_g_roll_right": TrackSection(
        "large_zero_g_roll_right", curves.large_zero_g_roll_right_curve,
        curves.LARGE_ZERO_G_ROLL_LENGTH,
        flags=TrackFlag.NO_SUPPORTS | TrackFlag.OFFSET_SPRITE_MASK
        | TrackFlag.ALT_INVERT | TrackFlag.ALT_PREFER_ODD,
    ),
    "dive_loop_45_left": TrackSection(
        "dive_loop_45_left", curves.dive_loop_45_left_curve, curves.DIVE_LOOP_45_LENGTH,
        flags=TrackFlag.DIAGONAL | TrackFlag.NO_SUPPORTS | TrackFlag.EXIT_45_DEG_LEFT,
    ),
    "dive_loop_45_right": TrackSection(
        "dive_loop_45_right", curves.dive_loop_45_right_curve, curves.DIVE_LOOP_45_LENGTH,
        flags=TrackFlag.DIAGONAL | TrackFlag.NO_SUPPORTS | TrackFlag.EXIT_45_DEG_RIGHT,
    ),
    "vertical_twist_left": TrackSection(
        "vertical_twist_left", curves.vertical_twist_left_curve, curves.VERTICAL_TWIST_LENGTH,
        flags=TrackFlag.VERTICAL | TrackFlag.OFFSET_SPRITE_MASK | TrackFlag.NO_SUPPORTS
        | TrackFlag.EXIT_90_DEG_LEFT,
    ),
    "vertical_twist_right": TrackSection(
        "vertical_twist_right", curves.vertical_twist_right_curve, curves.VERTICAL_TWIST_LENGTH,
        flags=TrackFlag.VERTICAL | TrackFlag.OFFSET_SPRITE_MASK | TrackFlag.NO_SUPPORTS
        | TrackFlag.EXIT_90_DEG_RIGHT,
    ),
    "vertical_twist_left_to_diag": TrackSection(
        "vertical_twist_left_to_diag", curves.vertical_twist_left_to_diag_curve,
        curves.VERTICAL_TWIST_45_LENGTH,
        flags=TrackFlag.OFFSET_SPRITE_MASK | TrackFlag.NO_SUPPORTS | TrackFlag.EXIT_45_DEG_LEFT,
    ),
    "vertical_twist_right_to_diag": TrackSection(
        "vertical_twist_right_to_diag", curves.vertical_twist_right_to_diag_curve,
        curves.VERTICAL_TWIST_45_LENGTH,
        flags=TrackFlag.OFFSET_SPRITE_MASK | TrackFlag.NO_SUPPORTS | TrackFlag.EXIT_45_DEG_RIGHT,
    ),
    "vertical_twist_left_to_orthogonal": TrackSection(
        "vertical_twist_left_to_orthogonal", curves.vertical_twist_left_to_orthogonal_curve,
        curves.VERTICAL_TWIST_45_LENGTH,
        flags=TrackFlag.OFFSET_SPRITE_MASK | TrackFlag.NO_SUPPORTS | TrackFlag.EXIT_45_DEG_LEFT,
    ),
    "vertical_twist_right_to_orthogonal": TrackSection(
        "vertical_twist_right_to_orthogonal", curves.vertical_twist_right_to_orthogonal_curve,
        curves.VERTICAL_TWIST_45_LENGTH,
        flags=TrackFlag.OFFSET_SPRITE_MASK | TrackFlag.NO_SUPPORTS | TrackFlag.EXIT_45_DEG_RIGHT,
    ),
    "vertical_loop_left": TrackSection(
        "vertical_loop_left", curves.vertical_loop_left_curve, curves.VERTICAL_LOOP_LENGTH,
        flags=TrackFlag.NO_SUPPORTS | TrackFlag.EXIT_180_DEG,
    ),
    "vertical_loop_right": TrackSection(
        "vertical_loop_right", curves.vertical_loop_right_curve, curves.VERTICAL_LOOP_LENGTH,
        flags=TrackFlag.NO_SUPPORTS | TrackFlag.EXIT_180_DEG,
    ),
}


def resolve_section(name: str) -> TrackSection:
    """Look up a section by name, raising ``KeyError`` with the known names if absent."""
    try:
        return SECTION_REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(SECTION_REGISTRY))
        raise KeyError(f'Unknown track section "{name}" (known: {known})') from None
