"""
Track geometry constants and flags, ported from RCTGen's ``track.h``.

``TILE_SIZE`` is the side length (in OBJ/model units) of one OpenRCT2 tile; the
deformation, curve library, and render scale all share it. ``CLEARANCE_HEIGHT`` is
one OpenRCT2 height step expressed in the same units, and ``BANK_ANGLE`` is the
default 45° bank. ``TrackFlag`` mirrors the rendering-flag bitfield from ``track.h``;
only a subset is exercised by the vertical slice, but the full set is carried so the
section registry can grow without re-numbering.
"""

from enum import IntEnum, IntFlag

__all__ = [
    "BANK_ANGLE",
    "CLEARANCE_HEIGHT",
    "SPECIAL_MODEL_KEY",
    "SPECIAL_RIGHT_NO_FLIP",
    "SPECIAL_TILED",
    "SUPPORT_BANK_DENOM",
    "SUPPORT_BANK_KEY",
    "SUPPORT_BASE_KEY",
    "TILE_SIZE",
    "SpecialModel",
    "TrackFlag",
]

# RCTGen track.h: TILE_SIZE 3.3, CLEARANCE_HEIGHT (0.20412414523*TILE_SIZE).
TILE_SIZE: float = 3.3
CLEARANCE_HEIGHT: float = 0.20412414523 * TILE_SIZE

# RCTGen track_sections.cpp: BANK_ANGLE 0.25*M_PI (a 45-degree bank).
BANK_ANGLE: float = 0.25 * 3.141592653589793


class TrackFlag(IntFlag):
    """Per-section rendering flags (mirrors the anonymous enum in ``track.h``).

    These describe how a section is oriented/handled at render time (diagonal
    rotation, verticality, banking at entry/exit, exit heading, and the special
    geometry family). The vertical slice only sets ``VERTICAL`` / ``ALT_*`` /
    ``EXIT_90_DEG_LEFT`` flags, but the rest are defined up front so the registry
    can scale to all 170 sections.
    """

    NONE = 0
    DIAGONAL = 1
    VERTICAL = 2
    ENTRY_BANK_LEFT = 8
    ENTRY_BANK_RIGHT = 16
    EXIT_BANK_LEFT = 32
    EXIT_BANK_RIGHT = 64
    EXIT_45_DEG_LEFT = 128
    EXIT_45_DEG_RIGHT = 256
    EXIT_90_DEG_LEFT = 512
    EXIT_90_DEG_RIGHT = 1024
    EXIT_180_DEG = 2048
    ALT_PREFER_ODD = 4096
    ALT_INVERT = 8192
    NO_SUPPORTS = 16384
    OFFSET_SPRITE_MASK = 32768
    SUPPORT_BASE = 65536
    DIAGONAL_2 = 131072


class SpecialModel(IntEnum):
    """Special-mechanism model selector (``track.h``'s ``TRACK_SPECIAL_*`` high byte).

    A section carrying one of these renders the corresponding mechanism mesh
    (brake / booster / inversion support) on top of the deformed track. The value
    equals ``(TRACK_SPECIAL_* >> 24)``; ``get_special_index`` (mirrored by
    :data:`SPECIAL_MODEL_KEY`) maps it to the model loaded for the track.
    """

    STEEP_TO_VERTICAL = 0x01
    VERTICAL_TO_STEEP = 0x02
    VERTICAL = 0x03
    VERTICAL_TWIST_LEFT = 0x04
    VERTICAL_TWIST_RIGHT = 0x05
    BARREL_ROLL_LEFT = 0x06
    BARREL_ROLL_RIGHT = 0x07
    HALF_LOOP = 0x08
    QUARTER_LOOP = 0x09
    CORKSCREW_LEFT = 0x0A
    CORKSCREW_RIGHT = 0x0B
    ZERO_G_ROLL_LEFT = 0x0C
    ZERO_G_ROLL_RIGHT = 0x0D
    LARGE_ZERO_G_ROLL_LEFT = 0x0E
    LARGE_ZERO_G_ROLL_RIGHT = 0x0F
    BRAKE = 0x10
    BLOCK_BRAKE = 0x11
    BOOSTER = 0x12
    MAGNETIC_BRAKE = 0x13
    LAUNCHED_LIFT = 0x14
    VERTICAL_BOOSTER = 0x15


# Special selector -> model key in ``Track.special_models`` (RCTGen ``get_special_index``
# + ``support_model_names``). The brake/booster family tiles; the inversion specials place a
# single rigid support model.
SPECIAL_MODEL_KEY: dict[SpecialModel, str] = {
    SpecialModel.BRAKE: "brake",
    SpecialModel.BLOCK_BRAKE: "block_brake",
    SpecialModel.BOOSTER: "booster",
    SpecialModel.MAGNETIC_BRAKE: "magnetic_brake",
    SpecialModel.LAUNCHED_LIFT: "booster",
    SpecialModel.VERTICAL_BOOSTER: "booster",
    SpecialModel.STEEP_TO_VERTICAL: "support_steep_to_vertical",
    SpecialModel.VERTICAL_TO_STEEP: "support_vertical_to_steep",
    SpecialModel.VERTICAL: "support_vertical",
    SpecialModel.VERTICAL_TWIST_LEFT: "support_vertical_twist",
    SpecialModel.VERTICAL_TWIST_RIGHT: "support_vertical_twist",
    SpecialModel.BARREL_ROLL_LEFT: "support_barrel_roll",
    SpecialModel.BARREL_ROLL_RIGHT: "support_barrel_roll",
    SpecialModel.HALF_LOOP: "support_half_loop",
    SpecialModel.QUARTER_LOOP: "support_quarter_loop",
    SpecialModel.CORKSCREW_LEFT: "support_corkscrew",
    SpecialModel.CORKSCREW_RIGHT: "support_corkscrew",
    SpecialModel.ZERO_G_ROLL_LEFT: "support_zero_g_roll",
    SpecialModel.ZERO_G_ROLL_RIGHT: "support_zero_g_roll",
    SpecialModel.LARGE_ZERO_G_ROLL_LEFT: "support_large_zero_g_roll",
    SpecialModel.LARGE_ZERO_G_ROLL_RIGHT: "support_large_zero_g_roll",
}

# Specials whose mechanism mesh is *tiled* along the section (``track.cpp:381``):
# brake/magnetic_brake/block_brake/booster. The rest place a single *rigid* support model
# (``track.cpp:401``) — see :data:`SPECIAL_RIGHT_NO_FLIP`.
SPECIAL_TILED: frozenset[SpecialModel] = frozenset(
    {
        SpecialModel.BRAKE,
        SpecialModel.MAGNETIC_BRAKE,
        SpecialModel.BLOCK_BRAKE,
        SpecialModel.BOOSTER,
    }
)

# Rigid specials place the model with ``views[1]``; all but these *_RIGHT variants also flip
# the matrix bottom row (``track.cpp:371-379``). These keep ``views[1]`` unflipped.
SPECIAL_RIGHT_NO_FLIP: frozenset[SpecialModel] = frozenset(
    {
        SpecialModel.VERTICAL_TWIST_RIGHT,
        SpecialModel.BARREL_ROLL_RIGHT,
        SpecialModel.CORKSCREW_RIGHT,
        SpecialModel.ZERO_G_ROLL_RIGHT,
        SpecialModel.LARGE_ZERO_G_ROLL_RIGHT,
    }
)

# Supports (track.cpp). The per-tile base model key, and the support-post model keyed by
# absolute bank step 0..6 (``get_support_index`` -> ``support_model_names``). DENOM is the
# integer bank resolution (6 = the number of steps from level to a full 45° bank).
SUPPORT_BASE_KEY = "support_base"
SUPPORT_BANK_DENOM = 6
SUPPORT_BANK_KEY: tuple[str, ...] = (
    "support_flat",            # 0
    "support_bank_sixth",      # 1
    "support_bank_third",      # 2
    "support_bank_half",       # 3
    "support_bank_two_thirds",  # 4
    "support_bank_five_sixths",  # 5
    "support_bank",            # 6
)
