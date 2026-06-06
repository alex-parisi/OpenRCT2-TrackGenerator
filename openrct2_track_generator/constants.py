"""
Track geometry constants and flags, ported from RCTGen's ``track.h``.

``TILE_SIZE`` is the side length (in OBJ/model units) of one OpenRCT2 tile; the
deformation, curve library, and render scale all share it. ``CLEARANCE_HEIGHT`` is
one OpenRCT2 height step expressed in the same units, and ``BANK_ANGLE`` is the
default 45° bank. ``TrackFlag`` mirrors the rendering-flag bitfield from ``track.h``;
only a subset is exercised by the vertical slice, but the full set is carried so the
section registry can grow without re-numbering.
"""

from enum import IntFlag

__all__ = ["BANK_ANGLE", "CLEARANCE_HEIGHT", "TILE_SIZE", "TrackFlag"]

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
