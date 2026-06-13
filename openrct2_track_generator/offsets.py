"""
Per-section-end offset table — maketrack's ``special_end_offsets`` smoothing.

Some track families (notably LIM/launched pieces) need each section's endpoints nudged so
adjacent pieces line up. maketrack drives this from an ``offsets`` object: 10 rows (one per
slope/bank *category*) × 8 columns (4 view angles × a z/y pair). For a section, each
endpoint is **classified** into a category by its tangent direction and bank
(:func:`offset_table_index`), the category's offset for the current view is looked up
(:func:`get_offset`), and the two endpoint offsets are blended into the deformation with a
Hermite weight (already implemented in :func:`deform.get_track_point_array`).

Faithful port of ``track.cpp``'s ``compare_vec`` / ``offset_table_index_with_rot`` /
``offset_table_index`` / ``get_offset`` / ``set_offset``. Offsets are returned in
*curve space* ``(x, y, z)``, matching what the deformation adds before ``change_coordinates``.

Active only when the track sets the ``special_end_offsets`` flag; otherwise the offsets are
zero and the deformation is unchanged.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from openrct2_x7_renderer.geometry import rotate_y
from openrct2_x7_renderer.ray_trace import VIEWS

from .constants import CLEARANCE_HEIGHT, TILE_SIZE
from .types import TrackSection

__all__ = [
    "NUM_OFFSET_ROWS",
    "OFFSET_ROW_NAMES",
    "get_offset",
    "offset_table_index",
    "set_offset",
]

# Offset categories (``track.h``'s ``enum offsets``), in the same order as the config rows
# (``main.cpp:load_offsets``) so a row loads into the index its enum value names.
OFFSET_ROW_NAMES: tuple[str, ...] = (
    "flat",            # 0  OFFSET_FLAT
    "gentle",          # 1  OFFSET_GENTLE
    "steep",           # 2  OFFSET_STEEP
    "flat_banked",     # 3  OFFSET_BANK
    "gentle_banked",   # 4  OFFSET_GENTLE_BANK
    "inverted",        # 5  OFFSET_INVERTED
    "diagonal",        # 6  OFFSET_DIAGONAL
    "diagonal_banked",  # 7  OFFSET_DIAGONAL_BANK
    "diagonal_gentle",  # 8  OFFSET_DIAGONAL_GENTLE
    "diagonal_steep",  # 9  OFFSET_DIAGONAL_STEEP
)
NUM_OFFSET_ROWS = len(OFFSET_ROW_NAMES)

_OFFSET_FLAT, _OFFSET_GENTLE, _OFFSET_STEEP, _OFFSET_BANK, _OFFSET_GENTLE_BANK = 0, 1, 2, 3, 4
_OFFSET_INVERTED, _OFFSET_DIAGONAL, _OFFSET_DIAGONAL_BANK = 5, 6, 7
_OFFSET_DIAGONAL_GENTLE, _OFFSET_DIAGONAL_STEEP = 8, 9

# Sentinel "no category matched" (``track.cpp`` uses 0xFF).
_NO_MATCH = 0xFF

_RIGHT_BIT = 0x10


def _compare_vec(tangent: NDArray[np.float64], reference: NDArray[np.float64], rot: int) -> bool:
    """``track.cpp:compare_vec``: is ``tangent`` ~ the view-rotated, normalized ``reference``?"""
    rotated = VIEWS[rot] @ reference
    norm = float(np.linalg.norm(rotated))
    if norm > 0.0:
        rotated = rotated / norm
    return bool(np.linalg.norm(tangent - rotated) < 0.15)


def _index_with_rot(
    tangent: NDArray[np.float64],
    normal: NDArray[np.float64],
    binormal: NDArray[np.float64],
    rot: int,
) -> int:
    """``track.cpp:offset_table_index_with_rot``: classify an endpoint frame at one rotation."""
    banked = abs(abs(math.asin(math.sqrt(normal[0] ** 2 + normal[2] ** 2))) - 0.25 * math.pi) < 0.1
    right = _RIGHT_BIT if (banked and binormal[1] < 0) else 0

    ch = CLEARANCE_HEIGHT
    if _compare_vec(tangent, np.array([0.0, 0.0, TILE_SIZE]), rot):
        if normal[1] < -0.9:
            return _OFFSET_INVERTED
        if banked:
            return right | _OFFSET_BANK
        return _OFFSET_FLAT
    if _compare_vec(tangent, np.array([0.0, 2.0 * ch, TILE_SIZE]), rot):
        if banked:
            return right | _OFFSET_GENTLE_BANK
        return _OFFSET_GENTLE
    if _compare_vec(tangent, np.array([0.0, 8.0 * ch, TILE_SIZE]), rot):
        return _OFFSET_STEEP
    if _compare_vec(tangent, np.array([-TILE_SIZE, 0.0, TILE_SIZE]), rot):
        if banked:
            return right | _OFFSET_DIAGONAL_BANK
        return _OFFSET_DIAGONAL
    if _compare_vec(tangent, np.array([-TILE_SIZE, 2.0 * ch, TILE_SIZE]), rot) and not banked:
        return _OFFSET_DIAGONAL_GENTLE
    if _compare_vec(tangent, np.array([-TILE_SIZE, 8.0 * ch, TILE_SIZE]), rot):
        return _OFFSET_DIAGONAL_STEEP
    return _NO_MATCH


def offset_table_index(
    tangent: NDArray[np.float64],
    normal: NDArray[np.float64],
    binormal: NDArray[np.float64],
) -> int:
    """``track.cpp:offset_table_index``: classify an endpoint, trying straight/left/right.

    Returns a packed code: low nibble = category, bit ``0x10`` = right-banked,
    bits ``>>5`` = end rotation (0/1/3 -> 0/3/1 via the ``0x60`` / ``0x20`` tags). ``0xFF``
    means unclassifiable (no offset applied).
    """
    index = _index_with_rot(tangent, normal, binormal, 0)
    if index != _NO_MATCH:
        return index
    index = _index_with_rot(tangent, normal, binormal, 1)
    if index != _NO_MATCH:
        return 0x60 | index
    index = _index_with_rot(tangent, normal, binormal, 3)
    if index != _NO_MATCH:
        return 0x20 | index
    return _NO_MATCH


def get_offset(
    code: int, view_angle: int, offset_table: NDArray[np.float64]
) -> NDArray[np.float64]:
    """``track.cpp:get_offset``: the curve-space offset for a classified endpoint + view.

    ``offset_table`` is ``(10, 8)``: row = category, columns = ``[z0,y0, z1,y1, z2,y2, z3,y3]``
    for the four (rotation-corrected) view angles, in the config's scaled units.
    """
    offset = np.zeros(3, dtype=np.float64)
    if code == _NO_MATCH:
        return offset

    index = code & 0xF
    end_angle = code >> 5
    right = (code & _RIGHT_BIT) >> 4
    rotated_view_angle = (view_angle + end_angle + 2 * right) % 4

    offset[2] = offset_table[index, 2 * rotated_view_angle] * TILE_SIZE / 32.0
    offset[1] = offset_table[index, 2 * rotated_view_angle + 1] * CLEARANCE_HEIGHT / 8.0

    if right:
        offset[2] *= -1.0

    if _OFFSET_DIAGONAL <= index <= _OFFSET_DIAGONAL_GENTLE:
        offset[2] *= math.sqrt(0.5)
        offset[0] = offset[2]

    if end_angle != 0:
        offset = rotate_y(-0.5 * math.pi * end_angle) @ offset
    return offset


def _endpoint_frame(
    section: TrackSection, distance: float
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Evaluate ``section``'s raw curve at ``distance`` -> (tangent, normal, binormal)."""
    tp = section.curve(np.array([distance], dtype=np.float64))
    return tp.tangent[0], tp.normal[0], tp.binormal[0]


def set_offset(
    view_angle: int, section: TrackSection, offset_table: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """``track.cpp:set_offset``: the (start, end) curve-space offsets for one view angle."""
    st, sn, sb = _endpoint_frame(section, 0.0)
    et, en, eb = _endpoint_frame(section, section.length)
    start = get_offset(offset_table_index(st, sn, sb), view_angle, offset_table)
    end = get_offset(offset_table_index(et, en, eb), view_angle, offset_table)
    return start, end
