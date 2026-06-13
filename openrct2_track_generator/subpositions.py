"""
Vehicle subposition tables — the real OpenRCT2 encoding (port of ``subposition.cpp``).

A subposition is a point a vehicle occupies along a track piece: an integer ``(x, y, z)``
in the tile's 1/32-tile grid plus a *discretized* orientation — a yaw sprite (0-31) and
pitch/bank sprite slots. maketrack's companion ``subposition`` tool produces these by
walking the section curve, snapping each sampled frame to the nearest of OpenRCT2's
sprite-rotation table entries (:mod:`._sprite_rotations`) and emitting the integer grid
coordinate whenever a coordinate ticks over.

This module ports that pipeline: :func:`track_point_get_rotation`,
:func:`get_rotation_distance`, :func:`get_closest_rotation`, :func:`get_subposition`,
:func:`calc_differing_coords` and :func:`generate_view_subposition_data`. The rotation
candidate matrices are precomputed once from the embedded tables. ``reverse`` (descending a
piece backwards) is supported by the core; the emitted sidecar uses the forward pass.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cache
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ._sprite_rotations import BANK_NAMES, PITCH_NAMES, SPRITE_GROUPS
from .constants import CLEARANCE_HEIGHT, TILE_SIZE
from .types import Track, TrackSection

__all__ = [
    "BANK_NAMES",
    "PITCH_NAMES",
    "SPRITE_GROUP_BASE",
    "Subposition",
    "build_subposition_data",
    "generate_view_subposition_data",
    "get_closest_rotation",
    "get_subposition",
]

# Sprite-group bit flags (subposition.cpp's anonymous enum).
SPRITE_GROUP_ORTHOGONAL = 1
SPRITE_GROUP_DIAGONAL = 2
SPRITE_GROUP_TURN = 4
SPRITE_GROUP_INLINE_TWIST = 8
SPRITE_GROUP_CORKSCREW = 16
SPRITE_GROUP_ZERO_G_ROLLS_ORTHOGONAL = 32
SPRITE_GROUP_ZERO_G_ROLLS_DIAGONAL = 64
SPRITE_GROUP_ZERO_G_ROLLS_OTHER = 128
SPRITE_GROUP_DIVE_LOOP = 256
SPRITE_GROUP_BASE = SPRITE_GROUP_ORTHOGONAL | SPRITE_GROUP_DIAGONAL | SPRITE_GROUP_TURN

# Group name (in :data:`SPRITE_GROUPS`) -> its bit flag, in the order subposition.cpp lists
# them (``sprite_group_rotations``).
_GROUP_BITS: tuple[tuple[str, int], ...] = (
    ("orthogonal", SPRITE_GROUP_ORTHOGONAL),
    ("diagonal", SPRITE_GROUP_DIAGONAL),
    ("turn", SPRITE_GROUP_TURN),
    ("inline_twist", SPRITE_GROUP_INLINE_TWIST),
    ("corkscrew", SPRITE_GROUP_CORKSCREW),
    ("zero_g_orthogonal", SPRITE_GROUP_ZERO_G_ROLLS_ORTHOGONAL),
    ("zero_g_diagonal", SPRITE_GROUP_ZERO_G_ROLLS_DIAGONAL),
    ("zero_g_other", SPRITE_GROUP_ZERO_G_ROLLS_OTHER),
    ("dive_loop", SPRITE_GROUP_DIVE_LOOP),
)

# Step used to walk the curve looking for the next grid tick (subposition.cpp uses 0.01).
_STEP = 0.01


@dataclass(frozen=True, slots=True)
class Subposition:
    """One emitted subposition point: integer grid coord + discretized orientation."""

    x: int
    y: int
    z: int
    yaw_sprite: int
    pitch_sprite: int
    bank_sprite: int


def _rotate_x(t: float) -> NDArray[np.float64]:
    c, s = math.cos(t), math.sin(t)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rotate_y(t: float) -> NDArray[np.float64]:
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rotate_z(t: float) -> NDArray[np.float64]:
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


@cache
def _group_candidates(name: str) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Precompute one group's candidate rotation matrices + their sprite indices.

    Returns ``(mats, sprites)`` where ``mats`` is ``(K, 3, 3)`` of
    ``rotate_y(yaw) @ rotate_z(pitch) @ rotate_x(roll)`` and ``sprites`` is ``(K, 3)`` of
    ``(yaw_sprite, pitch_sprite, bank_sprite)``.
    """
    rows = SPRITE_GROUPS[name]
    mats = np.empty((len(rows), 3, 3), dtype=np.float64)
    sprites = np.empty((len(rows), 3), dtype=np.int64)
    for i, (ys, ps, bs, yaw, pitch, roll) in enumerate(rows):
        mats[i] = _rotate_y(yaw) @ _rotate_z(pitch) @ _rotate_x(roll)
        sprites[i] = (ys, ps, bs)
    return mats, sprites


def track_point_get_rotation(
    tangent: NDArray[np.float64], normal: NDArray[np.float64], binormal: NDArray[np.float64]
) -> NDArray[np.float64]:
    """The orientation matrix of a track frame (``subposition.cpp:track_point_get_rotation``).

    Columns are change-coordinates'd tangent / normal / negated binormal.
    """
    return np.array(
        [
            [tangent[2], normal[2], -binormal[2]],
            [tangent[1], normal[1], -binormal[1]],
            [tangent[0], normal[0], -binormal[0]],
        ]
    )


def get_closest_rotation(rotation: NDArray[np.float64], groups: int) -> tuple[int, int, int]:
    """Nearest sprite rotation (by geodesic distance) over the enabled groups.

    Vectorized form of ``subposition.cpp:get_closest_rotation``: the rotation distance is
    monotonic in the Frobenius inner product ``sum(rotation * candidate)``, so the nearest
    candidate is the one maximizing it. Returns ``(yaw_sprite, pitch_sprite, bank_sprite)``.
    """
    best_score = -math.inf
    best = (0, 0, 0)
    for name, bit in _GROUP_BITS:
        if not (groups & bit):
            continue
        mats, sprites = _group_candidates(name)
        scores = np.einsum("ij,kij->k", rotation, mats)
        k = int(np.argmax(scores))
        if scores[k] > best_score:
            best_score = float(scores[k])
            best = (int(sprites[k, 0]), int(sprites[k, 1]), int(sprites[k, 2]))
    return best


def get_subposition(position: NDArray[np.float64], view: int, diag: int) -> tuple[int, int, int]:
    """Map a curve-space position to the integer grid coord for a view (``get_subposition``)."""
    x = int(round(32.0 * position[2] / TILE_SIZE))
    y = int(round(32.0 * position[0] / TILE_SIZE))
    z = int(round((16.0 * math.sqrt(6.0)) * position[1] / TILE_SIZE))
    case = view + 4 * diag
    if case == 0:
        x, y = 32 - x, 16 - y
    elif case == 1:
        x, y = 16 - y, x
    elif case == 2:
        y = 16 + y
    elif case == 3:
        x, y = 16 + y, 32 - x
    elif case == 4:
        x, y = 16 - x, 15 - y
    elif case == 5:
        x, y = 16 - y, 16 + x
    elif case == 6:
        x, y = 16 + x, 15 + y
    elif case == 7:
        x, y = 16 + y, 16 - x
    return x, y, z


def calc_differing_coords(a: tuple[int, int, int], b: tuple[int, int, int]) -> int:
    """How many coords differ by exactly 1; ``-1`` if any differs by more (``calc_differing``)."""
    dx, dy, dz = abs(a[0] - b[0]), abs(a[1] - b[1]), abs(a[2] - b[2])
    if dx > 1 or dy > 1 or dz > 1:
        return -1
    return (dx == 1) + (dy == 1) + (dz == 1)


@dataclass(frozen=True, slots=True)
class _Point:
    position: NDArray[np.float64]
    tangent: NDArray[np.float64]
    normal: NDArray[np.float64]
    binormal: NDArray[np.float64]


def _curve_point(section: TrackSection, progress: float) -> _Point:
    tp = section.curve(np.array([progress], dtype=np.float64))
    return _Point(tp.position[0], tp.tangent[0], tp.normal[0], tp.binormal[0])


def _get_track_point(
    section: TrackSection,
    progress: float,
    reverse: int,
    reverse_transform: NDArray[np.float64],
    reverse_offset: NDArray[np.float64],
) -> _Point:
    """Evaluate the curve, applying the reverse transform when descending backwards."""
    p = _curve_point(section, progress)
    if not reverse:
        return p
    pos = reverse_transform @ (p.position - reverse_offset)
    tangent = -(reverse_transform @ p.tangent)
    normal = reverse_transform @ p.normal
    binormal = -(reverse_transform @ p.binormal)
    return _Point(pos, tangent, normal, binormal)


def generate_view_subposition_data(
    section: TrackSection, groups: int, view: int, reverse: int = 0
) -> list[Subposition]:
    """Walk ``section``'s curve for one view, emitting subpositions (``generate_view_...``)."""
    start = _curve_point(section, 0.0)
    start_angle = int(round(4.0 * math.atan2(-start.tangent[0], start.tangent[2]) / math.pi))
    if start_angle < 0:
        start_angle += 8
    end = _curve_point(section, section.length)
    finish_angle = int(round(4.0 * math.atan2(-end.tangent[0], end.tangent[2]) / math.pi))
    if finish_angle < 0:
        finish_angle += 8

    reverse_transform = np.eye(3)
    reverse_offset = np.zeros(3)
    if reverse:
        reverse_transform = _rotate_y(0.25 * math.pi * ((finish_angle & 0xFE) + 4))
        reverse_offset = end.position.copy()
        steps = round(8.0 * reverse_offset[1] / CLEARANCE_HEIGHT - (reverse - 1))
        reverse_offset[1] = steps * CLEARANCE_HEIGHT / 8.0
        rot = (finish_angle & 0xFE) + 4
        tmp = start_angle
        start_angle = (finish_angle + rot) % 8
        finish_angle = (tmp + rot) % 8

    diag = start_angle & 1
    skip_start = view == 0 or view == 3
    skip_final = False

    points: list[Subposition] = []
    progress = 0.0
    done = False
    length = section.length
    while not done:
        if progress >= length:
            progress = length
            done = True
        point = _get_track_point(section, progress, reverse, reverse_transform, reverse_offset)
        cur_sub = get_subposition(point.position, view, diag)

        emit = ((reverse or not skip_start) and (not reverse or not skip_final)) or progress != 0
        if emit:
            ys, ps, bs = get_closest_rotation(
                track_point_get_rotation(point.tangent, point.normal, point.binormal), groups
            )
            yaw_sprite = (8 * view + ys) % 32
            points.append(
                Subposition(cur_sub[0], cur_sub[1], cur_sub[2], yaw_sprite, ps, bs)
            )

        # Advance to just before the next coordinate tick (>1 jump in any axis).
        i = 0
        while True:
            nxt = _get_track_point(
                section, progress + _STEP * i, reverse, reverse_transform, reverse_offset
            )
            if calc_differing_coords(cur_sub, get_subposition(nxt.position, view, diag)) < 0:
                i -= 1
                break
            i += 1
            if not (progress + _STEP * i < length):
                break
        progress += _STEP * i

    if (not reverse and skip_final) or (reverse and skip_start):
        if points:
            points.pop()
    if reverse:
        points.reverse()
    return points


def _groups_for_section(section: TrackSection) -> int:
    """Pick the sprite groups to snap a section against (best-effort; default = base set).

    The original ``subposition`` tool selected groups per piece by hand; this mirrors the
    common choices for the inversion families and falls back to the orthogonal/diagonal/turn
    base set for ordinary track.
    """
    name = section.name
    if "large_corkscrew" in name or "corkscrew" in name:
        return SPRITE_GROUP_CORKSCREW | SPRITE_GROUP_ORTHOGONAL
    if "zero_g_roll" in name:
        return (
            SPRITE_GROUP_ZERO_G_ROLLS_ORTHOGONAL
            | SPRITE_GROUP_INLINE_TWIST
            | SPRITE_GROUP_ORTHOGONAL
        )
    if "inline_twist" in name or "barrel_roll" in name:
        return SPRITE_GROUP_INLINE_TWIST | SPRITE_GROUP_BASE
    if "dive_loop" in name:
        return SPRITE_GROUP_DIVE_LOOP | SPRITE_GROUP_CORKSCREW | SPRITE_GROUP_ORTHOGONAL
    return SPRITE_GROUP_BASE


def build_subposition_data(track: Track) -> dict[str, Any]:
    """Build the JSON-able subposition sidecar: real discretized tables per section/view."""
    sections: list[dict[str, Any]] = []
    for section in track.sections:
        groups = _groups_for_section(section)
        views: list[list[dict[str, int]]] = []
        for view in range(4):
            pts = generate_view_subposition_data(section, groups, view)
            views.append(
                [
                    {
                        "x": p.x, "y": p.y, "z": p.z,
                        "yaw_sprite": p.yaw_sprite,
                        "pitch_sprite": p.pitch_sprite,
                        "bank_sprite": p.bank_sprite,
                    }
                    for p in pts
                ]
            )
        sections.append({"section": section.name, "groups": groups, "views": views})
    return {"id": track.id, "encoding": "openrct2_sprite_indices", "sections": sections}
