"""
Support-post placement, ported from ``track.cpp``'s support loop (lines 405-437).

A supported section drops vertical support posts at ``support_spacing`` intervals. Each
post is a *rigid* model (not deformed): it stands upright and yaws to follow the track
heading (``only_yaw``), with a bank-dependent model chosen by interpolating the entry and
exit bank across the section (``get_support_index``). Banked-right posts reuse the
banked-left model rotated 180° about Y (``views[2]``). The post is lowered under pitched
track by the pivot-height correction. The per-tile base model is handled separately in
``section_renderer`` (it is a curve-deformed mesh, not a rigid post).
"""

import numpy as np
from numpy.typing import NDArray

from .constants import (
    SUPPORT_BANK_DENOM,
    SUPPORT_BANK_KEY,
    TrackFlag,
)
from .deform import get_track_point_array
from .types import TrackSection

__all__ = ["SupportPost", "support_posts"]

# views[2] (libIsoRender): 180° about Y. Banked-right posts reuse the left model rotated.
_VIEWS2 = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float64)
_UP = np.array([0.0, 1.0, 0.0])


def _cc(v: NDArray[np.float64]) -> NDArray[np.float64]:
    """change_coordinates: curve-space (x,y,z) -> render-space (z,y,x)."""
    return v[::-1].copy()


def _normalize(v: NDArray[np.float64]) -> NDArray[np.float64]:
    n = float(np.linalg.norm(v))
    return v / n if n > 0.0 else v


class SupportPost:
    """One placed support post: its model key + rigid transform (matrix, translation)."""

    __slots__ = ("model_key", "matrix", "translation")

    def __init__(
        self, model_key: str, matrix: NDArray[np.float64], translation: NDArray[np.float64]
    ):
        self.model_key = model_key
        self.matrix = matrix
        self.translation = translation


def support_posts(
    section: TrackSection, z_offset: float, support_spacing: float, pivot: float
) -> list[SupportPost]:
    """Compute the support posts for ``section`` (``track.cpp:405-437``).

    ``num = round(length / support_spacing)`` posts (plus the closing one) are spaced
    evenly; the integer bank step at each (``-6..6``) interpolates entry->exit bank and
    selects the post model. Returns one :class:`SupportPost` per placement.
    """
    length = section.length
    num = max(1, int(np.floor(0.5 + length / support_spacing)))
    step = length / num
    flags = section.flags

    def _bank(left: TrackFlag, right: TrackFlag) -> int:
        if flags & left:
            return SUPPORT_BANK_DENOM
        if flags & right:
            return -SUPPORT_BANK_DENOM
        return 0

    entry = _bank(TrackFlag.ENTRY_BANK_LEFT, TrackFlag.ENTRY_BANK_RIGHT)
    exit_ = _bank(TrackFlag.EXIT_BANK_LEFT, TrackFlag.EXIT_BANK_RIGHT)

    distances = np.array([i * step for i in range(num + 1)], dtype=np.float64)
    zero = np.zeros(3)
    tp = get_track_point_array(section.curve, flags, z_offset, length, zero, zero, distances)

    posts: list[SupportPost] = []
    for i in range(num + 1):
        u = (i * SUPPORT_BANK_DENOM) // num
        bank = (entry * (SUPPORT_BANK_DENOM - u) + exit_ * u) // SUPPORT_BANK_DENOM

        tangent = tp.tangent[i]  # curve-space tangent (for the pivot pitch correction)
        # only_yaw: upright normal, horizontal binormal from the render-space tangent.
        ct = _cc(tangent)
        binormal = _normalize(np.cross(_UP, ct))
        tangent_yaw = _normalize(np.cross(_UP, binormal))  # = cross(normal, binormal)
        # matrix columns = (binormal, normal=up, tangent_yaw).
        matrix = np.column_stack((binormal, _UP, tangent_yaw))
        if bank >= 0:
            matrix = _VIEWS2 @ matrix

        translation = _cc(tp.position[i])
        horiz = float(np.hypot(tangent[0], tangent[2]))
        if horiz > 0.0:
            translation = translation.copy()
            translation[1] -= pivot / horiz - pivot

        posts.append(SupportPost(SUPPORT_BANK_KEY[abs(bank)], matrix, translation))
    return posts
