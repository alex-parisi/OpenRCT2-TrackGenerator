"""
Track-specific dataclasses.

A ``CurveFn`` evaluates a section's geometry at an array of arc-distances and
returns a vectorized ``TrackPointArray`` frame (position + tangent/normal/binormal
at each sample). ``TrackSection`` binds a name to its curve, arc length, and render
flags; ``Track`` is the whole authored object the loader produces and the exporter
consumes. Shared rendering primitives (``Mesh``, ``IndexedImage``) live in the
renderer package.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from .constants import TILE_SIZE, TrackFlag

if TYPE_CHECKING:
    from openrct2_x7_renderer.mesh import Mesh
    from openrct2_x7_renderer.types import IndexedImage

__all__ = ["CurveFn", "Track", "TrackPointArray", "TrackSection"]


@dataclass(frozen=True, slots=True)
class TrackPointArray:
    """A curve's moving frame evaluated at ``N`` arc-distances.

    Each field is an ``(N, 3)`` float64 array. ``tangent`` points along the track,
    ``normal`` is "up" out of the rails, and ``binormal`` points across the track;
    together they form the per-sample basis the deformation places mesh vertices in.
    """

    position: NDArray[np.float64]
    tangent: NDArray[np.float64]
    normal: NDArray[np.float64]
    binormal: NDArray[np.float64]


# A vectorized curve: arc-distances ``(N,)`` -> frame at each distance.
CurveFn = Callable[[NDArray[np.float64]], TrackPointArray]


@dataclass(frozen=True, slots=True)
class TrackSection:
    """One track piece type: its name, geometry curve, arc length, and flags.

    ``z_offset`` raises/lowers the piece on the tile (height-step adjustment from
    the legacy config's per-section ``offsets`` table); it defaults to zero for the
    vertical slice.
    """

    name: str
    curve: CurveFn
    length: float
    flags: TrackFlag = TrackFlag.NONE
    z_offset: float = 0.0


@dataclass
class Track:
    """An authored track object: metadata, input meshes, and the sections to render."""

    id: str = ""
    original_id: str = ""
    name: str = ""
    description: str = ""
    authors: list[str] = field(default_factory=list)
    version: str = "1.0"
    ride_type: str = ""

    # Model units per tile — the scale mapping OBJ units onto one OpenRCT2 tile.
    # Drives both the render projection and the curve/deform arc-length units.
    units_per_tile: float = TILE_SIZE

    # The deformable track-tile mesh lives in ``meshes[track_mesh_index]``. Kept as
    # a list (not a single mesh) so support/tie/mask/banked-variant meshes slot in
    # as additional indices later without reshaping Track.
    meshes: list[Mesh] = field(default_factory=list)
    track_mesh_index: int = 0
    # Separate solid occlusion-volume mesh for the track-mask silhouette (split/transfer
    # front/behind sub-sprites). -1 = none, so occlusion ops are skipped.
    mask_mesh_index: int = -1

    sections: list[TrackSection] = field(default_factory=list)
    flat_shaded: bool = False

    # Per-track vertical offset in 1/8-clearance-height units (maketrack's track z_offset);
    # the deform shifts y by ``(z_offset / 8) * CLEARANCE_HEIGHT``.
    z_offset: float = 0.0

    # Path to the masks JSON used to carve per-view sub-sprites; empty = bundled default.
    masks_path: str = ""

    preview: IndexedImage | None = None

    # Object-level flags (has_lift / has_supports / ...) — parsed-but-unused seam.
    flags: int = 0
