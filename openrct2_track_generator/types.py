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

from .constants import TILE_SIZE, SpecialModel, TrackFlag

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
    # Lift-hill chain pattern name ("flat"/"gentle"), or None if the piece has no chain.
    chain: str | None = None
    # Special-mechanism model (brake/booster/...) rendered on top of the track, or None.
    special: SpecialModel | None = None


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

    # Separate ties (maketrack's TRACK_SEPARATE_TIE / TRACK_TIE_AT_BOUNDARY). ``separate_tie``
    # places a rigid ``tie_mesh`` at each tile centre. ``tie_at_boundary`` (which implies
    # ``separate_tie``) instead retiles the section so ties sit at tile boundaries, alternating
    # the rigid tie + a deformed ``track_tie_mesh`` over each ``tie_length`` span with the plain
    # track over the between-tie span. Meshes are indices into ``meshes`` (-1 = none).
    separate_tie: bool = False
    tie_at_boundary: bool = False
    tie_mesh_index: int = -1
    track_tie_mesh_index: int = -1
    tie_length: float = TILE_SIZE

    # Special-mechanism meshes (brake/block_brake/booster/magnetic_brake), keyed by the
    # model name in :data:`constants.SPECIAL_MODEL_KEY`. A section with a ``special`` whose
    # key is present here renders that mesh tiled on top of the track (maketrack's
    # ``TRACK_SPECIAL_MASK`` branch); absent keys fall back to plain track.
    special_models: dict[str, Mesh] = field(default_factory=dict)
    # Tile length for tiled specials (brake mechanism); block_brake always tiles by one tile.
    brake_length: float = TILE_SIZE

    # Supports (track.cpp): with ``has_supports`` a section without TRACK_NO_SUPPORTS gets a
    # per-tile base model (``support_base`` in ``special_models``) plus support posts spaced
    # ``support_spacing`` apart, banked per the entry/exit bank. ``pivot`` is the post pivot
    # height used to lower posts under pitched track. (All inert without ``has_supports``.)
    has_supports: bool = False
    support_spacing: float = TILE_SIZE
    pivot: float = 0.0

    sections: list[TrackSection] = field(default_factory=list)
    flat_shaded: bool = False

    # Lift hill: overlays the chain pattern and expands flat's 2 views to 4 (one per
    # chain direction). Set from the config's ``flags`` list ("has_lift").
    has_lift: bool = False
    # maketrack's ``lift_offset`` (default 13). Loaded for config parity but, as in maketrack
    # itself, it is currently unused by the render path.
    lift_offset: int = 13

    # Per-track vertical offset in 1/8-clearance-height units (maketrack's track z_offset);
    # the deform shifts y by ``(z_offset / 8) * CLEARANCE_HEIGHT``.
    z_offset: float = 0.0

    # ``special_end_offsets`` smoothing (maketrack's TRACK_SPECIAL_OFFSETS): when set, each
    # section's endpoints are nudged per the ``offset_table`` (a (10, 8) array: slope/bank
    # category x 4 view angles x z/y pair) so adjacent pieces line up. Inert when the flag is
    # unset or the table is all zeros. See :mod:`offsets`.
    special_end_offsets: bool = False
    offset_table: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros((10, 8), dtype=np.float64)
    )

    # Path to the masks JSON used to carve per-view sub-sprites; empty = bundled default.
    masks_path: str = ""

    # Sprite-manifest output (maketrack's sprite_directory / spritefile_in / spritefile_out).
    # ``sprite_directory`` is the directory the PNGs go in and the prefix in each manifest
    # ``path`` (default "images"). ``spritefile_in``, when set, is an existing manifest array
    # the rendered sprites are appended to (the way maketrack places sprites at fixed global
    # image indices); ``spritefile_out`` is where the merged manifest is written (default
    # ``<id>.sprites.json``). Both resolve relative to the output directory if not absolute.
    sprite_directory: str = ""
    spritefile_in: str = ""
    spritefile_out: str = ""

    # Per-track sprite-filename suffix (maketrack appends the track's ``name`` as ``_<name>``).
    # Empty for a single-track config (filenames stay ``<section>_<view>_<sub>.png``); set per
    # entry in a multi-track file so colour-scheme/mesh variants share one manifest without
    # filename collisions. Includes its own leading underscore (e.g. ``"_alt"``).
    suffix: str = ""

    preview: IndexedImage | None = None

    # Object-level flags (has_lift / has_supports / ...) — parsed-but-unused seam.
    flags: int = 0
