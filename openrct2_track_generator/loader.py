"""
Load a track config (JSON or YAML) into a Track dataclass.
"""

import math
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from openrct2_object_common.config import (
    LoadError,
    load_preview,
    optional_bool,
    optional_int,
    optional_number,
    optional_string,
    optional_string_list,
    parse_config,
    require_string,
)
from openrct2_x7_renderer.geometry import rotate_y
from openrct2_x7_renderer.mesh import Mesh, load_mesh
from openrct2_x7_renderer.types import IndexedImage

from .constants import TILE_SIZE
from .groups import expand_groups, is_group
from .offsets import NUM_OFFSET_ROWS, OFFSET_ROW_NAMES
from .sections import resolve_section
from .types import Track, TrackSection

# maketrack's load_model applies rotate_y(-90deg) to every track mesh, so meshes are
# authored with +X along the track (and Z across, centred). Our deform expects +Z
# along-track, which this rotation produces: (x, y, z) -> (-z, y, x).
_MESH_LOAD_TRANSFORM = rotate_y(-0.5 * math.pi)


def load_track_meshes(root: dict[str, Any], base_dir: Path | None = None) -> list[Mesh]:
    """Load the config's ``meshes`` with maketrack's along-track (+X) load rotation."""
    mesh_paths = root.get("meshes")
    if not isinstance(mesh_paths, list):
        raise LoadError('Property "meshes" does not exist or is not an array')
    meshes: list[Mesh] = []
    for path in mesh_paths:
        if not isinstance(path, str):
            raise LoadError("Mesh path is not a string")
        resolved = Path(path) if base_dir is None or Path(path).is_absolute() else base_dir / path
        meshes.append(load_mesh(resolved, transform=_MESH_LOAD_TRANSFORM))
    return meshes


def _build_sections(root: dict[str, Any]) -> list[TrackSection]:
    """Resolve the config's ``sections`` list into ordered :class:`TrackSection` objects.

    Each entry is a maketrack *group* name (expanded via :func:`groups.expand_groups` into
    its constituent sections, in maketrack's canonical emission order) or an individual
    section name. Groups are expanded first; explicit section names follow, in listed order.
    Duplicates (a section reached via both a group and an explicit name) are dropped,
    keeping the first occurrence.
    """
    sections_raw = root.get("sections")
    if not isinstance(sections_raw, list):
        raise LoadError('Property "sections" does not exist or is not an array')
    if not sections_raw:
        raise LoadError('Property "sections" must list at least one section')
    for name in sections_raw:
        if not isinstance(name, str):
            raise LoadError('Array "sections" contains a non-string value')

    group_names = [name for name in sections_raw if is_group(name)]
    section_names = [name for name in sections_raw if not is_group(name)]

    ordered: list[str] = expand_groups(group_names)
    seen = set(ordered)
    for name in section_names:
        if name not in seen:
            seen.add(name)
            ordered.append(name)

    out: list[TrackSection] = []
    for name in ordered:
        try:
            out.append(resolve_section(name))
        except KeyError as e:
            # KeyError stringifies with surrounding quotes; unwrap to the message.
            raise LoadError(e.args[0]) from None
    return out


def load_special_models(
    root: dict[str, Any], base_dir: Path | None = None
) -> dict[str, Mesh]:
    """Load the optional ``special_models`` map (name -> mesh path) with the load rotation.

    Mirrors maketrack's named brake/booster model loading; keys are the model names in
    :data:`constants.SPECIAL_MODEL_KEY` (``brake`` / ``block_brake`` / ``booster`` /
    ``magnetic_brake``). Absent = no special meshes (specials fall back to plain track).
    """
    raw = root.get("special_models")
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise LoadError('Property "special_models" must be an object of name -> path')
    out: dict[str, Mesh] = {}
    for name, path in raw.items():
        if not isinstance(path, str):
            raise LoadError(f'special_models["{name}"] must be a path string')
        resolved = Path(path) if base_dir is None or Path(path).is_absolute() else base_dir / path
        out[name] = load_mesh(resolved, transform=_MESH_LOAD_TRANSFORM)
    return out


def load_offset_table(root: dict[str, Any]) -> NDArray[np.float64]:
    """Parse the optional ``offsets`` object into a ``(10, 8)`` table (zeros if absent).

    Mirrors maketrack's ``load_offsets`` (``main.cpp``): the object's keys are the slope/bank
    category names in :data:`offsets.OFFSET_ROW_NAMES`, each an array of 8 numbers
    (``[z0,y0, z1,y1, z2,y2, z3,y3]``). Missing rows stay zero.
    """
    table = np.zeros((NUM_OFFSET_ROWS, 8), dtype=np.float64)
    raw = root.get("offsets")
    if raw is None:
        return table
    if not isinstance(raw, dict):
        raise LoadError('Property "offsets" is not an object')
    for i, name in enumerate(OFFSET_ROW_NAMES):
        row = raw.get(name)
        if row is None:
            continue
        if not isinstance(row, list) or len(row) != 8:
            raise LoadError(f'Property "offsets.{name}" is not an array of length 8')
        for j, value in enumerate(row):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise LoadError(f'Array "offsets.{name}" contains a non-numeric value')
            table[i, j] = float(value)
    return table


def build_track(
    config: dict[str, Any],
    meshes: list[Mesh],
    preview: IndexedImage | None = None,
    special_models: dict[str, Mesh] | None = None,
) -> Track:
    """Build a Track from an already-parsed config dict + in-memory meshes."""
    root = config

    track = Track()
    track.id = require_string(root, "id")
    track.original_id = optional_string(root, "original_id")
    track.name = require_string(root, "name")
    track.description = optional_string(root, "description")
    track.authors = optional_string_list(root, "authors")
    v_str = optional_string(root, "version")
    if v_str:
        track.version = v_str

    track.ride_type = require_string(root, "ride_type")

    track.units_per_tile = optional_number(root, "units_per_tile", TILE_SIZE)
    if track.units_per_tile <= 0.0:
        raise LoadError('Property "units_per_tile" must be greater than 0')

    track.flat_shaded = optional_bool(root, "flat_shaded", False)
    track.z_offset = optional_number(root, "z_offset", 0.0)
    track.has_lift = "has_lift" in optional_string_list(root, "flags")
    # Parsed for parity with maketrack; unused by the render path (as in maketrack itself).
    track.lift_offset = optional_int(root, "lift_offset", 13)

    # brake_length is a tile fraction scaled to model units (default one tile, matching
    # maketrack's TILE_SIZE scale); drives the tiling of brake/booster special meshes.
    track.brake_length = optional_number(root, "brake_length", 1.0) * TILE_SIZE
    if special_models is not None:
        track.special_models = special_models

    # Supports: a "has_supports" flag enables the per-tile base + posts; support_spacing/pivot
    # are tile fractions scaled to model units (maketrack's TILE_SIZE scale).
    track.has_supports = "has_supports" in optional_string_list(root, "flags")
    track.support_spacing = optional_number(root, "support_spacing", 1.0) * TILE_SIZE
    track.pivot = optional_number(root, "pivot", 0.0) * TILE_SIZE

    # special_end_offsets smoothing (maketrack's TRACK_SPECIAL_OFFSETS): a flag + the offsets
    # table. Loaded unconditionally; inert unless the flag is set (see section_renderer).
    track.special_end_offsets = "special_end_offsets" in optional_string_list(root, "flags")
    track.offset_table = load_offset_table(root)

    # Separate ties (maketrack TRACK_SEPARATE_TIE / TRACK_TIE_AT_BOUNDARY). tie_at_boundary
    # implies separate_tie. tie_length is a tile fraction scaled to model units.
    flags_list = optional_string_list(root, "flags")
    track.tie_at_boundary = "tie_at_boundary" in flags_list
    track.separate_tie = "separate_tie" in flags_list or track.tie_at_boundary
    track.tie_length = optional_number(root, "tie_length", 1.0) * TILE_SIZE
    track.tie_mesh_index = optional_int(root, "tie_mesh_index", -1)
    track.track_tie_mesh_index = optional_int(root, "track_tie_mesh_index", -1)
    for label, idx in (("tie_mesh_index", track.tie_mesh_index),
                       ("track_tie_mesh_index", track.track_tie_mesh_index)):
        if idx >= 0 and not (meshes and idx < len(meshes)):
            raise LoadError(f'Property "{label}" ({idx}) is out of range')

    track.meshes = list(meshes)
    track.track_mesh_index = optional_int(root, "track_mesh_index", 0)
    if meshes and not (0 <= track.track_mesh_index < len(meshes)):
        raise LoadError(
            f'Property "track_mesh_index" ({track.track_mesh_index}) is out of range'
        )

    # Optional occlusion mask mesh (enables split/transfer front/behind sub-sprites).
    track.mask_mesh_index = optional_int(root, "mask_mesh_index", -1)
    if track.mask_mesh_index >= 0 and not (meshes and track.mask_mesh_index < len(meshes)):
        raise LoadError(
            f'Property "mask_mesh_index" ({track.mask_mesh_index}) is out of range'
        )

    track.preview = preview if preview is not None else IndexedImage.blank(1, 1)

    # Optional masks JSON override (carves per-view sub-sprites); empty = bundled default.
    track.masks_path = optional_string(root, "masks")

    # Sprite-manifest output + merge (maketrack's sprite_directory / spritefile_in/out).
    track.sprite_directory = optional_string(root, "sprite_directory")
    track.spritefile_in = optional_string(root, "spritefile_in")
    track.spritefile_out = optional_string(root, "spritefile_out")

    # Per-track filename suffix (maketrack's per-track ``name`` -> ``_<name>``). Empty unless a
    # ``suffix`` is given (multi-track variants); the leading underscore is added here.
    suffix = optional_string(root, "suffix")
    track.suffix = f"_{suffix}" if suffix else ""

    track.sections = _build_sections(root)
    return track


def _build_one(cfg: dict[str, Any]) -> Track:
    """Load a single config dict's meshes/preview/special-models and build its Track."""
    return build_track(
        cfg, load_track_meshes(cfg), load_preview(cfg), load_special_models(cfg)
    )


def build_tracks(root: dict[str, Any]) -> list[Track]:
    """Build the track(s) a config describes (maketrack's ``tracks`` array with inheritance).

    A plain config (no ``tracks`` key) yields a single Track. A ``tracks`` array yields one
    Track per entry, where each entry *inherits* the running config (top-level keys, then each
    prior entry) and overrides only the keys it sets — mirroring maketrack's ``preloaded``
    per-track loading. Shared keys (``sprite_directory`` / ``spritefile_in``/``out`` / ``id``)
    live at the top level; per-entry ``suffix`` keeps variant sprite filenames distinct.
    """
    entries = root.get("tracks")
    if entries is None:
        return [_build_one(root)]
    if not isinstance(entries, list) or not entries:
        raise LoadError('Property "tracks" must be a non-empty array')
    running = {k: v for k, v in root.items() if k != "tracks"}
    tracks: list[Track] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise LoadError('Array "tracks" contains a non-object value')
        running = {**running, **entry}
        tracks.append(_build_one(running))
    return tracks


def load_track(json_path: Path | str) -> Track:
    """Parse a config file, load its meshes + preview from disk, build a Track."""
    return _build_one(parse_config(json_path))
