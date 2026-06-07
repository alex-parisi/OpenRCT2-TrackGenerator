"""
Load a track config (JSON or YAML) into a Track dataclass.
"""

import math
from pathlib import Path
from typing import Any

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
from .sections import resolve_section
from .types import Track

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


def _build_sections(root: dict[str, Any]) -> list[Any]:
    sections_raw = root.get("sections")
    if not isinstance(sections_raw, list):
        raise LoadError('Property "sections" does not exist or is not an array')
    if not sections_raw:
        raise LoadError('Property "sections" must list at least one section')
    out = []
    for name in sections_raw:
        if not isinstance(name, str):
            raise LoadError('Array "sections" contains a non-string value')
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

    track.sections = _build_sections(root)
    return track


def load_track(json_path: Path | str) -> Track:
    """Parse a config file, load its meshes + preview from disk, build a Track."""
    root = parse_config(json_path)
    return build_track(
        root, load_track_meshes(root), load_preview(root), load_special_models(root)
    )
