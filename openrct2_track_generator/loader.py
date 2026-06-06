"""
Load a track config (JSON or YAML) into a Track dataclass.
"""

from pathlib import Path
from typing import Any

from openrct2_object_common.config import (
    LoadError,
    load_meshes,
    load_preview,
    optional_bool,
    optional_int,
    optional_number,
    optional_string,
    optional_string_list,
    parse_config,
    require_string,
)
from openrct2_x7_renderer.mesh import Mesh
from openrct2_x7_renderer.types import IndexedImage

from .constants import TILE_SIZE
from .sections import resolve_section
from .types import Track


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


def build_track(
    config: dict[str, Any], meshes: list[Mesh], preview: IndexedImage | None = None
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

    track.meshes = list(meshes)
    track.track_mesh_index = optional_int(root, "track_mesh_index", 0)
    if meshes and not (0 <= track.track_mesh_index < len(meshes)):
        raise LoadError(
            f'Property "track_mesh_index" ({track.track_mesh_index}) is out of range'
        )

    track.preview = preview if preview is not None else IndexedImage.blank(1, 1)

    track.sections = _build_sections(root)
    return track


def load_track(json_path: Path | str) -> Track:
    """Parse a config file, load its meshes + preview from disk, build a Track."""
    root = parse_config(json_path)
    return build_track(root, load_meshes(root), load_preview(root))
