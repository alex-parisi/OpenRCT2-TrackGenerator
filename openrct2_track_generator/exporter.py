"""
Render track sprites and emit an OpenRCT2 sprite manifest (+ subposition sidecar).

OpenRCT2 has no "track" object type: track sprites are baked graphics referenced by
hardcoded global image indices, compiled from a sprite manifest via
``openrct2 sprite build <out.dat> <manifest.json>``. So — like the original RCTGen
``maketrack`` — this generator writes palette-indexed PNGs plus a manifest array of
``{path, x, y, palette: "keep"}`` entries (the exact schema
``createImageImportMetaFromJson`` parses), rather than a ``.parkobj``. Sprites are
addressed by their order in the array, so the section/view emission order is the
contract with whatever paint code consumes them.

The deformation runs once per section; rendering and the manifest schema are otherwise
the shared X7 path.
"""

import json
import logging
from pathlib import Path
from typing import Any

from openrct2_object_common.parkobj import combine_indexed_images
from openrct2_x7_renderer.image import write_png
from openrct2_x7_renderer.ray_trace import VIEWS, Context

from .section_renderer import render_section
from .subpositions import build_subposition_data
from .types import Track

log = logging.getLogger(__name__)

# Sprites rendered per section (the four cardinal park-view rotations).
SPRITE_VIEWS = len(VIEWS)

# Subdirectory (relative to the manifest) that the PNGs are written into; the
# manifest's "path" values are resolved against the manifest's own directory by
# `openrct2 sprite build`.
IMAGES_DIRNAME = "images"


def expected_sprite_count(track: Track) -> int:
    """How many sprites ``export_track`` will write for ``track``."""
    return len(track.sections) * SPRITE_VIEWS


def _render_sprites(track: Track, context: Context, images_dir: Path) -> list[dict[str, Any]]:
    """Render every section's views to PNGs and return the manifest entries.

    Emission order is section-major, view-minor (``VIEWS[0..3]``); each sprite's
    manifest index is its position in the returned list.
    """
    images_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    for section in track.sections:
        log.info("Rendering track section %s", section.name)
        views = render_section(track, section, context)
        for view_index, image in enumerate(views):
            filename = f"{section.name}_{view_index + 1}.png"
            write_png(image, images_dir / filename)
            manifest.append(
                {
                    # write_png drops draw offsets, so carry them in the manifest,
                    # which is where `sprite build` reads them anyway.
                    "path": f"{images_dir.name}/{filename}",
                    "x": int(image.x_offset),
                    "y": int(image.y_offset),
                    "palette": "keep",
                }
            )
    return manifest


def _write_subpositions(track: Track, output_directory: Path) -> Path:
    """Write the ``<id>.subpositions.json`` sidecar and return its path."""
    path = output_directory / f"{track.id}.subpositions.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build_subposition_data(track), indent=4))
    log.info("wrote %s (%d sections)", path, len(track.sections))
    return path


def export_track(
    track: Track, context: Context, output_directory: Path | str, skip_render: bool = False
) -> None:
    """Render the track's sprites + manifest into ``output_directory``.

    Writes ``<output>/images/<section>_<view>.png`` (palette-indexed), a
    ``<output>/<id>.sprites.json`` manifest for ``openrct2 sprite build``, and the
    ``<id>.subpositions.json`` sidecar. ``skip_render`` is a no-op for tracks: there
    is no separate packaging step to repeat, and the manifest's draw offsets can only
    be recovered by rendering, so a prior run's outputs are simply left in place.
    """
    output_directory = Path(output_directory)
    if skip_render:
        log.info("skip_render: leaving existing sprites and manifest in place")
        return

    manifest = _render_sprites(track, context, output_directory / IMAGES_DIRNAME)

    manifest_path = output_directory / f"{track.id}.sprites.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=4))
    log.info("wrote %s (%d sprites)", manifest_path, len(manifest))

    _write_subpositions(track, output_directory)


def export_track_test(track: Track, context: Context, test_dir: Path | str = "test") -> None:
    """Four-direction render for fast iteration.

    Renders each section at the four park-view rotations and tiles them into one 2x2
    preview per section, so the test sprite shows every direction the piece is drawn
    at. This is the fastest way to validate the deformation and the render-space axis
    mapping before a full export.
    """
    test_dir = Path(test_dir)
    test_dir.mkdir(parents=True, exist_ok=True)
    for section in track.sections:
        log.info("Rendering track section %s", section.name)
        views = render_section(track, section, context)
        write_png(combine_indexed_images(views, columns=2), test_dir / f"{section.name}.png")
