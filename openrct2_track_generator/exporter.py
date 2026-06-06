"""
Render track sprites and emit an OpenRCT2 sprite manifest (+ subposition sidecar).

OpenRCT2 has no "track" object type: track sprites are baked graphics referenced by
hardcoded global image indices, compiled from a sprite manifest via
``openrct2 sprite build <out.dat> <manifest.json>``. So — like the original RCTGen
``maketrack`` — this generator writes palette-indexed PNGs plus a manifest array of
``{path, x, y, palette: "keep"}`` entries, rather than a ``.parkobj``.

Each section is deformed once, rendered per view, and carved into per-tile sub-sprites
by that view's mask (``section_renderer`` / ``masks``). Sprites are emitted in
section-major, view, then sub-sprite order — the manifest index order paint code
consumes them by.
"""

import json
import logging
from pathlib import Path
from typing import Any

from openrct2_object_common.parkobj import combine_indexed_images
from openrct2_x7_renderer.image import write_png
from openrct2_x7_renderer.ray_trace import Context

from .masks import ViewMask, load_section_masks
from .section_renderer import angle_plan, render_section
from .subpositions import build_subposition_data
from .types import Track, TrackSection

log = logging.getLogger(__name__)

# Subdirectory (relative to the manifest) the PNGs are written into; the manifest's
# "path" values are resolved against the manifest's own directory by `sprite build`.
IMAGES_DIRNAME = "images"


def _section_view_masks(track: Track, section: TrackSection) -> list[ViewMask]:
    # Occlusion ops (split/transfer front/behind) only apply with a mask mesh available.
    return load_section_masks(
        section.name, track.masks_path or None, occlusion=track.mask_mesh_index >= 0
    )


def _sprite_names(track: Track, section: TrackSection, view_masks: list[ViewMask]) -> list[str]:
    """Sprite filenames for a section, in angle-major then sub-sprite order."""
    return [
        f"{section.name}_{angle}_{sub}"
        for angle, view_mask, _chain in angle_plan(track, section, view_masks)
        for sub in range(len(view_mask.masks))
    ]


def expected_sprite_count(track: Track) -> int:
    """How many sprites ``export_track`` will write (sum over the per-angle plan)."""
    return sum(
        len(view_mask.masks)
        for section in track.sections
        for _angle, view_mask, _chain in angle_plan(
            track, section, _section_view_masks(track, section)
        )
    )


def _render_sprites(track: Track, context: Context, images_dir: Path) -> list[dict[str, Any]]:
    """Render+carve every section and return the manifest entries (in emission order)."""
    images_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    for section in track.sections:
        log.info("Rendering track section %s", section.name)
        view_masks = _section_view_masks(track, section)
        images = render_section(track, section, context, view_masks)
        names = _sprite_names(track, section, view_masks)
        for name, image in zip(names, images, strict=True):
            filename = f"{name}.png"
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

    Writes ``<output>/images/<section>_<view>_<sub>.png`` (palette-indexed), a
    ``<output>/<id>.sprites.json`` manifest for ``openrct2 sprite build``, and the
    ``<id>.subpositions.json`` sidecar. ``skip_render`` is a no-op for tracks: there is
    no separate packaging step to repeat, and the manifest's draw offsets can only be
    recovered by rendering, so a prior run's outputs are simply left in place.
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
    """Per-section preview for fast iteration.

    Renders+carves each section and tiles all its sub-sprites into one preview image,
    so the test sprite shows every view/tile the section produces. The fastest way to
    validate the deformation, the axis mapping, and the mask carve.
    """
    test_dir = Path(test_dir)
    test_dir.mkdir(parents=True, exist_ok=True)
    for section in track.sections:
        log.info("Rendering track section %s", section.name)
        view_masks = _section_view_masks(track, section)
        images = render_section(track, section, context, view_masks)
        write_png(combine_indexed_images(images, columns=2), test_dir / f"{section.name}.png")
