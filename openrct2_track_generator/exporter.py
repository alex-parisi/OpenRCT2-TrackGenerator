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
    """Sprite filenames for a section, in angle-major then sub-sprite order.

    The track's ``suffix`` (empty for single-track configs) is appended after the section
    name, matching maketrack's per-track ``name`` suffix so variants don't collide.
    """
    return [
        f"{section.name}{track.suffix}_{angle}_{sub}"
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


def _render_sprites(
    track: Track, context: Context, images_dir: Path, path_prefix: str
) -> list[dict[str, Any]]:
    """Render+carve every section and return the manifest entries (in emission order).

    PNGs are written under ``images_dir``; each manifest ``path`` is ``path_prefix`` joined
    to the filename (so the manifest stays relative to its own directory, as ``sprite build``
    expects).
    """
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
                    "path": f"{path_prefix}/{filename}",
                    "x": int(image.x_offset),
                    "y": int(image.y_offset),
                    "palette": "keep",
                }
            )
    return manifest


def _resolve(output_directory: Path, path: str) -> Path:
    """Resolve a configured manifest path against the output dir (absolute paths kept)."""
    p = Path(path)
    return p if p.is_absolute() else output_directory / p


def _load_spritefile_in(path: Path) -> list[dict[str, Any]]:
    """Load an existing sprite manifest array to append to (maketrack's ``spritefile_in``)."""
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError(f'spritefile_in "{path}" is not a JSON array')
    return data


def _write_subpositions(track: Track, output_directory: Path) -> Path:
    """Write the ``<id>.subpositions.json`` sidecar and return its path."""
    path = output_directory / f"{track.id}.subpositions.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build_subposition_data(track), indent=4))
    log.info("wrote %s (%d sections)", path, len(track.sections))
    return path


def export_tracks(
    tracks: list[Track],
    context: Context,
    output_directory: Path | str,
    skip_render: bool = False,
) -> None:
    """Render one or more track variants into a single merged sprite manifest.

    Each track is rendered (its ``suffix`` keeping variant filenames distinct) and its
    sprites appended, in order, to one manifest — mirroring maketrack processing a ``tracks``
    array into one spritefile. Shared output settings (``sprite_directory`` /
    ``spritefile_in``/``out`` / ``id``) are taken from the first track. Writes
    ``<output>/<sprite_dir>/<section><suffix>_<view>_<sub>.png`` (palette-indexed), the
    manifest for ``openrct2 sprite build``, and one ``<id>.subpositions.json`` sidecar.
    ``skip_render`` leaves any prior outputs in place (the draw offsets can only be recovered
    by rendering, and there is no separate packaging step to repeat).
    """
    output_directory = Path(output_directory)
    if skip_render:
        log.info("skip_render: leaving existing sprites and manifest in place")
        return
    if not tracks:
        return

    head = tracks[0]
    sprite_dir = head.sprite_directory or IMAGES_DIRNAME
    images_dir = output_directory / sprite_dir

    # maketrack appends the rendered sprites to an existing manifest (spritefile_in) so they
    # land at fixed global image indices; absent, we write a fresh standalone array.
    manifest: list[dict[str, Any]] = []
    if head.spritefile_in:
        manifest = _load_spritefile_in(_resolve(output_directory, head.spritefile_in))
    rendered_total = 0
    for track in tracks:
        rendered = _render_sprites(track, context, images_dir, sprite_dir)
        manifest += rendered
        rendered_total += len(rendered)

    if head.spritefile_out:
        manifest_path = _resolve(output_directory, head.spritefile_out)
    else:
        manifest_path = output_directory / f"{head.id}.sprites.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=4))
    log.info(
        "wrote %s (%d sprites, %d rendered from %d track(s))",
        manifest_path, len(manifest), rendered_total, len(tracks),
    )

    _write_subpositions(head, output_directory)


def export_track(
    track: Track, context: Context, output_directory: Path | str, skip_render: bool = False
) -> None:
    """Render a single track's sprites + manifest (see :func:`export_tracks`)."""
    export_tracks([track], context, output_directory, skip_render)


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
        out = test_dir / f"{section.name}{track.suffix}.png"
        write_png(combine_indexed_images(images, columns=2), out)
