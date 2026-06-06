"""
Render one track section into its per-view, per-tile sprites (maketrack-style).

Ported from ``track.cpp::render_track_section`` + the carve loop. A section is tiled
into ``round(length / tile_length)`` mesh copies (keeping tile-length proportions around
curves), with **ghost end-caps** one tile before/after (they occlude but aren't drawn, so
boundaries with neighbours hide correctly). Per-view boundary flags adjust this:
``extrude_*`` draw an extra real tile into the neighbour; ``mask_end`` renders the final
tile as a ghost. Each view is then rendered and carved by its mask; when a view needs
occlusion, the mask mesh's silhouette is rendered the same way and fed to the carve.
"""

import numpy as np
from openrct2_x7_renderer.constants import MeshFlag
from openrct2_x7_renderer.mesh import Mesh
from openrct2_x7_renderer.ray_trace import VIEWS, Context, FinalizedScene
from openrct2_x7_renderer.types import IndexedImage

from .constants import CLEARANCE_HEIGHT, TILE_SIZE, TrackFlag
from .deform import deform_mesh
from .lift import CHAIN_PATTERNS, ChainPattern, apply_lift
from .masks import ViewMask, carve
from .types import Track, TrackSection

__all__ = ["angle_plan", "render_section"]

_IDENTITY = np.eye(3, dtype=np.float64)
_ORIGIN = np.zeros(3, dtype=np.float64)
_GHOST = int(MeshFlag.GHOST)
_MASK = int(MeshFlag.MASK)


def _section_layout(mesh: Mesh, section: TrackSection) -> tuple[int, float, float]:
    """How the tile mesh tiles a section: (num_copies, per-copy scale, per-copy arc length).

    ``num_copies`` rounds the section's arc length to whole tile lengths
    (``track.cpp::render_track_section``); the scale absorbs the rounding so the copies
    exactly span the section.
    """
    if mesh.vertices.shape[0] == 0:
        return 1, 1.0, section.length
    z = mesh.vertices[:, 2].astype(np.float64)
    tile_length = float(z.max() - z.min())
    if tile_length <= 0.0:
        tile_length = TILE_SIZE
    num = max(1, int(np.floor(0.5 + section.length / tile_length)))
    scale = section.length / (num * tile_length)
    return num, scale, scale * tile_length


def _build_scene(
    context: Context,
    mesh: Mesh,
    section: TrackSection,
    view_mask: ViewMask,
    *,
    flat_shaded: bool,
    is_mask: bool,
    z_offset: float,
    occluder: Mesh | None = None,
) -> FinalizedScene:
    """Tile ``mesh`` along the section with ghost end-caps + extrude/mask_end handling.

    ``is_mask`` builds the occlusion silhouette: every copy (incl. end-caps) is drawn so
    the silhouette covers the neighbours; for the visible render the end-caps and the
    ``mask_end`` final tile are ghosts. For the silhouette, the track ``occluder`` mesh is
    added on each drawn tile as a ``MeshFlag.MASK`` occluder (``track.cpp`` adds the track
    mesh with the track-mask flag alongside the mask plate), so where the rail sits *in
    front of* the mask plate it punches it out of the silhouette — essential for the
    self-overlapping inverted pieces. End-caps stay mask-plate only.
    """
    num, scale, length = _section_layout(mesh, section)
    builder = context.begin_render()

    def add(offset: float, flag: int, *, with_occluder: bool = False) -> None:
        copy = deform_mesh(
            mesh, section, scale=scale, offset=offset, track_length=section.length,
            z_offset=z_offset, flat_shaded=flat_shaded,
        )
        builder.add_model(copy, _IDENTITY, _ORIGIN, flag)
        if with_occluder and occluder is not None:
            occ = deform_mesh(
                occluder, section, scale=scale, offset=offset, track_length=section.length,
                z_offset=z_offset, flat_shaded=flat_shaded,
            )
            builder.add_model(occ, _IDENTITY, _ORIGIN, _MASK)

    # Ghost end-caps one tile before/after (drawn for the silhouette so it spans neighbours).
    if is_mask or not view_mask.extrude_behind:
        add(-length, 0 if is_mask else _GHOST)
    if is_mask or not (view_mask.extrude_in_front or view_mask.mask_end):
        add(section.length, 0 if is_mask else _GHOST)

    # The drawn copies (extended by one for each extrude flag / mask_end).
    count = num + int(view_mask.extrude_behind) + int(view_mask.extrude_in_front)
    count += int(view_mask.mask_end)
    for i in range(count):
        offset = (i - (1 if view_mask.extrude_behind else 0)) * length
        ghost_last = (not is_mask) and view_mask.mask_end and (i + 1 == count)
        add(offset, _GHOST if ghost_last else 0, with_occluder=is_mask)

    return builder.finalize()


def angle_plan(
    track: Track, section: TrackSection, view_masks: list[ViewMask]
) -> list[tuple[int, ViewMask, ChainPattern | None]]:
    """The per-angle render plan: ``(view-direction, mask, chain-stamp-or-None)``.

    Normally one angle per mask entry. A lift hill emits all four directions for *every*
    section (maketrack's view loop expands under lift regardless of chain) — a section
    with fewer masks than four reuses ``view_masks[angle % len]``, so a 2-view flat or
    S-bend becomes 4 sprites. The chain stamp is overlaid per direction only when the
    section actually has one.
    """
    chain = CHAIN_PATTERNS[section.chain] if (track.has_lift and section.chain) else None
    num_angles = 4 if track.has_lift else len(view_masks)
    return [
        (angle, view_masks[angle % len(view_masks)], chain[angle] if chain is not None else None)
        for angle in range(num_angles)
    ]


def render_section(
    track: Track, section: TrackSection, context: Context, view_masks: list[ViewMask]
) -> list[IndexedImage]:
    """Render+carve a section per its :func:`angle_plan` (angle-major sub-sprites)."""
    track_mesh = track.meshes[track.track_mesh_index]
    images: list[IndexedImage] = []

    # maketrack's per-track vertical offset (track.cpp): (z_offset / 8) * CLEARANCE_HEIGHT,
    # plus any per-section adjustment.
    z_offset = (track.z_offset / 8.0) * CLEARANCE_HEIGHT + section.z_offset

    # OFFSET_SPRITE_MASK nudges the mask-layer sampling by (z_offset - 8) px (track.cpp);
    # z_offset there is the integer-rounded config value. The silhouette is not shifted.
    z_offset_int = int(track.z_offset + 0.499999)
    mask_dy = (z_offset_int - 8) if (section.flags & TrackFlag.OFFSET_SPRITE_MASK) else 0

    for angle, view_mask, chain in angle_plan(track, section, view_masks):
        silhouette: IndexedImage | None = None
        if track.mask_mesh_index >= 0 and view_mask.needs_silhouette:
            mask_mesh = track.meshes[track.mask_mesh_index]
            sil_scene = _build_scene(
                context, mask_mesh, section, view_mask,
                flat_shaded=track.flat_shaded, is_mask=True, z_offset=z_offset,
                occluder=track_mesh,
            )
            try:
                silhouette = sil_scene.render_silhouette(VIEWS[angle])
            finally:
                sil_scene.end_render()

        scene = _build_scene(
            context, track_mesh, section, view_mask,
            flat_shaded=track.flat_shaded, is_mask=False, z_offset=z_offset,
        )
        try:
            full = scene.render_view(VIEWS[angle])
        finally:
            scene.end_render()

        subs = carve(full, view_mask, silhouette, mask_dy)
        if chain is not None:
            subs = [apply_lift(s, chain) for s in subs]
        images.extend(subs)
    return images
