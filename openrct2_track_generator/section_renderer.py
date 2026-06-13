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
from openrct2_x7_renderer.ray_trace import VIEWS, Context, FinalizedScene, SceneBuilder
from openrct2_x7_renderer.types import IndexedImage

from .constants import (
    CLEARANCE_HEIGHT,
    SPECIAL_MODEL_KEY,
    SPECIAL_RIGHT_NO_FLIP,
    SPECIAL_TILED,
    SUPPORT_BASE_KEY,
    TILE_SIZE,
    SpecialModel,
    TrackFlag,
)
from .deform import deform_mesh, get_track_point_array
from .lift import CHAIN_PATTERNS, ChainPattern, apply_lift
from .masks import ViewMask, carve
from .offsets import set_offset
from .supports import support_posts
from .types import Track, TrackSection

__all__ = ["angle_plan", "render_section"]

_IDENTITY = np.eye(3, dtype=np.float64)
_ORIGIN = np.zeros(3, dtype=np.float64)
_GHOST = int(MeshFlag.GHOST)
_MASK = int(MeshFlag.MASK)


def _cc(v: np.ndarray) -> np.ndarray:
    """change_coordinates: curve-space (x, y, z) -> render-space (z, y, x)."""
    return v[::-1].copy()


def _rigid_tie_transform(
    section: TrackSection,
    z_offset: float,
    u: float,
    start_offset: np.ndarray,
    end_offset: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Placement (matrix, translation) for a rigid tie at arc-distance ``u`` (``track.cpp``).

    The tie sits in the curve's moving frame: matrix columns are the change-coordinates'd
    binormal/normal/tangent, translation is the change-coordinates'd position.
    """
    tp = get_track_point_array(
        section.curve, section.flags, z_offset, section.length,
        start_offset, end_offset, np.array([u], dtype=np.float64),
    )
    mat = np.column_stack((_cc(tp.binormal[0]), _cc(tp.normal[0]), _cc(tp.tangent[0])))
    return mat, _cc(tp.position[0])


def _tie_boundary_flags(flags: TrackFlag, view_angle: int) -> tuple[bool, bool]:
    """Whether a tie sits at the section's start/end boundary for a view (``track.cpp:251-267``).

    The entry heading is the view angle (turned one step for diagonals); the exit heading is
    the entry adjusted by the section's exit-direction flags. A boundary gets a tie when its
    heading is one of the two "tie" orientations (angle <= 1 at the start, > 1 at the end).
    """
    diag = bool(flags & (TrackFlag.DIAGONAL | TrackFlag.DIAGONAL_2))
    start_angle = (view_angle + 1) % 4 if diag else view_angle
    end_angle = start_angle
    if flags & TrackFlag.EXIT_90_DEG_LEFT:
        end_angle -= 1
    elif flags & TrackFlag.EXIT_90_DEG_RIGHT:
        end_angle += 1
    elif flags & TrackFlag.EXIT_180_DEG:
        end_angle += 2
    elif (flags & TrackFlag.EXIT_45_DEG_LEFT) and diag:
        end_angle -= 1
    elif (flags & TrackFlag.EXIT_45_DEG_RIGHT) and not diag:
        end_angle += 1
    end_angle %= 4
    return start_angle <= 1, end_angle > 1


def _build_boundary_ties(
    builder: SceneBuilder,
    mesh: Mesh,
    occluder: Mesh | None,
    section: TrackSection,
    view_mask: ViewMask,
    *,
    flat_shaded: bool,
    is_mask: bool,
    z_offset: float,
    start_offset: np.ndarray,
    end_offset: np.ndarray,
    tie_mesh: Mesh,
    track_tie_mesh: Mesh,
    tie_length: float,
    view_angle: int,
    num: int,
    tile_length: float,
) -> None:
    """Retile a section with ties at tile boundaries (``track.cpp:249-327``).

    The section is re-divided so a tie sits at each boundary the entry/exit heading puts one
    at (``start_tie`` / ``end_tie``), and a corrected scale absorbs the resulting length
    change. Even segments place a rigid ``tie_mesh`` + a deformed ``track_tie_mesh`` over a
    ``tie_length`` span; odd segments place the plain track over the between-tie span. (No
    example track uses this mode, so it is a faithful structural port, not golden-validated;
    alt track meshes and VIEW_MASK_END are not modelled, matching the unused C++ paths.)
    """
    sflag = _MASK if is_mask else 0
    track_solid = occluder if (is_mask and occluder is not None) else mesh  # the track mesh
    plate = mesh if is_mask else None  # the mask plate (silhouette only)

    start_tie, end_tie = _tie_boundary_flags(section.flags, view_angle)

    corrected_length = num * tile_length
    if not start_tie:
        corrected_length -= tie_length
    if end_tie:
        corrected_length += tie_length
    corrected_scale = section.length / corrected_length
    tie_len = corrected_scale * tie_length
    inter_length = corrected_scale * (tile_length - tie_length)

    offset = 0.0
    nm = num
    if view_mask.extrude_behind:
        nm += 1
        offset -= corrected_scale * tile_length
    if view_mask.extrude_in_front:
        nm += 1

    def _deform(m: Mesh, off: float) -> Mesh:
        return deform_mesh(
            m, section, scale=corrected_scale, offset=off, track_length=section.length,
            z_offset=z_offset, flat_shaded=flat_shaded,
            start_offset=start_offset, end_offset=end_offset,
        )

    for i in range(2 * nm + 1):
        is_tie = (i % 2 == 0) and (i != 0 or start_tie) and (i != 2 * nm or end_tie)
        if is_tie:
            # The rigid tie centres at the (unscaled) tie midpoint, matching track.cpp:298.
            mat, trans = _rigid_tie_transform(
                section, z_offset, offset + tie_length / 2.0, start_offset, end_offset
            )
            builder.add_model(tie_mesh, mat, trans, sflag)
            builder.add_model(
                _deform(track_tie_mesh, offset), _IDENTITY, _ORIGIN, sflag
            )
            offset += tie_len
        elif i % 2 == 1:
            # use_alt is always false here (the port has no alt track mesh).
            builder.add_model(
                _deform(track_solid, offset), _IDENTITY, _ORIGIN, sflag
            )
            if is_mask and plate is not None:
                plate_off = offset - tie_len if start_tie else offset
                builder.add_model(
                    _deform(plate, plate_off), _IDENTITY, _ORIGIN, 0
                )
            offset += inter_length


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
    special_mesh: Mesh | None = None,
    special_length: float = TILE_SIZE,
    rigid_special: tuple[Mesh, np.ndarray, np.ndarray] | None = None,
    base_mesh: Mesh | None = None,
    posts: list[tuple[Mesh, np.ndarray, np.ndarray]] | None = None,
    start_offset: np.ndarray = _ORIGIN,
    end_offset: np.ndarray = _ORIGIN,
    tie_mesh: Mesh | None = None,
    track_tie_mesh: Mesh | None = None,
    tie_length: float = TILE_SIZE,
    tie_at_boundary: bool = False,
    view_angle: int = 0,
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
            start_offset=start_offset, end_offset=end_offset,
        )
        builder.add_model(copy, _IDENTITY, _ORIGIN, flag)
        if with_occluder and occluder is not None:
            occ = deform_mesh(
                occluder, section, scale=scale, offset=offset, track_length=section.length,
                z_offset=z_offset, flat_shaded=flat_shaded,
                start_offset=start_offset, end_offset=end_offset,
            )
            builder.add_model(occ, _IDENTITY, _ORIGIN, _MASK)

    # Ghost end-caps one tile before/after (drawn for the silhouette so it spans neighbours).
    if is_mask or not view_mask.extrude_behind:
        add(-length, 0 if is_mask else _GHOST)
    if is_mask or not (view_mask.extrude_in_front or view_mask.mask_end):
        add(section.length, 0 if is_mask else _GHOST)

    sflag = _MASK if is_mask else 0
    if tie_at_boundary and tie_mesh is not None and track_tie_mesh is not None:
        _build_boundary_ties(
            builder, mesh, occluder, section, view_mask,
            flat_shaded=flat_shaded, is_mask=is_mask, z_offset=z_offset,
            start_offset=start_offset, end_offset=end_offset,
            tie_mesh=tie_mesh, track_tie_mesh=track_tie_mesh, tie_length=tie_length,
            view_angle=view_angle, num=num, tile_length=length / scale,
        )
    else:
        # The drawn copies (extended by one for each extrude flag / mask_end).
        count = num + int(view_mask.extrude_behind) + int(view_mask.extrude_in_front)
        count += int(view_mask.mask_end)
        for i in range(count):
            offset = (i - (1 if view_mask.extrude_behind else 0)) * length
            ghost_last = (not is_mask) and view_mask.mask_end and (i + 1 == count)
            add(offset, _GHOST if ghost_last else 0, with_occluder=is_mask)
            # Per-tile support base (track.cpp:356-357): a curve-deformed model on its own
            # upright frame, drawn with the track (occluder in the silhouette).
            if base_mesh is not None and not ghost_last:
                bcopy = deform_mesh(
                    base_mesh, section, scale=scale, offset=offset, track_length=section.length,
                    z_offset=z_offset, flat_shaded=flat_shaded, base=True,
                    start_offset=start_offset, end_offset=end_offset,
                )
                builder.add_model(bcopy, _IDENTITY, _ORIGIN, sflag)
            # Separate rigid tie at the tile centre (track.cpp:358-362).
            if tie_mesh is not None and not ghost_last:
                mat, trans = _rigid_tie_transform(
                    section, z_offset, offset + 0.5 * length, start_offset, end_offset
                )
                builder.add_model(tie_mesh, mat, trans, sflag)

    # Special-mechanism mesh (brake/booster) tiled on top of the track by its own length
    # (track.cpp:381-400). Drawn for the visible render, an occluder for the silhouette.
    if special_mesh is not None:
        snum = max(1, int(np.floor(0.5 + section.length / special_length)))
        sscale = section.length / (snum * special_length)
        sstep = sscale * special_length
        for i in range(snum):
            scopy = deform_mesh(
                special_mesh, section, scale=sscale, offset=i * sstep,
                track_length=section.length, z_offset=z_offset, flat_shaded=flat_shaded,
                start_offset=start_offset, end_offset=end_offset,
            )
            builder.add_model(scopy, _IDENTITY, _ORIGIN, sflag)

    # Rigid special support model (inversions / launched_lift): a single model placed with a
    # fixed view[1]-based transform, not deformed along the curve (track.cpp:401).
    if rigid_special is not None:
        rmesh, rmat, rtrans = rigid_special
        builder.add_model(rmesh, rmat, rtrans, sflag)

    # Support posts (track.cpp:405-437): rigid models placed at intervals along the section.
    if posts is not None:
        for pmesh, pmat, ptrans in posts:
            builder.add_model(pmesh, pmat, ptrans, sflag)

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

    # Tiled special-mechanism mesh (brake/booster): resolve the model + its tile length
    # (block_brake tiles by one tile, others by brake_length). Missing model -> plain track.
    special_mesh: Mesh | None = None
    special_length = track.brake_length
    rigid_special: tuple[Mesh, np.ndarray, np.ndarray] | None = None
    if section.special in SPECIAL_TILED:
        special_mesh = track.special_models.get(SPECIAL_MODEL_KEY[section.special])
        if section.special is SpecialModel.BLOCK_BRAKE:
            special_length = TILE_SIZE
    elif section.special is not None:
        # Rigid support model (inversions / launched_lift). Placed with views[1]; all but the
        # *_RIGHT variants flip the matrix bottom row. Translation centres across-track (unless
        # vertical) and drops by 2 clearance heights below the track z_offset (track.cpp:401).
        rmesh = track.special_models.get(SPECIAL_MODEL_KEY[section.special])
        if rmesh is not None:
            rmat = VIEWS[1].copy()
            if section.special not in SPECIAL_RIGHT_NO_FLIP:
                rmat[2] = -rmat[2]
            tx = 0.0 if (section.flags & TrackFlag.VERTICAL) else -0.5 * TILE_SIZE
            rtrans = np.array([tx, z_offset - 2.0 * CLEARANCE_HEIGHT, 0.0], dtype=np.float64)
            rigid_special = (rmesh, rmat, rtrans)

    # Supports: per-tile base model + support posts, when the track enables supports and the
    # section permits them (track.cpp:356-357, 405-437). Posts whose bank model isn't supplied
    # are dropped (maketrack leaves them as empty meshes).
    supports_enabled = track.has_supports and not (section.flags & TrackFlag.NO_SUPPORTS)
    base_mesh: Mesh | None = (
        track.special_models.get(SUPPORT_BASE_KEY) if supports_enabled else None
    )

    def _resolve_posts(
        start_off: np.ndarray, end_off: np.ndarray
    ) -> list[tuple[Mesh, np.ndarray, np.ndarray]] | None:
        if not supports_enabled:
            return None
        resolved = [
            (track.special_models[p.model_key], p.matrix, p.translation)
            for p in support_posts(
                section, z_offset, track.support_spacing, track.pivot, start_off, end_off
            )
            if p.model_key in track.special_models
        ]
        return resolved or None

    # Separate ties: the rigid tie mesh (both modes) + the deformable track-tie mesh
    # (boundary mode). Absent indices disable the respective tie path.
    tie_mesh = (
        track.meshes[track.tie_mesh_index]
        if track.separate_tie and track.tie_mesh_index >= 0
        else None
    )
    track_tie_mesh = (
        track.meshes[track.track_tie_mesh_index]
        if track.tie_at_boundary and track.track_tie_mesh_index >= 0
        else None
    )

    # End offsets are per view angle. Without special_end_offsets they're zero for every
    # angle, so the posts are identical and computed once; otherwise they're recomputed per
    # angle from set_offset (track.cpp sets the same blend on the per-view post placement).
    static_posts = None if track.special_end_offsets else _resolve_posts(_ORIGIN, _ORIGIN)

    for angle, view_mask, chain in angle_plan(track, section, view_masks):
        if track.special_end_offsets:
            start_off, end_off = set_offset(angle, section, track.offset_table)
            posts = _resolve_posts(start_off, end_off)
        else:
            start_off = end_off = _ORIGIN
            posts = static_posts

        silhouette: IndexedImage | None = None
        if track.mask_mesh_index >= 0 and view_mask.needs_silhouette:
            mask_mesh = track.meshes[track.mask_mesh_index]
            sil_scene = _build_scene(
                context, mask_mesh, section, view_mask,
                flat_shaded=track.flat_shaded, is_mask=True, z_offset=z_offset,
                occluder=track_mesh, special_mesh=special_mesh, special_length=special_length,
                rigid_special=rigid_special, base_mesh=base_mesh, posts=posts,
                start_offset=start_off, end_offset=end_off,
                tie_mesh=tie_mesh, track_tie_mesh=track_tie_mesh, tie_length=track.tie_length,
                tie_at_boundary=track.tie_at_boundary, view_angle=angle,
            )
            try:
                silhouette = sil_scene.render_silhouette(VIEWS[angle])
            finally:
                sil_scene.end_render()

        scene = _build_scene(
            context, track_mesh, section, view_mask,
            flat_shaded=track.flat_shaded, is_mask=False, z_offset=z_offset,
            special_mesh=special_mesh, special_length=special_length,
            rigid_special=rigid_special, base_mesh=base_mesh, posts=posts,
            start_offset=start_off, end_offset=end_off,
            tie_mesh=tie_mesh, track_tie_mesh=track_tie_mesh, tie_length=track.tie_length,
            tie_at_boundary=track.tie_at_boundary, view_angle=angle,
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
