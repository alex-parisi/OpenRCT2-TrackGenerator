"""
Per-view sprite masks, ported from RCTGen's ``mask.cpp`` / ``masks/default.json``.

maketrack renders each section's full deformed mesh once per view (angle), then carves
that render into per-tile sub-sprites using hand-authored mask PNGs. Each mask pixel
encodes its sub-sprite in the low 3 bits (``&0x7``, 1..N; 0 = background) and a
*secondary* layer in bits 3-5 (``&0x38``), with bit ``0x40`` flagging the origin pixel.

Some sub-sprites carry an occlusion **op** (``mask.cpp::process_mask``): ``split`` divides
a region into a front half (``INTERSECT`` the track-mask silhouette) and a behind half
(``DIFFERENCE``); ``transfer`` pulls the next region's silhouette-covered pixels in
(``TRANSFER_NEXT``). The silhouette is a separate solid *mask mesh* rendered per view;
occlusion is therefore only applied when a mask mesh is configured.

Carving uses edge-clamped sampling, reproducing ``mask.cpp``'s open-edge rectangle
behaviour (a region touching the mask border extends outward) — so a trivial all-ones
``single_tile`` mask keeps the whole render. ``mask_end`` / ``extrude`` (boundary render
trims) and ``mirror`` (right-hand reuse) are not yet ported.
"""

import json
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from openrct2_x7_renderer.types import IndexedImage
from PIL import Image as PILImage

__all__ = [
    "MaskOp",
    "OutputMask",
    "ViewMask",
    "carve",
    "default_masks_path",
    "load_section_masks",
]

# Mask pixel bit layout (mask.cpp::mask_index).
_PRIMARY_MASK = 0x7
_SECONDARY_MASK = 0x38
_SECONDARY_SHIFT = 3
_ORIGIN_BIT = 0x40


class MaskOp(IntEnum):
    """How a sub-sprite's region combines with the track-mask silhouette."""

    NONE = 0
    DIFFERENCE = 1  # behind: region minus silhouette
    INTERSECT = 2  # front: region within silhouette
    TRANSFER_NEXT = 3  # add the next region's silhouette-covered pixels


def default_masks_path() -> Path:
    """Path to the bundled ``masks/default.json``."""
    return Path(__file__).parent / "masks" / "default.json"


@dataclass(frozen=True, slots=True)
class OutputMask:
    """One carved sub-sprite: which mask region, its op, and its screen offset (tile)."""

    value: int  # sub-sprite index to match
    secondary: bool  # match the secondary layer instead of the primary
    op: MaskOp
    off_x: int
    off_y: int


@dataclass(frozen=True, slots=True)
class ViewMask:
    """A view's mask: per-pixel sub-sprite ids (both layers), origin, sub-sprites, flags.

    The boundary flags drive how the section's mesh is tiled when rendered (see
    ``section_renderer``): ``extrude_behind``/``extrude_in_front`` extend the drawn track
    one tile into the neighbour; ``mask_end`` renders the final tile as a ghost (occludes
    but isn't drawn) so the neighbouring piece draws that boundary instead.
    """

    primary: NDArray[np.uint8]  # (H, W) low-3-bit sub-sprite id (0 = background)
    secondary: NDArray[np.uint8]  # (H, W) bits 3-5 sub-sprite id
    origin: tuple[int, int]  # (ox, oy) mask-pixel coords of the origin marker
    masks: tuple[OutputMask, ...]
    extrude_behind: bool = False
    extrude_in_front: bool = False
    mask_end: bool = False

    @property
    def needs_silhouette(self) -> bool:
        return any(m.op is not MaskOp.NONE for m in self.masks)


def _load_mask_png(path: Path) -> tuple[NDArray[np.uint8], NDArray[np.uint8], tuple[int, int]]:
    """Read a mask PNG into (primary layer, secondary layer, origin). Raises if no origin."""
    arr = np.asarray(PILImage.open(path), dtype=np.uint8)
    if arr.ndim == 3:
        arr = arr[..., 0]
    origin = np.argwhere((arr & _ORIGIN_BIT) != 0)
    if origin.shape[0] != 1:
        raise ValueError(
            f"mask {path} must have exactly one origin pixel (found {origin.shape[0]})"
        )
    oy, ox = (int(origin[0, 0]), int(origin[0, 1]))
    primary = (arr & _PRIMARY_MASK).astype(np.uint8)
    secondary = ((arr & _SECONDARY_MASK) >> _SECONDARY_SHIFT).astype(np.uint8)
    return primary, secondary, (ox, oy)


def _build_output_masks(
    primary: NDArray[np.uint8],
    secondary: NDArray[np.uint8],
    *,
    split: list[bool],
    transfer: list[bool],
    split_ends: bool,
    offsets: list[tuple[int, int]],
    occlusion: bool,
) -> tuple[OutputMask, ...]:
    """Port of ``mask.cpp::process_mask``: turn the mask layers + flags into sub-sprites.

    With ``occlusion`` False (no mask mesh), the split/transfer ops are dropped and each
    sub-sprite is a plain region (the base carve).
    """
    num_sprites = max(int(primary.max()), int(secondary.max()), len(split), len(transfer))
    num_sprites = max(num_sprites, len(offsets))

    nontrivial = bool(np.any(primary != 1) or np.any(secondary != 1))
    if not nontrivial and not (split and split[0]) and num_sprites == 1:
        off = offsets[0] if offsets else (0, 0)
        return (OutputMask(1, False, MaskOp.NONE, off[0], off[1]),)

    def _flag(arr: list[bool], i: int) -> bool:
        return bool(arr[i]) if i < len(arr) else False

    def _off(i: int) -> tuple[int, int]:
        return offsets[i] if i < len(offsets) else (0, 0)

    out: list[OutputMask] = []
    for sprite in range(num_sprites):
        value = sprite + 1
        ox, oy = _off(sprite)

        if split_ends and sprite == num_sprites - 1:
            end_op = MaskOp.DIFFERENCE if occlusion else MaskOp.NONE
            out.append(OutputMask(1, False, end_op, ox, oy))
            break

        if not occlusion:
            out.append(OutputMask(value, False, MaskOp.NONE, ox, oy))
            continue

        if not _flag(transfer, sprite):
            if split_ends and sprite == 0:
                out.append(OutputMask(value, False, MaskOp.INTERSECT, ox, oy))
            elif sprite > 0 and _flag(transfer, sprite - 1):
                out.append(OutputMask(value, False, MaskOp.DIFFERENCE, ox, oy))
            elif _flag(split, sprite):
                out.append(OutputMask(value, False, MaskOp.INTERSECT, ox, oy))
                identical = bool(np.array_equal(primary == value, secondary == value))
                out.append(OutputMask(value, not identical, MaskOp.DIFFERENCE, ox, oy))
            else:
                out.append(OutputMask(value, False, MaskOp.NONE, ox, oy))
        else:
            out.append(OutputMask(value, False, MaskOp.TRANSFER_NEXT, ox, oy))
    return tuple(out)


def load_section_masks(
    section_name: str, masks_path: Path | str | None = None, *, occlusion: bool = False
) -> list[ViewMask]:
    """Load every view's mask for ``section_name`` from a ``masks`` JSON file.

    Mask image paths are resolved relative to the JSON file's directory. With
    ``occlusion`` set (a mask mesh is available), ``split``/``transfer`` ops expand into
    front/behind sub-sprites; otherwise each view yields plain regions.
    """
    masks_path = Path(masks_path) if masks_path is not None else default_masks_path()
    data = json.loads(masks_path.read_text())
    if section_name not in data:
        raise KeyError(f'No masks defined for section "{section_name}" in {masks_path}')

    views: list[ViewMask] = []
    for entry in data[section_name]:
        primary, secondary, origin = _load_mask_png(masks_path.parent / entry["mask"])
        if entry.get("mirror"):
            # Reflect the mask about its origin column (right-hand pieces reuse the
            # left mask mirrored); flipping the array + moving the origin keeps the
            # carve's sampling aligned. (mask.cpp add_rect mirror branch.)
            primary = np.fliplr(primary).copy()
            secondary = np.fliplr(secondary).copy()
            origin = (primary.shape[1] - 1 - origin[0], origin[1])
        offsets = [(int(x), int(y)) for x, y in entry.get("offset", [])]
        masks = _build_output_masks(
            primary,
            secondary,
            split=list(entry.get("split", [])),
            transfer=list(entry.get("transfer", [])),
            split_ends=bool(entry.get("split_ends", False)),
            offsets=offsets,
            occlusion=occlusion,
        )
        views.append(
            ViewMask(
                primary=primary,
                secondary=secondary,
                origin=origin,
                masks=masks,
                extrude_behind=bool(entry.get("extrude_behind", False)),
                extrude_in_front=bool(entry.get("extrude_in_front", False)),
                mask_end=bool(entry.get("mask_end", False)),
            )
        )
    return views


def _crop(x_off: int, y_off: int, pixels: NDArray[np.uint8]) -> IndexedImage:
    """Crop ``pixels`` to its non-transparent bounding box, shifting the draw offset."""
    nz = np.argwhere(pixels != 0)
    if nz.shape[0] == 0:
        return IndexedImage.blank(1, 1)
    (y0, x0), (y1, x1) = nz.min(axis=0), nz.max(axis=0) + 1
    return IndexedImage(
        width=int(x1 - x0),
        height=int(y1 - y0),
        x_offset=x_off + int(x0),
        y_offset=y_off + int(y0),
        pixels=pixels[y0:y1, x0:x1].copy(),
    )


def _sample_layer(
    layer: NDArray[np.uint8], origin: tuple[int, int], full: IndexedImage, dy: int = 0
) -> NDArray[np.uint8]:
    """Edge-clamp sample a mask layer onto the full sprite's pixel grid.

    ``dy`` shifts the mask-layer sampling vertically (``OFFSET_SPRITE_MASK``'s
    ``z_offset - 8`` nudge in ``track.cpp``); the track-mask silhouette is *not*
    shifted, matching the engine.
    """
    ox, oy = origin
    mh, mw = layer.shape
    cols = np.arange(full.pixels.shape[1])[None, :]
    rows = np.arange(full.pixels.shape[0])[:, None]
    mx = np.clip(cols + full.x_offset + ox, 0, mw - 1)
    my = np.clip(rows + full.y_offset + oy + dy, 0, mh - 1)
    sampled: NDArray[np.uint8] = layer[my, mx]
    return sampled


def _in_silhouette(silhouette: IndexedImage, full: IndexedImage) -> NDArray[np.bool_]:
    """Boolean grid (full-sprite shape) of whether each pixel falls inside the silhouette."""
    sh, sw = silhouette.pixels.shape
    shape = full.pixels.shape
    cols = np.broadcast_to(np.arange(shape[1])[None, :], shape)
    rows = np.broadcast_to(np.arange(shape[0])[:, None], shape)
    sx = cols + full.x_offset - silhouette.x_offset
    sy = rows + full.y_offset - silhouette.y_offset
    inside: NDArray[np.bool_] = np.zeros(shape, dtype=bool)
    valid = (sx >= 0) & (sx < sw) & (sy >= 0) & (sy < sh)
    inside[valid] = silhouette.pixels[sy[valid], sx[valid]] != 0
    return inside


def carve(
    full: IndexedImage,
    view_mask: ViewMask,
    silhouette: IndexedImage | None = None,
    mask_dy: int = 0,
) -> list[IndexedImage]:
    """Carve a full render into its sub-sprites using ``view_mask`` (+ ``silhouette`` for ops).

    Mask alignment via the origin: full-sprite pixel ``(c, r)`` maps to mask pixel
    ``(c + x_offset + ox, r + y_offset + oy)``, edge-clamped. Ops combine each region
    with the silhouette (DIFFERENCE=behind, INTERSECT=front, TRANSFER_NEXT pulls the next
    region's silhouette-covered pixels in).
    """
    prim = _sample_layer(view_mask.primary, view_mask.origin, full, mask_dy)
    sec = _sample_layer(view_mask.secondary, view_mask.origin, full, mask_dy)
    in_sil = _in_silhouette(silhouette, full) if silhouette is not None else None

    def region(m: OutputMask) -> NDArray[np.bool_]:
        matched: NDArray[np.bool_] = (sec if m.secondary else prim) == m.value
        return matched

    out: list[IndexedImage] = []
    for i, m in enumerate(view_mask.masks):
        keep = region(m)
        if m.op is not MaskOp.NONE and in_sil is not None:
            if m.op is MaskOp.DIFFERENCE:
                keep = keep & ~in_sil
            elif m.op is MaskOp.INTERSECT:
                keep = keep & in_sil
            elif m.op is MaskOp.TRANSFER_NEXT and i + 1 < len(view_mask.masks):
                keep = keep | (in_sil & region(view_mask.masks[i + 1]))
        pixels = np.where(keep, full.pixels, np.uint8(0)).astype(np.uint8)
        out.append(_crop(full.x_offset + m.off_x, full.y_offset + m.off_y, pixels))
    return out
