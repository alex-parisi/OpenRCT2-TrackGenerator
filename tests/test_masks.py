"""Mask loading (incl. occlusion ops) and the numpy carve."""

import numpy as np
import pytest
from openrct2_track_generator.masks import (
    MaskOp,
    OutputMask,
    ViewMask,
    _build_output_masks,
    carve,
    default_masks_path,
    load_section_masks,
)
from openrct2_x7_renderer.types import IndexedImage


def test_default_masks_path_exists():
    assert default_masks_path().is_file()


def test_flat_is_two_trivial_views():
    views = load_section_masks("flat")
    assert len(views) == 2
    for vm in views:
        assert len(vm.masks) == 1  # single_tile -> one sub-sprite, kept whole
        assert vm.masks[0].op is MaskOp.NONE
        assert not vm.needs_silhouette


def test_small_turn_has_four_views_of_three_subsprites():
    views = load_section_masks("small_turn_left")
    assert len(views) == 4
    for vm in views:
        assert [m.value for m in vm.masks] == [1, 2, 3]
        assert all(m.op is MaskOp.NONE for m in vm.masks)  # turns are a pure carve
    assert [(m.off_x, m.off_y) for m in views[0].masks] == [(0, 0), (-32, 16), (0, 32)]


def test_split_expands_to_front_and_behind_only_with_occlusion():
    # gentle_to_steep's split views become INTERSECT (front) + DIFFERENCE (behind).
    occ = load_section_masks("gentle_to_steep", occlusion=True)
    split_views = [vm for vm in occ if len(vm.masks) == 2]
    assert split_views, "expected split views to expand to two sub-sprites"
    for vm in split_views:
        assert {m.op for m in vm.masks} == {MaskOp.INTERSECT, MaskOp.DIFFERENCE}
        assert vm.needs_silhouette

    # Without a mask mesh, the split collapses to a plain single sub-sprite.
    base = load_section_masks("gentle_to_steep", occlusion=False)
    assert all(len(vm.masks) == 1 and not vm.needs_silhouette for vm in base)


def test_plain_multi_sprite_occlusion_keeps_ops_none():
    # An occluding load of a no-split section (the turn) keeps every op NONE.
    views = load_section_masks("small_turn_left", occlusion=True)
    assert all(m.op is MaskOp.NONE for vm in views for m in vm.masks)


def test_build_output_masks_split_ends():
    # split_ends: first sub-sprite INTERSECTs, last DIFFERENCEs.
    prim = np.array([[1, 2]], dtype=np.uint8)
    masks = _build_output_masks(
        prim, prim, split=[], transfer=[], split_ends=True, offsets=[], occlusion=True
    )
    assert [m.op for m in masks] == [MaskOp.INTERSECT, MaskOp.DIFFERENCE]


def test_build_output_masks_transfer():
    # transfer on sub-sprite 0 -> TRANSFER_NEXT; the following one DIFFERENCEs.
    prim = np.array([[1, 2]], dtype=np.uint8)
    masks = _build_output_masks(
        prim, prim, split=[], transfer=[True, False], split_ends=False, offsets=[], occlusion=True
    )
    assert masks[0].op is MaskOp.TRANSFER_NEXT
    assert masks[1].op is MaskOp.DIFFERENCE


def test_mirror_flips_mask_about_origin(tmp_path):
    import json

    from PIL import Image

    # 4-wide mask: sub-sprite 1 on the left two columns, origin at column 1.
    # Pixel value = sub-sprite (low 3 bits); 0x40 marks the origin.
    row = [1 | 0x40, 1, 0, 0]  # origin (0x40) at col 0
    im = Image.fromarray(np.array([row], dtype=np.uint8), mode="P")
    im.putpalette([c for i in range(256) for c in (i, i, i)])  # identity, preserves indices
    im.save(tmp_path / "m.png")
    (tmp_path / "j.json").write_text(
        json.dumps({"flat": [{"mask": "m.png", "mirror": True}]})
    )
    (view,) = load_section_masks("flat", tmp_path / "j.json")
    # fliplr turns [1,1,0,0] into [0,0,1,1]; the origin moves col 0 -> col 3.
    assert view.primary.tolist() == [[0, 0, 1, 1]]
    assert view.origin == (3, 0)


def test_unknown_section_raises():
    with pytest.raises(KeyError, match="No masks defined"):
        load_section_masks("loop_de_loop")


def test_load_mask_png_requires_one_origin(tmp_path):
    import json

    from PIL import Image

    Image.fromarray(np.full((4, 4), 9, dtype=np.uint8), mode="P").save(tmp_path / "bad.png")
    (tmp_path / "m.json").write_text(json.dumps({"flat": [{"mask": "bad.png"}]}))
    with pytest.raises(ValueError, match="origin"):
        load_section_masks("flat", tmp_path / "m.json")


def test_load_rgb_mask_uses_red_channel(tmp_path):
    import json

    from PIL import Image

    arr = np.full((3, 3, 3), 0, dtype=np.uint8)
    arr[..., 0] = 9  # sub-sprite 1
    arr[0, 0, 0] = 73  # origin (0x40 | 9)
    Image.fromarray(arr, mode="RGB").save(tmp_path / "rgb.png")
    (tmp_path / "m.json").write_text(json.dumps({"flat": [{"mask": "rgb.png"}]}))
    views = load_section_masks("flat", tmp_path / "m.json")
    assert views[0].origin == (0, 0)
    assert [m.value for m in views[0].masks] == [1]


def _full(pixels, x_offset=0, y_offset=0):
    arr = np.asarray(pixels, dtype=np.uint8)
    return IndexedImage(arr.shape[1], arr.shape[0], x_offset, y_offset, arr)


def _trivial_mask(*masks):
    ones = np.ones((1, 1), dtype=np.uint8)
    return ViewMask(primary=ones, secondary=ones, origin=(0, 0), masks=masks)


def test_carve_splits_by_subsprite_and_crops():
    low3 = np.array([[1, 1, 2, 2], [1, 1, 2, 2]], dtype=np.uint8)
    vm = ViewMask(
        primary=low3,
        secondary=low3,
        origin=(0, 0),
        masks=(
            OutputMask(1, False, MaskOp.NONE, 0, 0),
            OutputMask(2, False, MaskOp.NONE, 5, 0),
        ),
    )
    left, right = carve(_full([[10, 11, 12, 13], [14, 15, 16, 17]]), vm)
    assert left.pixels.tolist() == [[10, 11], [14, 15]]
    assert (left.x_offset, left.y_offset) == (0, 0)
    assert right.pixels.tolist() == [[12, 13], [16, 17]]
    assert (right.x_offset, right.y_offset) == (5 + 2, 0)  # JSON offset + crop shift


def test_carve_edge_clamp_keeps_whole_for_trivial_mask():
    vm = _trivial_mask(OutputMask(1, False, MaskOp.NONE, 0, 0))
    full = _full([[1, 2, 3], [4, 5, 6], [7, 8, 9]])  # bigger than the 1x1 mask
    (only,) = carve(full, vm)
    assert only.pixels.tolist() == full.pixels.tolist()


def test_carve_empty_subsprite_returns_blank():
    low3 = np.array([[1, 1]], dtype=np.uint8)
    vm = ViewMask(
        primary=low3,
        secondary=low3,
        origin=(0, 0),
        masks=(OutputMask(2, False, MaskOp.NONE, 0, 0),),
    )
    (img,) = carve(_full([[7, 8]]), vm)
    assert (img.width, img.height) == (1, 1)
    assert img.pixels.tolist() == [[0]]


def test_carve_intersect_and_difference_split_by_silhouette():
    # A full 1x4 strip, all sub-sprite 1; silhouette covers the left two columns.
    low3 = np.ones((1, 4), dtype=np.uint8)
    vm = ViewMask(
        primary=low3,
        secondary=low3,
        origin=(0, 0),
        masks=(
            OutputMask(1, False, MaskOp.INTERSECT, 0, 0),  # front
            OutputMask(1, False, MaskOp.DIFFERENCE, 0, 0),  # behind
        ),
    )
    full = _full([[10, 11, 12, 13]])
    sil = _full([[9, 9, 0, 0]])  # inside on the left half
    front, behind = carve(full, vm, sil)
    assert front.pixels.tolist() == [[10, 11]]  # intersect = covered (left)
    assert behind.pixels.tolist() == [[12, 13]]  # difference = uncovered (right)


def test_carve_transfer_next_pulls_covered_pixels_from_next():
    # Sub-sprite 1 (cols 0-1) transfers in sub-sprite 2's (cols 2-3) silhouette-covered pixels.
    low3 = np.array([[1, 1, 2, 2]], dtype=np.uint8)
    vm = ViewMask(
        primary=low3,
        secondary=low3,
        origin=(0, 0),
        masks=(
            OutputMask(1, False, MaskOp.TRANSFER_NEXT, 0, 0),
            OutputMask(2, False, MaskOp.NONE, 0, 0),
        ),
    )
    full = _full([[10, 11, 12, 13]])
    sil = _full([[0, 0, 1, 0]])  # only col 2 covered
    first, second = carve(full, vm, sil)
    # First keeps its own region (cols 0-1) plus the covered col 2.
    assert first.pixels.tolist() == [[10, 11, 12]]
    assert second.pixels.tolist() == [[12, 13]]
