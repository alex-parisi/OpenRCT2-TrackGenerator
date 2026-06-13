"""Sprite manifest emission, indexed-PNG output, and subposition sidecars.

The native ray tracer is stubbed with ``FakeContext`` (every view renders a 1x1
dummy); the real masks still carve it, so counts/ordering are exercised end to end.
The deformation runs for real before the stubbed render, and everything downstream
(PNGs, manifest JSON, subposition JSON) runs against ``tmp_path``.
"""

import json

import pytest
from openrct2_object_common.config import LoadError
from openrct2_object_common.testing import FakeContext
from openrct2_track_generator.exporter import (
    expected_sprite_count,
    export_track,
    export_track_test,
    export_tracks,
)
from openrct2_track_generator.loader import build_track, build_tracks
from openrct2_x7_renderer.mesh import load_mesh

# A tiny track tile spanning z in [0, 1] so the deform's auto-fit has a real extent.
_OBJ = "v 0 0 0\nv 0 0 1\nv 1 0 0\nf 1 2 3\n"


def _build(tmp_path, **overrides):
    (tmp_path / "m.obj").write_text(_OBJ)
    base = {
        "id": "test.track.x",
        "name": "X",
        "description": "desc",
        "authors": ["A"],
        "ride_type": "steel_roller_coaster",
        "meshes": [str(tmp_path / "m.obj")],
        "sections": ["flat", "gentle"],
    }
    base.update(overrides)
    meshes = [load_mesh(p) for p in base["meshes"]]
    return build_track(base, meshes)


def test_expected_sprite_count_is_mask_driven(tmp_path):
    # flat = 2 views x 1 sub-sprite; gentle = 4 views x 1 sub-sprite.
    track = _build(tmp_path)
    assert expected_sprite_count(track) == 2 + 4


def test_small_turn_is_twelve_sprites(tmp_path):
    # small_turn_left = 4 views x 3 tiles -> 12 sprites (the multi-tile headline).
    track = _build(tmp_path, sections=["small_turn_left"])
    assert expected_sprite_count(track) == 12


def test_mask_mesh_enables_occlusion_split(tmp_path):
    # With a mask mesh, gentle_to_steep's split views expand: views 0,3 (mask_end) = 1
    # sub-sprite each; views 1,2 (split) = 2 each (front/behind) -> 6. Exercises the
    # silhouette render path. (Mask mesh reuses the track mesh here.)
    track = _build(tmp_path, sections=["gentle_to_steep"], mask_mesh_index=0)
    out = tmp_path / "dist"
    export_track(track, FakeContext(), out)
    manifest = json.loads((out / "test.track.x.sprites.json").read_text())
    assert len(manifest) == 6 == expected_sprite_count(track)


def test_no_mask_mesh_collapses_split(tmp_path):
    # Without a mask mesh the same section is a plain carve: 4 views -> 4 sprites.
    track = _build(tmp_path, sections=["gentle_to_steep"])
    assert expected_sprite_count(track) == 4


def test_has_lift_expands_flat_to_four(tmp_path):
    # With a lift hill, flat's 2 views become 4 (one chain direction each).
    assert expected_sprite_count(_build(tmp_path, sections=["flat"])) == 2
    assert expected_sprite_count(_build(tmp_path, sections=["flat"], flags=["has_lift"])) == 4


def test_export_track_writes_manifest_pngs_and_sidecar(tmp_path):
    track = _build(tmp_path)
    out = tmp_path / "dist"
    export_track(track, FakeContext(), out)

    manifest = json.loads((out / "test.track.x.sprites.json").read_text())
    assert isinstance(manifest, list)
    assert len(manifest) == expected_sprite_count(track) == 6
    # Emission order: section-major, then view, then sub-sprite.
    assert [e["path"] for e in manifest] == [
        "images/flat_0_0.png",
        "images/flat_1_0.png",
        "images/gentle_0_0.png",
        "images/gentle_1_0.png",
        "images/gentle_2_0.png",
        "images/gentle_3_0.png",
    ]
    assert all(e["palette"] == "keep" for e in manifest)
    for entry in manifest:
        assert (out / entry["path"]).exists()

    sidecar = json.loads((out / "test.track.x.subpositions.json").read_text())
    assert [s["section"] for s in sidecar["sections"]] == ["flat", "gentle"]


def test_spritefile_in_merge_appends_after_existing(tmp_path):
    # An existing manifest is loaded and the rendered sprites appended after it, mirroring
    # maketrack placing sprites at fixed global image indices.
    out = tmp_path / "dist"
    out.mkdir()
    existing = [{"path": "preexisting_0.png", "x": 1, "y": 2, "palette": "keep"}]
    (out / "in.json").write_text(json.dumps(existing))

    track = _build(
        tmp_path, sections=["flat"], spritefile_in="in.json", spritefile_out="out.json"
    )
    export_track(track, FakeContext(), out)

    merged = json.loads((out / "out.json").read_text())
    assert merged[0] == existing[0]  # preexisting entry kept first
    assert [e["path"] for e in merged[1:]] == ["images/flat_0_0.png", "images/flat_1_0.png"]
    assert not (out / "test.track.x.sprites.json").exists()  # spritefile_out used instead


def test_spritefile_out_without_in_writes_fresh_array(tmp_path):
    out = tmp_path / "dist"
    track = _build(tmp_path, sections=["flat"], spritefile_out="sprites.json")
    export_track(track, FakeContext(), out)
    manifest = json.loads((out / "sprites.json").read_text())
    assert [e["path"] for e in manifest] == ["images/flat_0_0.png", "images/flat_1_0.png"]


def test_sprite_directory_prefixes_paths_and_pngs(tmp_path):
    out = tmp_path / "dist"
    track = _build(tmp_path, sections=["flat"], sprite_directory="object/images")
    export_track(track, FakeContext(), out)
    manifest = json.loads((out / "test.track.x.sprites.json").read_text())
    assert [e["path"] for e in manifest] == [
        "object/images/flat_0_0.png",
        "object/images/flat_1_0.png",
    ]
    for entry in manifest:
        assert (out / entry["path"]).exists()


def test_spritefile_in_not_array_raises(tmp_path):
    out = tmp_path / "dist"
    out.mkdir()
    (out / "in.json").write_text(json.dumps({"not": "an array"}))
    track = _build(tmp_path, sections=["flat"], spritefile_in="in.json")
    with pytest.raises(ValueError, match="not a JSON array"):
        export_track(track, FakeContext(), out)


def test_export_track_skip_render_is_noop(tmp_path):
    track = _build(tmp_path)
    out = tmp_path / "dist"
    ctx = FakeContext()
    export_track(track, ctx, out, skip_render=True)
    assert ctx.events == []  # nothing rendered
    assert not (out / "test.track.x.sprites.json").exists()


def test_export_track_test_writes_one_png_per_section(tmp_path):
    track = _build(tmp_path)
    test_dir = tmp_path / "test"
    export_track_test(track, FakeContext(), test_dir)
    assert (test_dir / "flat.png").exists()
    assert (test_dir / "gentle.png").exists()


# ---------- multi-track (build_tracks / export_tracks) ----------


def _multi_config(tmp_path, **overrides):
    (tmp_path / "m.obj").write_text(_OBJ)
    base = {
        "id": "test.track.x",
        "name": "X",
        "ride_type": "steel_roller_coaster",
        "meshes": [str(tmp_path / "m.obj")],
        "sections": ["flat"],
    }
    base.update(overrides)
    return base


def test_build_tracks_single_when_no_tracks_key(tmp_path):
    tracks = build_tracks(_multi_config(tmp_path))
    assert len(tracks) == 1
    assert tracks[0].suffix == ""


def test_build_tracks_inheritance_and_suffix(tmp_path):
    cfg = _multi_config(tmp_path, tracks=[{"suffix": "base"}, {"suffix": "alt"}])
    tracks = build_tracks(cfg)
    assert [t.suffix for t in tracks] == ["_base", "_alt"]
    # Entry 2 inherits id + sections from the shared top level.
    assert tracks[1].id == "test.track.x"
    assert [s.name for s in tracks[1].sections] == ["flat"]


def test_build_tracks_rejects_empty_array(tmp_path):
    with pytest.raises(LoadError, match="non-empty"):
        build_tracks(_multi_config(tmp_path, tracks=[]))


def test_build_tracks_rejects_non_object_entry(tmp_path):
    with pytest.raises(LoadError, match="non-object"):
        build_tracks(_multi_config(tmp_path, tracks=[1]))


def test_export_tracks_merges_variants_with_suffix(tmp_path):
    t0 = build_track(_multi_config(tmp_path, suffix="base"), [load_mesh(tmp_path / "m.obj")])
    t1 = build_track(_multi_config(tmp_path, suffix="alt"), [load_mesh(tmp_path / "m.obj")])
    out = tmp_path / "dist"
    export_tracks([t0, t1], FakeContext(), out)
    manifest = json.loads((out / "test.track.x.sprites.json").read_text())
    assert [e["path"] for e in manifest] == [
        "images/flat_base_0_0.png",
        "images/flat_base_1_0.png",
        "images/flat_alt_0_0.png",
        "images/flat_alt_1_0.png",
    ]


def test_export_tracks_empty_list_is_noop(tmp_path):
    out = tmp_path / "dist"
    export_tracks([], FakeContext(), out)
    assert not out.exists() or not any(out.iterdir())
