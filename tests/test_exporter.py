"""Sprite manifest emission, indexed-PNG output, and subposition sidecars.

The native ray tracer is stubbed with ``FakeContext`` (shared by the other
generators): every view renders a 1x1 dummy. The deformation runs for real on the
mesh before the stubbed render, and everything downstream (PNGs, the manifest JSON,
the subposition JSON) runs against ``tmp_path``.
"""

import json

from openrct2_object_common.testing import FakeContext
from openrct2_track_generator.exporter import (
    SPRITE_VIEWS,
    expected_sprite_count,
    export_track,
    export_track_test,
)
from openrct2_track_generator.loader import build_track
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


def test_expected_sprite_count():
    track = _build_count_only()
    assert expected_sprite_count(track) == 2 * SPRITE_VIEWS


def _build_count_only():
    from openrct2_track_generator.sections import SECTION_REGISTRY
    from openrct2_track_generator.types import Track

    return Track(sections=[SECTION_REGISTRY["flat"], SECTION_REGISTRY["gentle"]])


def test_export_track_writes_manifest_pngs_and_sidecar(tmp_path):
    track = _build(tmp_path)
    out = tmp_path / "dist"
    export_track(track, FakeContext(), out)

    # Manifest: one entry per section/view, in section-major order.
    manifest_path = out / "test.track.x.sprites.json"
    manifest = json.loads(manifest_path.read_text())
    assert isinstance(manifest, list)
    assert len(manifest) == expected_sprite_count(track)
    assert manifest[0] == {"path": "images/flat_1.png", "x": 0, "y": 0, "palette": "keep"}
    assert [e["path"] for e in manifest[:SPRITE_VIEWS]] == [
        f"images/flat_{i + 1}.png" for i in range(SPRITE_VIEWS)
    ]
    assert manifest[SPRITE_VIEWS]["path"] == "images/gentle_1.png"
    assert all(e["palette"] == "keep" for e in manifest)

    # PNGs exist on disk under images/.
    for entry in manifest:
        assert (out / entry["path"]).exists()

    # Subposition sidecar.
    sidecar = json.loads((out / "test.track.x.subpositions.json").read_text())
    assert [s["section"] for s in sidecar["sections"]] == ["flat", "gentle"]


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
