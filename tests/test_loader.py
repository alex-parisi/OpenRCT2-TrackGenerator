"""Config -> Track loading and validation."""

import json

import pytest
from openrct2_object_common.config import LoadError
from openrct2_track_generator.loader import build_track, load_track
from openrct2_x7_renderer.mesh import Mesh


def _config(**overrides):
    base = {
        "id": "test.track.x",
        "name": "X",
        "description": "desc",
        "authors": ["A"],
        "ride_type": "steel_roller_coaster",
        "meshes": ["unused.obj"],
        "sections": ["flat", "gentle"],
    }
    base.update(overrides)
    return base


def _meshes(n=1):
    return [Mesh.empty() for _ in range(n)]


def test_build_track_minimal():
    track = build_track(_config(), _meshes())
    assert track.id == "test.track.x"
    assert track.ride_type == "steel_roller_coaster"
    assert [s.name for s in track.sections] == ["flat", "gentle"]
    assert track.units_per_tile == 3.3
    assert track.track_mesh_index == 0


def test_missing_id_raises():
    cfg = _config()
    del cfg["id"]
    with pytest.raises(LoadError, match="id"):
        build_track(cfg, _meshes())


def test_missing_ride_type_raises():
    cfg = _config()
    del cfg["ride_type"]
    with pytest.raises(LoadError, match="ride_type"):
        build_track(cfg, _meshes())


def test_unknown_section_raises():
    with pytest.raises(LoadError, match="Unknown track section"):
        build_track(_config(sections=["flat", "nonexistent"]), _meshes())


def test_empty_sections_raises():
    with pytest.raises(LoadError, match="at least one section"):
        build_track(_config(sections=[]), _meshes())


def test_non_array_sections_raises():
    with pytest.raises(LoadError, match="sections"):
        build_track(_config(sections="flat"), _meshes())


def test_non_string_section_value_raises():
    with pytest.raises(LoadError, match="non-string"):
        build_track(_config(sections=["flat", 5]), _meshes())


def test_bad_units_per_tile_raises():
    with pytest.raises(LoadError, match="units_per_tile"):
        build_track(_config(units_per_tile=0), _meshes())


def test_track_mesh_index_out_of_range_raises():
    with pytest.raises(LoadError, match="track_mesh_index"):
        build_track(_config(track_mesh_index=5), _meshes(1))


def test_mask_mesh_index_out_of_range_raises():
    with pytest.raises(LoadError, match="mask_mesh_index"):
        build_track(_config(mask_mesh_index=5), _meshes(1))


def test_mask_mesh_index_defaults_to_none():
    assert build_track(_config(), _meshes()).mask_mesh_index == -1


def test_z_offset_optional_and_read():
    assert build_track(_config(), _meshes()).z_offset == 0.0
    assert build_track(_config(z_offset=3), _meshes()).z_offset == 3.0


def test_version_override():
    track = build_track(_config(version="2.0"), _meshes())
    assert track.version == "2.0"


def test_masks_path_optional_and_override():
    assert build_track(_config(), _meshes()).masks_path == ""
    assert build_track(_config(masks="custom/m.json"), _meshes()).masks_path == "custom/m.json"


def test_load_track_from_disk(tmp_path):
    (tmp_path / "m.obj").write_text("v 0 0 0\nv 0 0 1\nv 1 0 0\nf 1 2 3\n")
    cfg = tmp_path / "track.json"
    cfg.write_text(json.dumps(_config(meshes=[str(tmp_path / "m.obj")])))
    track = load_track(cfg)
    assert track.id == "test.track.x"
    assert len(track.meshes) == 1
    assert [s.name for s in track.sections] == ["flat", "gentle"]
