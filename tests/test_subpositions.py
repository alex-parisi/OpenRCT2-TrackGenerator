"""Subposition sampling: counts, monotonic distance, and derived angles."""

import math

import pytest
from openrct2_track_generator.constants import CLEARANCE_HEIGHT, TILE_SIZE
from openrct2_track_generator.sections import SECTION_REGISTRY
from openrct2_track_generator.subpositions import (
    build_subposition_data,
    sample_subpositions,
)


def test_sample_count_and_monotonic_distance():
    subs = sample_subpositions(SECTION_REGISTRY["gentle"], num_samples=12)
    assert len(subs) == 12
    distances = [s.distance for s in subs]
    assert distances == sorted(distances)
    assert distances[0] == 0.0
    assert math.isclose(distances[-1], SECTION_REGISTRY["gentle"].length)


def test_gentle_pitch_and_zero_roll():
    subs = sample_subpositions(SECTION_REGISTRY["gentle"])
    expected_pitch = math.atan(2.0 * CLEARANCE_HEIGHT / TILE_SIZE)
    assert all(math.isclose(s.pitch, expected_pitch, abs_tol=1e-6) for s in subs)
    assert all(abs(s.roll) < 1e-6 for s in subs)


def test_flat_has_no_pitch_or_roll():
    subs = sample_subpositions(SECTION_REGISTRY["flat"])
    assert all(abs(s.pitch) < 1e-6 and abs(s.roll) < 1e-6 for s in subs)


def test_small_turn_left_yaw_sweeps_quarter_turn():
    subs = sample_subpositions(SECTION_REGISTRY["small_turn_left"])
    assert math.isclose(subs[0].yaw, 0.0, abs_tol=1e-6)
    assert math.isclose(subs[-1].yaw, math.pi / 2, abs_tol=1e-5)
    yaws = [s.yaw for s in subs]
    assert yaws == sorted(yaws)


def test_too_few_samples_raises():
    with pytest.raises(ValueError, match="at least 2"):
        sample_subpositions(SECTION_REGISTRY["flat"], num_samples=1)


def test_build_subposition_data_shape():
    from openrct2_track_generator.types import Track

    track = Track(id="t.x", sections=[SECTION_REGISTRY["flat"], SECTION_REGISTRY["gentle"]])
    data = build_subposition_data(track, num_samples=5)
    assert data["id"] == "t.x"
    assert data["angle_units"] == "radians"
    assert [s["section"] for s in data["sections"]] == ["flat", "gentle"]
    assert len(data["sections"][0]["subpositions"]) == 5
    assert set(data["sections"][0]["subpositions"][0]) == {
        "distance",
        "x",
        "y",
        "z",
        "yaw",
        "pitch",
        "roll",
    }
