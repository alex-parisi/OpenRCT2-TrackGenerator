"""Support-post placement geometry (track.cpp:405-437)."""

import numpy as np
from openrct2_track_generator.constants import TILE_SIZE
from openrct2_track_generator.sections import resolve_section
from openrct2_track_generator.supports import _cc, _normalize, support_posts

_SPACING = 0.8 * TILE_SIZE
_PIVOT = 0.116897727273 * TILE_SIZE


def _posts(name):
    return support_posts(resolve_section(name), 0.0, _SPACING, _PIVOT)


def test_flat_posts_count_model_and_orthonormal():
    posts = _posts("flat")
    assert len(posts) == 2  # round(FLAT_LENGTH / spacing) + 1
    assert {p.model_key for p in posts} == {"support_flat"}
    for p in posts:
        assert np.allclose(p.matrix @ p.matrix.T, np.eye(3), atol=1e-6)


def test_left_bank_uses_full_bank_model():
    # Entry+exit fully banked left -> bank step 6 at every post -> support_bank.
    assert [p.model_key for p in _posts("left_bank")] == ["support_bank", "support_bank"]


def test_bank_interpolates_entry_to_exit():
    # flat_to_right_bank: entry level (support_flat) -> exit fully banked (support_bank).
    keys = [p.model_key for p in _posts("flat_to_right_bank")]
    assert keys[0] == "support_flat"
    assert keys[-1] == "support_bank"


def test_pivot_lowers_posts_on_sloped_track():
    # On a climbing section horiz<1, so the correction pivot/horiz - pivot > 0 lowers the
    # post vs pivot=0. (On flat, horiz==1, the correction is zero — no effect.)
    sec = resolve_section("gentle")
    with_pivot = support_posts(sec, 0.0, _SPACING, _PIVOT)
    no_pivot = support_posts(sec, 0.0, _SPACING, 0.0)
    assert with_pivot[1].translation[1] < no_pivot[1].translation[1]
    flat_with = _posts("flat")
    flat_no = support_posts(resolve_section("flat"), 0.0, _SPACING, 0.0)
    assert flat_with[1].translation[1] == flat_no[1].translation[1]  # flat: no correction


def test_helpers():
    assert np.allclose(_cc(np.array([1.0, 2.0, 3.0])), [3.0, 2.0, 1.0])
    assert np.allclose(_normalize(np.array([0.0, 0.0, 0.0])), [0.0, 0.0, 0.0])  # zero stays zero
    assert np.allclose(_normalize(np.array([3.0, 0.0, 0.0])), [1.0, 0.0, 0.0])
