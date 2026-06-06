"""Lift-hill chain overlay (apply_lift) and the chain pattern data."""

import numpy as np
from openrct2_track_generator.lift import CHAIN_PATTERNS, FLAT_CHAIN, GENTLE_CHAIN, apply_lift
from openrct2_x7_renderer.types import IndexedImage


def test_chain_patterns_registry():
    assert CHAIN_PATTERNS["flat"] is FLAT_CHAIN
    assert CHAIN_PATTERNS["gentle"] is GENTLE_CHAIN
    assert len(FLAT_CHAIN) == 4 and len(GENTLE_CHAIN) == 4
    assert (FLAT_CHAIN[0].height, FLAT_CHAIN[0].width) == (6, 3)


def test_apply_lift_replaces_only_rail_pixels():
    # A constant chain stamp (all 2s) replaces rail pixels (1-3), leaves the rest.
    from openrct2_track_generator.lift import ChainPattern

    pattern = ChainPattern(0, 0, np.full((1, 1), 2, dtype=np.uint8))
    img = IndexedImage(4, 1, 0, 0, np.array([[0, 1, 3, 9]], dtype=np.uint8))
    out = apply_lift(img, pattern)
    # rails 1 and 3 -> 2; background 0 and non-rail 9 untouched.
    assert out.pixels.tolist() == [[0, 2, 2, 9]]


def test_apply_lift_tiles_pattern_by_offset():
    # A 1x2 stamp [5, 6] tiles across the row; offsets shift the phase.
    from openrct2_track_generator.lift import ChainPattern

    pattern = ChainPattern(0, 0, np.array([[5, 6]], dtype=np.uint8))
    img = IndexedImage(4, 1, 0, 0, np.array([[1, 1, 1, 1]], dtype=np.uint8))
    out = apply_lift(img, pattern)
    assert out.pixels.tolist() == [[5, 6, 5, 6]]
    # A draw-offset shifts which pattern column each pixel samples.
    shifted = apply_lift(IndexedImage(4, 1, 1, 0, np.ones((1, 4), dtype=np.uint8)), pattern)
    assert shifted.pixels.tolist() == [[6, 5, 6, 5]]
