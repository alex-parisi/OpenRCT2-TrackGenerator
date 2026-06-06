"""Curve-math correctness: endpoints, slopes, and frame orthonormality."""

import numpy as np
from openrct2_track_generator import curves
from openrct2_track_generator.constants import CLEARANCE_HEIGHT, TILE_SIZE


def _assert_orthonormal(tp) -> None:
    n = tp.position.shape[0]
    for arr in (tp.tangent, tp.normal, tp.binormal):
        np.testing.assert_allclose(np.linalg.norm(arr, axis=1), np.ones(n), atol=1e-6)
    np.testing.assert_allclose(np.sum(tp.tangent * tp.normal, axis=1), 0.0, atol=1e-6)
    np.testing.assert_allclose(np.sum(tp.tangent * tp.binormal, axis=1), 0.0, atol=1e-6)
    np.testing.assert_allclose(np.sum(tp.normal * tp.binormal, axis=1), 0.0, atol=1e-6)


def test_flat_curve_position_and_frame():
    d = np.array([0.0, 1.0, 2.0, TILE_SIZE])
    tp = curves.flat_curve(d)
    np.testing.assert_allclose(tp.position[:, 0], 0.0)
    np.testing.assert_allclose(tp.position[:, 1], 0.0)
    np.testing.assert_allclose(tp.position[:, 2], d)
    np.testing.assert_allclose(tp.tangent, np.tile([0.0, 0.0, 1.0], (4, 1)))
    np.testing.assert_allclose(tp.normal, np.tile([0.0, 1.0, 0.0], (4, 1)))
    np.testing.assert_allclose(tp.binormal, np.tile([-1.0, 0.0, 0.0], (4, 1)))


def test_gentle_curve_slope():
    # Over one section length the piece rises 2 clearance-heights per tile.
    d = np.array([0.0, curves.GENTLE_LENGTH])
    tp = curves.gentle_curve(d)
    np.testing.assert_allclose(tp.position[1, 1], 2.0 * CLEARANCE_HEIGHT)
    np.testing.assert_allclose(tp.position[1, 2], TILE_SIZE)
    expected = np.array([0.0, 2.0 * CLEARANCE_HEIGHT / TILE_SIZE, 1.0])
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(tp.tangent[0], expected, atol=1e-9)
    _assert_orthonormal(tp)


def test_steep_curve_slope():
    d = np.array([0.0, curves.STEEP_LENGTH])
    tp = curves.steep_curve(d)
    np.testing.assert_allclose(tp.position[1, 1], 8.0 * CLEARANCE_HEIGHT)
    np.testing.assert_allclose(tp.position[1, 2], TILE_SIZE)
    _assert_orthonormal(tp)


def test_flat_to_gentle_starts_at_origin_and_rises():
    # reparameterize is constant-free, so u(0) = 0 and the cubic hits (0,0,0).
    d = np.linspace(0.0, curves.FLAT_TO_GENTLE_LENGTH, 8)
    tp = curves.flat_to_gentle_curve(d)
    np.testing.assert_allclose(tp.position[0], [0.0, 0.0, 0.0], atol=1e-6)
    assert tp.position[-1, 2] > tp.position[0, 2]  # advances along the track
    assert tp.position[-1, 1] > tp.position[0, 1]  # gains height
    _assert_orthonormal(tp)


def test_gentle_to_steep_frame_orthonormal():
    d = np.linspace(0.0, curves.GENTLE_TO_STEEP_LENGTH, 8)
    _assert_orthonormal(curves.gentle_to_steep_curve(d))


def test_small_turn_left_sweeps_quarter_turn():
    d = np.array([0.0, curves.SMALL_TURN_LENGTH])
    tp = curves.small_turn_left_curve(d)
    # Starts at the origin heading +Z.
    np.testing.assert_allclose(tp.position[0], [0.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(tp.tangent[0], [0.0, 0.0, 1.0], atol=1e-6)
    # Ends a quarter turn later heading +X, displaced by the turn radius.
    r = 1.5 * TILE_SIZE
    np.testing.assert_allclose(tp.position[1], [r, 0.0, r], atol=1e-5)
    np.testing.assert_allclose(tp.tangent[1], [1.0, 0.0, 0.0], atol=1e-6)
    _assert_orthonormal(tp)


def test_cubic_matches_scalar_reference():
    x = np.array([-1.0, 0.0, 0.5, 2.0])
    np.testing.assert_allclose(curves.cubic(2.0, -3.0, 1.0, 5.0, x), 2 * x**3 - 3 * x**2 + x + 5)
    np.testing.assert_allclose(curves.cubic_derivative(2.0, -3.0, 1.0, x), 6 * x**2 - 6 * x + 1)


def test_reparameterize_old_is_constant_free():
    # All-ones coefficients: at x=0 the degree-7 Horner with no constant term is 0.
    coeffs = (1.0,) * 7
    out = curves.reparameterize_old(coeffs, np.array([0.0]))
    np.testing.assert_allclose(out, 0.0)
    # Monotonic for positive distances.
    d = np.array([0.1, 0.2, 0.3]) * TILE_SIZE
    vals = curves.reparameterize_old(coeffs, d)
    assert np.all(np.diff(vals) > 0)
