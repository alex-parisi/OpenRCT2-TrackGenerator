"""
Vectorized track-curve library, ported from RCTGen's ``track_sections.cpp``.

Each section curve is a :data:`~openrct2_track_generator.types.CurveFn`: given an
``(N,)`` array of arc-distances it returns a :class:`TrackPointArray` holding the
moving frame (position + tangent/normal/binormal) at every sample. The legacy code
evaluates one scalar distance per call; here the building blocks are numpy-vectorized
so a whole mesh's vertices (or a subposition sweep) are evaluated in one shot.

The cubic-spline sections reuse the exact polynomial coefficients from the C source,
including the ``reparameterize_old`` arc-length correction
(``3.674234614174767 * x / TILE_SIZE``).
"""

import math

import numpy as np
from numpy.typing import NDArray

from .constants import CLEARANCE_HEIGHT, TILE_SIZE
from .types import TrackPointArray

__all__ = [
    "FLAT_LENGTH",
    "FLAT_TO_GENTLE_LENGTH",
    "GENTLE_LENGTH",
    "GENTLE_TO_STEEP_LENGTH",
    "SMALL_TURN_LENGTH",
    "STEEP_LENGTH",
    "flat_curve",
    "flat_to_gentle_curve",
    "gentle_curve",
    "gentle_to_steep_curve",
    "small_turn_left_curve",
    "steep_curve",
]

# --- Section arc lengths (track_sections.cpp #defines) -----------------------
FLAT_LENGTH: float = TILE_SIZE
FLAT_TO_GENTLE_LENGTH: float = 1.027122 * TILE_SIZE
GENTLE_LENGTH: float = 1.080123 * TILE_SIZE
GENTLE_TO_STEEP_LENGTH: float = 1.314179 * TILE_SIZE
STEEP_LENGTH: float = 1.914854 * TILE_SIZE
# SMALL_TURN_LENGTH is 0.75*pi*TILE_SIZE; with the curve's angle = d/(1.5*TILE_SIZE)
# this makes the piece sweep exactly a quarter turn over its length.
SMALL_TURN_LENGTH: float = 0.75 * math.pi * TILE_SIZE

_Quad = tuple[float, float, float, float]
_Hept = tuple[float, float, float, float, float, float, float]


# --- Vectorized building blocks ----------------------------------------------
def _normalize(v: NDArray[np.float64]) -> NDArray[np.float64]:
    """Row-wise L2 normalization of an ``(N, 3)`` array (zero rows left as-is)."""
    length = np.linalg.norm(v, axis=1, keepdims=True)
    length[length == 0.0] = 1.0
    result: NDArray[np.float64] = v / length
    return result


def cubic(a: float, b: float, c: float, d: float, x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Evaluate the cubic ``a*x^3 + b*x^2 + c*x + d`` (Horner form)."""
    out: NDArray[np.float64] = x * (x * (x * a + b) + c) + d
    return out


def cubic_derivative(
    a: float, b: float, c: float, x: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Evaluate the derivative ``3a*x^2 + 2b*x + c`` of :func:`cubic`."""
    out: NDArray[np.float64] = x * (3.0 * x * a + 2.0 * b) + c
    return out


def _horner7(coeffs: _Hept, x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Evaluate the constant-free degree-7 polynomial used to reparameterize arc length."""
    a, b, c, d, e, f, g = coeffs
    out: NDArray[np.float64] = x * (x * (x * (x * (x * (x * (x * a + b) + c) + d) + e) + f) + g)
    return out


def reparameterize_old(coeffs: _Hept, x: NDArray[np.float64]) -> NDArray[np.float64]:
    """The legacy arc-length reparameterization (track_sections.cpp::reparameterize_old).

    Applies the ``3.674234614174767 / TILE_SIZE`` distance correction before the
    degree-7 polynomial, mapping true arc distance to the spline parameter ``u``.
    """
    return _horner7(coeffs, 3.674234614174767 * x / TILE_SIZE)


def plane_curve_vertical(
    position: NDArray[np.float64], tangent: NDArray[np.float64]
) -> TrackPointArray:
    """Frame for a curve confined to the vertical (Y/Z) plane.

    normal = ``(0, tz, -ty)`` (tangent rotated 90° in the Y/Z plane); binormal is the
    constant ``(-1, 0, 0)`` across-track axis (track_sections.cpp::plane_curve_vertical).
    """
    n = position.shape[0]
    normal = np.stack([np.zeros(n), tangent[:, 2], -tangent[:, 1]], axis=1)
    binormal = np.tile(np.array([-1.0, 0.0, 0.0]), (n, 1))
    return TrackPointArray(position, tangent, normal, binormal)


def cubic_curve_vertical_old(
    x_coeffs: _Quad, y_coeffs: _Quad, p: _Hept, distance: NDArray[np.float64]
) -> TrackPointArray:
    """A vertical-plane spline: cubic Z (x_coeffs) and Y (y_coeffs) in the reparam ``u``.

    Mirrors track_sections.cpp::cubic_curve_vertical_old — Z uses ``x_coeffs`` and Y
    uses ``y_coeffs`` (the legacy naming, where the spline's "x" is the along-track Z).
    """
    u = reparameterize_old(p, distance)
    xa, xb, xc, _xd = x_coeffs
    ya, yb, yc, _yd = y_coeffs
    position = np.stack(
        [np.zeros_like(u), cubic(*y_coeffs, u), cubic(*x_coeffs, u)], axis=1
    )
    tangent = _normalize(
        np.stack(
            [np.zeros_like(u), cubic_derivative(ya, yb, yc, u), cubic_derivative(xa, xb, xc, u)],
            axis=1,
        )
    )
    return plane_curve_vertical(position, tangent)


# --- Concrete section curves -------------------------------------------------
def flat_curve(distance: NDArray[np.float64]) -> TrackPointArray:
    """A level straight piece along +Z."""
    n = distance.shape[0]
    position = np.stack([np.zeros(n), np.zeros(n), distance], axis=1)
    tangent = np.tile(np.array([0.0, 0.0, 1.0]), (n, 1))
    return plane_curve_vertical(position, tangent)


def gentle_curve(distance: NDArray[np.float64]) -> TrackPointArray:
    """A constant gentle (2 height-step per tile) slope."""
    u = distance / GENTLE_LENGTH
    n = distance.shape[0]
    position = np.stack([np.zeros(n), 2.0 * CLEARANCE_HEIGHT * u, u * TILE_SIZE], axis=1)
    tangent = _normalize(
        np.tile(np.array([0.0, 2.0 * CLEARANCE_HEIGHT / TILE_SIZE, 1.0]), (n, 1))
    )
    return plane_curve_vertical(position, tangent)


def steep_curve(distance: NDArray[np.float64]) -> TrackPointArray:
    """A constant steep (8 height-step per tile) slope."""
    u = distance / STEEP_LENGTH
    n = distance.shape[0]
    position = np.stack([np.zeros(n), 8.0 * CLEARANCE_HEIGHT * u, u * TILE_SIZE], axis=1)
    tangent = _normalize(
        np.tile(np.array([0.0, 8.0 * CLEARANCE_HEIGHT / TILE_SIZE, 1.0]), (n, 1))
    )
    return plane_curve_vertical(position, tangent)


def flat_to_gentle_curve(distance: NDArray[np.float64]) -> TrackPointArray:
    """Transition from flat to a gentle slope (track_sections.cpp coefficients)."""
    return cubic_curve_vertical_old(
        (0.0, 0.0, TILE_SIZE, 0.0),
        (0.0, CLEARANCE_HEIGHT, 0.0, 0.0),
        (
            1.53990713e-09,
            -3.71353195e-07,
            5.71497773e-06,
            -2.15973089e-06,
            -5.57959067e-04,
            -9.43549275e-07,
            2.72165680e-01,
        ),
        distance,
    )


def gentle_to_steep_curve(distance: NDArray[np.float64]) -> TrackPointArray:
    """Transition from a gentle slope to a steep slope (track_sections.cpp coefficients)."""
    return cubic_curve_vertical_old(
        (-0.5 * TILE_SIZE, TILE_SIZE, 0.5 * TILE_SIZE, 0.0),
        (CLEARANCE_HEIGHT, 2.0 * CLEARANCE_HEIGHT, CLEARANCE_HEIGHT, 0.0),
        (
            1.03809332e-04,
            -2.00628254e-03,
            1.59305808e-02,
            -6.75327840e-02,
            1.68115244e-01,
            -2.67819389e-01,
            4.74022260e-01,
        ),
        distance,
    )


def small_turn_left_curve(distance: NDArray[np.float64]) -> TrackPointArray:
    """A flat 90° left turn of radius ``1.5*TILE_SIZE`` (track_sections.cpp)."""
    angle = distance / (1.5 * TILE_SIZE)
    n = distance.shape[0]
    position = np.stack(
        [1.5 * TILE_SIZE * (1.0 - np.cos(angle)), np.zeros(n), 1.5 * TILE_SIZE * np.sin(angle)],
        axis=1,
    )
    tangent = np.stack([np.sin(angle), np.zeros(n), np.cos(angle)], axis=1)
    normal = np.tile(np.array([0.0, 1.0, 0.0]), (n, 1))
    binormal = np.cross(tangent, normal)
    return TrackPointArray(position, tangent, normal, binormal)
