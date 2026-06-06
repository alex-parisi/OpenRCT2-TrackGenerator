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

from .constants import BANK_ANGLE, CLEARANCE_HEIGHT, TILE_SIZE
from .types import TrackPointArray

__all__ = [
    "BARREL_ROLL_LENGTH",
    "CORKSCREW_LENGTH",
    "INLINE_TWIST_LENGTH",
    "LARGE_CORKSCREW_LENGTH",
    "QUARTER_LOOP_LENGTH",
    "HALF_LOOP_LENGTH",
    "MEDIUM_HALF_LOOP_LENGTH",
    "LARGE_HALF_LOOP_LENGTH",
    "ZERO_G_ROLL_LENGTH",
    "LARGE_ZERO_G_ROLL_LENGTH",
    "DIVE_LOOP_45_LENGTH",
    "VERTICAL_TWIST_LENGTH",
    "VERTICAL_TWIST_45_LENGTH",
    "VERTICAL_LOOP_LENGTH",
    "FLAT_LENGTH",
    "FLAT_TO_GENTLE_LENGTH",
    "FLAT_TO_STEEP_LENGTH",
    "GENTLE_LENGTH",
    "GENTLE_TO_STEEP_LENGTH",
    "FLAT_DIAG_LENGTH",
    "FLAT_TO_GENTLE_DIAG_LENGTH",
    "GENTLE_DIAG_LENGTH",
    "GENTLE_TO_STEEP_DIAG_LENGTH",
    "LARGE_TURN_LENGTH",
    "MEDIUM_HELIX_LENGTH",
    "MEDIUM_TURN_LENGTH",
    "SMALL_HELIX_LENGTH",
    "SMALL_TURN_LENGTH",
    "S_BEND_LENGTH",
    "STEEP_DIAG_LENGTH",
    "STEEP_LENGTH",
    "STEEP_TO_VERTICAL_LENGTH",
    "VERTICAL_LENGTH",
    "barrel_roll_left_curve",
    "barrel_roll_right_curve",
    "corkscrew_left_curve",
    "corkscrew_right_curve",
    "flat_curve",
    "flat_to_left_bank_curve",
    "flat_to_left_bank_diag_curve",
    "flat_to_right_bank_curve",
    "flat_to_right_bank_diag_curve",
    "gentle_to_left_bank_curve",
    "gentle_to_left_bank_diag_curve",
    "gentle_to_right_bank_curve",
    "gentle_to_right_bank_diag_curve",
    "large_turn_left_to_diag_bank_curve",
    "large_turn_right_to_diag_bank_curve",
    "left_bank_curve",
    "left_bank_diag_curve",
    "left_bank_to_gentle_curve",
    "left_bank_to_gentle_diag_curve",
    "medium_helix_left_curve",
    "medium_helix_right_curve",
    "medium_turn_left_bank_curve",
    "right_bank_to_gentle_curve",
    "right_bank_to_gentle_diag_curve",
    "s_bend_left_bank_curve",
    "s_bend_left_curve",
    "s_bend_right_bank_curve",
    "s_bend_right_curve",
    "small_helix_left_curve",
    "small_helix_right_curve",
    "small_turn_left_bank_curve",
    "flat_diag_curve",
    "flat_to_gentle_curve",
    "flat_to_gentle_diag_curve",
    "flat_to_steep_curve",
    "gentle_curve",
    "gentle_diag_curve",
    "gentle_to_flat_curve",
    "gentle_to_flat_diag_curve",
    "gentle_to_steep_curve",
    "gentle_to_steep_diag_curve",
    "large_turn_left_to_diag_curve",
    "large_turn_right_to_diag_curve",
    "medium_turn_left_curve",
    "small_turn_left_curve",
    "steep_curve",
    "steep_diag_curve",
    "steep_to_flat_curve",
    "steep_to_gentle_curve",
    "steep_to_gentle_diag_curve",
    "steep_to_vertical_curve",
    "vertical_curve",
    "vertical_to_steep_curve",
]

# --- Section arc lengths (track_sections.cpp #defines) -----------------------
FLAT_LENGTH: float = TILE_SIZE
FLAT_TO_GENTLE_LENGTH: float = 1.027122 * TILE_SIZE
GENTLE_LENGTH: float = 1.080123 * TILE_SIZE
GENTLE_TO_STEEP_LENGTH: float = 1.314179 * TILE_SIZE
STEEP_LENGTH: float = 1.914854 * TILE_SIZE
FLAT_TO_STEEP_LENGTH: float = 4.792426 * TILE_SIZE
STEEP_TO_VERTICAL_LENGTH: float = 1.531568 * TILE_SIZE
VERTICAL_LENGTH: float = 0.816497 * TILE_SIZE
# SMALL_TURN_LENGTH is 0.75*pi*TILE_SIZE; with the curve's angle = d/(1.5*TILE_SIZE)
# this makes the piece sweep exactly a quarter turn over its length.
SMALL_TURN_LENGTH: float = 0.75 * math.pi * TILE_SIZE
MEDIUM_TURN_LENGTH: float = 1.25 * math.pi * TILE_SIZE
LARGE_TURN_LENGTH: float = 2.757100 * TILE_SIZE
FLAT_DIAG_LENGTH: float = 1.414213 * TILE_SIZE
FLAT_TO_GENTLE_DIAG_LENGTH: float = 1.433617 * TILE_SIZE
GENTLE_DIAG_LENGTH: float = 1.471960 * TILE_SIZE
GENTLE_TO_STEEP_DIAG_LENGTH: float = 1.656243 * TILE_SIZE
STEEP_DIAG_LENGTH: float = 2.160247 * TILE_SIZE
S_BEND_LENGTH: float = 3.240750 * TILE_SIZE
SMALL_HELIX_LENGTH: float = 2.365020 * TILE_SIZE
MEDIUM_HELIX_LENGTH: float = 3.932292 * TILE_SIZE
BARREL_ROLL_LENGTH: float = 3.091882 * TILE_SIZE
CORKSCREW_SEGMENT1_LENGTH: float = 1.682311 * TILE_SIZE
CORKSCREW_SEGMENT2_LENGTH: float = 1.744083 * TILE_SIZE
CORKSCREW_LENGTH: float = CORKSCREW_SEGMENT1_LENGTH + CORKSCREW_SEGMENT2_LENGTH
INLINE_TWIST_LENGTH: float = 3.001903 * TILE_SIZE
LARGE_CORKSCREW_SEGMENT1_LENGTH: float = 2.665302 * TILE_SIZE
LARGE_CORKSCREW_SEGMENT2_LENGTH: float = 2.665301 * TILE_SIZE
LARGE_CORKSCREW_LENGTH: float = LARGE_CORKSCREW_SEGMENT1_LENGTH + LARGE_CORKSCREW_SEGMENT2_LENGTH
QUARTER_LOOP_LENGTH: float = 4.253756 * TILE_SIZE
HALF_LOOP_SEGMENT1_LENGTH: float = 0.540062 * TILE_SIZE
HALF_LOOP_SEGMENT2_LENGTH: float = HALF_LOOP_SEGMENT1_LENGTH + 2.685141 * TILE_SIZE
HALF_LOOP_LENGTH: float = HALF_LOOP_SEGMENT2_LENGTH + 1.956695 * TILE_SIZE
_MEDIUM_HALF_LOOP_FACTOR: float = 1.0086337001020873
_MEDIUM_HALF_LOOP_SEGMENT1_LENGTH: float = 4.605006 * TILE_SIZE
_MEDIUM_HALF_LOOP_SEGMENT2_LENGTH: float = 2.988654 * TILE_SIZE
MEDIUM_HALF_LOOP_LENGTH: float = (
    (_MEDIUM_HALF_LOOP_SEGMENT1_LENGTH + _MEDIUM_HALF_LOOP_SEGMENT2_LENGTH)
    * _MEDIUM_HALF_LOOP_FACTOR
)
_LARGE_HALF_LOOP_FACTOR: float = 1.0050562625650226
_LARGE_HALF_LOOP_SEGMENT1_LENGTH: float = 1.5 * GENTLE_LENGTH
_LARGE_HALF_LOOP_SEGMENT2_LENGTH: float = _LARGE_HALF_LOOP_SEGMENT1_LENGTH + 4.766127 * TILE_SIZE
LARGE_HALF_LOOP_LENGTH: float = (
    (_LARGE_HALF_LOOP_SEGMENT2_LENGTH + 3.545350 * TILE_SIZE) * _LARGE_HALF_LOOP_FACTOR
)
ZERO_G_ROLL_BASE_LENGTH: float = 3.083249 * TILE_SIZE
ZERO_G_ROLL_LENGTH: float = 3.266924 * TILE_SIZE
LARGE_ZERO_G_ROLL_BASE_LENGTH: float = 5.385804 * TILE_SIZE
LARGE_ZERO_G_ROLL_LENGTH: float = 5.568164 * TILE_SIZE
DIVE_LOOP_45_LENGTH: float = 5.335896 * TILE_SIZE
VERTICAL_TWIST_LENGTH: float = 2.449490 * TILE_SIZE
VERTICAL_TWIST_45_LENGTH: float = 1.632993 * TILE_SIZE
_VERTICAL_LOOP_FACTOR: float = 1.006604
_VERTICAL_LOOP_SEGMENT1_LENGTH: float = 0.540062 * TILE_SIZE
_VERTICAL_LOOP_SEGMENT2_LENGTH: float = _VERTICAL_LOOP_SEGMENT1_LENGTH + 2.686603 * TILE_SIZE
VERTICAL_LOOP_LENGTH: float = (
    (_VERTICAL_LOOP_SEGMENT2_LENGTH + 1.730928 * TILE_SIZE) * _VERTICAL_LOOP_FACTOR
)

_SQRT2 = math.sqrt(2.0)
_SQRT_HALF = math.sqrt(0.5)

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


def cubic_second_derivative(a: float, b: float, x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Evaluate the second derivative ``6a*x + 2b`` of :func:`cubic`."""
    out: NDArray[np.float64] = 6.0 * x * a + 2.0 * b
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


def reparameterize(coeffs: _Hept, x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Arc-length reparameterization without the legacy ``_old`` TILE correction."""
    return _horner7(coeffs, x)


def plane_curve_horizontal(
    position: NDArray[np.float64], tangent: NDArray[np.float64]
) -> TrackPointArray:
    """Frame for a curve confined to the horizontal (X/Z) plane (turns).

    normal is the constant up axis ``(0, 1, 0)``; binormal = ``(-tz, 0, tx)``
    (track_sections.cpp::plane_curve_horizontal).
    """
    n = position.shape[0]
    normal = np.tile(np.array([0.0, 1.0, 0.0]), (n, 1))
    binormal = np.stack([-tangent[:, 2], np.zeros(n), tangent[:, 0]], axis=1)
    return TrackPointArray(position, tangent, normal, binormal)


def cubic_curve_horizontal(
    x_coeffs: _Quad, y_coeffs: _Quad, p: _Hept, distance: NDArray[np.float64]
) -> TrackPointArray:
    """A horizontal-plane spline: cubic X (y_coeffs) and Z (x_coeffs) in reparam ``u``.

    Mirrors track_sections.cpp::cubic_curve_horizontal (which uses the non-``_old``
    reparameterize): position is ``(cubic(y_coeffs), 0, cubic(x_coeffs))``.
    """
    u = reparameterize(p, distance)
    xa, xb, xc, _xd = x_coeffs
    ya, yb, yc, _yd = y_coeffs
    position = np.stack(
        [cubic(*y_coeffs, u), np.zeros_like(u), cubic(*x_coeffs, u)], axis=1
    )
    tangent = _normalize(
        np.stack(
            [cubic_derivative(ya, yb, yc, u), np.zeros_like(u), cubic_derivative(xa, xb, xc, u)],
            axis=1,
        )
    )
    return plane_curve_horizontal(position, tangent)


def bezier3d(
    x_coeffs: _Quad, y_coeffs: _Quad, z_coeffs: _Quad, roll: _Quad, p: _Hept,
    distance: NDArray[np.float64],
) -> TrackPointArray:
    """A full 3D cubic Bézier with a Frenet frame + superimposed roll (track_sections.cpp).

    Used by inversions (corkscrews, large zero-G rolls): the path is a 3D cubic, the
    natural frame comes from the first/second derivatives, and ``roll`` rolls it.
    """
    u = reparameterize(p, distance)
    point = np.stack([cubic(*x_coeffs, u), cubic(*y_coeffs, u), cubic(*z_coeffs, u)], axis=1)
    tangent = _normalize(
        np.stack(
            [
                cubic_derivative(x_coeffs[0], x_coeffs[1], x_coeffs[2], u),
                cubic_derivative(y_coeffs[0], y_coeffs[1], y_coeffs[2], u),
                cubic_derivative(z_coeffs[0], z_coeffs[1], z_coeffs[2], u),
            ],
            axis=1,
        )
    )
    second = np.stack(
        [
            cubic_second_derivative(x_coeffs[0], x_coeffs[1], u),
            cubic_second_derivative(y_coeffs[0], y_coeffs[1], u),
            cubic_second_derivative(z_coeffs[0], z_coeffs[1], u),
        ],
        axis=1,
    )
    proj = np.sum(tangent * second, axis=1)[:, None]
    fn = _normalize(second - tangent * proj)
    fb = np.cross(fn, tangent)
    angle = cubic(*roll, u)
    c = np.cos(angle)[:, None]
    s = np.sin(angle)[:, None]
    return TrackPointArray(point, tangent, fn * c + fb * s, fn * s - fb * c)


def roll_curve(
    radius: float, x_coeffs: _Quad, y_coeffs: _Quad, roll_coeffs: _Quad, p: _Hept,
    distance: NDArray[np.float64],
) -> TrackPointArray:
    """A vertical-plane spline with a superimposed roll offset by ``radius`` (track_sections.cpp).

    The unbanked path is a vertical-plane cubic (Z=``x_coeffs``, Y=``y_coeffs`` in the
    reparam ``u``); the roll angle is ``cubic(roll_coeffs, distance)`` evaluated on the
    *passed* ``distance`` (callers pass an already-reparameterized distance). The track is
    displaced by ``radius`` around the rolling frame — used by the zero-G rolls.
    """
    u = reparameterize(p, distance)
    pos = np.stack([np.zeros_like(u), cubic(*y_coeffs, u), cubic(*x_coeffs, u)], axis=1)
    tan = _normalize(
        np.stack(
            [
                np.zeros_like(u),
                cubic_derivative(y_coeffs[0], y_coeffs[1], y_coeffs[2], u),
                cubic_derivative(x_coeffs[0], x_coeffs[1], x_coeffs[2], u),
            ],
            axis=1,
        )
    )
    unbanked = plane_curve_vertical(pos, tan)
    roll = cubic(*roll_coeffs, distance)
    roll_rate = cubic_derivative(roll_coeffs[0], roll_coeffs[1], roll_coeffs[2], distance)
    cosr = np.cos(roll)[:, None]
    sinr = np.sin(roll)[:, None]
    rr = roll_rate[:, None]
    position = (
        unbanked.position
        + unbanked.normal * (radius * (1 - cosr))
        + unbanked.binormal * (radius * sinr)
    )
    tangent = _normalize(
        unbanked.tangent
        + unbanked.normal * (radius * rr * sinr)
        + unbanked.binormal * (radius * rr * cosr)
    )
    normal = unbanked.normal * cosr - unbanked.binormal * sinr
    return TrackPointArray(position, tangent, normal, np.cross(tangent, normal))


def _select(
    mask: NDArray[np.bool_], a: TrackPointArray, b: TrackPointArray
) -> TrackPointArray:
    """Per-sample pick between two frames (``True`` -> ``a``) for piecewise curves."""
    m = mask[:, None]
    return TrackPointArray(
        np.where(m, a.position, b.position),
        np.where(m, a.tangent, b.tangent),
        np.where(m, a.normal, b.normal),
        np.where(m, a.binormal, b.binormal),
    )


def banked_curve(tp: TrackPointArray, angle: NDArray[np.float64]) -> TrackPointArray:
    """Roll a curve's frame about its tangent by ``angle`` radians (per-sample).

    Rotates normal/binormal (position + tangent unchanged), matching
    track_sections.cpp::banked_curve — the building block for every banked piece.
    """
    c = np.cos(angle)[:, None]
    s = np.sin(angle)[:, None]
    normal = tp.normal * c - tp.binormal * s
    binormal = tp.normal * s + tp.binormal * c
    return TrackPointArray(tp.position, tp.tangent, normal, binormal)


def plane_curve_vertical_diagonal(
    position: NDArray[np.float64], tangent: NDArray[np.float64]
) -> TrackPointArray:
    """Frame for a 45°-diagonal vertical-plane curve (track_sections.cpp).

    normal = ``(ty/√2, tz·√2, -ty/√2)``; binormal is the constant diagonal across-track
    axis ``(-√0.5, 0, -√0.5)``.
    """
    n = position.shape[0]
    normal = np.stack(
        [tangent[:, 1] / _SQRT2, tangent[:, 2] * _SQRT2, -tangent[:, 1] / _SQRT2], axis=1
    )
    binormal = np.tile(np.array([-_SQRT_HALF, 0.0, -_SQRT_HALF]), (n, 1))
    return TrackPointArray(position, tangent, normal, binormal)


def cubic_curve_vertical_diagonal_old(
    x_coeffs: _Quad, y_coeffs: _Quad, p: _Hept, distance: NDArray[np.float64]
) -> TrackPointArray:
    """A 45°-diagonal vertical-plane spline (track_sections.cpp::cubic_curve_vertical_diagonal_old).

    The cubic's "x" is the along-diagonal distance (split into ∓x/√2, +x/√2 across the
    two horizontal axes); "y" is height. Uses the ``_old`` reparameterization.
    """
    u = reparameterize_old(p, distance)
    xa, xb, xc, _xd = x_coeffs
    ya, yb, yc, _yd = y_coeffs
    x = cubic(*x_coeffs, u)
    y = cubic(*y_coeffs, u)
    dx = cubic_derivative(xa, xb, xc, u)
    dy = cubic_derivative(ya, yb, yc, u)
    position = np.stack([-x / _SQRT2, y, x / _SQRT2], axis=1)
    tangent = _normalize(np.stack([-dx / _SQRT2, dy, dx / _SQRT2], axis=1))
    return plane_curve_vertical_diagonal(position, tangent)


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


def cubic_curve_vertical(
    x_coeffs: _Quad, y_coeffs: _Quad, p: _Hept, distance: NDArray[np.float64]
) -> TrackPointArray:
    """A vertical-plane spline like :func:`cubic_curve_vertical_old` but using the
    non-legacy ``reparameterize`` (no TILE correction). Used by the vertical loop's
    closing arc (track_sections.cpp::cubic_curve_vertical)."""
    u = reparameterize(p, distance)
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


def gentle_to_flat_curve(distance: NDArray[np.float64]) -> TrackPointArray:
    """Transition from a gentle slope back to flat (track_sections.cpp coefficients)."""
    return cubic_curve_vertical_old(
        (0.0, 0.0, TILE_SIZE, 0.0),
        (0.0, -CLEARANCE_HEIGHT, 2.0 * CLEARANCE_HEIGHT, 0.0),
        (
            1.53990713e-09, -3.71353195e-07, 5.71497773e-06, -2.15973089e-06,
            -5.57959067e-04, -9.43549275e-07, 2.72165680e-01,
        ),
        distance,
    )


def steep_to_gentle_curve(distance: NDArray[np.float64]) -> TrackPointArray:
    """Transition from a steep slope to a gentle slope (track_sections.cpp coefficients)."""
    return cubic_curve_vertical_old(
        (-0.5 * TILE_SIZE, 0.5 * TILE_SIZE, TILE_SIZE, 0.0),
        (CLEARANCE_HEIGHT, -5.0 * CLEARANCE_HEIGHT, 8.0 * CLEARANCE_HEIGHT, 0.0),
        (
            1.03809332e-04, -1.50249574e-03, 8.63282156e-03, -2.44630312e-02,
            3.57654761e-02, -1.76496309e-02, 1.47759227e-01,
        ),
        distance,
    )


def flat_to_steep_curve(distance: NDArray[np.float64]) -> TrackPointArray:
    """Large transition from flat directly to a steep slope (track_sections.cpp)."""
    return cubic_curve_vertical_old(
        (-0.5 * TILE_SIZE, -0.5 * TILE_SIZE, 5.0 * TILE_SIZE, 0.0),
        (-2.0 * CLEARANCE_HEIGHT, 13.0 * CLEARANCE_HEIGHT, 0.0, 0.0),
        (
            9.15921042e-12, -7.64176033e-10, 1.91775506e-08, -6.20557903e-08,
            -1.07682192e-05, 2.95991517e-04, 5.44333126e-02,
        ),
        distance,
    )


def steep_to_flat_curve(distance: NDArray[np.float64]) -> TrackPointArray:
    """Large transition from a steep slope back to flat (track_sections.cpp)."""
    return cubic_curve_vertical_old(
        (-0.5 * TILE_SIZE, 2.0 * TILE_SIZE, 2.5 * TILE_SIZE, 0.0),
        (-2.0 * CLEARANCE_HEIGHT, -7.0 * CLEARANCE_HEIGHT, 20.0 * CLEARANCE_HEIGHT, 0.0),
        (
            9.15921034e-12, -3.64783573e-10, -1.92055403e-09, 1.77492141e-07,
            -8.30160814e-06, 1.17635683e-04, 5.68534428e-02,
        ),
        distance,
    )


def vertical_curve(distance: NDArray[np.float64]) -> TrackPointArray:
    """A vertical (90°) section: the track goes straight up."""
    n = distance.shape[0]
    position = np.stack([np.zeros(n), distance, np.zeros(n)], axis=1)
    tangent = np.tile(np.array([0.0, 1.0, 0.0]), (n, 1))
    return plane_curve_vertical(position, tangent)


def steep_to_vertical_curve(distance: NDArray[np.float64]) -> TrackPointArray:
    """Transition from a steep slope to vertical (track_sections.cpp)."""
    return cubic_curve_vertical_old(
        (-TILE_SIZE / 6, -TILE_SIZE / 6, 5.0 * TILE_SIZE / 6, -TILE_SIZE / 2),
        (2.0 * CLEARANCE_HEIGHT / 3, -CLEARANCE_HEIGHT / 3, 20.0 * CLEARANCE_HEIGHT / 3, 0.0),
        (
            -1.27409679e-07, 1.66746015e-06, -6.60784097e-06, 2.87367949e-06,
            5.24290249e-04, -1.54572818e-03, 1.70550125e-01,
        ),
        distance,
    )


def vertical_to_steep_curve(distance: NDArray[np.float64]) -> TrackPointArray:
    """Transition from vertical back to a steep slope (track_sections.cpp)."""
    return cubic_curve_vertical_old(
        (-TILE_SIZE / 6, 2.0 * TILE_SIZE / 3, 0.0, 0.0),
        (-2.0 * CLEARANCE_HEIGHT / 3, CLEARANCE_HEIGHT, 20.0 * CLEARANCE_HEIGHT / 3, 0.0),
        (
            -1.27409680e-07, 3.35138224e-06, -3.50358430e-05, 1.85655626e-04,
            -3.24817476e-05, -6.05934285e-03, 2.00014155e-01,
        ),
        distance,
    )


def _flat_turn_left(distance: NDArray[np.float64], radius: float) -> TrackPointArray:
    """A flat 90° left turn of the given radius (shared by small/medium turns)."""
    angle = distance / radius
    n = distance.shape[0]
    position = np.stack(
        [radius * (1.0 - np.cos(angle)), np.zeros(n), radius * np.sin(angle)], axis=1
    )
    tangent = np.stack([np.sin(angle), np.zeros(n), np.cos(angle)], axis=1)
    normal = np.tile(np.array([0.0, 1.0, 0.0]), (n, 1))
    binormal = np.cross(tangent, normal)
    return TrackPointArray(position, tangent, normal, binormal)


def small_turn_left_curve(distance: NDArray[np.float64]) -> TrackPointArray:
    """A flat 90° left turn of radius ``1.5*TILE_SIZE`` (track_sections.cpp)."""
    return _flat_turn_left(distance, 1.5 * TILE_SIZE)


def medium_turn_left_curve(distance: NDArray[np.float64]) -> TrackPointArray:
    """A flat 90° left turn of radius ``2.5*TILE_SIZE`` (track_sections.cpp)."""
    return _flat_turn_left(distance, 2.5 * TILE_SIZE)


def large_turn_left_to_diag_curve(distance: NDArray[np.float64]) -> TrackPointArray:
    """A large left turn transitioning from orthogonal to diagonal (track_sections.cpp)."""
    return cubic_curve_horizontal(
        (
            68 * CLEARANCE_HEIGHT / 3 - 5 * TILE_SIZE,
            7.5 * TILE_SIZE - 112 * CLEARANCE_HEIGHT / 3,
            44 * CLEARANCE_HEIGHT / 3,
            0.0,
        ),
        (8 * CLEARANCE_HEIGHT - 2 * TILE_SIZE, 3 * TILE_SIZE - 8 * CLEARANCE_HEIGHT, 0.0, 0.0),
        (
            4.13551374e-09, -6.73303292e-08, 5.27398800e-07, 4.75914480e-06,
            -4.03664043e-06, 4.03835068e-04, 1.01222332e-01,
        ),
        distance,
    )


def large_turn_right_to_diag_curve(distance: NDArray[np.float64]) -> TrackPointArray:
    """A large right turn transitioning from orthogonal to diagonal (track_sections.cpp).

    Same as the left variant with the across-track (Y-coefficient) component negated.
    """
    return cubic_curve_horizontal(
        (
            68 * CLEARANCE_HEIGHT / 3 - 5 * TILE_SIZE,
            7.5 * TILE_SIZE - 112 * CLEARANCE_HEIGHT / 3,
            44 * CLEARANCE_HEIGHT / 3,
            0.0,
        ),
        (2 * TILE_SIZE - 8 * CLEARANCE_HEIGHT, 8 * CLEARANCE_HEIGHT - 3 * TILE_SIZE, 0.0, 0.0),
        (
            4.13551374e-09, -6.73303292e-08, 5.27398800e-07, 4.75914480e-06,
            -4.03664043e-06, 4.03835068e-04, 1.01222332e-01,
        ),
        distance,
    )


def flat_diag_curve(distance: NDArray[np.float64]) -> TrackPointArray:
    """A flat diagonal (45°) straight (track_sections.cpp)."""
    n = distance.shape[0]
    position = np.stack([-distance / _SQRT2, np.zeros(n), distance / _SQRT2], axis=1)
    tangent = np.tile(np.array([-_SQRT_HALF, 0.0, _SQRT_HALF]), (n, 1))
    return plane_curve_horizontal(position, tangent)


def gentle_diag_curve(distance: NDArray[np.float64]) -> TrackPointArray:
    """A constant gentle diagonal slope (track_sections.cpp)."""
    u = distance / GENTLE_DIAG_LENGTH
    n = distance.shape[0]
    position = np.stack([-TILE_SIZE * u, 2.0 * CLEARANCE_HEIGHT * u, TILE_SIZE * u], axis=1)
    tangent = _normalize(
        np.tile(
            np.array([-1.0 / _SQRT2, 2.0 * CLEARANCE_HEIGHT / (_SQRT2 * TILE_SIZE), 1.0 / _SQRT2]),
            (n, 1),
        )
    )
    return plane_curve_vertical_diagonal(position, tangent)


def steep_diag_curve(distance: NDArray[np.float64]) -> TrackPointArray:
    """A constant steep diagonal slope (track_sections.cpp)."""
    u = distance / STEEP_DIAG_LENGTH
    n = distance.shape[0]
    position = np.stack([-TILE_SIZE * u, 8.0 * CLEARANCE_HEIGHT * u, TILE_SIZE * u], axis=1)
    tangent = _normalize(
        np.tile(np.array([-1.0, 8.0 * CLEARANCE_HEIGHT / TILE_SIZE, 1.0]), (n, 1))
    )
    return plane_curve_vertical_diagonal(position, tangent)


def flat_to_gentle_diag_curve(distance: NDArray[np.float64]) -> TrackPointArray:
    """Diagonal transition flat -> gentle (track_sections.cpp)."""
    return cubic_curve_vertical_diagonal_old(
        (0.0, 0.0, FLAT_DIAG_LENGTH, 0.0),
        (0.0, CLEARANCE_HEIGHT, 0.0, 0.0),
        (
            -1.92303122e-10, -3.87922510e-09, 2.17273953e-07, -4.77178399e-08,
            -9.89319820e-05, -4.25613783e-08, 1.92450100e-01,
        ),
        distance,
    )


def gentle_to_flat_diag_curve(distance: NDArray[np.float64]) -> TrackPointArray:
    """Diagonal transition gentle -> flat (track_sections.cpp)."""
    return cubic_curve_vertical_diagonal_old(
        (0.0, 0.0, FLAT_DIAG_LENGTH, 0.0),
        (0.0, -CLEARANCE_HEIGHT, 2.0 * CLEARANCE_HEIGHT, 0.0),
        (
            -1.92302736e-10, 1.09698407e-08, -1.73760158e-08, -3.07650071e-06,
            -5.61731001e-05, 1.31496758e-03, 1.84900055e-01,
        ),
        distance,
    )


def gentle_to_steep_diag_curve(distance: NDArray[np.float64]) -> TrackPointArray:
    """Diagonal transition gentle -> steep (track_sections.cpp)."""
    return cubic_curve_vertical_diagonal_old(
        (-0.5 * FLAT_DIAG_LENGTH, FLAT_DIAG_LENGTH, 0.5 * FLAT_DIAG_LENGTH, 0.0),
        (CLEARANCE_HEIGHT, 2.0 * CLEARANCE_HEIGHT, CLEARANCE_HEIGHT, 0.0),
        (
            1.78635273e-05, -4.35933078e-04, 4.37240480e-03, -2.34451422e-02,
            7.41421344e-02, -1.50070476e-01, 3.50024806e-01,
        ),
        distance,
    )


def steep_to_gentle_diag_curve(distance: NDArray[np.float64]) -> TrackPointArray:
    """Diagonal transition steep -> gentle (track_sections.cpp)."""
    return cubic_curve_vertical_diagonal_old(
        (-0.5 * FLAT_DIAG_LENGTH, 0.5 * FLAT_DIAG_LENGTH, FLAT_DIAG_LENGTH, 0.0),
        (CLEARANCE_HEIGHT, -5.0 * CLEARANCE_HEIGHT, 8.0 * CLEARANCE_HEIGHT, 0.0),
        (
            1.78635273e-05, -3.25016888e-04, 2.34748867e-03, -8.33887962e-03,
            1.52652332e-02, -1.07964578e-02, 1.29831889e-01,
        ),
        distance,
    )


def _const_bank(distance: NDArray[np.float64], sign: float = 1.0) -> NDArray[np.float64]:
    return np.full(distance.shape[0], sign * BANK_ANGLE)


# --- Banking (compose banked_curve over the existing base curves) ---
def flat_to_left_bank_curve(distance: NDArray[np.float64]) -> TrackPointArray:
    """Flat track banking up to a left bank over the tile (track_sections.cpp)."""
    return banked_curve(flat_curve(distance), BANK_ANGLE * distance / FLAT_LENGTH)


def flat_to_right_bank_curve(distance: NDArray[np.float64]) -> TrackPointArray:
    """Flat track banking up to a right bank over the tile."""
    return banked_curve(flat_curve(distance), -BANK_ANGLE * distance / FLAT_LENGTH)


def left_bank_curve(distance: NDArray[np.float64]) -> TrackPointArray:
    """A fully left-banked flat straight."""
    return banked_curve(flat_curve(distance), _const_bank(distance))


def left_bank_to_gentle_curve(distance: NDArray[np.float64]) -> TrackPointArray:
    """Left bank easing out as the track rises to a gentle slope."""
    return banked_curve(
        flat_to_gentle_curve(distance), BANK_ANGLE * (1.0 - distance / FLAT_TO_GENTLE_LENGTH)
    )


def right_bank_to_gentle_curve(distance: NDArray[np.float64]) -> TrackPointArray:
    """Right bank easing out as the track rises to a gentle slope."""
    return banked_curve(
        flat_to_gentle_curve(distance), -BANK_ANGLE * (1.0 - distance / FLAT_TO_GENTLE_LENGTH)
    )


def gentle_to_left_bank_curve(distance: NDArray[np.float64]) -> TrackPointArray:
    """Gentle slope easing down to a left-banked flat."""
    return banked_curve(
        gentle_to_flat_curve(distance), BANK_ANGLE * distance / FLAT_TO_GENTLE_LENGTH
    )


def gentle_to_right_bank_curve(distance: NDArray[np.float64]) -> TrackPointArray:
    """Gentle slope easing down to a right-banked flat."""
    return banked_curve(
        gentle_to_flat_curve(distance), -BANK_ANGLE * distance / FLAT_TO_GENTLE_LENGTH
    )


def small_turn_left_bank_curve(distance: NDArray[np.float64]) -> TrackPointArray:
    """A small left turn, fully banked."""
    return banked_curve(small_turn_left_curve(distance), _const_bank(distance))


def medium_turn_left_bank_curve(distance: NDArray[np.float64]) -> TrackPointArray:
    """A medium left turn, fully banked."""
    return banked_curve(medium_turn_left_curve(distance), _const_bank(distance))


def large_turn_left_to_diag_bank_curve(distance: NDArray[np.float64]) -> TrackPointArray:
    """A large left orthogonal->diagonal turn, fully banked."""
    return banked_curve(large_turn_left_to_diag_curve(distance), _const_bank(distance))


def large_turn_right_to_diag_bank_curve(distance: NDArray[np.float64]) -> TrackPointArray:
    """A large right orthogonal->diagonal turn, fully banked."""
    return banked_curve(large_turn_right_to_diag_curve(distance), _const_bank(distance, -1.0))


# --- S-bends (horizontal cubic) ---
def s_bend_left_curve(distance: NDArray[np.float64]) -> TrackPointArray:
    """An S-bend stepping one tile left then back (track_sections.cpp)."""
    return cubic_curve_horizontal(
        (
            152 * CLEARANCE_HEIGHT / 3 - 6 * TILE_SIZE,
            9 * TILE_SIZE - 76 * CLEARANCE_HEIGHT,
            76 * CLEARANCE_HEIGHT / 3,
            0.0,
        ),
        (-2 * TILE_SIZE, 3 * TILE_SIZE, 0.0, 0.0),
        (
            -3.83794701e-07, 1.43656901e-05, -1.92240010e-04, 1.03219045e-03,
            -1.92626934e-03, 6.80531878e-03, 5.77160750e-02,
        ),
        distance,
    )


def s_bend_right_curve(distance: NDArray[np.float64]) -> TrackPointArray:
    """An S-bend stepping one tile right then back (the left S-bend mirrored in X)."""
    return cubic_curve_horizontal(
        (
            152 * CLEARANCE_HEIGHT / 3 - 6 * TILE_SIZE,
            9 * TILE_SIZE - 76 * CLEARANCE_HEIGHT,
            76 * CLEARANCE_HEIGHT / 3,
            0.0,
        ),
        (2 * TILE_SIZE, -3 * TILE_SIZE, 0.0, 0.0),
        (
            -3.83794701e-07, 1.43656901e-05, -1.92240010e-04, 1.03219045e-03,
            -1.92626934e-03, 6.80531878e-03, 5.77160750e-02,
        ),
        distance,
    )


def s_bend_left_bank_curve(distance: NDArray[np.float64]) -> TrackPointArray:
    """A left S-bend that banks then unbanks across it (track_sections.cpp)."""
    return banked_curve(
        s_bend_left_curve(distance), BANK_ANGLE * (1.0 - 2.0 * distance / S_BEND_LENGTH)
    )


def s_bend_right_bank_curve(distance: NDArray[np.float64]) -> TrackPointArray:
    """A right S-bend that banks then unbanks across it (track_sections.cpp)."""
    return banked_curve(
        s_bend_right_curve(distance), -BANK_ANGLE * (1.0 - 2.0 * distance / S_BEND_LENGTH)
    )


# --- Diagonal banking (banked_curve over the diagonal base curves) ---
def flat_to_left_bank_diag_curve(distance: NDArray[np.float64]) -> TrackPointArray:
    return banked_curve(flat_diag_curve(distance), BANK_ANGLE * distance / FLAT_DIAG_LENGTH)


def flat_to_right_bank_diag_curve(distance: NDArray[np.float64]) -> TrackPointArray:
    return banked_curve(flat_diag_curve(distance), -BANK_ANGLE * distance / FLAT_DIAG_LENGTH)


def left_bank_diag_curve(distance: NDArray[np.float64]) -> TrackPointArray:
    return banked_curve(flat_diag_curve(distance), _const_bank(distance))


def left_bank_to_gentle_diag_curve(distance: NDArray[np.float64]) -> TrackPointArray:
    return banked_curve(
        flat_to_gentle_diag_curve(distance),
        BANK_ANGLE * (1.0 - distance / FLAT_TO_GENTLE_DIAG_LENGTH),
    )


def right_bank_to_gentle_diag_curve(distance: NDArray[np.float64]) -> TrackPointArray:
    return banked_curve(
        flat_to_gentle_diag_curve(distance),
        -BANK_ANGLE * (1.0 - distance / FLAT_TO_GENTLE_DIAG_LENGTH),
    )


def gentle_to_left_bank_diag_curve(distance: NDArray[np.float64]) -> TrackPointArray:
    return banked_curve(
        gentle_to_flat_diag_curve(distance), BANK_ANGLE * distance / FLAT_TO_GENTLE_DIAG_LENGTH
    )


def gentle_to_right_bank_diag_curve(distance: NDArray[np.float64]) -> TrackPointArray:
    return banked_curve(
        gentle_to_flat_diag_curve(distance), -BANK_ANGLE * distance / FLAT_TO_GENTLE_DIAG_LENGTH
    )


# --- Helices (a banked sloped turn) ---
def _sloped_turn(
    distance: NDArray[np.float64], radius: float, gradient: float, sign: float
) -> TrackPointArray:
    """A turn that also climbs (track_sections.cpp::sloped_turn_*); ``sign`` +1 left / -1 right."""
    g2 = math.sqrt(1.0 + gradient * gradient)
    angle = distance / (radius * g2)
    n = distance.shape[0]
    position = np.stack(
        [sign * radius * (1.0 - np.cos(angle)), angle * radius * gradient, radius * np.sin(angle)],
        axis=1,
    )
    tz = 1.0 / g2
    ty = gradient / g2
    tangent = _normalize(
        np.stack([sign * tz * np.sin(angle), np.full(n, ty), tz * np.cos(angle)], axis=1)
    )
    normal = _normalize(
        np.stack([-sign * ty * np.sin(angle), np.full(n, tz), -ty * np.cos(angle)], axis=1)
    )
    binormal = np.cross(tangent, normal)
    return TrackPointArray(position, tangent, normal, binormal)


def small_helix_left_curve(distance: NDArray[np.float64]) -> TrackPointArray:
    grad = CLEARANCE_HEIGHT / (0.75 * math.pi * TILE_SIZE)
    return banked_curve(_sloped_turn(distance, 1.5 * TILE_SIZE, grad, 1.0), _const_bank(distance))


def small_helix_right_curve(distance: NDArray[np.float64]) -> TrackPointArray:
    grad = CLEARANCE_HEIGHT / (0.75 * math.pi * TILE_SIZE)
    return banked_curve(
        _sloped_turn(distance, 1.5 * TILE_SIZE, grad, -1.0), _const_bank(distance, -1.0)
    )


def medium_helix_left_curve(distance: NDArray[np.float64]) -> TrackPointArray:
    grad = CLEARANCE_HEIGHT / (1.25 * math.pi * TILE_SIZE)
    return banked_curve(_sloped_turn(distance, 2.5 * TILE_SIZE, grad, 1.0), _const_bank(distance))


def medium_helix_right_curve(distance: NDArray[np.float64]) -> TrackPointArray:
    grad = CLEARANCE_HEIGHT / (1.25 * math.pi * TILE_SIZE)
    return banked_curve(
        _sloped_turn(distance, 2.5 * TILE_SIZE, grad, -1.0), _const_bank(distance, -1.0)
    )


# --- Inversions: barrel rolls (analytic) and corkscrews (piecewise bezier3d) ---
_MIRROR_X = np.array([-1.0, 1.0, 1.0])


def barrel_roll_left_curve(distance: NDArray[np.float64]) -> TrackPointArray:
    """A left barrel roll: a 360°-ish roll while advancing 3 tiles (track_sections.cpp)."""
    n = distance.shape[0]
    u = distance / BARREL_ROLL_LENGTH
    radius = 7 * CLEARANCE_HEIGHT / 6
    pu = math.pi * u
    position = np.stack(
        [-radius * np.sin(pu), radius * (1.0 - np.cos(pu)), 3 * TILE_SIZE * u], axis=1
    )
    tangent = _normalize(
        np.stack(
            [
                -radius * math.pi * np.cos(pu) / BARREL_ROLL_LENGTH,
                radius * math.pi * np.sin(pu) / BARREL_ROLL_LENGTH,
                np.ones(n),
            ],
            axis=1,
        )
    )
    ends = (distance < 1e-4) | (distance > BARREL_ROLL_LENGTH - 1e-4)
    tangent[ends] = [0.0, 0.0, 1.0]
    normal = np.stack([np.sin(pu), np.cos(pu), np.zeros(n)], axis=1)
    binormal = np.cross(tangent, normal)
    return TrackPointArray(position, tangent, normal, binormal)


def barrel_roll_right_curve(distance: NDArray[np.float64]) -> TrackPointArray:
    """A right barrel roll: the left roll mirrored across the track."""
    left = barrel_roll_left_curve(distance)
    tangent = left.tangent * _MIRROR_X
    ends = (distance < 1e-4) | (distance > BARREL_ROLL_LENGTH - 1e-4)
    tangent[ends] = [0.0, 0.0, 1.0]
    normal = left.normal * _MIRROR_X
    return TrackPointArray(left.position * _MIRROR_X, tangent, normal, np.cross(tangent, normal))


def corkscrew_left_curve(distance: NDArray[np.float64]) -> TrackPointArray:
    """A left corkscrew inversion: two 3D Bézier segments (track_sections.cpp)."""
    seg1 = bezier3d(
        (1.030829, -0.535829, 0.0, 0.0),
        (-1.571756, 4.378463, 0.0, 0.0),
        (0.535829, -3.505829, 7.425000, 0.0),
        (0.104345, -0.906517, 0.500000, 0.121773),
        (
            8.82243634e-08, 2.23254996e-06, -4.18114465e-05, 6.87173643e-05,
            1.08563628e-04, 8.30764584e-03, 1.44263227e-01,
        ),
        distance,
    )
    seg2 = bezier3d(
        (0.535829, 1.898342, 2.020829, 0.495000),
        (-1.571756, 0.336805, 4.041658, 2.806707),
        (1.030829, -2.556658, 2.020829, 4.455000),
        (0.729345, -0.031517, -1.000000, 0.180399),
        (
            -8.84322683e-07, 1.94659293e-05, -1.48815075e-04, 4.72284342e-04,
            -1.52382973e-03, 4.02066359e-03, 1.83543852e-01,
        ),
        distance - CORKSCREW_SEGMENT1_LENGTH,
    )
    return _select(distance < CORKSCREW_SEGMENT1_LENGTH, seg1, seg2)


def corkscrew_right_curve(distance: NDArray[np.float64]) -> TrackPointArray:
    """A right corkscrew: the left corkscrew mirrored (track_sections.cpp)."""
    left = corkscrew_left_curve(distance)
    return TrackPointArray(
        left.position * _MIRROR_X,
        left.tangent * _MIRROR_X,
        left.normal * _MIRROR_X,
        left.binormal * np.array([1.0, -1.0, -1.0]),
    )


def inline_twist_left_curve(distance: NDArray[np.float64]) -> TrackPointArray:
    """A left in-line twist: a gentle roll over 3 tiles (track_sections.cpp).

    Same analytic family as the barrel roll but with a much smaller radius
    (``CLEARANCE_HEIGHT/6`` vs ``7·CLEARANCE_HEIGHT/6``), so the rail barely bows.
    """
    n = distance.shape[0]
    u = distance / INLINE_TWIST_LENGTH
    radius = CLEARANCE_HEIGHT / 6
    pu = math.pi * u
    position = np.stack(
        [-radius * np.sin(pu), radius * (1.0 - np.cos(pu)), 3 * TILE_SIZE * u], axis=1
    )
    tangent = _normalize(
        np.stack(
            [
                -radius * math.pi * np.cos(pu) / INLINE_TWIST_LENGTH,
                radius * math.pi * np.sin(pu) / INLINE_TWIST_LENGTH,
                np.ones(n),
            ],
            axis=1,
        )
    )
    ends = (distance < 1e-4) | (distance > INLINE_TWIST_LENGTH - 1e-4)
    tangent[ends] = [0.0, 0.0, 1.0]
    normal = np.stack([np.sin(pu), np.cos(pu), np.zeros(n)], axis=1)
    return TrackPointArray(position, tangent, normal, np.cross(tangent, normal))


def inline_twist_right_curve(distance: NDArray[np.float64]) -> TrackPointArray:
    """A right in-line twist: the left twist mirrored across the track."""
    left = inline_twist_left_curve(distance)
    tangent = left.tangent * _MIRROR_X
    ends = (distance < 1e-4) | (distance > INLINE_TWIST_LENGTH - 1e-4)
    tangent[ends] = [0.0, 0.0, 1.0]
    normal = left.normal * _MIRROR_X
    return TrackPointArray(left.position * _MIRROR_X, tangent, normal, np.cross(tangent, normal))


def large_corkscrew_left_curve(distance: NDArray[np.float64]) -> TrackPointArray:
    """A left large corkscrew: two 3D Bézier segments, like the corkscrew but bigger."""
    seg1 = bezier3d(
        (0.933626, 0.221374, 0.0, 0.0),
        (-2.469325, 6.623252, 0.0, 0.0),
        (0.438626, -4.728626, 11.385000, 0.0),
        (0.474996, -0.837494, 0.250000, -0.033411),
        (
            -2.12934033e-08, 5.76526391e-07, -4.79935988e-06, 5.11070674e-06,
            2.88738739e-05, 3.22777231e-03, 8.78272624e-02,
        ),
        distance,
    )
    seg2 = bezier3d(
        (0.438626, 3.412747, 3.243626, 1.155000),
        (-2.469325, 0.784724, 5.838527, 4.153926),
        (0.933626, -3.022253, 3.243626, 7.095000),
        (1.376400, -1.314599, 0.000000, -0.028389),
        (
            -2.12934347e-08, 7.34476868e-07, -8.96710411e-06, 4.40460099e-05,
            -1.18670859e-04, -1.92288999e-03, 1.34679889e-01,
        ),
        distance - LARGE_CORKSCREW_SEGMENT1_LENGTH,
    )
    return _select(distance < LARGE_CORKSCREW_SEGMENT1_LENGTH, seg1, seg2)


def large_corkscrew_right_curve(distance: NDArray[np.float64]) -> TrackPointArray:
    """A right large corkscrew: the left large corkscrew mirrored (track_sections.cpp)."""
    left = large_corkscrew_left_curve(distance)
    return TrackPointArray(
        left.position * _MIRROR_X,
        left.tangent * _MIRROR_X,
        left.normal * _MIRROR_X,
        left.binormal * np.array([1.0, -1.0, -1.0]),
    )


def half_loop_curve(distance: NDArray[np.float64]) -> TrackPointArray:
    """A half loop: a short ramp into a vertical-plane loop, exiting inverted 180°.

    Three segments (``track_sections.cpp``): a straight launch ramp, then two
    ``cubic_curve_vertical_old`` spline arcs that carry the track over the top.
    """
    n = distance.shape[0]
    ramp = distance / HALF_LOOP_SEGMENT1_LENGTH
    seg1_pos = np.stack(
        [np.zeros(n), CLEARANCE_HEIGHT * ramp, 0.5 * TILE_SIZE * ramp], axis=1
    )
    seg1_tan = _normalize(
        np.tile(np.array([0.0, 2 * CLEARANCE_HEIGHT / TILE_SIZE, 1.0]), (n, 1))
    )
    seg1 = plane_curve_vertical(seg1_pos, seg1_tan)
    seg2 = cubic_curve_vertical_old(
        (3 * TILE_SIZE - 32 * CLEARANCE_HEIGHT / 3, 16 * CLEARANCE_HEIGHT - 6.5 * TILE_SIZE,
         4 * TILE_SIZE, 0.5 * TILE_SIZE),
        (-14 * CLEARANCE_HEIGHT / 3, 19 * CLEARANCE_HEIGHT / 3, 8 * CLEARANCE_HEIGHT,
         CLEARANCE_HEIGHT),
        (
            2.60735963e-07, -7.42305927e-06, 8.47657813e-05, -4.81686166e-04,
            1.50741670e-03, 3.64826748e-04, 6.3993545e-02,
        ),
        distance - HALF_LOOP_SEGMENT1_LENGTH,
    )
    seg3 = cubic_curve_vertical_old(
        (0.0, -16 * CLEARANCE_HEIGHT / 3, 0.0, 16 * CLEARANCE_HEIGHT / 3 + TILE_SIZE),
        (-8 * CLEARANCE_HEIGHT / 3, -4 * CLEARANCE_HEIGHT / 3, 32 * CLEARANCE_HEIGHT / 3,
         32 * CLEARANCE_HEIGHT / 3),
        (
            4.98545231e-07, -8.90813710e-06, 4.65444884e-05, -1.03595298e-04,
            3.16464106e-04, 1.97218182e-03, 1.24963548e-01,
        ),
        distance - HALF_LOOP_SEGMENT2_LENGTH,
    )
    tail = _select(distance < HALF_LOOP_SEGMENT2_LENGTH, seg2, seg3)
    return _select(distance < HALF_LOOP_SEGMENT1_LENGTH, seg1, tail)


def quarter_loop_curve(distance: NDArray[np.float64]) -> TrackPointArray:
    """A quarter loop: a vertical-plane spline that pitches up to vertical (track_sections.cpp)."""
    return cubic_curve_vertical_old(
        (5 * TILE_SIZE - 64 * CLEARANCE_HEIGHT / 3, -7.5 * TILE_SIZE + 64 * CLEARANCE_HEIGHT / 3,
         0.0, 0.0),
        (-22 * CLEARANCE_HEIGHT / 3, CLEARANCE_HEIGHT / 3.0, 64 * CLEARANCE_HEIGHT / 3.0, 0.0),
        (
            7.18882561e-10, -3.95603532e-08, 6.77429101e-07, -4.86060220e-06,
            3.29621469e-05, -1.32508457e-04, 6.25520141e-02,
        ),
        distance,
    )


def _pin_launch_start(tp: TrackPointArray, distance: NDArray[np.float64]) -> TrackPointArray:
    """Pin the very first point(s) to the clean launch frame the engine hard-codes.

    For ``distance < 0.001`` several inversions force position ``0``, the gentle launch
    tangent ``normalize(0, 2·CLEARANCE/TILE, 1)`` and binormal ``(-1, 0, 0)`` to avoid the
    spline's degenerate frame at the start.
    """
    start = distance < 1e-3
    if not bool(np.any(start)):
        return tp
    st = np.array([0.0, 2 * CLEARANCE_HEIGHT / TILE_SIZE, 1.0])
    st = st / np.linalg.norm(st)
    position = tp.position.copy()
    tangent = tp.tangent.copy()
    normal = tp.normal.copy()
    binormal = tp.binormal.copy()
    position[start] = 0.0
    tangent[start] = st
    normal[start] = _normalize(np.array([[0.0, st[2], -st[1]]]))[0]
    binormal[start] = [-1.0, 0.0, 0.0]
    return TrackPointArray(position, tangent, normal, binormal)


def _finalize_half_loop(
    distance: NDArray[np.float64],
    position: NDArray[np.float64],
    tangent_yz: NDArray[np.float64],
    drift: float,
) -> TrackPointArray:
    """Shared frame finish for the medium/large half loops (track_sections.cpp).

    The piecewise body builds a vertical-plane ``position`` and a Y/Z ``tangent``; this
    adds the small sideways ``drift`` to ``tangent.x``, renormalizes, derives the frame
    (``normal = (0, tz, -ty)``, ``binormal = tangent × normal``), and pins the launch frame.
    """
    n = distance.shape[0]
    tangent = tangent_yz.copy()
    tangent[:, 0] = drift
    tangent = _normalize(tangent)
    normal = _normalize(np.stack([np.zeros(n), tangent[:, 2], -tangent[:, 1]], axis=1))
    binormal = np.cross(tangent, normal)
    return _pin_launch_start(TrackPointArray(position, tangent, normal, binormal), distance)


def medium_half_loop_left_curve(distance: NDArray[np.float64]) -> TrackPointArray:
    """A medium left half loop: two spline arcs that drift sideways over the top."""
    proj = distance / _MEDIUM_HALF_LOOP_FACTOR
    px = TILE_SIZE * distance / MEDIUM_HALF_LOOP_LENGTH
    u1 = reparameterize(
        (1.52767386e-08, -6.76167426e-07, 1.19808364e-05, -1.04744877e-04,
         4.98755117e-04, -1.11193799e-04, 4.08073528e-02),
        proj,
    )
    ya1 = (-22.0 * CLEARANCE_HEIGHT / 3.0, 28.0 * CLEARANCE_HEIGHT / 3.0, 14 * CLEARANCE_HEIGHT)
    za1 = (1.2 * TILE_SIZE, -5.3 * TILE_SIZE, 7 * TILE_SIZE)
    pos1 = np.stack([px, cubic(*ya1, 0.0, u1), cubic(*za1, 0.0, u1)], axis=1)
    tan1 = np.stack(
        [np.zeros_like(u1), cubic_derivative(*ya1, u1), cubic_derivative(*za1, u1)], axis=1
    )
    u2 = reparameterize(
        (7.17265729e-09, -2.45217193e-07, 2.74827040e-06, -8.83261672e-06,
         -1.32828600e-04, 1.70980960e-03, 9.62026021e-02),
        proj - _MEDIUM_HALF_LOOP_SEGMENT1_LENGTH,
    )
    ya2 = (-56 * CLEARANCE_HEIGHT / 3.0 + TILE_SIZE * 3.15, 28 * CLEARANCE_HEIGHT - TILE_SIZE * 6.3,
           TILE_SIZE * 3.15)
    za2 = (0.65 * TILE_SIZE, -2.55 * TILE_SIZE, 0.0)
    pos2 = np.stack([px, cubic(*ya2, 16 * CLEARANCE_HEIGHT, u2), cubic(*za2, 2.9 * TILE_SIZE, u2)],
                    axis=1)
    tan2 = np.stack(
        [np.zeros_like(u2), cubic_derivative(*ya2, u2), cubic_derivative(*za2, u2)], axis=1
    )
    pick = proj < _MEDIUM_HALF_LOOP_SEGMENT1_LENGTH
    m = pick[:, None]
    position = np.where(m, pos1, pos2)
    tangent_yz = np.where(m, tan1, tan2)
    drift = _MEDIUM_HALF_LOOP_FACTOR / MEDIUM_HALF_LOOP_LENGTH
    return _finalize_half_loop(distance, position, tangent_yz, drift)


def medium_half_loop_right_curve(distance: NDArray[np.float64]) -> TrackPointArray:
    """A medium right half loop: the left half loop mirrored (track_sections.cpp)."""
    left = medium_half_loop_left_curve(distance)
    return TrackPointArray(
        left.position * _MIRROR_X,
        left.tangent * _MIRROR_X,
        left.normal * _MIRROR_X,
        left.binormal * np.array([1.0, -1.0, -1.0]),
    )


def large_half_loop_left_curve(distance: NDArray[np.float64]) -> TrackPointArray:
    """A large left half loop: a launch ramp + two spline arcs, drifting sideways."""
    proj = distance / _LARGE_HALF_LOOP_FACTOR
    px = TILE_SIZE * proj / LARGE_HALF_LOOP_LENGTH
    u0 = proj / GENTLE_LENGTH
    pos0 = np.stack([px, 2 * CLEARANCE_HEIGHT * u0, u0 * TILE_SIZE], axis=1)
    tan0 = np.tile(np.array([0.0, 2 * CLEARANCE_HEIGHT, TILE_SIZE]), (distance.shape[0], 1))
    u1 = reparameterize(
        (4.92552773e-08, -3.05251408e-06, 7.72071092e-05, -1.03134802e-03,
         7.91192883e-03, -3.64962014e-02, 1.60807957e-01),
        proj - _LARGE_HALF_LOOP_SEGMENT1_LENGTH,
    )
    ya1 = (-0.403193, 10.731874, 2.020829)
    za1 = (-11.880000, 15.345000, 4.950000)
    pos1 = np.stack([px, cubic(*ya1, 2.020829, u1), cubic(*za1, 4.950000, u1)], axis=1)
    tan1 = np.stack(
        [np.zeros_like(u1), cubic_derivative(*ya1, u1), cubic_derivative(*za1, u1)], axis=1
    )
    u2 = reparameterize(
        (1.99640047e-07, -6.93285191e-06, 9.60806400e-05, -6.7055459e-04,
         2.46312103e-03, -1.96991475e-03, 5.27553949e-02),
        proj - _LARGE_HALF_LOOP_SEGMENT2_LENGTH,
    )
    ya2 = (3.633368, -15.350052, 19.800000)
    za2 = (8.580000, -15.345000, 0.000000)
    pos2 = np.stack([px, cubic(*ya2, 14.370340, u2), cubic(*za2, 13.365000, u2)], axis=1)
    tan2 = np.stack(
        [np.zeros_like(u2), cubic_derivative(*ya2, u2), cubic_derivative(*za2, u2)], axis=1
    )
    in0 = (proj < _LARGE_HALF_LOOP_SEGMENT1_LENGTH)[:, None]
    in1 = (distance < _LARGE_HALF_LOOP_SEGMENT2_LENGTH)[:, None]
    position = np.where(in0, pos0, np.where(in1, pos1, pos2))
    tangent_yz = np.where(in0, tan0, np.where(in1, tan1, tan2))
    return _finalize_half_loop(distance, position, tangent_yz, 0.1006880872852946)


def large_half_loop_right_curve(distance: NDArray[np.float64]) -> TrackPointArray:
    """A large right half loop: the left half loop mirrored (track_sections.cpp)."""
    left = large_half_loop_left_curve(distance)
    return TrackPointArray(
        left.position * _MIRROR_X,
        left.tangent * _MIRROR_X,
        left.normal * _MIRROR_X,
        left.binormal * np.array([1.0, -1.0, -1.0]),
    )


def _zero_g_roll_coeffs(rate_initial: float, rate_final: float, base: float) -> _Quad:
    """The roll-angle cubic (a·x³+b·x²+c·x) for a zero-G roll given its end roll rates."""
    a = (rate_final + rate_initial - 2 * math.pi / base) / (base * base)
    b = (3 * math.pi / base - 2 * rate_initial - rate_final) / base
    return (a, b, rate_initial, 0.0)


def zero_g_roll_left_curve(distance: NDArray[np.float64]) -> TrackPointArray:
    """A left zero-G roll: a rolling vertical-plane arc (track_sections.cpp)."""
    roll = _zero_g_roll_coeffs(
        0.5 * math.pi / (3.0 * TILE_SIZE), 0.75 * math.pi / (3.0 * TILE_SIZE),
        ZERO_G_ROLL_BASE_LENGTH,
    )
    d2 = reparameterize(
        (-1.70840591e-06, 6.64971489e-05, -1.00667206e-03, 7.50562783e-03,
         -2.86012281e-02, 4.62022020e-02, 9.61979416e-01),
        distance,
    )
    tp = roll_curve(
        7 * CLEARANCE_HEIGHT / 6,
        (-0.5 * TILE_SIZE, -1.5 * TILE_SIZE, 5 * TILE_SIZE, 0.0),
        (4 * CLEARANCE_HEIGHT, -11 * CLEARANCE_HEIGHT, 10 * CLEARANCE_HEIGHT, 0.0),
        roll,
        (2.72452673e-06, -8.60587142e-05, 1.06785619e-03, -6.53445874e-03,
         2.04313108e-02, -2.83005236e-02, 7.09176768e-02),
        d2,
    )
    return _pin_launch_start(tp, distance)


def zero_g_roll_right_curve(distance: NDArray[np.float64]) -> TrackPointArray:
    """A right zero-G roll: the left roll mirrored (track_sections.cpp)."""
    left = zero_g_roll_left_curve(distance)
    return TrackPointArray(
        left.position * _MIRROR_X,
        left.tangent * _MIRROR_X,
        left.normal * _MIRROR_X,
        left.binormal * np.array([1.0, -1.0, -1.0]),
    )


def large_zero_g_roll_left_curve(distance: NDArray[np.float64]) -> TrackPointArray:
    """A left large zero-G roll: a longer rolling vertical-plane arc (track_sections.cpp)."""
    roll = _zero_g_roll_coeffs(
        0.15 * math.pi / (3.0 * TILE_SIZE), 0.85 * math.pi / (3.0 * TILE_SIZE),
        LARGE_ZERO_G_ROLL_BASE_LENGTH,
    )
    d2 = reparameterize(
        (1.28813475e-09, 4.41476438e-09, -1.60801085e-06, 2.32101117e-05,
         -1.72800658e-04, 3.66637278e-04, 9.99343058e-01),
        distance,
    )
    return roll_curve(
        4 * CLEARANCE_HEIGHT / 6,
        (0.0, 1.0 * TILE_SIZE, 3.0 * TILE_SIZE, 0.0),
        (-8 * CLEARANCE_HEIGHT, 0.0 * CLEARANCE_HEIGHT, 24 * CLEARANCE_HEIGHT, 0.0),
        roll,
        (4.56294513e-11, -9.86183787e-09, 3.08458335e-07, -4.08131118e-06,
         5.38263660e-05, -3.00786755e-04, 5.27935955e-02),
        d2,
    )


def large_zero_g_roll_right_curve(distance: NDArray[np.float64]) -> TrackPointArray:
    """A right large zero-G roll: the left roll mirrored (track_sections.cpp)."""
    left = large_zero_g_roll_left_curve(distance)
    return TrackPointArray(
        left.position * _MIRROR_X,
        left.tangent * _MIRROR_X,
        left.normal * _MIRROR_X,
        left.binormal * np.array([1.0, -1.0, -1.0]),
    )


def dive_loop_45_left_curve(distance: NDArray[np.float64]) -> TrackPointArray:
    """A left 45° dive loop: a single 3D Bézier with a rolled frame (track_sections.cpp)."""
    return bezier3d(
        (-3.30000000e00, 9.90000000e00, -9.90000000e00, 0.0),
        (-7.18516991e00, 2.69443872e00, 1.61666323e01, 0.0),
        (1.65000000e00, 0.0, 9.90000000e00, 0.0),
        (-2.29510887e00, 3.70037965e00, -6.23809670e-01, -7.81461115e-01),
        (-1.45512874e-09, 6.16508770e-08, -1.00546234e-06, 9.05852906e-06,
         -3.03257464e-05, 3.23153975e-04, 4.67191926e-02),
        distance,
    )


def dive_loop_45_right_curve(distance: NDArray[np.float64]) -> TrackPointArray:
    """A right 45° dive loop: the left dive loop rotated 90° (x↔z swap), not mirrored."""
    left = dive_loop_45_left_curve(distance)

    def swap_xz(v: NDArray[np.float64]) -> NDArray[np.float64]:
        out = v.copy()
        out[:, 0] = -v[:, 2]
        out[:, 2] = -v[:, 0]
        return out

    position = swap_xz(left.position)
    normal = swap_xz(left.normal)
    tangent = swap_xz(left.tangent)
    return TrackPointArray(position, tangent, normal, np.cross(tangent, normal))


def _vertical_twist_curve(
    distance: NDArray[np.float64], sign: float, base: float, rate: float
) -> TrackPointArray:
    """A vertical climb whose normal twists in the X/Z plane (track_sections.cpp).

    The twist angle is ``base + rate·distance``; ``sign`` selects the direction
    (left = -1, right = +1). The full twist sweeps 90°; the to_diag/to_orthogonal
    variants sweep the first/second 45°.
    """
    n = distance.shape[0]
    angle = base + rate * distance
    position = np.stack([np.zeros(n), distance, np.zeros(n)], axis=1)
    tangent = np.tile(np.array([0.0, 1.0, 0.0]), (n, 1))
    normal = np.stack([sign * np.sin(angle), np.zeros(n), -np.cos(angle)], axis=1)
    return TrackPointArray(position, tangent, normal, np.cross(tangent, normal))


def vertical_twist_left_curve(distance: NDArray[np.float64]) -> TrackPointArray:
    """A left vertical twist: climbs straight up while rotating 90° (track_sections.cpp)."""
    return _vertical_twist_curve(distance, -1.0, 0.0, 0.5 * math.pi / VERTICAL_TWIST_LENGTH)


def vertical_twist_right_curve(distance: NDArray[np.float64]) -> TrackPointArray:
    """A right vertical twist: climbs straight up while rotating 90° the other way."""
    return _vertical_twist_curve(distance, 1.0, 0.0, 0.5 * math.pi / VERTICAL_TWIST_LENGTH)


def vertical_twist_left_to_diag_curve(distance: NDArray[np.float64]) -> TrackPointArray:
    """A left vertical twist into the diagonal: the first 45° of the twist."""
    return _vertical_twist_curve(distance, -1.0, 0.0, 0.25 * math.pi / VERTICAL_TWIST_45_LENGTH)


def vertical_twist_right_to_diag_curve(distance: NDArray[np.float64]) -> TrackPointArray:
    """A right vertical twist into the diagonal: the first 45° of the twist."""
    return _vertical_twist_curve(distance, 1.0, 0.0, 0.25 * math.pi / VERTICAL_TWIST_45_LENGTH)


def vertical_twist_left_to_orthogonal_curve(distance: NDArray[np.float64]) -> TrackPointArray:
    """A left vertical twist back to orthogonal: the second 45° of the twist (starts at 45°)."""
    return _vertical_twist_curve(
        distance, -1.0, 0.25 * math.pi, 0.25 * math.pi / VERTICAL_TWIST_45_LENGTH
    )


def vertical_twist_right_to_orthogonal_curve(distance: NDArray[np.float64]) -> TrackPointArray:
    """A right vertical twist back to orthogonal: the second 45° of the twist (starts at 45°)."""
    return _vertical_twist_curve(
        distance, 1.0, 0.25 * math.pi, 0.25 * math.pi / VERTICAL_TWIST_45_LENGTH
    )


def vertical_loop_left_curve(distance: NDArray[np.float64]) -> TrackPointArray:
    """A left vertical loop: a launch ramp + two vertical-plane arcs that drift sideways."""
    proj = distance / _VERTICAL_LOOP_FACTOR
    ramp = proj / _VERTICAL_LOOP_SEGMENT1_LENGTH
    seg1 = plane_curve_vertical(
        np.stack([np.zeros_like(proj), CLEARANCE_HEIGHT * ramp, 0.5 * TILE_SIZE * ramp], axis=1),
        _normalize(np.tile(np.array([0.0, 2 * CLEARANCE_HEIGHT / TILE_SIZE, 1.0]),
                           (distance.shape[0], 1))),
    )
    seg2 = cubic_curve_vertical_old(
        (TILE_SIZE, -3.5 * TILE_SIZE, 4 * TILE_SIZE, 0.5 * TILE_SIZE),
        (-20 * CLEARANCE_HEIGHT / 3, 26 * CLEARANCE_HEIGHT / 3, 8 * CLEARANCE_HEIGHT,
         CLEARANCE_HEIGHT),
        (
            2.60735963e-07, -7.42305927e-06, 8.47657813e-05, -4.81686166e-04,
            1.50741670e-03, 3.64826748e-04, 6.3993545e-02,
        ),
        proj - _VERTICAL_LOOP_SEGMENT1_LENGTH,
    )
    u3 = (proj - _VERTICAL_LOOP_SEGMENT2_LENGTH) / (1.730928 * TILE_SIZE)
    seg3 = cubic_curve_vertical(
        (0.0, -TILE_SIZE, 0.0, 2 * TILE_SIZE),
        (-11 * CLEARANCE_HEIGHT / 3, 9 * CLEARANCE_HEIGHT / 6, 8 * CLEARANCE_HEIGHT,
         33 * CLEARANCE_HEIGHT / 3),
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        u3,
    )
    tail = _select(proj < _VERTICAL_LOOP_SEGMENT2_LENGTH, seg2, seg3)
    curve = _select(proj < _VERTICAL_LOOP_SEGMENT1_LENGTH, seg1, tail)

    position = curve.position.copy()
    position[:, 0] += 0.5 * TILE_SIZE * distance / VERTICAL_LOOP_LENGTH
    tangent = curve.tangent.copy()
    tangent[:, 0] += 0.5 * _VERTICAL_LOOP_FACTOR / VERTICAL_LOOP_LENGTH
    tangent = _normalize(tangent)
    normal = _normalize(
        np.stack([np.zeros(distance.shape[0]), tangent[:, 2], -tangent[:, 1]], axis=1)
    )
    return TrackPointArray(position, tangent, normal, np.cross(tangent, normal))


def vertical_loop_right_curve(distance: NDArray[np.float64]) -> TrackPointArray:
    """A right vertical loop: the left loop mirrored (track_sections.cpp)."""
    left = vertical_loop_left_curve(distance)
    return TrackPointArray(
        left.position * _MIRROR_X,
        left.tangent * _MIRROR_X,
        left.normal * _MIRROR_X,
        left.binormal * np.array([1.0, -1.0, -1.0]),
    )
