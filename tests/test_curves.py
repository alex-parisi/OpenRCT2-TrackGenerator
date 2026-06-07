"""Curve-math correctness: endpoints, slopes, and frame orthonormality."""

import numpy as np
import pytest
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


def test_vertical_curve_goes_straight_up():
    d = np.array([0.0, curves.VERTICAL_LENGTH])
    tp = curves.vertical_curve(d)
    np.testing.assert_allclose(tp.position[:, 1], d)  # y == distance (straight up)
    np.testing.assert_allclose(tp.position[:, 0], 0.0)
    np.testing.assert_allclose(tp.position[:, 2], 0.0)
    np.testing.assert_allclose(tp.tangent, np.tile([0.0, 1.0, 0.0], (2, 1)))  # pointing up
    _assert_orthonormal(tp)


@pytest.mark.parametrize(
    "curve,length,from_origin",
    [
        (curves.gentle_to_flat_curve, curves.FLAT_TO_GENTLE_LENGTH, True),
        (curves.steep_to_gentle_curve, curves.GENTLE_TO_STEEP_LENGTH, True),
        (curves.flat_to_steep_curve, curves.FLAT_TO_STEEP_LENGTH, True),
        (curves.steep_to_flat_curve, curves.FLAT_TO_STEEP_LENGTH, True),
        # steep_to_vertical has a non-zero cubic constant (xd = -TILE/2), so it does
        # not start at the origin; vertical_to_steep does.
        (curves.steep_to_vertical_curve, curves.STEEP_TO_VERTICAL_LENGTH, False),
        (curves.vertical_to_steep_curve, curves.STEEP_TO_VERTICAL_LENGTH, True),
    ],
)
def test_batch_a_cubic_curves_orthonormal(curve, length, from_origin):
    d = np.linspace(0.0, length, 8)
    tp = curve(d)
    if from_origin:
        np.testing.assert_allclose(tp.position[0], [0.0, 0.0, 0.0], atol=1e-6)
    _assert_orthonormal(tp)


def test_medium_turn_left_sweeps_quarter_turn():
    d = np.array([0.0, curves.MEDIUM_TURN_LENGTH])
    tp = curves.medium_turn_left_curve(d)
    r = 2.5 * TILE_SIZE
    np.testing.assert_allclose(tp.position[0], [0.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(tp.position[1], [r, 0.0, r], atol=1e-4)
    np.testing.assert_allclose(tp.tangent[1], [1.0, 0.0, 0.0], atol=1e-5)
    _assert_orthonormal(tp)


def test_large_turns_orthonormal_and_mirror_in_x():
    d = np.linspace(0.0, curves.LARGE_TURN_LENGTH, 8)
    left = curves.large_turn_left_to_diag_curve(d)
    right = curves.large_turn_right_to_diag_curve(d)
    _assert_orthonormal(left)
    _assert_orthonormal(right)
    # The right turn is the left turn reflected across the track (X negated).
    np.testing.assert_allclose(right.position[:, 0], -left.position[:, 0], atol=1e-6)
    np.testing.assert_allclose(right.position[:, 2], left.position[:, 2], atol=1e-6)


def test_flat_diag_runs_along_diagonal():
    d = np.array([0.0, curves.FLAT_DIAG_LENGTH])
    tp = curves.flat_diag_curve(d)
    # Moves equally in -X and +Z (the 45° diagonal), stays level.
    np.testing.assert_allclose(tp.position[1, 0], -curves.FLAT_DIAG_LENGTH / np.sqrt(2), atol=1e-5)
    np.testing.assert_allclose(tp.position[1, 2], curves.FLAT_DIAG_LENGTH / np.sqrt(2), atol=1e-5)
    np.testing.assert_allclose(tp.position[:, 1], 0.0)
    _assert_orthonormal(tp)


@pytest.mark.parametrize(
    "curve,length,from_origin",
    [
        (curves.gentle_diag_curve, curves.GENTLE_DIAG_LENGTH, False),
        (curves.steep_diag_curve, curves.STEEP_DIAG_LENGTH, False),
        (curves.flat_to_gentle_diag_curve, curves.FLAT_TO_GENTLE_DIAG_LENGTH, True),
        (curves.gentle_to_flat_diag_curve, curves.FLAT_TO_GENTLE_DIAG_LENGTH, True),
        (curves.gentle_to_steep_diag_curve, curves.GENTLE_TO_STEEP_DIAG_LENGTH, True),
        (curves.steep_to_gentle_diag_curve, curves.GENTLE_TO_STEEP_DIAG_LENGTH, True),
    ],
)
def test_diagonal_curves_orthonormal(curve, length, from_origin):
    d = np.linspace(0.0, length, 8)
    tp = curve(d)
    if from_origin:
        np.testing.assert_allclose(tp.position[0], [0.0, 0.0, 0.0], atol=1e-6)
    _assert_orthonormal(tp)


@pytest.mark.parametrize(
    "left_curve,right_curve,length",
    [
        (curves.gentle_left_bank_curve, curves.gentle_right_bank_curve, curves.GENTLE_LENGTH),
        (curves.gentle_to_gentle_left_bank_curve, curves.gentle_to_gentle_right_bank_curve,
         curves.GENTLE_LENGTH),
        (curves.gentle_left_bank_to_gentle_curve, curves.gentle_right_bank_to_gentle_curve,
         curves.GENTLE_LENGTH),
        (curves.flat_to_gentle_left_bank_curve, curves.flat_to_gentle_right_bank_curve,
         curves.FLAT_TO_GENTLE_LENGTH),
        (curves.gentle_left_bank_to_flat_curve, curves.gentle_right_bank_to_flat_curve,
         curves.FLAT_TO_GENTLE_LENGTH),
        (curves.left_bank_to_gentle_left_bank_curve, curves.right_bank_to_gentle_right_bank_curve,
         curves.FLAT_TO_GENTLE_LENGTH),
        (curves.gentle_left_bank_to_left_bank_curve, curves.gentle_right_bank_to_right_bank_curve,
         curves.FLAT_TO_GENTLE_LENGTH),
    ],
)
def test_gentle_bank_transitions_orthonormal_and_mirror(left_curve, right_curve, length):
    d = np.linspace(0.0, length, 8)
    left = left_curve(d)
    right = right_curve(d)
    _assert_orthonormal(left)
    _assert_orthonormal(right)
    # Left/right banks are roll-mirrored: same path, opposite normal-x tilt.
    np.testing.assert_allclose(left.position, right.position, atol=1e-9)
    np.testing.assert_allclose(left.normal[:, 0], -right.normal[:, 0], atol=1e-9)


@pytest.mark.parametrize(
    "left_curve,right_curve,length",
    [
        (curves.gentle_left_bank_diag_curve, curves.gentle_right_bank_diag_curve,
         curves.GENTLE_DIAG_LENGTH),
        (curves.gentle_to_gentle_left_bank_diag_curve,
         curves.gentle_to_gentle_right_bank_diag_curve, curves.GENTLE_DIAG_LENGTH),
        (curves.gentle_left_bank_to_gentle_diag_curve,
         curves.gentle_right_bank_to_gentle_diag_curve, curves.GENTLE_DIAG_LENGTH),
        (curves.flat_to_gentle_left_bank_diag_curve, curves.flat_to_gentle_right_bank_diag_curve,
         curves.FLAT_TO_GENTLE_DIAG_LENGTH),
        (curves.gentle_left_bank_to_flat_diag_curve, curves.gentle_right_bank_to_flat_diag_curve,
         curves.FLAT_TO_GENTLE_DIAG_LENGTH),
        (curves.left_bank_to_gentle_left_bank_diag_curve,
         curves.right_bank_to_gentle_right_bank_diag_curve, curves.FLAT_TO_GENTLE_DIAG_LENGTH),
        (curves.gentle_left_bank_to_left_bank_diag_curve,
         curves.gentle_right_bank_to_right_bank_diag_curve, curves.FLAT_TO_GENTLE_DIAG_LENGTH),
    ],
)
def test_gentle_bank_diag_transitions_orthonormal_and_mirror(left_curve, right_curve, length):
    d = np.linspace(0.0, length, 8)
    left = left_curve(d)
    right = right_curve(d)
    _assert_orthonormal(left)
    _assert_orthonormal(right)
    np.testing.assert_allclose(left.position, right.position, atol=1e-9)


def test_gentle_bank_endpoints_match_bank_angle():
    from openrct2_track_generator.constants import BANK_ANGLE

    # A fully-banked gentle slope holds BANK_ANGLE throughout; the ramp-in version
    # starts unbanked and reaches BANK_ANGLE at the end.
    d = np.array([0.0, curves.GENTLE_LENGTH])
    held = curves.gentle_left_bank_curve(d)
    base = curves.gentle_curve(d)
    cos_held = np.sum(held.normal * base.normal, axis=1)
    np.testing.assert_allclose(cos_held, np.cos(BANK_ANGLE), atol=1e-6)
    ramp = curves.gentle_to_gentle_left_bank_curve(d)
    cos_ramp = np.sum(ramp.normal * base.normal, axis=1)
    np.testing.assert_allclose(cos_ramp, [1.0, np.cos(BANK_ANGLE)], atol=1e-6)


_BATCH_H_CURVES = [
    (curves.small_turn_left_gentle_curve, curves.SMALL_TURN_GENTLE_LENGTH),
    (curves.small_turn_right_gentle_curve, curves.SMALL_TURN_GENTLE_LENGTH),
    (curves.medium_turn_left_gentle_curve, curves.MEDIUM_TURN_GENTLE_LENGTH),
    (curves.medium_turn_right_gentle_curve, curves.MEDIUM_TURN_GENTLE_LENGTH),
    (curves.large_turn_left_to_diag_gentle_curve, curves.LARGE_TURN_GENTLE_LENGTH),
    (curves.large_turn_right_to_diag_gentle_curve, curves.LARGE_TURN_GENTLE_LENGTH),
    (curves.large_turn_left_to_orthogonal_gentle_curve, curves.LARGE_TURN_GENTLE_LENGTH),
    (curves.large_turn_right_to_orthogonal_gentle_curve, curves.LARGE_TURN_GENTLE_LENGTH),
    (curves.small_turn_left_bank_gentle_curve, curves.SMALL_TURN_GENTLE_LENGTH),
    (curves.small_turn_right_bank_gentle_curve, curves.SMALL_TURN_GENTLE_LENGTH),
    (curves.medium_turn_left_bank_gentle_curve, curves.MEDIUM_TURN_GENTLE_LENGTH),
    (curves.medium_turn_right_bank_gentle_curve, curves.MEDIUM_TURN_GENTLE_LENGTH),
    (curves.large_turn_left_bank_to_diag_gentle_curve, curves.LARGE_TURN_GENTLE_LENGTH),
    (curves.large_turn_right_bank_to_diag_gentle_curve, curves.LARGE_TURN_GENTLE_LENGTH),
    (curves.large_turn_left_bank_to_orthogonal_gentle_curve, curves.LARGE_TURN_GENTLE_LENGTH),
    (curves.large_turn_right_bank_to_orthogonal_gentle_curve, curves.LARGE_TURN_GENTLE_LENGTH),
    (curves.small_turn_left_bank_to_gentle_curve, curves.TURN_BANK_TRANSITION_LENGTH),
    (curves.small_turn_right_bank_to_gentle_curve, curves.TURN_BANK_TRANSITION_LENGTH),
]


@pytest.mark.parametrize("curve,length", _BATCH_H_CURVES)
def test_gentle_turn_curves_orthonormal_and_climb(curve, length):
    d = np.linspace(0.0, length, 10)
    tp = curve(d)
    _assert_orthonormal(tp)
    assert tp.position[-1, 1] > tp.position[0, 1]  # net climb


_BATCH_I_CURVES = [
    (curves.very_small_turn_left_steep_curve, curves.VERY_SMALL_TURN_STEEP_LENGTH),
    (curves.very_small_turn_right_steep_curve, curves.VERY_SMALL_TURN_STEEP_LENGTH),
    (curves.small_turn_left_steep_curve, curves.SMALL_TURN_STEEP_LENGTH),
    (curves.small_turn_right_steep_curve, curves.SMALL_TURN_STEEP_LENGTH),
    (curves.large_turn_left_to_diag_steep_curve, curves.LARGE_TURN_STEEP_LENGTH),
    (curves.large_turn_right_to_diag_steep_curve, curves.LARGE_TURN_STEEP_LENGTH),
    (curves.large_turn_left_to_orthogonal_steep_curve, curves.LARGE_TURN_STEEP_LENGTH),
    (curves.large_turn_right_to_orthogonal_steep_curve, curves.LARGE_TURN_STEEP_LENGTH),
]


@pytest.mark.parametrize("curve,length", _BATCH_I_CURVES)
def test_steep_turn_curves_orthonormal_and_climb(curve, length):
    d = np.linspace(0.0, length, 10)
    tp = curve(d)
    _assert_orthonormal(tp)
    assert tp.position[-1, 1] > tp.position[0, 1]  # net climb (steep)


@pytest.mark.parametrize(
    "curve,length",
    [
        (curves.small_flat_to_steep_curve, curves.SMALL_FLAT_TO_STEEP_LENGTH),
        (curves.small_steep_to_flat_curve, curves.SMALL_FLAT_TO_STEEP_LENGTH),
        (curves.small_flat_to_steep_diag_curve, curves.SMALL_FLAT_TO_STEEP_DIAG_LENGTH),
        (curves.small_steep_to_flat_diag_curve, curves.SMALL_FLAT_TO_STEEP_DIAG_LENGTH),
        (curves.flat_to_steep_diag_curve, curves.FLAT_TO_STEEP_DIAG_LENGTH),
        (curves.steep_to_flat_diag_curve, curves.FLAT_TO_STEEP_DIAG_LENGTH),
        (curves.steep_to_vertical_diag_curve, curves.STEEP_TO_VERTICAL_DIAG_LENGTH),
        (curves.vertical_to_steep_diag_curve, curves.STEEP_TO_VERTICAL_DIAG_LENGTH),
        (curves.vertical_diag_curve, curves.VERTICAL_LENGTH),
    ],
)
def test_batch_j_slope_transitions_orthonormal_and_climb(curve, length):
    d = np.linspace(0.0, length, 10)
    tp = curve(d)
    _assert_orthonormal(tp)
    assert tp.position[-1, 1] > tp.position[0, 1]  # net climb


@pytest.mark.parametrize(
    "left_curve,right_curve,length",
    [
        (curves.gentle_left_bank_to_steep_curve, curves.gentle_right_bank_to_steep_curve,
         curves.GENTLE_TO_STEEP_LENGTH),
        (curves.steep_to_gentle_left_bank_curve, curves.steep_to_gentle_right_bank_curve,
         curves.GENTLE_TO_STEEP_LENGTH),
        (curves.gentle_left_bank_to_steep_diag_curve, curves.gentle_right_bank_to_steep_diag_curve,
         curves.GENTLE_TO_STEEP_DIAG_LENGTH),
        (curves.steep_to_gentle_left_bank_diag_curve, curves.steep_to_gentle_right_bank_diag_curve,
         curves.GENTLE_TO_STEEP_DIAG_LENGTH),
    ],
)
def test_steep_bank_transitions_orthonormal_and_mirror(left_curve, right_curve, length):
    d = np.linspace(0.0, length, 8)
    left = left_curve(d)
    right = right_curve(d)
    _assert_orthonormal(left)
    _assert_orthonormal(right)
    # Same base path, opposite bank: position is shared; the banks tilt opposite ways.
    np.testing.assert_allclose(left.position, right.position, atol=1e-9)
    assert not np.allclose(left.normal, right.normal)


@pytest.mark.parametrize(
    "curve,length",
    [
        (curves.banked_barrel_roll_left_curve, curves.BARREL_ROLL_LENGTH),
        (curves.banked_barrel_roll_right_curve, curves.BARREL_ROLL_LENGTH),
        (curves.banked_inline_twist_left_curve, curves.INLINE_TWIST_LENGTH),
        (curves.banked_inline_twist_right_curve, curves.INLINE_TWIST_LENGTH),
        (curves.banked_zero_g_roll_left_curve, curves.ZERO_G_ROLL_LENGTH),
        (curves.banked_zero_g_roll_right_curve, curves.ZERO_G_ROLL_LENGTH),
        (curves.dive_loop_90_left_curve, curves.DIVE_LOOP_90_LENGTH),
        (curves.dive_loop_90_right_curve, curves.DIVE_LOOP_90_LENGTH),
    ],
)
def test_batch_l_inversions_orthonormal(curve, length):
    _assert_orthonormal(curve(np.linspace(0.0, length, 16)))


def test_dive_loop_90_mirrors():
    d = np.linspace(0.0, curves.DIVE_LOOP_90_LENGTH, 8)
    left = curves.dive_loop_90_left_curve(d)
    right = curves.dive_loop_90_right_curve(d)
    np.testing.assert_allclose(right.position[:, 0], -left.position[:, 0], atol=1e-9)


def test_gentle_turns_sweep_yaw_and_mirror():
    d = np.linspace(0.0, curves.SMALL_TURN_GENTLE_LENGTH, 8)
    left = curves.small_turn_left_gentle_curve(d)
    right = curves.small_turn_right_gentle_curve(d)
    # Left turns toward -... and right mirrors its X position.
    np.testing.assert_allclose(right.position[:, 0], -left.position[:, 0], atol=1e-9)
    np.testing.assert_allclose(right.position[:, 1], left.position[:, 1], atol=1e-9)


def test_banked_curve_rolls_frame_and_stays_orthonormal():
    from openrct2_track_generator.constants import BANK_ANGLE

    d = np.linspace(0.0, curves.FLAT_LENGTH, 6)
    flat = curves.flat_curve(d)
    left = curves.left_bank_curve(d)
    # Banking leaves position + tangent, rolls normal/binormal by BANK_ANGLE.
    np.testing.assert_allclose(left.position, flat.position)
    np.testing.assert_allclose(left.tangent, flat.tangent)
    assert not np.allclose(left.normal, flat.normal)  # tilted
    _assert_orthonormal(left)
    # Roll angle: the new normal makes BANK_ANGLE with the old one.
    cosang = np.sum(left.normal * flat.normal, axis=1)
    np.testing.assert_allclose(cosang, np.cos(BANK_ANGLE), atol=1e-6)


@pytest.mark.parametrize(
    "curve,length",
    [
        (curves.flat_to_left_bank_curve, curves.FLAT_LENGTH),
        (curves.flat_to_right_bank_curve, curves.FLAT_LENGTH),
        (curves.left_bank_to_gentle_curve, curves.FLAT_TO_GENTLE_LENGTH),
        (curves.right_bank_to_gentle_curve, curves.FLAT_TO_GENTLE_LENGTH),
        (curves.gentle_to_left_bank_curve, curves.FLAT_TO_GENTLE_LENGTH),
        (curves.gentle_to_right_bank_curve, curves.FLAT_TO_GENTLE_LENGTH),
        (curves.small_turn_left_bank_curve, curves.SMALL_TURN_LENGTH),
        (curves.medium_turn_left_bank_curve, curves.MEDIUM_TURN_LENGTH),
        (curves.large_turn_left_to_diag_bank_curve, curves.LARGE_TURN_LENGTH),
        (curves.large_turn_right_to_diag_bank_curve, curves.LARGE_TURN_LENGTH),
        (curves.s_bend_left_curve, curves.S_BEND_LENGTH),
        (curves.s_bend_right_curve, curves.S_BEND_LENGTH),
    ],
)
def test_batch_d_curves_orthonormal(curve, length):
    _assert_orthonormal(curve(np.linspace(0.0, length, 8)))


def test_s_bends_mirror_in_x():
    d = np.linspace(0.0, curves.S_BEND_LENGTH, 8)
    left = curves.s_bend_left_curve(d)
    right = curves.s_bend_right_curve(d)
    np.testing.assert_allclose(right.position[:, 0], -left.position[:, 0], atol=1e-6)
    np.testing.assert_allclose(right.position[:, 2], left.position[:, 2], atol=1e-6)


def test_small_helix_climbs_while_turning():
    d = np.linspace(0.0, curves.SMALL_HELIX_LENGTH, 8)
    tp = curves.small_helix_left_curve(d)
    # A helix gains height monotonically as it turns.
    assert np.all(np.diff(tp.position[:, 1]) > 0)
    _assert_orthonormal(tp)


@pytest.mark.parametrize(
    "curve,length",
    [
        (curves.flat_to_left_bank_diag_curve, curves.FLAT_DIAG_LENGTH),
        (curves.flat_to_right_bank_diag_curve, curves.FLAT_DIAG_LENGTH),
        (curves.left_bank_diag_curve, curves.FLAT_DIAG_LENGTH),
        (curves.left_bank_to_gentle_diag_curve, curves.FLAT_TO_GENTLE_DIAG_LENGTH),
        (curves.right_bank_to_gentle_diag_curve, curves.FLAT_TO_GENTLE_DIAG_LENGTH),
        (curves.gentle_to_left_bank_diag_curve, curves.FLAT_TO_GENTLE_DIAG_LENGTH),
        (curves.gentle_to_right_bank_diag_curve, curves.FLAT_TO_GENTLE_DIAG_LENGTH),
        (curves.s_bend_left_bank_curve, curves.S_BEND_LENGTH),
        (curves.s_bend_right_bank_curve, curves.S_BEND_LENGTH),
        (curves.small_helix_right_curve, curves.SMALL_HELIX_LENGTH),
        (curves.medium_helix_left_curve, curves.MEDIUM_HELIX_LENGTH),
        (curves.medium_helix_right_curve, curves.MEDIUM_HELIX_LENGTH),
    ],
)
def test_batch_d_followup_curves_orthonormal(curve, length):
    _assert_orthonormal(curve(np.linspace(0.0, length, 8)))


def test_barrel_roll_inverts_and_orthonormal():
    d = np.linspace(0.0, curves.BARREL_ROLL_LENGTH, 16)
    tp = curves.barrel_roll_left_curve(d)
    _assert_orthonormal(tp)
    assert tp.normal[:, 1].min() < -0.9  # the roll carries the track fully inverted
    # Tangent is forced straight (+Z) at both ends so it joins flat track.
    np.testing.assert_allclose(tp.tangent[0], [0.0, 0.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(tp.tangent[-1], [0.0, 0.0, 1.0], atol=1e-6)


@pytest.mark.parametrize(
    "curve,length",
    [
        (curves.barrel_roll_right_curve, curves.BARREL_ROLL_LENGTH),
        (curves.corkscrew_left_curve, curves.CORKSCREW_LENGTH),
        (curves.corkscrew_right_curve, curves.CORKSCREW_LENGTH),
        (curves.inline_twist_left_curve, curves.INLINE_TWIST_LENGTH),
        (curves.inline_twist_right_curve, curves.INLINE_TWIST_LENGTH),
        (curves.large_corkscrew_left_curve, curves.LARGE_CORKSCREW_LENGTH),
        (curves.large_corkscrew_right_curve, curves.LARGE_CORKSCREW_LENGTH),
        (curves.quarter_loop_curve, curves.QUARTER_LOOP_LENGTH),
        (curves.half_loop_curve, curves.HALF_LOOP_LENGTH),
        (curves.medium_half_loop_left_curve, curves.MEDIUM_HALF_LOOP_LENGTH),
        (curves.medium_half_loop_right_curve, curves.MEDIUM_HALF_LOOP_LENGTH),
        (curves.large_half_loop_left_curve, curves.LARGE_HALF_LOOP_LENGTH),
        (curves.large_half_loop_right_curve, curves.LARGE_HALF_LOOP_LENGTH),
        (curves.zero_g_roll_left_curve, curves.ZERO_G_ROLL_LENGTH),
        (curves.zero_g_roll_right_curve, curves.ZERO_G_ROLL_LENGTH),
        (curves.large_zero_g_roll_left_curve, curves.LARGE_ZERO_G_ROLL_LENGTH),
        (curves.large_zero_g_roll_right_curve, curves.LARGE_ZERO_G_ROLL_LENGTH),
        (curves.dive_loop_45_left_curve, curves.DIVE_LOOP_45_LENGTH),
        (curves.dive_loop_45_right_curve, curves.DIVE_LOOP_45_LENGTH),
        (curves.vertical_twist_left_curve, curves.VERTICAL_TWIST_LENGTH),
        (curves.vertical_twist_right_curve, curves.VERTICAL_TWIST_LENGTH),
        (curves.vertical_twist_left_to_diag_curve, curves.VERTICAL_TWIST_45_LENGTH),
        (curves.vertical_twist_right_to_diag_curve, curves.VERTICAL_TWIST_45_LENGTH),
        (curves.vertical_twist_left_to_orthogonal_curve, curves.VERTICAL_TWIST_45_LENGTH),
        (curves.vertical_twist_right_to_orthogonal_curve, curves.VERTICAL_TWIST_45_LENGTH),
        (curves.vertical_loop_left_curve, curves.VERTICAL_LOOP_LENGTH),
        (curves.vertical_loop_right_curve, curves.VERTICAL_LOOP_LENGTH),
    ],
)
def test_inversion_curves_orthonormal(curve, length):
    _assert_orthonormal(curve(np.linspace(0.0, length, 16)))


def test_vertical_twist_45_variants_sweep_correct_halves():
    # to_diag sweeps the first 45° (normal x reaches ~ -sin45 at the end);
    # to_orthogonal starts at 45° and sweeps to 90° (normal x reaches ~ -1).
    diag = curves.vertical_twist_left_to_diag_curve(np.array([curves.VERTICAL_TWIST_45_LENGTH]))
    np.testing.assert_allclose(diag.normal[0], [-np.sin(np.pi / 4), 0.0, -np.cos(np.pi / 4)],
                               atol=1e-6)
    ortho_start = curves.vertical_twist_left_to_orthogonal_curve(np.array([0.0]))
    np.testing.assert_allclose(ortho_start.normal[0],
                               [-np.sin(np.pi / 4), 0.0, -np.cos(np.pi / 4)], atol=1e-6)
    ortho_end = curves.vertical_twist_left_to_orthogonal_curve(
        np.array([curves.VERTICAL_TWIST_45_LENGTH])
    )
    np.testing.assert_allclose(ortho_end.normal[0], [-1.0, 0.0, 0.0], atol=1e-6)


def test_vertical_loop_inverts_and_mirrors():
    d = np.linspace(0.0, curves.VERTICAL_LOOP_LENGTH, 24)
    left = curves.vertical_loop_left_curve(d)
    assert left.normal[:, 1].min() < -0.9  # fully inverted over the top of the loop
    right = curves.vertical_loop_right_curve(d)
    np.testing.assert_allclose(right.position[:, 0], -left.position[:, 0], atol=1e-6)


def test_launch_pinned_curve_sampled_away_from_start():
    # Sampling strictly past the launch zone skips the start-pin override entirely
    # (the `distance < 0.001` branch is empty), exercising the fast path.
    d = np.linspace(0.5, curves.ZERO_G_ROLL_LENGTH, 8)
    assert d.min() > 1e-3
    _assert_orthonormal(curves.zero_g_roll_left_curve(d))


def test_vertical_twist_climbs_and_rotates():
    d = np.linspace(0.0, curves.VERTICAL_TWIST_LENGTH, 9)
    left = curves.vertical_twist_left_curve(d)
    # Climbs straight up: position is purely +Y, tangent is +Y throughout.
    np.testing.assert_allclose(left.position[:, [0, 2]], 0.0, atol=1e-12)
    np.testing.assert_allclose(left.tangent, np.tile([0.0, 1.0, 0.0], (9, 1)), atol=1e-12)
    # The normal twists from -Z toward -X (left) / +X (right) over the climb.
    right = curves.vertical_twist_right_curve(d)
    np.testing.assert_allclose(right.normal[:, 0], -left.normal[:, 0], atol=1e-9)


def test_zero_g_roll_rolls_and_mirrors():
    d = np.linspace(0.0, curves.ZERO_G_ROLL_LENGTH, 16)
    left = curves.zero_g_roll_left_curve(d)
    # The roll carries the frame past vertical (normal tips well off +Y).
    assert left.normal[:, 1].min() < 0.0
    right = curves.zero_g_roll_right_curve(d)
    np.testing.assert_allclose(right.position[:, 0], -left.position[:, 0], atol=1e-6)


def test_dive_loop_45_right_is_xz_rotation_of_left():
    d = np.linspace(0.0, curves.DIVE_LOOP_45_LENGTH, 8)
    left = curves.dive_loop_45_left_curve(d)
    right = curves.dive_loop_45_right_curve(d)
    # Right rotates the left 90° in the ground plane: x' = -z, z' = -x.
    np.testing.assert_allclose(right.position[:, 0], -left.position[:, 2], atol=1e-6)
    np.testing.assert_allclose(right.position[:, 2], -left.position[:, 0], atol=1e-6)


@pytest.mark.parametrize(
    "left_curve,right_curve,length",
    [
        (curves.medium_half_loop_left_curve, curves.medium_half_loop_right_curve,
         curves.MEDIUM_HALF_LOOP_LENGTH),
        (curves.large_half_loop_left_curve, curves.large_half_loop_right_curve,
         curves.LARGE_HALF_LOOP_LENGTH),
    ],
)
def test_half_loops_invert_and_mirror(left_curve, right_curve, length):
    d = np.linspace(0.0, length, 24)
    left = left_curve(d)
    assert left.normal[:, 1].min() < -0.9  # fully inverted over the top
    np.testing.assert_allclose(right_curve(d).position[:, 0], -left.position[:, 0], atol=1e-6)


def test_half_loop_inverts_and_exits_backward():
    d = np.linspace(0.0, curves.HALF_LOOP_LENGTH, 24)
    tp = curves.half_loop_curve(d)
    assert tp.normal[:, 1].min() < -0.9  # fully inverted over the top
    # Exits inverted, heading back along -Z (a 180° exit).
    np.testing.assert_allclose(tp.tangent[-1], [0.0, 0.0, -1.0], atol=1e-3)


def test_corkscrew_left_right_mirror_in_x():
    d = np.linspace(0.0, curves.CORKSCREW_LENGTH, 8)
    left = curves.corkscrew_left_curve(d)
    right = curves.corkscrew_right_curve(d)
    np.testing.assert_allclose(right.position[:, 0], -left.position[:, 0], atol=1e-6)


@pytest.mark.parametrize(
    "left_curve,right_curve,length",
    [
        (curves.inline_twist_left_curve, curves.inline_twist_right_curve,
         curves.INLINE_TWIST_LENGTH),
        (curves.large_corkscrew_left_curve, curves.large_corkscrew_right_curve,
         curves.LARGE_CORKSCREW_LENGTH),
    ],
)
def test_new_inversions_mirror_in_x(left_curve, right_curve, length):
    d = np.linspace(0.0, length, 8)
    np.testing.assert_allclose(
        right_curve(d).position[:, 0], -left_curve(d).position[:, 0], atol=1e-6
    )


def test_inline_twist_ends_join_flat_track():
    d = np.linspace(0.0, curves.INLINE_TWIST_LENGTH, 16)
    tp = curves.inline_twist_left_curve(d)
    np.testing.assert_allclose(tp.tangent[0], [0.0, 0.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(tp.tangent[-1], [0.0, 0.0, 1.0], atol=1e-6)


def test_quarter_loop_pitches_from_vertical_to_inverted():
    d = np.linspace(0.0, curves.QUARTER_LOOP_LENGTH, 16)
    tp = curves.quarter_loop_curve(d)
    # The top quarter of a loop: enters heading straight up (+Y) and exits inverted,
    # heading back along -Z (a 180° exit).
    np.testing.assert_allclose(tp.tangent[0], [0.0, 1.0, 0.0], atol=1e-3)
    np.testing.assert_allclose(tp.tangent[-1], [0.0, 0.0, -1.0], atol=1e-3)


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
