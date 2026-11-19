"""Tests for format_paired_one_d.get_distance_angle — the signed angular-
difference helper used by current direction pairing/skill — and for
``_direction_bias``, the array form that builds the DIR_BIAS column."""
import unittest

import numpy as np

from ofs_skill.skill_assessment.format_paired_one_d import (
    _direction_bias,
    get_distance_angle,
)


class TestGetDistanceAngle(unittest.TestCase):
    """
    Unit tests for the `get_distance_angle` function.

    The `get_distance_angle(ofs_angle, obs_angle)` function computes the signed
    angular difference between a model (OFS) direction and an observation (OBS)
    direction in degrees.

    A positive result indicates that the OFS angle is clockwise relative to
    the OBS angle, while a negative result indicates it is counter-clockwise.
    Crucially, it must correctly handle the 0/360-degree wraparound.

    This test suite verifies:
    - Identical angles evaluating to 0.
    - Standard angular differences (no boundary crossing).
    - Wraparound angular differences (crossing the 360/0 boundary).
    - Edge cases where angles are exactly 180 degrees apart.
    """

    def test_identical_angles(self):
        """
        Test that identical OFS and OBS angles return a 0.0 degree difference.
        Evaluates various quadrants including exactly 0 and 360.
        """
        self.assertEqual(get_distance_angle(90.0, 90.0), 0.0)
        self.assertEqual(get_distance_angle(0.0, 0.0), 0.0)
        self.assertEqual(get_distance_angle(360.0, 360.0), 0.0)

    def test_standard_clockwise(self):
        """
        Test cases where the OFS angle is clockwise relative to the OBS angle,
        without crossing the 0/360 boundary.
        Expects a positive float result.
        """
        # OFS is 100, OBS is 90 -> Difference is +10 degrees
        self.assertEqual(get_distance_angle(100.0, 90.0), 10.0)
        self.assertEqual(get_distance_angle(180.0, 90.0), 90.0)

    def test_standard_counter_clockwise(self):
        """
        Test cases where the OFS angle is counter-clockwise relative to the OBS angle,
        without crossing the 0/360 boundary.
        Expects a negative float result.
        """
        # OFS is 80, OBS is 90 -> Difference is -10 degrees
        self.assertEqual(get_distance_angle(80.0, 90.0), -10.0)
        self.assertEqual(get_distance_angle(90.0, 180.0), -90.0)

    def test_wraparound_clockwise(self):
        """
        Test angular differences that cross the 0/360 boundary in a clockwise
        direction. Even though the OFS numerical value is smaller, it is physically
        clockwise of the OBS value.
        Expects a positive float result.
        """
        # OBS is 350, OFS is 10. Physically, OFS is 20 degrees clockwise.
        self.assertEqual(get_distance_angle(10.0, 350.0), 20.0)

    def test_wraparound_counter_clockwise(self):
        """
        Test angular differences that cross the 0/360 boundary in a counter-clockwise
        direction. Even though the OFS numerical value is larger, it is physically
        counter-clockwise of the OBS value.
        Expects a negative float result.
        """
        # OBS is 10, OFS is 350. Physically, OFS is 20 degrees counter-clockwise.
        self.assertEqual(get_distance_angle(350.0, 10.0), -20.0)

    def test_opposite_directions(self):
        """
        Test edge cases where the angles are exactly 180 degrees apart.
        The function should return exactly 180 (sign does not physically matter
        at exactly opposite angles, but we verify absolute magnitude).
        """
        self.assertEqual(abs(get_distance_angle(180.0, 0.0)), 180.0)
        self.assertEqual(abs(get_distance_angle(0.0, 180.0)), 180.0)


class TestDirectionBiasVectorized(unittest.TestCase):
    """`_direction_bias` must reproduce `get_distance_angle` exactly.

    The DIR_BIAS column is written verbatim into every currents
    ``*_pair.int`` file and feeds the direction skill metrics, so the
    array form is held to bit-for-bit equality with the scalar helper it
    replaced — not to a floating-point tolerance. Comparison is done on
    the raw bit patterns so that a sign flip on a zero result, or a
    NaN payload change, would fail rather than compare equal.
    """

    @staticmethod
    def _bits(values):
        """Raw IEEE-754 bit patterns of a float array."""
        return np.asarray(values, dtype=float).view(np.int64)

    def _assert_matches_scalar(self, ofs_dir, obs_dir):
        """Assert the array form equals the scalar loop bit-for-bit.

        NaN results are compared by NaN-ness rather than by bits: the
        two paths can land on differently signed quiet NaNs, which both
        render as ``nan`` in the ``.int`` file and behave identically in
        every downstream comparison. Every finite result, signed zeros
        included, must match exactly.
        """
        expected = np.array(
            [get_distance_angle(a, b) for a, b in zip(ofs_dir, obs_dir)],
            dtype=float,
        )
        actual = _direction_bias(np.asarray(ofs_dir), np.asarray(obs_dir))
        self.assertEqual(actual.dtype, np.dtype('float64'))
        self.assertEqual(len(actual), len(expected))

        nan_expected = np.isnan(expected)
        nan_actual = np.isnan(actual)
        nan_mismatch = np.flatnonzero(nan_expected != nan_actual)
        self.assertEqual(
            nan_mismatch.size, 0,
            f'NaN disagreement at index {nan_mismatch[:1]}'
        )

        finite = ~nan_expected
        mismatch = np.flatnonzero(
            self._bits(actual[finite]) != self._bits(expected[finite]))
        self.assertEqual(
            mismatch.size, 0,
            f'first mismatch at finite index {mismatch[:1]}: '
            f'expected={expected[finite][mismatch[:1]]} '
            f'got={actual[finite][mismatch[:1]]}'
        )
        # The rendered form is what reaches disk, so check it too.
        self.assertEqual(
            [repr(value) for value in actual.tolist()],
            [repr(value) for value in expected.tolist()],
        )

    def test_edge_angle_grid(self):
        """Every pairing of boundary, wrapped, signed-zero, and non-finite
        angles matches the scalar helper."""
        edges = [
            0.0, -0.0, 1e-12, 90.0, 179.999999, 180.0, 180.000001, 270.0,
            359.999999, 360.0, 360.000001, 540.0, 720.0, -1.0, -180.0,
            -360.0, -720.0, float('nan'), float('inf'), float('-inf'),
        ]
        ofs = [a for a in edges for _ in edges]
        obs = [b for _ in edges for b in edges]
        self._assert_matches_scalar(ofs, obs)

    def test_large_random_sweep(self):
        """A wide seeded sweep, including out-of-range angles, matches."""
        rng = np.random.default_rng(20237)
        ofs = rng.uniform(-720.0, 1080.0, 50000)
        obs = rng.uniform(-720.0, 1080.0, 50000)
        self._assert_matches_scalar(ofs, obs)

    def test_missing_directions_propagate_as_nan(self):
        """A missing direction on either side yields NaN, as before."""
        result = _direction_bias(
            np.array([10.0, float('nan'), 10.0]),
            np.array([float('nan'), 10.0, 350.0]),
        )
        self.assertTrue(np.isnan(result[0]))
        self.assertTrue(np.isnan(result[1]))
        self.assertEqual(result[2], 20.0)

    def test_empty_input_returns_empty_float_array(self):
        """A window that filtered out every row must not raise."""
        result = _direction_bias(np.array([]), np.array([]))
        self.assertEqual(result.shape, (0,))
        self.assertEqual(result.dtype, np.dtype('float64'))

    def test_does_not_warn_on_non_finite_input(self):
        """Infinite directions must not leak a numpy RuntimeWarning.

        The pairing call sites sit under broad exception handling, so a
        warning escaping from here would be easy to miss and hard to
        trace back.
        """
        with np.errstate(all='raise'):
            result = _direction_bias(
                np.array([float('inf'), float('-inf')]),
                np.array([10.0, 10.0]),
            )
        self.assertTrue(np.isnan(result).all())


if __name__ == '__main__':
    unittest.main()
