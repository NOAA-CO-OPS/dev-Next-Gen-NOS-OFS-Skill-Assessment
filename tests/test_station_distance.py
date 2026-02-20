"""
Pytest unit tests for calculate_station_distance.

Tests geographic distance calculation (Haversine) in
ofs_skill.model_processing.station_distance.
"""

import math

import pytest

from ofs_skill.model_processing.station_distance import calculate_station_distance


class TestCalculateStationDistance:
    """Tests for calculate_station_distance."""

    def test_identical_coordinates_returns_zero(self):
        """Identical coordinates must yield distance 0.0 km."""
        dist = calculate_station_distance(37.0, -76.0, 37.0, -76.0)
        assert dist == 0.0

    def test_one_degree_longitude_at_equator_approx_111_km(self):
        """1 degree longitude at equator (0,0) to (0,1) is ~111.19 km."""
        dist = calculate_station_distance(0.0, 0.0, 0.0, 1.0)
        assert math.isclose(dist, 111.19, rel_tol=0.01)

    def test_distance_is_non_negative(self):
        """Distance must always be non-negative."""
        pairs = [
            (0.0, 0.0, 1.0, 1.0),
            (-90.0, 0.0, 90.0, 180.0),
            (36.9, -76.3, 39.3, -76.6),
        ]
        for lat1, lon1, lat2, lon2 in pairs:
            dist = calculate_station_distance(lat1, lon1, lat2, lon2)
            assert dist >= 0.0, f"distance({lat1},{lon1},{lat2},{lon2}) = {dist}"

    def test_symmetry_distance_a_b_equals_distance_b_a(self):
        """distance(A, B) must equal distance(B, A)."""
        a = (36.9, -76.3)
        b = (39.3, -76.6)
        d_ab = calculate_station_distance(a[0], a[1], b[0], b[1])
        d_ba = calculate_station_distance(b[0], b[1], a[0], a[1])
        assert d_ab == d_ba


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
