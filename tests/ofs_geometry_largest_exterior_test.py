"""
Regression: ofs_geometry must select an exterior ring, never a hole, for the
global STOFS-2D-Global extent.

ofs_extents/stofs_2d_glo.shp is a single polygon covering the whole mesh:
one exterior ring (67,797 points) and 6,885 holes (land). Its largest hole,
the Americas coastline, has 241,976 points. The original selection
(get_response_1 / get_response_2) takes the ring with the most points, so the
station inventory was filtered against the Americas land outline: inland
stations (Great Lakes, St. Lawrence) passed, and island and non-American
stations (Key West, Honolulu) were dropped.
"""
from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
import shapefile
from shapely.geometry import Point, Polygon

from ofs_skill.obs_retrieval.ofs_geometry import (
    get_largest_exterior,
    get_response_1,
    get_response_2,
    normalize_longitudes,
    ofs_geometry,
)

_LOGGER = logging.getLogger('ofs_geometry_largest_exterior_test')


def _square(x0, y0, size, n_per_side):
    """Closed square ring as (lon, lat) tuples with n_per_side points per side."""
    step = size / n_per_side
    ring = [(x0 + i * step, y0) for i in range(n_per_side)]
    ring += [(x0 + size, y0 + i * step) for i in range(n_per_side)]
    ring += [(x0 + size - i * step, y0 + size) for i in range(n_per_side)]
    ring += [(x0, y0 + size - i * step) for i in range(n_per_side)]
    return ring + [ring[0]]


def test_polygon_hole_with_more_points_is_not_selected():
    exterior = _square(0.0, 0.0, 10.0, 2)
    hole = _square(2.0, 2.0, 4.0, 50)
    geo = {'type': 'Polygon', 'coordinates': (tuple(exterior), tuple(hole))}

    # The original helper picks the hole because it has more points.
    assert get_response_2(geo)[4] == tuple(hole)

    lat_1, lat_2, lon_1, lon_2, mask = get_largest_exterior(geo)
    assert mask == exterior
    assert (lat_1, lat_2, lon_1, lon_2) == (0.0, 10.0, 0.0, 10.0)


def test_multipolygon_selects_largest_exterior():
    small = _square(20.0, 20.0, 1.0, 2)
    large = _square(0.0, 0.0, 10.0, 5)
    hole = _square(2.0, 2.0, 4.0, 50)
    geo = {
        'type': 'MultiPolygon',
        'coordinates': ((tuple(small),), (tuple(large), tuple(hole))),
    }
    assert get_largest_exterior(geo)[4] == large


def test_stofs_2d_glo_uses_global_exterior():
    mask, lat_1, lat_2, lon_1, lon_2 = ofs_geometry('stofs_2d_glo', None, _LOGGER)

    assert len(mask) == 67797
    assert (lon_1, lon_2) == (-180.0, 180.0)
    assert lat_1 < -85.0
    assert lat_2 == 90.0

    # Stations the hole-based selection excluded now pass the inventory's
    # polygon filter (ofs_inventory_stations buffers the polygon by 0.04).
    polygon = Polygon(mask).buffer(0.04)
    for lon, lat in [(-81.8079, 24.5508),    # Key West, 8724580
                     (-157.8670, 21.3033),   # Honolulu, 1612340
                     (-87.8876, 43.0020)]:   # Milwaukee, 9087057
        assert Point(lon, lat).within(polygon)


def _fake_reader(geo):
    """Stand-in for shapefile.Reader returning one shape with ``geo``."""
    shape = SimpleNamespace(__geo_interface__=geo)
    record = SimpleNamespace(shape=shape)
    return lambda *a, **k: SimpleNamespace(shapeRecords=lambda: [record])


@pytest.mark.parametrize('ofs, expect_exterior', [
    ('stofs_2d_glo', True),
    ('STOFS_2D_GLO', True),
    ('cbofs', False),
    ('stofs_3d_atl', False),
])
def test_exterior_selection_is_gated_on_stofs_2d_glo(monkeypatch, ofs,
                                                     expect_exterior):
    # Same polygon for every OFS name: its hole has more points than its
    # exterior. Only stofs_2d_glo (case-insensitive) selects the exterior;
    # every other OFS keeps the original most-points ring, the hole.
    exterior = _square(0.0, 0.0, 10.0, 2)
    hole = _square(2.0, 2.0, 4.0, 50)
    geo = {'type': 'Polygon', 'coordinates': (tuple(exterior), tuple(hole))}
    # ofs_geometry calls shapefile.Reader through the shapefile module.
    monkeypatch.setattr(shapefile, 'Reader', _fake_reader(geo))

    mask, lat_1, lat_2, lon_1, lon_2 = ofs_geometry(ofs, None, _LOGGER)
    expected = exterior if expect_exterior else hole
    assert [tuple(map(float, p)) for p in mask] == [
        tuple(map(float, p)) for p in expected]
    size = 10.0 if expect_exterior else 4.0
    origin = 0.0 if expect_exterior else 2.0
    assert (lat_1, lat_2, lon_1, lon_2) == (
        origin, origin + size, origin, origin + size)


_OTHER_OFS = sorted(
    p.stem for p in (Path(__file__).resolve().parents[1] / 'ofs_extents').glob('*.shp')
    if p.stem != 'stofs_2d_glo'
)


@pytest.mark.parametrize('ofs', _OTHER_OFS)
def test_other_ofs_selection_unchanged(ofs):
    # On the real extent shapefiles, every OFS other than stofs_2d_glo gets
    # exactly what the original selection (get_response_1, falling back to
    # get_response_2, then the STOFS longitude normalisation) gives.
    path = Path(__file__).resolve().parents[1] / 'ofs_extents' / f'{ofs}.shp'
    first = shapefile.Reader(str(path)).shapeRecords()[0].shape.__geo_interface__
    lat_1, lat_2, lon_1, lon_2, mask = (
        get_response_1(first) or get_response_2(first))
    if 'stofs' in ofs:
        lons = normalize_longitudes([pt[0] for pt in mask])
        mask = [(float(lon), pt[1]) for lon, pt in zip(lons, mask)]
        lon_1, lon_2 = float(min(lons)), float(max(lons))

    assert ofs_geometry(ofs, None, _LOGGER) == (mask, lat_1, lat_2, lon_1, lon_2)
