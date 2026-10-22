"""
ADCIRC (STOFS-2D-Global) fields-mode nearest-node search.

index_nearest_node's ADCIRC branch:
- matches on 'x'/'y' (node,) coordinates with a unit-sphere KD-tree, so it is
  exact for great-circle distance and continuous across the antimeridian;
- excludes nodes whose zeta is the ADCIRC fill value (-99999) at any
  time step, so only nodes wet for the whole record are matched (the file
  has no _FillValue attribute, so these are not NaN after decoding); every
  unique time step is checked;
- returns NaN for stations with no eligible node within max_dist_km
  (default STATION_MATCH_MAX_DIST_KM).
"""
from __future__ import annotations

import logging

import numpy as np
import pytest
import xarray as xr

from ofs_skill.model_processing.get_node_ofs import _mask_adcirc_fill
from ofs_skill.model_processing.indexing import (
    ADCIRC_FILL_VALUE,
    STATION_MATCH_MAX_DIST_KM,
    index_nearest_node,
)
from ofs_skill.model_processing.station_distance import calculate_station_distance

_LOGGER = logging.getLogger('indexing_adcirc_nearest_node_test')

# Node layout (lon, lat). Node 1 sits right next to the station at
# (-70.0, 40.0) but is permanently dry; node 5 is dry at one step only;
# node 2 is the nearest always-wet node.
_NODES = [
    (-70.10, 40.00),   # 0 wet, ~8.5 km west of station A
    (-70.001, 40.0),   # 1 permanently dry, ~0.09 km from station A
    (-70.02, 40.00),   # 2 wet, ~1.7 km from station A
    (179.99, -17.0),   # 3 wet, just east of the antimeridian
    (-60.00, 10.00),   # 4 wet, isolated
    (-69.99, 40.00),   # 5 partially dry (fill at step 1 only), ~0.85 km
]


def _dataset(n_time=4, nodes=None, xy_layout='node'):
    """Synthetic fields dataset mirroring the real file's types.

    x/y are float64 in -180..180; zeta is float32 (time, node) with dry
    values exactly ADCIRC_FILL_VALUE (no _FillValue, no NaN). With the
    default ``nodes`` (_NODES), node 1 is dry at every step and node 5 at
    step 1; other node lists are wet everywhere. ``xy_layout='time_node'``
    replicates x/y along time, the legacy data_vars='all' intake layout.
    """
    default_nodes = nodes is None
    nodes = _NODES if default_nodes else nodes
    x = np.array([p[0] for p in nodes], dtype='f8')
    y = np.array([p[1] for p in nodes], dtype='f8')
    zeta = np.tile(np.linspace(0.1, 0.6, len(nodes)), (n_time, 1)).astype('f4')
    if default_nodes:
        zeta[:, 1] = ADCIRC_FILL_VALUE
        zeta[1, 5] = ADCIRC_FILL_VALUE
    if xy_layout == 'node':
        xy = {'x': ('node', x), 'y': ('node', y)}
    elif xy_layout == 'time_node':
        xy = {'x': (('time', 'node'), np.tile(x, (n_time, 1))),
              'y': (('time', 'node'), np.tile(y, (n_time, 1)))}
    else:
        raise ValueError(xy_layout)
    return xr.Dataset(
        {'zeta': (('time', 'node'), zeta)},
        coords={**xy, 'time': np.arange(n_time)},
    )


def _ctl(*lat_lon):
    return [[str(lat), str(lon), '0.0'] for lat, lon in lat_lon]


# Stations of test_mixed_stations_keep_order: A (node 2), one beyond the
# cutoff (NaN), one across the antimeridian (node 3).
_MIXED = ((40.0, -70.0), (11.0, -60.0), (-17.0, 179.995))


def _assert_return_contract(result, n_stations, n_nodes):
    """One entry per station; each a Python int node index or a float NaN."""
    assert len(result) == n_stations
    for v in result:
        if type(v) is int:
            assert v in range(n_nodes)
        else:
            assert type(v) is float and np.isnan(v), (type(v), v)
        assert not (type(v) is int and v == -1)


# dry-rule test: depends on "exclude nodes dry at any time step".
def test_dry_nodes_are_skipped():
    # Station A at (-70.0, 40.0): node 1 (0.09 km) is permanently dry and
    # node 5 (0.85 km) is dry at one step, so both are excluded and the
    # always-wet node 2 (1.7 km) is matched.
    result = index_nearest_node(
        _ctl((40.0, -70.0)), _dataset(), 'adcirc', 'wl', 'stofs_2d_glo', _LOGGER)
    assert result == [2]


def test_cutoff_returns_nan_and_default_is_module_constant():
    # Node 4 is ~111 km from (11.0, -60.0): beyond the default 4 km cutoff.
    ds = _dataset()
    result = index_nearest_node(
        _ctl((11.0, -60.0)), ds, 'adcirc', 'wl', 'stofs_2d_glo', _LOGGER)
    assert len(result) == 1 and np.isnan(result[0])

    km = calculate_station_distance(11.0, -60.0, 10.0, -60.0)
    assert km > STATION_MATCH_MAX_DIST_KM
    result = index_nearest_node(
        _ctl((11.0, -60.0)), ds, 'adcirc', 'wl', 'stofs_2d_glo', _LOGGER,
        max_dist_km=km + 1.0)
    assert result == [4]


def test_antimeridian_neighbour_is_found():
    # Station at -179.99 is ~2.1 km from node 3 at +179.99 across the
    # antimeridian; a degree-box or planar search would not find it.
    result = index_nearest_node(
        _ctl((-17.0, -179.99)), _dataset(), 'adcirc', 'wl', 'stofs_2d_glo',
        _LOGGER)
    assert result == [3]


def test_mixed_stations_keep_order():
    result = index_nearest_node(
        _ctl((40.0, -70.0), (11.0, -60.0), (-17.0, 179.995)),
        _dataset(), 'adcirc', 'wl', 'stofs_2d_glo', _LOGGER)
    assert result[0] == 2
    assert np.isnan(result[1])
    assert result[2] == 3


def test_dry_scan_logs_step_count_and_estimate(caplog):
    with caplog.at_level(logging.INFO, logger=_LOGGER.name):
        index_nearest_node(_ctl((40.0, -70.0)), _dataset(n_time=4), 'adcirc',
                           'wl', 'stofs_2d_glo', _LOGGER)
    start_msgs = [r.getMessage() for r in caplog.records
                  if 'dry node scan: checking' in r.getMessage()]
    assert len(start_msgs) == 1
    assert 'checking 4 unique time steps' in start_msgs[0]
    assert 'estimated time' in start_msgs[0]


# dry-rule test: depends on "exclude nodes dry at any time step".
def test_duplicated_timestamps_are_read_once(caplog):
    # intake lists a single fields file twice, so every time value appears
    # twice on the time axis. The second copy here marks node 2 dry at every
    # step: if the repeated steps were read, node 2 would be excluded and
    # station A would get NaN (node 0 is ~8.5 km away). Reading each unique
    # time once keeps node 2.
    first = _dataset(n_time=4)
    repeat = first.copy(deep=True)
    repeat['zeta'][:, 2] = ADCIRC_FILL_VALUE
    ds = xr.concat([first, repeat], dim='time')
    assert ds.sizes['time'] == 8
    ctl = _ctl((40.0, -70.0))
    with caplog.at_level(logging.INFO, logger=_LOGGER.name):
        assert index_nearest_node(ctl, ds, 'adcirc', 'wl', 'stofs_2d_glo',
                                  _LOGGER) == [2]
    msgs = [r.getMessage() for r in caplog.records]
    assert any('4 of 8 loaded time steps repeat' in m for m in msgs)
    assert any('checking 4 unique time steps' in m for m in msgs)


# 1. Time-replicated coordinates.
@pytest.mark.parametrize('xy_layout', ['node', 'time_node'])
def test_xy_layouts_give_same_result(xy_layout):
    result = index_nearest_node(
        _ctl(*_MIXED), _dataset(xy_layout=xy_layout), 'adcirc', 'wl',
        'stofs_2d_glo', _LOGGER)
    reference = index_nearest_node(
        _ctl(*_MIXED), _dataset(xy_layout='node'), 'adcirc', 'wl',
        'stofs_2d_glo', _LOGGER)
    assert result[0] == reference[0] == 2
    assert np.isnan(result[1]) and np.isnan(reference[1])
    assert result[2] == reference[2] == 3


# 2. Every node dry. contract test.
def test_every_node_dry_returns_all_nan():
    ds = _dataset()
    ds['zeta'][:] = ADCIRC_FILL_VALUE
    result = index_nearest_node(
        _ctl(*_MIXED), ds, 'adcirc', 'wl', 'stofs_2d_glo', _LOGGER)
    assert len(result) == len(_MIXED)
    assert all(isinstance(v, float) and np.isnan(v) for v in result)
    assert -1 not in result


# 3. No zeta variable.
def test_no_zeta_warns_and_uses_all_nodes(caplog):
    ds = _dataset().drop_vars('zeta')
    with caplog.at_level(logging.WARNING, logger=_LOGGER.name):
        result = index_nearest_node(
            _ctl((40.0, -70.0)), ds, 'adcirc', 'wl', 'stofs_2d_glo', _LOGGER)
    assert any(r.levelno == logging.WARNING and 'no zeta' in r.getMessage()
               for r in caplog.records)
    # Node 1 (dry when zeta is present) is the nearest node and is eligible.
    assert result == [1]


def _contract_case(name):
    if name == 'time_node_layout':
        return _dataset(xy_layout='time_node')
    ds = _dataset()
    if name == 'all_dry':
        ds['zeta'][:] = ADCIRC_FILL_VALUE
    elif name == 'no_zeta':
        ds = ds.drop_vars('zeta')
    return ds


# 4. Return contract. contract test.
@pytest.mark.parametrize(
    'case', ['node_layout', 'time_node_layout', 'all_dry', 'no_zeta'])
def test_return_contract(case):
    stations = _MIXED + ((40.0, -70.0),)
    result = index_nearest_node(
        _ctl(*stations), _contract_case(case), 'adcirc', 'wl', 'stofs_2d_glo',
        _LOGGER)
    _assert_return_contract(result, len(stations), len(_NODES))


def _point_at_distance(lat, lon, km, northward):
    """Station due north/south of (lat, lon) at ``km`` (6371 km sphere)."""
    dlat = np.degrees(km / 6371.0)
    return (lat + dlat if northward else lat - dlat), lon


# 5. Cutoff boundary.
def test_cutoff_boundary():
    lat4, lon4 = _NODES[4][1], _NODES[4][0]
    inside = _point_at_distance(lat4, lon4, STATION_MATCH_MAX_DIST_KM - 0.1, True)
    outside = _point_at_distance(lat4, lon4, STATION_MATCH_MAX_DIST_KM + 0.1, False)
    km_in = calculate_station_distance(*inside, lat4, lon4)
    km_out = calculate_station_distance(*outside, lat4, lon4)
    assert 3.85 < km_in < STATION_MATCH_MAX_DIST_KM < km_out < 4.15
    result = index_nearest_node(
        _ctl(inside, outside), _dataset(), 'adcirc', 'wl', 'stofs_2d_glo',
        _LOGGER)
    assert result[0] == 4
    assert np.isnan(result[1])


# 6. High latitude. contract test.
def test_high_latitude_uses_great_circle_distance():
    # At 85N a degree of longitude is ~9.7 km but a degree of latitude is
    # ~111 km. Nodes 0 and 1 are at 85N, 10 deg of longitude apart; node 2 is
    # 0.6 deg north of the station. In degrees node 2 is nearest (0.6 vs 4),
    # but on the sphere node 0 is (~39 km vs ~67 km).
    nodes = [(0.0, 85.0), (10.0, 85.0), (4.0, 85.6)]
    station = (85.0, 4.0)
    km = [calculate_station_distance(*station, la, lo) for lo, la in nodes]
    deg = [np.hypot(lo - station[1], la - station[0]) for lo, la in nodes]
    gc_nearest, deg_nearest = int(np.argmin(km)), int(np.argmin(deg))
    assert (gc_nearest, deg_nearest) == (0, 2)
    assert gc_nearest != deg_nearest
    result = index_nearest_node(
        _ctl(station), _dataset(nodes=nodes), 'adcirc', 'wl', 'stofs_2d_glo',
        _LOGGER, max_dist_km=100.0)
    assert result == [gc_nearest]


# 7. Duplicated timestamps. contract test; dry-rule test.
def test_identical_duplicated_timestamps_match_deduplicated(caplog):
    # Single-file intake repeats every step with identical values.
    dedup = _dataset(n_time=4)
    doubled = xr.concat([dedup, dedup], dim='time')
    assert doubled.sizes['time'] == 8
    expected = index_nearest_node(
        _ctl(*_MIXED), dedup, 'adcirc', 'wl', 'stofs_2d_glo', _LOGGER)
    with caplog.at_level(logging.INFO, logger=_LOGGER.name):
        result = index_nearest_node(
            _ctl(*_MIXED), doubled, 'adcirc', 'wl', 'stofs_2d_glo', _LOGGER)
    assert result[0] == expected[0] and result[2] == expected[2]
    assert np.isnan(result[1]) and np.isnan(expected[1])
    start = [r.getMessage() for r in caplog.records
             if 'dry node scan: checking' in r.getMessage()]
    assert len(start) == 1 and 'checking 4 unique time steps' in start[0]


def test_mask_adcirc_fill():
    arr = np.array([0.5, ADCIRC_FILL_VALUE, -0.25], dtype='f4')
    out = _mask_adcirc_fill(arr, '8443970', 'stofs_2d_glo', _LOGGER)
    assert out[0] == pytest.approx(0.5)
    assert np.isnan(out[1])
    assert out[2] == pytest.approx(-0.25)
    # Input is not modified.
    assert arr[1] == ADCIRC_FILL_VALUE
