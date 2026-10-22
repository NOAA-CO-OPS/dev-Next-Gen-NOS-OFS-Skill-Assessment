"""
ADCIRC (STOFS-2D-Global) fields-mode nearest-node search.

index_nearest_node's ADCIRC branch:
- matches on 'x'/'y' (node,) coordinates with a unit-sphere KD-tree, so it is
  exact for great-circle distance and continuous across the antimeridian;
- excludes nodes whose zeta is the ADCIRC fill value (-99999) at every
  checked time step (permanently dry; the file has no _FillValue attribute,
  so these are not NaN after decoding); every dry_check_stride-th step is
  checked, default every step;
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
    ADCIRC_DRY_CHECK_STRIDE,
    ADCIRC_FILL_VALUE,
    STATION_MATCH_MAX_DIST_KM,
    index_nearest_node,
)
from ofs_skill.model_processing.station_distance import calculate_station_distance
from ofs_skill.obs_retrieval.utils import get_adcirc_dry_check_stride

_LOGGER = logging.getLogger('indexing_adcirc_nearest_node_test')

# Node layout (lon, lat). Node 1 sits right next to the station at
# (-70.0, 40.0) but is permanently dry; node 2 is the nearest wet node.
_NODES = [
    (-70.10, 40.00),   # 0 wet, ~8.5 km west of station A
    (-70.001, 40.0),   # 1 permanently dry, ~0.09 km from station A
    (-70.02, 40.00),   # 2 wet, ~1.7 km from station A
    (179.99, -17.0),   # 3 wet, just east of the antimeridian
    (-60.00, 10.00),   # 4 wet, isolated
    (-69.99, 40.00),   # 5 partially dry (fill at one step only)
]


def _dataset(n_time=4):
    x = np.array([p[0] for p in _NODES])
    y = np.array([p[1] for p in _NODES])
    zeta = np.tile(np.linspace(0.1, 0.6, len(_NODES)), (n_time, 1)).astype('f4')
    zeta[:, 1] = ADCIRC_FILL_VALUE
    zeta[0, 5] = ADCIRC_FILL_VALUE
    return xr.Dataset(
        {'zeta': (('time', 'node'), zeta)},
        coords={'x': ('node', x), 'y': ('node', y),
                'time': np.arange(n_time)},
    )


def _ctl(*lat_lon):
    return [[str(lat), str(lon), '0.0'] for lat, lon in lat_lon]


def test_permanently_dry_node_is_skipped():
    # Station A at (-70.0, 40.0): node 5 (partially dry, 0.85 km) is eligible
    # and nearer than node 2; node 1 (0.09 km) is permanently dry.
    result = index_nearest_node(
        _ctl((40.0, -70.0)), _dataset(), 'adcirc', 'wl', 'stofs_2d_glo', _LOGGER)
    assert result == [5]


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
    assert result[0] == 5
    assert np.isnan(result[1])
    assert result[2] == 3


def test_stride_trades_accuracy_for_speed(caplog):
    # Node 5 is the fill value at step 0 only (of 4). Strides 1 and 2 check
    # a step where it is wet, so it stays eligible. Stride 4 checks step 0
    # only, so it is classified as always dry and the match moves to node 2.
    ds = _dataset(n_time=4)
    ctl = _ctl((40.0, -70.0))
    with caplog.at_level(logging.INFO, logger=_LOGGER.name):
        assert index_nearest_node(ctl, ds, 'adcirc', 'wl', 'stofs_2d_glo',
                                  _LOGGER, dry_check_stride=1) == [5]
    start_msgs = [r.getMessage() for r in caplog.records
                  if 'permanently-dry node scan: checking' in r.getMessage()]
    assert start_msgs and 'checking 4 of 4 time steps' in start_msgs[0]
    assert 'estimated time' in start_msgs[0]
    assert 'adcirc_dry_check_stride > 1' in start_msgs[0]

    assert index_nearest_node(ctl, ds, 'adcirc', 'wl', 'stofs_2d_glo',
                              _LOGGER, dry_check_stride=2) == [5]
    assert index_nearest_node(ctl, ds, 'adcirc', 'wl', 'stofs_2d_glo',
                              _LOGGER, dry_check_stride=4) == [2]


def test_stride_must_be_positive():
    with pytest.raises(ValueError, match='dry_check_stride'):
        index_nearest_node(_ctl((40.0, -70.0)), _dataset(), 'adcirc', 'wl',
                           'stofs_2d_glo', _LOGGER, dry_check_stride=0)


@pytest.mark.parametrize('body, expected', [
    ('[settings]\nadcirc_dry_check_stride=3\n', 3),
    ('[settings]\nstatic_plots=False\n', ADCIRC_DRY_CHECK_STRIDE),
    ('[settings]\nadcirc_dry_check_stride=\n', ADCIRC_DRY_CHECK_STRIDE),
    ('[settings]\nadcirc_dry_check_stride=0\n', ADCIRC_DRY_CHECK_STRIDE),
    ('[settings]\nadcirc_dry_check_stride=2.5\n', ADCIRC_DRY_CHECK_STRIDE),
])
def test_dry_check_stride_config(tmp_path, body, expected):
    cfg = tmp_path / 'ofs_dps.conf'
    cfg.write_text(body)
    assert get_adcirc_dry_check_stride(_LOGGER, config_file=str(cfg)) == expected


def test_mask_adcirc_fill():
    arr = np.array([0.5, ADCIRC_FILL_VALUE, -0.25], dtype='f4')
    out = _mask_adcirc_fill(arr, '8443970', 'stofs_2d_glo', _LOGGER)
    assert out[0] == pytest.approx(0.5)
    assert np.isnan(out[1])
    assert out[2] == pytest.approx(-0.25)
    # Input is not modified.
    assert arr[1] == ADCIRC_FILL_VALUE
