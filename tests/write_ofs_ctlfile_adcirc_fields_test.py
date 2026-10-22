"""``write_ofs_ctlfile`` and water-level extraction for STOFS-2D-Global fields.

Runs the ADCIRC fields branch end to end on a small synthetic file that
mirrors the real one (see tests/helpers/adcirc_minimal.py):

1. ``write_ofs_ctlfile`` writes ``{ofs}_wl_model.ctl`` with one
   ``node 0 lat lon id 0.0`` line per matched station, lat/lon taken from
   the matched node's y/x.
2. A station whose only nearby node is always dry and a station beyond the
   km cutoff are left out; no line has node -1.
3. If no station matches, the variable is skipped without raising.
4. ``index_nearest_depth`` gives layer 0 / depth 0.0 for water level.
5. ``format_waterlevel`` (per-station and batch paths) turns a -99999.0
   step of the matched node into NaN via ``_mask_adcirc_fill`` and shifts
   the other steps by the datum offset.

Datum is MSL, for which ``get_datum_offset`` returns 0 for stofs_2d_glo
without calling vdatum.
"""

from __future__ import annotations

import importlib
import logging
from types import SimpleNamespace

import numpy as np
import pytest

from ofs_skill.model_processing.indexing import index_nearest_depth
from ofs_skill.model_processing.write_ofs_ctlfile import write_ofs_ctlfile
from tests.helpers.adcirc_minimal import (
    ADCIRC_FILL,
    ADCIRC_NODES,
    ALWAYS_DRY_NODE,
    ONE_STEP_DRY_NODE,
    ONE_STEP_DRY_STEP,
    build_adcirc_minimal_dataset,
    write_adcirc_config,
)
from tests.helpers.fvcom_minimal import write_obs_station_ctl

# The package re-exports a get_node_ofs *function* under the module's name,
# so import the module itself.
get_node_ofs = importlib.import_module('ofs_skill.model_processing.get_node_ofs')

OFS = 'stofs_2d_glo'
_LOGGER = logging.getLogger('write_ofs_ctlfile_adcirc_fields_test')


def _node_station(sid, node, dlat=0.0):
    lon, lat = ADCIRC_NODES[node]
    return (sid, lat + dlat, lon, 0.0)


# (id, lat, lon, depth). S_A is ~1.1 km north of node 0, so the ctl line
# must carry node 0's coordinates, not the station's.
S_A = _node_station('9000001', 0, dlat=0.01)
S_B = _node_station('9000002', 1)
S_DRY = _node_station('9000003', ALWAYS_DRY_NODE)      # only nearby node always dry
S_ONE = _node_station('9000004', ONE_STEP_DRY_NODE)    # node dry at one step
S_FAR = _node_station('9000005', 4, dlat=0.1)          # ~11 km: beyond 4 km


def _setup(tmp_path, stations, dry_check_stride=1):
    model = build_adcirc_minimal_dataset(tmp_path)
    cfg = write_adcirc_config(tmp_path, max_dist_km=4.0,
                              dry_check_stride=dry_check_stride)
    control_dir = tmp_path / 'control_files'
    control_dir.mkdir()
    write_obs_station_ctl(control_dir, OFS, 'wl', stations)
    prop = SimpleNamespace(
        config_file=str(cfg),
        ofs=OFS,
        var_list=['water_level'],
        ofsfiletype='fields',
        user_input_location=False,
        model_source='adcirc',
        control_files_path=str(control_dir),
        datum='MSL',
        whichcast='nowcast',
        start_date_full='2026-10-20T00:00:00Z',
        end_date_full='2026-10-20T05:00:00Z',
        ctl_flag=1,
    )
    return model, prop, control_dir / f'{OFS}_wl_model.ctl'


def _ctl_rows(path):
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    return [ln.split() for ln in lines[1:]]   # skip the single header row


@pytest.mark.integration
def test_ctl_file_written_in_adcirc_format(tmp_path):
    # Tests 1 and 2. S_ONE is absent here because its node is dry at one
    # step and the default stride checks every step (dry-rule dependent).
    stations = [S_A, S_DRY, S_B, S_ONE, S_FAR]
    model, prop, out_path = _setup(tmp_path, stations)
    write_ofs_ctlfile(prop, model, _LOGGER)

    assert out_path.exists()
    rows = _ctl_rows(out_path)
    assert [r[4] for r in rows] == [S_A[0], S_B[0]]
    x = model['x'].values
    y = model['y'].values
    for row, node in zip(rows, [0, 1]):
        assert len(row) == 6, row
        assert int(row[0]) == node
        assert row[1] == '0'
        assert float(row[2]) == pytest.approx(round(float(y[node]), 3), abs=5e-4)
        assert float(row[3]) == pytest.approx(round(float(x[node]), 3), abs=5e-4)
        assert row[5] == '0.0'
    # S_A sits 0.01 deg north of node 0; the line has the node's latitude.
    assert float(rows[0][2]) != pytest.approx(S_A[1], abs=5e-4)
    written_ids = {r[4] for r in rows}
    assert S_DRY[0] not in written_ids and S_FAR[0] not in written_ids
    assert all(int(r[0]) != -1 for r in rows)


@pytest.mark.integration
def test_all_stations_unmatched_skips_variable(tmp_path, caplog):
    # Test 3.
    model, prop, out_path = _setup(tmp_path, [S_DRY, S_FAR])
    with caplog.at_level(logging.WARNING, logger=_LOGGER.name):
        write_ofs_ctlfile(prop, model, _LOGGER)
    assert any('No user input locations or model nodes found for wl'
               in r.getMessage() for r in caplog.records)
    assert not out_path.exists()


def test_index_nearest_depth_water_level_is_layer_0(tmp_path):
    # Test 4: matched nodes get layer 0 / depth 0.0; unmatched stay NaN.
    model = build_adcirc_minimal_dataset(tmp_path)
    prop = SimpleNamespace(ofsfiletype='fields', ofs=OFS)
    extract = [[str(s[1]), str(s[2]), '0.0'] for s in (S_A, S_B, S_FAR)]
    layers, depths = index_nearest_depth(
        prop, [0, 1, np.nan], model, extract, 'adcirc', 'wl', OFS, _LOGGER)
    assert layers[:2] == [0, 0]
    assert list(depths[:2]) == [0.0, 0.0]
    assert np.isnan(layers[2]) and np.isnan(depths[2])


def _values(formatted_series):
    return np.array([float(line.split()[-1]) for line in formatted_series])


@pytest.mark.integration
@pytest.mark.parametrize('batch', [False, True], ids=['per_station', 'batch'])
@pytest.mark.parametrize('offset', [None, 0.25], ids=['msl_0', 'patched_0.25'])
def test_fill_step_is_nan_and_others_shifted(tmp_path, monkeypatch, batch, offset):
    # Test 5. adcirc_dry_check_stride=2 checks steps 0, 2, 4, so node 3
    # (dry at step 1 only) is matched (dry-rule dependent) and its -99999.0
    # step reaches extraction.
    model, prop, _ = _setup(tmp_path, [S_A, S_ONE], dry_check_stride=2)
    write_ofs_ctlfile(prop, model, _LOGGER)
    ofs_ctlfile = get_node_ofs.ofs_ctlfile_extract(prop, 'wl', model, _LOGGER)
    assert ofs_ctlfile is not None
    i = ofs_ctlfile[4].index(S_ONE[0])
    assert ofs_ctlfile[1][i] == ONE_STEP_DRY_NODE

    if offset is not None:
        monkeypatch.setattr(get_node_ofs, 'get_datum_offset_func',
                            lambda *a, **k: offset)
    expected_offset = 0 if offset is None else offset

    masked = []
    real_mask = get_node_ofs._mask_adcirc_fill

    def spy(arr, station_id, ofs, log):
        out = real_mask(arr, station_id, ofs, log)
        masked.append((station_id, out))
        return out

    monkeypatch.setattr(get_node_ofs, '_mask_adcirc_fill', spy)

    precomputed = None
    if batch:
        precomputed = get_node_ofs._precompute_stations_data(
            prop, model, ofs_ctlfile, 'zeta', _LOGGER)
        assert precomputed is not None
    series, datum_offset = get_node_ofs.format_waterlevel(
        prop, model, ofs_ctlfile, 'zeta', i, _LOGGER, precomputed=precomputed)
    assert datum_offset == expected_offset

    zeta = model['zeta'].values[:, ONE_STEP_DRY_NODE].astype(float)
    assert zeta[ONE_STEP_DRY_STEP] == ADCIRC_FILL

    # The mask ran for this station and NaN'd exactly the fill step.
    assert [sid for sid, _ in masked] == [S_ONE[0]]
    out = masked[0][1]
    assert np.isnan(out[ONE_STEP_DRY_STEP])
    assert np.isnan(out).sum() == 1

    values = _values(series)
    assert len(values) == len(zeta)
    assert np.isnan(values[ONE_STEP_DRY_STEP])
    others = np.arange(len(zeta)) != ONE_STEP_DRY_STEP
    np.testing.assert_allclose(values[others],
                               zeta[others] - expected_offset, atol=5e-5)
