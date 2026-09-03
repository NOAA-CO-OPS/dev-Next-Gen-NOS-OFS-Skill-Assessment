"""Regression test for CHS water-level datum handling in write_obs_ctlfile.

CHS observed water level is labeled with the Great-Lakes datum 'IGLD'. For a
non-Great-Lakes OFS (e.g. stofs_3d_atl) requesting a tidal datum such as MLLW
there is no conversion path, and the underlying coastalmodeling_vdatum package
raises a cryptic ``UnboundLocalError`` ('h_g'). ``_process_chs_station`` must
skip such stations cleanly with an informative warning instead of letting the
crash surface as a misleading "data not found" message.
"""
from __future__ import annotations

import importlib
import logging
from datetime import datetime
from unittest import mock

import pandas as pd

# The package re-exports the ``write_obs_ctlfile`` *function*, shadowing the
# submodule of the same name, so resolve the module via its dotted path.
write_obs_ctlfile = importlib.import_module(
    'ofs_skill.obs_retrieval.write_obs_ctlfile')


def _fake_chs_wl_dataset():
    # data_station['Datum'][1] is read by _process_chs_station, so index 1
    # must carry the datum label CHS assigns to all water-level stations.
    return pd.DataFrame({'Datum': ['IGLD', 'IGLD'], 'WL': [1.1, 1.2]})


def test_chs_water_level_non_gl_skips_when_no_datum_path(caplog):
    """Non-GL OFS requesting MLLW: station is skipped, not crashed."""
    logger = logging.getLogger('chs_datum_skip_test')
    with mock.patch.object(
            write_obs_ctlfile, 'retrieve_chs_station',
            return_value=_fake_chs_wl_dataset()):
        with caplog.at_level(logging.WARNING):
            result = write_obs_ctlfile._process_chs_station(
                id_number='5cebf1e0',
                name='Some Atlantic Canada Station',
                x_value=-65.0,
                y_value=45.0,
                start_date=datetime(2026, 5, 26),
                end_date=datetime(2026, 6, 2),
                variable='water_level',
                name_var='wl',
                datum='MLLW',
                ofs='stofs_3d_atl',
                logger=logger,
            )

    # Skipped cleanly -> empty entry list, no exception propagated.
    assert result == []
    # The warning explains the real reason (no conversion path), not the
    # misleading h_g/'data not found' message.
    msgs = [r.message for r in caplog.records]
    assert any('no datum conversion path' in m for m in msgs), msgs
    assert not any('h_g' in m for m in msgs), msgs


def test_chs_wl_datum_precheck_rejects_non_gl_tidal_datum():
    """The (ofs, datum) verdict is reachable without retrieving anything."""
    logger = logging.getLogger('chs_datum_precheck_test')
    supported, reason = write_obs_ctlfile._chs_water_level_datum_supported(
        ofs='necofs', datum='MLLW', x_value=-65.0, y_value=45.0,
        logger=logger)

    assert supported is False
    assert reason is not None


def test_chs_wl_datum_precheck_allows_great_lakes_ofs():
    """GL OFS have an explicit offset path and must not be skipped."""
    logger = logging.getLogger('chs_datum_precheck_test')
    supported, reason = write_obs_ctlfile._chs_water_level_datum_supported(
        ofs='loofs2', datum='IGLD', x_value=-77.0, y_value=43.5,
        logger=logger)

    assert supported is True
    assert reason is None


def test_chs_stations_skipped_before_retrieval_with_explanation(
        caplog, tmp_path):
    """No CHS retrieval happens, and the log says what was skipped and why.

    The cost this avoids is the point: each CHS station is ~29 requests
    against a 30 req/min budget, so a station the datum check will discard
    anyway still costs about a minute before it is discarded.
    """
    logger = logging.getLogger('chs_skip_bulk_test')
    inventory = pd.DataFrame({
        'ID': ['00052', '00053', '00055'],
        'X': [-65.0, -64.5, -66.1],
        'Y': [45.0, 45.2, 44.8],
        'Source': ['CHS', 'CHS', 'CHS'],
        'Name': ['A', 'B', 'C'],
        'has_wl': [True, True, True],
    })

    with mock.patch.object(
            write_obs_ctlfile, 'retrieve_chs_station') as mock_retrieve:
        with caplog.at_level(logging.WARNING):
            write_obs_ctlfile._process_variable(
                variable='water_level',
                inventory=inventory,
                var_to_col={'water_level': 'has_wl'},
                start_date=datetime(2026, 3, 1),
                end_date=datetime(2026, 9, 1),
                datum='MLLW',
                datum_list=None,
                ofs='necofs',
                usgs_max_workers=1,
                control_files_path=str(tmp_path),
                logger=logger,
            )

    # Nothing was retrieved -- that is the saving.
    mock_retrieve.assert_not_called()

    msgs = [r.getMessage() for r in caplog.records]
    joined = '\n'.join(msgs)
    # One message, not one per station.
    skip_msgs = [m for m in msgs if 'Skipping all' in m]
    assert len(skip_msgs) == 1, msgs
    # It must say how many, why, and what the user can do about it.
    assert '3 CHS water level station(s)' in joined
    assert 'chart datum' in joined
    assert 'MLLW' in joined
    assert 'necofs' in joined
    assert '-so' in joined
