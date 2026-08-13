"""Tests for the control-file builder distance report (issue #189).

``ctl_builder_report`` joins the observation station ctl and the model
ctl written by ``write_ofs_ctlfile``, computes the obs-model great-circle
distance for each matched station, and writes a CSV report (and,
optionally, an interactive plotly map). These tests exercise the parse +
join + distance + threshold-flag logic against on-disk ctl fixtures
without any network access or model loading.
"""

from __future__ import annotations

import csv
import logging

from ofs_skill.model_processing.ctl_builder_report import (
    build_station_pair_records,
    report_ctl_matches,
    write_distance_report,
)
from ofs_skill.model_processing.station_ledger import StationLedger

logger = logging.getLogger('ctl_builder_report_test')


class _Prop:
    """Minimal stand-in for ModelProperties."""

    def __init__(self, control_files_path, ofs='cbofs',
                 ofsfiletype='stations',
                 var_list=('water_level',)):
        self.control_files_path = str(control_files_path)
        self.ofs = ofs
        self.ofsfiletype = ofsfiletype
        self.var_list = list(var_list)
        self.plotly_maps = str(control_files_path)


def _write_obs_ctl(path, stations):
    """stations: list of (id, name, lat, lon)."""
    with open(path, 'w', encoding='utf-8') as fh:
        for sid, name, lat, lon in stations:
            fh.write(f'{sid} {sid}_wl_cbofs_CO-OPS "{name}"\n')
            fh.write(f'  {lat} {lon} 0  0.0  MLLW\n')


def _write_model_ctl(path, rows):
    """rows: list of (node, layer, lat, lon, station_id, shift)."""
    with open(path, 'w', encoding='utf-8') as fh:
        for node, layer, lat, lon, sid, shift in rows:
            fh.write(f'{node} {layer} {lat}  {lon}  {sid}  {shift}\n')


def test_records_join_and_distance(tmp_path):
    _write_obs_ctl(
        tmp_path / 'cbofs_wl_station.ctl',
        [
            ('8571421', 'Bishops Head', 38.220, -76.038),
            ('8575512', 'Annapolis', 38.984, -76.480),
        ],
    )
    # Model node coords slightly offset from the obs coords.
    _write_model_ctl(
        tmp_path / 'cbofs_wl_model_station.ctl',
        [
            (29, 0, 38.227, -76.036, '8571421', 0.0),
            (14, 0, 38.985, -76.475, '8575512', 0.0),
        ],
    )
    prop = _Prop(tmp_path)
    records = build_station_pair_records(prop, logger, max_dist_km=4.0)

    assert len(records) == 2
    by_id = {r['station_id']: r for r in records}
    assert by_id['8571421']['station_name'] == 'Bishops Head'
    assert by_id['8571421']['matched'] == 'yes'
    assert by_id['8571421']['model_node_index'] == 29
    # Bishops Head obs->node is well under a km.
    assert by_id['8571421']['distance_km'] < 1.0
    assert by_id['8571421']['beyond_threshold'] == 'no'


def test_beyond_threshold_flagged(tmp_path):
    _write_obs_ctl(
        tmp_path / 'cbofs_wl_station.ctl',
        [('8571421', 'Bishops Head', 38.220, -76.038)],
    )
    # Put the model node ~10 km away.
    _write_model_ctl(
        tmp_path / 'cbofs_wl_model_station.ctl',
        [(29, 0, 38.310, -76.038, '8571421', 0.0)],
    )
    prop = _Prop(tmp_path)
    records = build_station_pair_records(prop, logger, max_dist_km=4.0)

    assert len(records) == 1
    assert records[0]['matched'] == 'yes'
    assert records[0]['beyond_threshold'] == 'yes'
    assert float(records[0]['distance_km']) > 4.0


def test_unmatched_station_gets_generic_reason(tmp_path):
    # Obs station present, but no model ctl match for it.
    _write_obs_ctl(
        tmp_path / 'cbofs_wl_station.ctl',
        [('8571421', 'Bishops Head', 38.220, -76.038)],
    )
    # Model ctl exists but references a different station.
    _write_model_ctl(
        tmp_path / 'cbofs_wl_model_station.ctl',
        [(29, 0, 38.985, -76.475, '8575512', 0.0)],
    )
    prop = _Prop(tmp_path)
    records = build_station_pair_records(prop, logger, max_dist_km=4.0)

    by_id = {r['station_id']: r for r in records}
    # Both the unmatched obs station and the orphan model station appear.
    unmatched = by_id['8571421']
    assert unmatched['matched'] == 'no'
    assert unmatched['model_node_index'] == ''
    assert unmatched['distance_km'] == ''
    assert '4.0 km cutoff' in unmatched['reason']


def test_unmatched_station_uses_ledger_reason(tmp_path):
    _write_obs_ctl(
        tmp_path / 'cbofs_wl_station.ctl',
        [('8571421', 'Bishops Head', 38.220, -76.038)],
    )
    # No model match at all -> record with ledger reason.
    _write_model_ctl(tmp_path / 'cbofs_wl_model_station.ctl', [])
    ledger = StationLedger(ofs='cbofs')
    ledger.drop('8571421', stage='node_match',
                reason='nearest model location 6.2 km away (> 4.0 km cutoff)')

    prop = _Prop(tmp_path)
    records = build_station_pair_records(
        prop, logger, max_dist_km=4.0, ledger=ledger)

    assert len(records) == 1
    r = records[0]
    assert r['matched'] == 'no'
    assert r['reason'] == 'nearest model location 6.2 km away (> 4.0 km cutoff)'


def test_missing_obs_ctl_yields_no_records(tmp_path):
    # No obs ctl written at all.
    _write_model_ctl(
        tmp_path / 'cbofs_wl_model_station.ctl',
        [(29, 0, 38.227, -76.036, '8571421', 0.0)],
    )
    prop = _Prop(tmp_path)
    records = build_station_pair_records(prop, logger, max_dist_km=4.0)
    assert records == []


def test_model_station_absent_from_obs_ctl(tmp_path):
    # Obs ctl has one station; model ctl references a different one. The
    # obs station is unmatched; the orphan model station is still listed.
    _write_obs_ctl(
        tmp_path / 'cbofs_wl_station.ctl',
        [('8575512', 'Annapolis', 38.984, -76.480)],
    )
    _write_model_ctl(
        tmp_path / 'cbofs_wl_model_station.ctl',
        [(29, 0, 38.227, -76.036, '8571421', 0.0)],
    )
    prop = _Prop(tmp_path)
    records = build_station_pair_records(prop, logger, max_dist_km=4.0)
    by_id = {r['station_id']: r for r in records}
    # Annapolis obs station has no model match.
    assert by_id['8575512']['matched'] == 'no'
    # The orphan model station 8571421 is still reported.
    orphan = by_id['8571421']
    assert orphan['obs_lat'] == ''
    assert orphan['model_node_index'] == 29


def test_fields_filetype_reads_model_ctl(tmp_path):
    _write_obs_ctl(
        tmp_path / 'cbofs_wl_station.ctl',
        [('8571421', 'Bishops Head', 38.220, -76.038)],
    )
    _write_model_ctl(
        tmp_path / 'cbofs_wl_model.ctl',
        [(29, 0, 38.227, -76.036, '8571421', 0.0)],
    )
    prop = _Prop(tmp_path, ofsfiletype='fields')
    records = build_station_pair_records(prop, logger, max_dist_km=4.0)
    assert len(records) == 1
    assert records[0]['model_node_index'] == 29


def test_write_distance_report_csv(tmp_path):
    _write_obs_ctl(
        tmp_path / 'cbofs_wl_station.ctl',
        [
            ('8571421', 'Bishops Head', 38.220, -76.038),
            ('8575512', 'Annapolis', 38.984, -76.480),
        ],
    )
    # Only the first station matches.
    _write_model_ctl(
        tmp_path / 'cbofs_wl_model_station.ctl',
        [(29, 0, 38.227, -76.036, '8571421', 0.0)],
    )
    prop = _Prop(tmp_path)
    records = build_station_pair_records(prop, logger, max_dist_km=4.0)
    out = write_distance_report(prop, records, logger)
    assert out is not None

    with open(out, encoding='utf-8', newline='') as fh:
        rows = list(csv.DictReader(fh))
    by_id = {r['station_id']: r for r in rows}
    assert len(rows) == 2
    assert by_id['8571421']['matched'] == 'yes'
    assert by_id['8571421']['model_node_index'] == '29'
    assert by_id['8571421']['beyond_threshold'] == 'no'
    # Unmatched station carries a reason and no model columns.
    assert by_id['8575512']['matched'] == 'no'
    assert by_id['8575512']['model_node_index'] == ''
    assert by_id['8575512']['reason'] != ''


def test_report_ctl_matches_no_map(tmp_path):
    _write_obs_ctl(
        tmp_path / 'cbofs_wl_station.ctl',
        [('8571421', 'Bishops Head', 38.220, -76.038)],
    )
    _write_model_ctl(
        tmp_path / 'cbofs_wl_model_station.ctl',
        [(29, 0, 38.227, -76.036, '8571421', 0.0)],
    )
    prop = _Prop(tmp_path)
    records = report_ctl_matches(prop, logger, max_dist_km=4.0,
                                 make_map=False)
    assert len(records) == 1
    # CSV should be written next to the ctl files.
    assert (tmp_path / 'cbofs_ctl_station_pairs.csv').is_file()
    # Map should NOT have been written.
    assert not (tmp_path / 'cbofs_ctl_station_pairs.html').is_file()
