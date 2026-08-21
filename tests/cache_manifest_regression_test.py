"""Regression tests for the stale-cache guard on the ctl/obs reuse gates.

Reproduces the reported failure mode: a user runs the pipeline for one date
window, then re-runs for a different window (or different station-owner /
datum) in the SAME working directory without deleting ./data and
./control_files. Before the cache-manifest guard, the second run reused
the first run's control files verbatim -- pinning the wrong station/node set
-- and produced incorrect results.

These tests drive cache_manifest.ensure_fresh the same way the real
gates do (write_obs_ctlfile, write_ofs_ctlfile,
get_station_observations, get_skill, create_1dplot) and assert
that a parameter change forces regeneration while an identical rerun reuses.
"""

import os
from types import SimpleNamespace

from ofs_skill.utils import cache_manifest as cm


def _prop(**overrides):
    base = dict(
        ofs='cbofs',
        start_date_full='2026-06-01T00:00:00Z',
        end_date_full='2026-06-02T00:00:00Z',
        ofsfiletype='stations',
        datum='MLLW',
        stationowner='co-ops,ndbc,usgs,chs',
        currents_bins_csv=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _build_ctl(path, signature, base_dir, contents='node depth ... id shift\n'):
    """Mimic a gate writing a ctl and stamping its signature."""
    with open(path, 'w', encoding='utf-8') as handle:
        handle.write(contents)
    cm.record_artifact(path, signature, base_dir)


def test_second_window_regenerates_stale_ctl(tmp_path):
    """Run A builds the ctl; run B (new window) must NOT reuse it."""
    ctl = str(tmp_path / 'cbofs_wl_station.ctl')

    # --- Run A: window 2026-06-01..02 ---
    prop_a = _prop()
    sig_a = cm.run_signature(prop_a, variable='water_level')
    _build_ctl(ctl, sig_a, str(tmp_path), contents='RUN_A_STATIONS\n')

    # Same-parameter rerun reuses verbatim (fast path, no regeneration).
    assert cm.ensure_fresh(ctl, sig_a, str(tmp_path), 'obs ctl') is True
    with open(ctl, encoding='utf-8') as handle:
        assert 'RUN_A_STATIONS' in handle.read()

    # --- Run B: different window ---
    prop_b = _prop(start_date_full='2026-07-01T00:00:00Z',
                   end_date_full='2026-07-02T00:00:00Z')
    sig_b = cm.run_signature(prop_b, variable='water_level')

    # The gate sees a signature mismatch, deletes the stale ctl, returns
    # False so the caller rebuilds it.
    assert cm.ensure_fresh(ctl, sig_b, str(tmp_path), 'obs ctl') is False
    assert not os.path.exists(ctl)

    # Caller rebuilds for window B and stamps the new signature.
    _build_ctl(ctl, sig_b, str(tmp_path), contents='RUN_B_STATIONS\n')
    assert cm.ensure_fresh(ctl, sig_b, str(tmp_path), 'obs ctl') is True
    with open(ctl, encoding='utf-8') as handle:
        assert 'RUN_B_STATIONS' in handle.read()


def test_stationowner_change_regenerates(tmp_path):
    """Narrowing -so from all providers to co-ops only rebuilds the ctl."""
    ctl = str(tmp_path / 'cbofs_wl_station.ctl')
    sig_all = cm.run_signature(
        _prop(stationowner='co-ops,ndbc,usgs,chs'), variable='water_level')
    _build_ctl(ctl, sig_all, str(tmp_path), contents='ALL_PROVIDERS\n')

    sig_coops = cm.run_signature(
        _prop(stationowner='co-ops'), variable='water_level')
    assert cm.ensure_fresh(ctl, sig_coops, str(tmp_path), 'obs ctl') is False
    assert not os.path.exists(ctl)


def test_inventory_regenerates_on_window_change(tmp_path):
    """The inventory CSV gate (its own signature shape) also regenerates."""
    inv = str(tmp_path / 'inventory_all_cbofs.csv')

    def inv_sig(start, end, owner):
        return {
            'ofs': cm._normalize('cbofs'),
            'start_date': cm._normalize(start),
            'end_date': cm._normalize(end),
            'stationowner': cm._normalize(owner),
            'currents_bins_csv': cm.file_fingerprint(None),
        }

    sig_a = inv_sig('20260601', '20260602', 'co-ops,ndbc')
    with open(inv, 'w', encoding='utf-8') as handle:
        handle.write('ID,X,Y,Source,Name\n8638610,-76,37,CO-OPS,x\n')
    cm.record_artifact(inv, sig_a, str(tmp_path))
    assert cm.ensure_fresh(inv, sig_a, str(tmp_path), 'inventory') is True

    sig_b = inv_sig('20260701', '20260702', 'co-ops,ndbc')
    assert cm.ensure_fresh(inv, sig_b, str(tmp_path), 'inventory') is False
    assert not os.path.exists(inv)


def test_prd_int_regenerate_on_datum_change(tmp_path):
    """Model .prd and paired .int gates key on datum + whichcast too."""
    prd = str(tmp_path / 's_cbofs_wl_0_nowcast_stations_model.prd')
    intf = str(tmp_path / 'cbofs_wl_s_0_nowcast_stations_pair.int')

    prd_sig = cm.run_signature(
        _prop(datum='MLLW'), variable='water_level',
        extra={'whichcast': 'nowcast'})
    int_sig = cm.run_signature(
        _prop(datum='MLLW'), variable='water_level',
        extra={'whichcast': 'nowcast'})
    for path, sig in ((prd, prd_sig), (intf, int_sig)):
        with open(path, 'w', encoding='utf-8') as handle:
            handle.write('DNUM YEAR MONTH DAY HOUR MINUTE VAL\n')
        cm.record_artifact(path, sig, str(tmp_path))

    # Same params -> reuse.
    assert cm.artifact_is_fresh(prd, prd_sig) is True
    assert cm.artifact_is_fresh(intf, int_sig) is True

    # Change datum -> both read stale.
    prd_sig2 = cm.run_signature(
        _prop(datum='NAVD88'), variable='water_level',
        extra={'whichcast': 'nowcast'})
    int_sig2 = cm.run_signature(
        _prop(datum='NAVD88'), variable='water_level',
        extra={'whichcast': 'nowcast'})
    assert cm.artifact_is_fresh(prd, prd_sig2) is False
    assert cm.artifact_is_fresh(intf, int_sig2) is False


def test_identical_rerun_reuses_all(tmp_path):
    """A byte-for-byte identical rerun reuses every artifact (no churn)."""
    cm.reset_stale_counter()
    prop = _prop()
    paths = {
        'cbofs_wl_station.ctl': cm.run_signature(prop, variable='water_level'),
        'cbofs_wl_model_station.ctl': cm.run_signature(
            prop, variable='water_level'),
    }
    for name, sig in paths.items():
        p = str(tmp_path / name)
        _build_ctl(p, sig, str(tmp_path))
    # Second run, same params: everything fresh, nothing tallied stale.
    for name, sig in paths.items():
        p = str(tmp_path / name)
        assert cm.ensure_fresh(p, sig, str(tmp_path), 'ctl') is True
    assert cm._stale_counter == {}
