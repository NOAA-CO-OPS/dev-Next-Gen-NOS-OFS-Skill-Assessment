"""Tests for model display names and merge keys in compare_ofs.

``_resolve_model_keys`` returns merge-column suffixes (keys) and display
names. A STOFS-2D-Global stations model gets a ' (bias-corrected)' name
when every requested cast is bias-corrected (first v2.1 cycle
2024-05-14 12Z). Keys never change.
"""
import importlib.util
import logging
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
COMPARE_OFS_PATH = REPO_ROOT / 'bin' / 'skill_assessment' / 'compare_ofs.py'

AFTER = ('2024-06-01T00:00:00Z', '2024-06-02T00:00:00Z')
BEFORE = ('2024-04-01T00:00:00Z', '2024-04-02T00:00:00Z')
CASTS = 'nowcast,forecast_b'


@pytest.fixture(scope='module')
def compare_ofs():
    """Import the compare_ofs script as a module."""
    spec = importlib.util.spec_from_file_location(
        'compare_ofs_under_test', COMPARE_OFS_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules['compare_ofs_under_test'] = mod
    spec.loader.exec_module(mod)
    return mod


def test_stofs_stations_after_v2p1_labeled(compare_ofs):
    keys_names = compare_ofs._resolve_model_keys(
        'cbofs', 'stofs_2d_glo', 'stations', 'stations', *AFTER, CASTS)
    assert keys_names == (
        'cbofs', 'stofs_2d_glo', 'CBOFS', 'STOFS_2D_GLO (bias-corrected)')


def test_stofs_stations_as_first_model_labeled(compare_ofs):
    keys_names = compare_ofs._resolve_model_keys(
        'stofs_2d_glo', 'cbofs', 'stations', 'fields', *AFTER, CASTS)
    assert keys_names == (
        'stofs_2d_glo', 'cbofs', 'STOFS_2D_GLO (bias-corrected)', 'CBOFS')


def test_stofs_stations_before_v2p1_not_labeled(compare_ofs):
    keys_names = compare_ofs._resolve_model_keys(
        'cbofs', 'stofs_2d_glo', 'stations', 'stations', *BEFORE, CASTS)
    assert keys_names == ('cbofs', 'stofs_2d_glo', 'CBOFS', 'STOFS_2D_GLO')


def test_stofs_fields_never_labeled(compare_ofs):
    keys_names = compare_ofs._resolve_model_keys(
        'cbofs', 'stofs_2d_glo', 'stations', 'fields', *AFTER, CASTS)
    assert keys_names == ('cbofs', 'stofs_2d_glo', 'CBOFS', 'STOFS_2D_GLO')


@pytest.mark.parametrize('dates', [AFTER, BEFORE])
def test_non_stofs_unchanged(compare_ofs, dates):
    keys_names = compare_ofs._resolve_model_keys(
        'cbofs', 'dbofs', 'stations', 'fields', *dates, CASTS)
    assert keys_names == ('cbofs', 'dbofs', 'CBOFS', 'DBOFS')


def test_non_stofs_does_not_parse_dates(compare_ofs):
    """Dates are only parsed when a STOFS stations model is compared."""
    keys_names = compare_ofs._resolve_model_keys(
        'cbofs', 'dbofs', 'stations', 'stations', 'not-a-date',
        'not-a-date', CASTS)
    assert keys_names == ('cbofs', 'dbofs', 'CBOFS', 'DBOFS')


def test_both_stofs_only_stations_labeled(compare_ofs):
    keys_names = compare_ofs._resolve_model_keys(
        'stofs_2d_glo', 'stofs_2d_glo', 'stations', 'fields', *AFTER, CASTS)
    assert keys_names == (
        'stofs_2d_glo_stations',
        'stofs_2d_glo_fields',
        'STOFS_2D_GLO (stations) (bias-corrected)',
        'STOFS_2D_GLO (fields)',
    )


def test_both_stofs_before_v2p1_not_labeled(compare_ofs):
    keys_names = compare_ofs._resolve_model_keys(
        'stofs_2d_glo', 'stofs_2d_glo', 'fields', 'stations', *BEFORE, CASTS)
    assert keys_names == (
        'stofs_2d_glo_fields',
        'stofs_2d_glo_stations',
        'STOFS_2D_GLO (fields)',
        'STOFS_2D_GLO (stations)',
    )


def test_compact_date_format_accepted(compare_ofs):
    keys_names = compare_ofs._resolve_model_keys(
        'cbofs', 'stofs_2d_glo', 'stations', 'stations',
        '20240601-00:00:00', '20240602-00:00:00', CASTS)
    assert keys_names[3] == 'STOFS_2D_GLO (bias-corrected)'


def test_mixed_casts_raise(compare_ofs):
    """Same as the single-model helper: a mixed request raises."""
    with pytest.raises(ValueError, match='would mix'):
        compare_ofs._resolve_model_keys(
            'cbofs', 'stofs_2d_glo', 'stations', 'stations',
            '2024-05-10T00:00:00Z', '2024-05-20T00:00:00Z', CASTS)


# --- forecast_a: label uses the cycle from get_fcst_dates -------------------

def _fake_fcst_dates(calls, start, end):
    def fake(prop, logger):
        calls.append({
            'ofs': prop.ofs,
            'start_date_full': prop.start_date_full,
            'forecast_hr': prop.forecast_hr,
            'config_file': prop.config_file,
        })
        return start, end
    return fake


def test_forecast_a_uses_cycle_before_v2p1(compare_ofs, monkeypatch):
    """-s 12Z on 2024-05-14 would be labeled, but the cycle is 00Z."""
    calls = []
    monkeypatch.setattr(compare_ofs, 'get_fcst_dates', _fake_fcst_dates(
        calls, '2024-05-14T00:00:00Z', '2024-05-21T12:00:00Z'))
    keys_names = compare_ofs._resolve_model_keys(
        'cbofs', 'stofs_2d_glo', 'stations', 'stations',
        '2024-05-14T12:00:00Z', '2024-05-15T00:00:00Z', 'forecast_a',
        config_file='my.conf')
    assert keys_names[3] == 'STOFS_2D_GLO'
    assert calls == [{
        'ofs': 'stofs_2d_glo',
        'start_date_full': '2024-05-14T12:00:00Z',
        'forecast_hr': compare_ofs._FORECAST_HR,
        'config_file': 'my.conf',
    }]


def test_forecast_a_uses_cycle_after_v2p1(compare_ofs, monkeypatch):
    monkeypatch.setattr(compare_ofs, 'get_fcst_dates', _fake_fcst_dates(
        [], '2024-05-14T12:00:00Z', '2024-05-22T00:00:00Z'))
    keys_names = compare_ofs._resolve_model_keys(
        'cbofs', 'stofs_2d_glo', 'stations', 'stations',
        '2024-05-14T00:00:00Z', '2024-05-15T00:00:00Z', 'forecast_a')
    assert keys_names[3] == 'STOFS_2D_GLO (bias-corrected)'


def test_forecast_a_window_applies_to_all_casts(compare_ofs, monkeypatch):
    """create_1dplot judges every cast on the forecast_a window."""
    monkeypatch.setattr(compare_ofs, 'get_fcst_dates', _fake_fcst_dates(
        [], '2024-06-01T00:00:00Z', '2024-06-08T12:00:00Z'))
    keys_names = compare_ofs._resolve_model_keys(
        'cbofs', 'stofs_2d_glo', 'stations', 'stations',
        '2024-05-10T00:00:00Z', '2024-05-20T00:00:00Z',
        'nowcast,forecast_a')
    assert keys_names[3] == 'STOFS_2D_GLO (bias-corrected)'


def test_forecast_a_matches_create_1dplot_without_s3(
        compare_ofs, tmp_path):
    """Real get_fcst_dates, S3 fallback off: 'now' becomes the 00Z cycle
    of the start day, so a 12Z start on 2024-05-14 is not labeled."""
    conf = tmp_path / 'ofs_dps.conf'
    conf.write_text('[settings]\nuse_s3_fallback = False\n',
                    encoding='utf-8')
    keys_names = compare_ofs._resolve_model_keys(
        'cbofs', 'stofs_2d_glo', 'stations', 'stations',
        '2024-05-14T12:00:00Z', '2024-05-15T00:00:00Z', 'forecast_a',
        config_file=str(conf), logger=logging.getLogger('test'))
    assert keys_names[3] == 'STOFS_2D_GLO'


@pytest.mark.parametrize('ofs2, filetype2, casts', [
    ('stofs_2d_glo', 'stations', 'nowcast,forecast_b'),
    ('stofs_2d_glo', 'fields', 'forecast_a'),
    ('dbofs', 'stations', 'forecast_a'),
])
def test_get_fcst_dates_only_for_stofs_stations_forecast_a(
        compare_ofs, monkeypatch, ofs2, filetype2, casts):
    def fail(prop, logger):
        raise AssertionError('get_fcst_dates should not be called')
    monkeypatch.setattr(compare_ofs, 'get_fcst_dates', fail)
    compare_ofs._resolve_model_keys(
        'cbofs', ofs2, 'stations', filetype2, *AFTER, casts)


# --- Errors stop the run before any plotting ---------------------------------

MIXED = ('2024-05-10T00:00:00Z', '2024-05-20T00:00:00Z')
UNREADABLE = ('06/01/2024', '06/02/2024')


def _run_generate_comparisons(compare_ofs, tmp_path, ofs2, dates):
    compare_ofs.generate_comparisons(
        'cbofs', ofs2, str(tmp_path / 'missing_inventory.csv'),
        'water_level', CASTS, str(tmp_path), 'MLLW', *dates,
        'stations', 'stations', {}, logging.getLogger('test'))


def _run_generate_stat_comparisons(compare_ofs, tmp_path, ofs2, dates):
    compare_ofs.generate_stat_comparisons(
        'cbofs', ofs2, 'water_level', CASTS, str(tmp_path),
        *dates, 'stations', 'stations', {}, logging.getLogger('test'))


RUNNERS = [_run_generate_comparisons, _run_generate_stat_comparisons]


@pytest.mark.parametrize('runner', RUNNERS)
@pytest.mark.parametrize('dates, cause', [
    (MIXED, 'mix bias-corrected and uncorrected'),
    (UNREADABLE, 'unreadable start/end date'),
])
def test_label_error_exits_before_plotting(
        compare_ofs, tmp_path, caplog, runner, dates, cause):
    with pytest.raises(SystemExit) as exc:
        runner(compare_ofs, tmp_path, 'stofs_2d_glo', dates)
    assert exc.value.code == 1
    errors = [r.getMessage() for r in caplog.records
              if r.levelno == logging.ERROR]
    # One error, from the label step: the inventory / stats files (missing
    # here) were never looked for.
    assert len(errors) == 1
    assert errors[0].startswith('Comparison plots not made: ')
    assert cause in errors[0]


@pytest.mark.parametrize('runner', RUNNERS)
def test_non_stofs_bad_dates_not_checked(
        compare_ofs, tmp_path, caplog, runner):
    """Non-STOFS runs never parse the dates; the functions go on to their
    usual missing-file exits."""
    with pytest.raises(SystemExit):
        runner(compare_ofs, tmp_path, 'dbofs', UNREADABLE)
    messages = [r.getMessage() for r in caplog.records]
    assert messages
    assert not any('Comparison plots not made' in m for m in messages)
