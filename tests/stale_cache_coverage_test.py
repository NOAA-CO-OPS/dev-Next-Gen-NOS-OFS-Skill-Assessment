"""
Regression tests for the stale-cache one-point-plot bug.

Cached pipeline artifacts (``*_station.obs``, ``*_model.prd``,
``*_pair.int``) are keyed by filenames that do not encode the run window,
so a persistent working directory can serve files left over from an
earlier run. The plotting step crops every series to the current window
(``combine_obs_across_casts``), so a stale pair file renders as a
one-point plot when the two windows are adjacent (daily operational runs
share a boundary timestamp) or a blank plot when they are disjoint.

Covers:
- ``ofs_skill.utils.timeseries_coverage`` helpers,
- stale-pair detection/regeneration in ``_ensure_paired_data_exists``,
- the plot-time guard ``_drop_stale_casts``,
- the left merge with the model filename key (a stale/partial key must
  not drop data rows).
"""

import importlib.util
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from ofs_skill.utils.timeseries_coverage import (
    covers_run_window,
    parse_run_window,
    read_first_last_timestamps,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
CREATE_1DPLOT_PATH = REPO_ROOT / 'bin' / 'visualization' / 'create_1dplot.py'

WINDOW_START = datetime(2026, 3, 28, 18, 0)
WINDOW_END = datetime(2026, 3, 29, 18, 0)


@pytest.fixture(scope='module', name='create_1dplot_mod')
def fixture_create_1dplot_mod():
    """Import bin/visualization/create_1dplot.py as a module."""
    spec = importlib.util.spec_from_file_location(
        'create_1dplot_stale_cache_under_test', CREATE_1DPLOT_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules['create_1dplot_stale_cache_under_test'] = mod
    spec.loader.exec_module(mod)
    return mod


def _write_pair_file(path, start, hours, header=True):
    """Write a minimal pair.int-shaped file with hourly rows."""
    lines = []
    if header:
        lines.append('DNUM_JAN1 YEAR MONTH DAY HOUR MINUTE OBS OFS BIAS')
    for i in range(hours):
        stamp = start + timedelta(hours=i)
        lines.append(
            f'2461127.25 {stamp.year} {stamp.month} {stamp.day} '
            f'{stamp.hour} {stamp.minute} 7.4 5.3 -2.1')
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def _make_logger():
    """Return a plain logger for functions that require one."""
    return logging.getLogger('stale_cache_test')


def _paired_df(start, hours):
    """Build a paired-data-shaped DataFrame with hourly timestamps."""
    stamps = [start + timedelta(hours=i) for i in range(hours)]
    return pd.DataFrame({
        'DateTime': stamps,
        'OBS': [7.4] * hours,
        'OFS': [5.3] * hours,
    })


class _WindowProp:
    """Minimum prop surface for the run-window helpers."""

    def __init__(self, start='2026-03-28T18:00:00Z',
                 end='2026-03-29T18:00:00Z'):
        self.start_date_full = start
        self.end_date_full = end
        self.whichcasts = []


class TestTimeseriesCoverage:
    """Unit tests for the timeseries_coverage helper module."""

    def test_parse_run_window_iso_format(self):
        """ISO-format dates parse to the expected window."""
        window = parse_run_window(_WindowProp())
        assert window == (WINDOW_START, WINDOW_END)

    def test_parse_run_window_compact_format(self):
        """Compact YYYYMMDD-HH:MM:SS dates parse identically."""
        window = parse_run_window(
            _WindowProp('20260328-18:00:00', '20260329-18:00:00'))
        assert window == (WINDOW_START, WINDOW_END)

    def test_parse_run_window_unparseable_returns_none(self):
        """Unparseable date strings yield None, not an exception."""
        assert parse_run_window(_WindowProp('garbage', 'garbage')) is None

    def test_parse_run_window_missing_attributes_returns_none(self):
        """Prop objects without the date attributes yield None.

        Some callers (and test stubs) hand in prop objects without the
        date attributes; staleness checks must be skipped, not crash.
        """
        assert parse_run_window(object()) is None

    def test_read_first_last_skips_header(self, tmp_path):
        """The header row is skipped when reading first/last stamps."""
        pair = tmp_path / 'pair.int'
        _write_pair_file(pair, WINDOW_START, hours=25)
        first, last = read_first_last_timestamps(pair)
        assert first == WINDOW_START
        assert last == WINDOW_START + timedelta(hours=24)

    def test_fresh_file_covers_window(self, tmp_path):
        """A file spanning the full window passes the coverage check."""
        pair = tmp_path / 'pair.int'
        _write_pair_file(pair, WINDOW_START, hours=25)
        assert covers_run_window(pair, WINDOW_START, WINDOW_END)

    def test_adjacent_stale_file_fails(self, tmp_path):
        """Yesterday's file, ending exactly at the new window's start,
        fails the check -- the daily-operations signature that produced
        one-point plots."""
        pair = tmp_path / 'pair.int'
        _write_pair_file(pair, WINDOW_START - timedelta(hours=24), hours=25)
        assert not covers_run_window(pair, WINDOW_START, WINDOW_END)

    def test_disjoint_stale_file_fails(self, tmp_path):
        """A file from a disjoint window fails the check."""
        pair = tmp_path / 'pair.int'
        _write_pair_file(pair, WINDOW_START - timedelta(days=5), hours=25)
        assert not covers_run_window(pair, WINDOW_START, WINDOW_END)

    def test_partial_coverage_from_data_gap_is_tolerated(self, tmp_path):
        """A file covering most of the window (e.g. the last model cycle
        not yet available) must NOT trigger regeneration churn."""
        pair = tmp_path / 'pair.int'
        _write_pair_file(pair, WINDOW_START, hours=19)
        assert covers_run_window(pair, WINDOW_START, WINDOW_END)

    def test_unparseable_file_fails_open(self, tmp_path):
        """A file with no parseable data rows is treated as covering."""
        pair = tmp_path / 'pair.int'
        pair.write_text('not a data row\nstill not one\n', encoding='utf-8')
        assert covers_run_window(pair, WINDOW_START, WINDOW_END)


class _PairCheckProp:
    """Minimum prop surface for _ensure_paired_data_exists."""

    def __init__(self, pair_dir):
        self.ofs = 'cbofs'
        self.whichcasts = ['nowcast']
        self.whichcast = 'nowcast'
        self.ofsfiletype = 'stations'
        self.data_skill_1d_pair_path = str(pair_dir)
        self.start_date_full = '2026-03-28T18:00:00Z'
        self.end_date_full = '2026-03-29T18:00:00Z'
        self.start_date_full_before = self.start_date_full
        self.end_date_full_before = self.end_date_full


VAR_INFO = ['water_temperature', 'temp',
            ['Julian', 'year', 'month', 'day', 'hour', 'minute',
             'OBS', 'OFS', 'BIAS']]

# [lines, nodes, depths, shifts, ids] -- only [1] and [-1] are used here.
OFS_CTL = [None, [29], None, None, ['8571421']]


class TestEnsurePairedDataStaleness:
    """Stale-pair handling in _ensure_paired_data_exists."""

    def test_stale_pair_is_deleted_and_regenerated(
            self, create_1dplot_mod, tmp_path, monkeypatch):
        """A pair file from the previous window is deleted and its cast
        queued for regeneration through get_skill."""
        pair = tmp_path / 'cbofs_temp_8571421_29_nowcast_stations_pair.int'
        _write_pair_file(pair, WINDOW_START - timedelta(hours=24), hours=25)

        calls = []
        monkeypatch.setattr(create_1dplot_mod, 'get_skill',
                            lambda p, lg: calls.append(p.whichcast))
        prop = _PairCheckProp(tmp_path)
        create_1dplot_mod._ensure_paired_data_exists(
            OFS_CTL, prop, VAR_INFO, _make_logger())

        assert not pair.exists(), 'stale pair file should be deleted'
        assert calls == ['nowcast'], 'get_skill should regenerate the cast'

    def test_fresh_pair_is_left_alone(
            self, create_1dplot_mod, tmp_path, monkeypatch):
        """A pair file covering the current window is reused as-is."""
        pair = tmp_path / 'cbofs_temp_8571421_29_nowcast_stations_pair.int'
        _write_pair_file(pair, WINDOW_START, hours=25)

        calls = []
        monkeypatch.setattr(create_1dplot_mod, 'get_skill',
                            lambda p, lg: calls.append(p.whichcast))
        prop = _PairCheckProp(tmp_path)
        create_1dplot_mod._ensure_paired_data_exists(
            OFS_CTL, prop, VAR_INFO, _make_logger())

        assert pair.exists()
        assert not calls


class TestDropStaleCasts:
    """Plot-time guard against stale paired series."""

    def test_stale_cast_dropped_fresh_cast_kept(self, create_1dplot_mod):
        """Only the cast whose series intersects the window survives."""
        prop = _WindowProp()
        prop.whichcasts = ['nowcast', 'forecast_b']
        fresh = _paired_df(WINDOW_START, 25)
        # Adjacent-window stale series: only the boundary stamp survives
        # the plot-time crop -- the one-point-plot signature.
        stale = _paired_df(WINDOW_START - timedelta(hours=24), 25)
        pairs, casts = create_1dplot_mod._drop_stale_casts(
            [stale, fresh], ['nowcast', 'forecast_b'], prop, '8571421',
            _make_logger())
        assert casts == ['forecast_b']
        assert len(pairs) == 1 and pairs[0] is fresh

    def test_all_stale_returns_empty(self, create_1dplot_mod):
        """When every cast is stale, nothing is left to plot."""
        prop = _WindowProp()
        prop.whichcasts = ['nowcast']
        stale = _paired_df(WINDOW_START - timedelta(days=3), 25)
        pairs, casts = create_1dplot_mod._drop_stale_casts(
            [stale], ['nowcast'], prop, '8571421', _make_logger())
        assert not pairs and not casts

    def test_nowcast_forecast_a_combo_is_exempt(self, create_1dplot_mod):
        """combine_obs_across_casts applies no crop for this combo, so
        the guard must not drop anything either."""
        prop = _WindowProp()
        prop.whichcasts = ['nowcast', 'forecast_a']
        stale = _paired_df(WINDOW_START - timedelta(days=3), 25)
        pairs, casts = create_1dplot_mod._drop_stale_casts(
            [stale, stale], ['nowcast', 'forecast_a'], prop, '8571421',
            _make_logger())
        assert casts == ['nowcast', 'forecast_a']
        assert len(pairs) == 2

    def test_unparseable_window_keeps_everything(self, create_1dplot_mod):
        """Without a parseable window the guard is skipped entirely."""
        prop = _WindowProp('garbage', 'garbage')
        prop.whichcasts = ['nowcast']
        stale = _paired_df(WINDOW_START - timedelta(days=3), 25)
        _, casts = create_1dplot_mod._drop_stale_casts(
            [stale], ['nowcast'], prop, '8571421', _make_logger())
        assert casts == ['nowcast']


class _PlotProp(_PairCheckProp):
    """Minimum prop surface for _process_station_plot."""

    def __init__(self, pair_dir, node_dir, visual_dir):
        super().__init__(pair_dir)
        self.data_model_1d_node_path = str(node_dir)
        self.visuals_1d_station_path = str(visual_dir)


# [0]: obs rows [id, name, source]; [1]: rows whose [2] is the datum.
STATION_CTL = [[['8571421', 'name_Solomons', 'CO-OPS']],
               [[0.0, 0.0, 'MLLW']]]


class TestFilenameKeyLeftMerge:
    """Integration through _process_station_plot."""

    def test_partial_key_does_not_drop_rows(
            self, create_1dplot_mod, tmp_path, monkeypatch):
        """A filename key covering only part of the series must not prune
        data rows from the plot (regression: how='inner' collapsed the
        series to the key's coverage)."""
        pair_dir = tmp_path / 'pair'
        node_dir = tmp_path / 'node'
        visual_dir = tmp_path / 'visual'
        for directory in (pair_dir, node_dir, visual_dir):
            directory.mkdir()

        pair = pair_dir / 'cbofs_temp_8571421_29_nowcast_stations_pair.int'
        _write_pair_file(pair, WINDOW_START, hours=25)
        # Key covers only the first 5 hours of the series.
        key = pd.DataFrame({
            'DateTime': [WINDOW_START + timedelta(hours=i)
                         for i in range(5)],
            'filename': ['cbofs.t00z.20260328.stations.nowcast.nc'] * 5,
        })
        key.to_csv(node_dir / 'cbofs_nowcast_filename_key.csv', index=False)

        captured = {}

        def _capture_plot(now_fores_paired, *_args, **_kwargs):
            captured['pairs'] = now_fores_paired

        monkeypatch.setattr(
            create_1dplot_mod.plotting_scalar, 'oned_scalar_plot',
            _capture_plot)

        prop = _PlotProp(pair_dir, node_dir, visual_dir)
        result = create_1dplot_mod._process_station_plot(
            0, OFS_CTL, STATION_CTL, prop, VAR_INFO, _make_logger())

        assert result == '8571421'
        assert len(captured['pairs']) == 1
        assert len(captured['pairs'][0]) == 25, (
            'left merge with the filename key must keep all data rows')

    def test_stale_pair_reaches_no_plot(
            self, create_1dplot_mod, tmp_path, monkeypatch):
        """End-to-end through _process_station_plot: a stale pair file
        must not produce a plot call at all."""
        pair_dir = tmp_path / 'pair'
        node_dir = tmp_path / 'node'
        visual_dir = tmp_path / 'visual'
        for directory in (pair_dir, node_dir, visual_dir):
            directory.mkdir()

        pair = pair_dir / 'cbofs_temp_8571421_29_nowcast_stations_pair.int'
        _write_pair_file(pair, WINDOW_START - timedelta(hours=24), hours=25)

        calls = []
        monkeypatch.setattr(
            create_1dplot_mod.plotting_scalar, 'oned_scalar_plot',
            lambda *a, **k: calls.append(a))

        prop = _PlotProp(pair_dir, node_dir, visual_dir)
        create_1dplot_mod._process_station_plot(
            0, OFS_CTL, STATION_CTL, prop, VAR_INFO, _make_logger())

        assert not calls, (
            'a stale pair file must be dropped by the plot-time guard, '
            'not rendered as a one-point plot')
