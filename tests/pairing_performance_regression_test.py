"""Byte-identity regression tests for the pairing/skill fast paths.

Issue #237 replaced four hot spots in the 1D pairing and skill path with
faster equivalents: the per-timestep direction-bias loop, the per-row
``.int`` row munging, the re-parsing of cast-independent ``.obs`` files,
and the O(n_mod x n_obs) water-level extrema matcher. None of them may
change a single byte of a produced artifact, so each is pinned here
against a reference implementation of the behaviour it replaced -- and,
for the ``.int`` writer, against a stored golden file.

The ``.int`` delimiter deserves a note, because it constrains what the
writer is allowed to be. The historical writer rendered each row with
``str(row).replace(',', ' ')``. ``str`` of a list joins on ``', '``, so
the on-disk separator is *two* spaces and every number carries Python's
``repr``. ``DataFrame.to_csv`` cannot reproduce that: its ``sep`` must be
a single character, and it writes an empty field where these files carry
``nan``.
"""

from __future__ import annotations

import hashlib
import importlib
import logging
import math
import os
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ofs_skill.skill_assessment.format_paired_one_d import (
    get_distance_angle,
    paired_scalar,
    paired_vector,
)
from ofs_skill.skill_assessment.get_skill import (
    _INT_HEADERS,
    _OBS_FRAME_CACHE,
    _clear_obs_frame_cache,
    _format_int_rows,
    _pair_extrema,
    _pair_extrema_scan,
    _read_obs_frame,
    _write_int_file,
)
from ofs_skill.utils.file_headers import series_rows_to_skip
from tests.helpers.api_mocks import PIPELINE_FIXTURES, load_julian_disk_series

# The package re-exports the ``get_skill`` *function* under the module's
# own name, so the module object has to be fetched explicitly.
get_skill_module = importlib.import_module(
    'ofs_skill.skill_assessment.get_skill')

_OBS_PATH = PIPELINE_FIXTURES / '8637689_cbofs_wl_station.obs'
_PRD_PATH = PIPELINE_FIXTURES / '8637689_cbofs_wl_45_nowcast_stations_model.prd'
_GOLDEN_ROWS = (
    PIPELINE_FIXTURES
    / 'cbofs_wl_8637689_45_nowcast_stations_pair.golden_rows.int'
)
_GOLDEN_ROWS_SHA256 = (
    'beb0e33108a0177970e95a82c0b1e583a4f8fc018cf6509e8d56231ed677a338'  # pragma: allowlist secret
)
_WINDOW_START = '20260327-18:00:00'
_WINDOW_END = '20260327-22:42:00'

# The .int headers, reproduced here rather than stored in the golden
# file: they end in a space, which the repository's trailing-whitespace
# hook would strip out of any committed fixture. TestIntHeaders pins the
# shipped headers against these literals, so a drift between the two is
# a test failure rather than a silently weakened fixture.
_SCALAR_HEADER = (
    'DNUM_JAN1 YEAR MONTH DAY HOUR MINUTE VAL_OB VAL_MODEL BIAS \n'
)
_VECTOR_HEADER = (
    'DNUM_JAN1 YEAR MONTH DAY HOUR MINUTE SPEED_OB SPEED_MODEL BIAS_SPEED '
    'DIR_OB DIR_MODEL BIAS_DIR \n'
)


@pytest.fixture(name='logger')
def fixture_logger():
    """Quiet logger for the pairing functions."""
    log = logging.getLogger('pairing_performance_regression')
    log.setLevel(logging.CRITICAL)
    return log


def _legacy_obs_read(obs_path) -> pd.DataFrame:
    """Reference ``.obs`` parse: the uncached read this cache replaced."""
    return pd.read_csv(
        obs_path,
        sep=r'\s+',
        header=None,
        skiprows=series_rows_to_skip(obs_path),
    )


def _legacy_int_body(rows) -> str:
    """Reference ``.int`` body: the row munging this writer replaced."""
    return ''.join(
        str(row).replace(',', ' ').replace('[', '').replace(']', '') + '\n'
        for row in rows
    )


def _write_series(path: Path, rows) -> None:
    """Write an ``.obs``/``.prd``-shaped whitespace-delimited file."""
    lines = []
    for row in rows:
        fields = ' '.join(
            'nan' if value != value else f'{value:.4f}' for value in row[6:])
        lines.append(
            f'{row[0]:.8f} {int(row[1])} {int(row[2])} {int(row[3])} '
            f'{int(row[4])} {int(row[5])} {fields}')
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def _synthetic_currents(seed: float, count: int, observed: bool):
    """Deterministic currents rows with missing speeds and directions."""
    rows = []
    for k in range(count):
        minutes = 6 * k
        hour, minute = divmod(minutes, 60)
        day, hour = divmod(hour, 24)
        julian = 2461127.25 + k * 0.00417
        phase = k * 0.1
        speed = abs(1.1 * math.sin(phase / 2.07 + seed)) + 0.05
        direction = (360.0 * math.sin(phase / 3.3 + seed) + 180.0) % 360.0
        if observed and k % 29 == 3:
            direction = float('nan')
        if observed and k % 37 == 11:
            speed = float('nan')
        if direction == direction:
            u_component = speed * math.sin(math.radians(direction))
            v_component = speed * math.cos(math.radians(direction))
        else:
            u_component = float('nan')
            v_component = float('nan')
        rows.append([
            julian, 2026, 3, 27 + day, hour, minute,
            speed, direction, u_component, v_component,
        ])
    return rows


class TestIntWriter:
    """``_format_int_rows`` must reproduce the legacy row munging."""

    def test_matches_legacy_rendering_for_awkward_values(self):
        """NaN, signed zero, infinities, and exponent forms all survive.

        These are exactly the values a naive ``to_csv`` rewrite would
        change: ``nan`` would become an empty field, and fixed-precision
        formatting would truncate the long repr of a float bias.
        """
        rows = [
            [2461127.25, 2026, 3, 27, 18, 0, 0.269, 0.3842,
             0.11519999999999997],
            [2461127.2542, 2026, 3, 27, 18, 6, float('nan'), -0.0,
             float('nan')],
            [2461127.2583, 2026, 3, 27, 18, 12, 1e-07, 1e12,
             float('inf')],
            [2461127.2625, 2026, 3, 27, 18, 18, -1e-07, float('-inf'),
             0.3333333333333333],
        ]
        assert _format_int_rows(rows) == _legacy_int_body(rows)

    def test_delimiter_is_two_spaces(self):
        """The on-disk separator is two spaces, not one.

        This is what rules out ``DataFrame.to_csv``, whose ``sep`` must
        be a single character.
        """
        line = _format_int_rows([[1.0, 2, 3]]).rstrip('\n')
        assert line == '1.0  2  3'

    def test_nan_is_written_not_blanked(self):
        """A missing value is the literal ``nan``, as downstream readers
        (and every existing ``.int`` on disk) expect."""
        assert _format_int_rows([[1.0, float('nan')]]) == '1.0  nan\n'

    def test_no_rows_writes_nothing(self):
        """An empty series must not emit a stray blank line."""
        assert _format_int_rows([]) == ''

    def test_single_row_is_newline_terminated(self):
        """Every file ends with exactly one trailing newline."""
        assert _format_int_rows([[1.0, 2]]).endswith('\n')
        assert not _format_int_rows([[1.0, 2]]).endswith('\n\n')


class TestIntHeaders:
    """The ``.int`` header lines are load-bearing bytes of their own."""

    def test_scalar_header_is_unchanged(self):
        """Trailing space included -- every existing ``.int`` has it."""
        assert _INT_HEADERS['scalar'] == _SCALAR_HEADER
        assert _INT_HEADERS['scalar'].endswith('BIAS \n')

    def test_vector_header_is_unchanged(self):
        """Same for the currents header."""
        assert _INT_HEADERS['cu'] == _VECTOR_HEADER
        assert _INT_HEADERS['cu'].endswith('BIAS_DIR \n')

    def test_writer_selects_the_header_by_variable(self, tmp_path):
        """``cu`` gets the currents header; everything else the scalar
        one, which is how ``_process_station_pair`` branches."""
        for name_var, expected in (
                ('cu', _VECTOR_HEADER),
                ('wl', _SCALAR_HEADER),
                ('temp', _SCALAR_HEADER),
                ('salt', _SCALAR_HEADER),
        ):
            out = tmp_path / f'{name_var}.int'
            _write_int_file(str(out), name_var, [[1.0, 2]])
            assert out.read_text(encoding='utf-8') == expected + '1.0  2\n'

    def test_writer_is_the_only_place_the_header_is_written(self):
        """The pairing write site must go through ``_write_int_file``.

        The headers live in exactly one place so that a test can assert
        them; a future edit that inlines a header string again would
        make these assertions cosmetic, so guard against it.
        """
        source = Path(get_skill_module.__file__).read_text(encoding='utf-8')
        assert source.count('DNUM_JAN1') == 2, (
            'header text should appear only in _INT_HEADERS')
        assert '_write_int_file(int_path, name_var' in source


class TestPairedScalarIntBytes:
    """End-to-end ``.int`` bytes for the committed CBOFS wl fixtures."""

    def _rows(self, logger):
        result = paired_scalar(
            load_julian_disk_series(_OBS_PATH),
            load_julian_disk_series(_PRD_PATH),
            _WINDOW_START,
            _WINDOW_END,
            logger,
            lookback_hours=0,
        )
        assert isinstance(result, tuple), 'fixtures must pair successfully'
        return result[0]

    def test_matches_golden_file(self, tmp_path, logger):
        """The written file is byte-identical to the stored golden.

        This goes through ``_write_int_file`` -- the function
        ``_process_station_pair`` calls -- so the header it emits is
        asserted against the golden rather than against a copy of
        itself.
        """
        rows = self._rows(logger)
        expected = _SCALAR_HEADER.encode('utf-8') + _GOLDEN_ROWS.read_bytes()

        out = tmp_path / 'pair.int'
        _write_int_file(str(out), 'wl', rows)

        assert out.read_bytes() == expected
        # Tripwire on the fixture itself: the data rows end in digits, so
        # no whitespace hook can reach them, but a well-meaning reformat
        # of the file would otherwise silently weaken this test.
        assert hashlib.sha256(
            _GOLDEN_ROWS.read_bytes()).hexdigest() == _GOLDEN_ROWS_SHA256

    def test_matches_legacy_writer(self, logger):
        """Same bytes as the row-by-row writer it replaced."""
        rows = self._rows(logger)
        assert _format_int_rows(rows) == _legacy_int_body(rows)
        assert len(rows) == 48


class TestPairedVectorIntBytes:
    """Currents pairing: vectorized DIR_BIAS must not move any byte."""

    def _paired(self, tmp_path, logger):
        obs_path = tmp_path / 'obs.txt'
        prd_path = tmp_path / 'prd.txt'
        _write_series(obs_path, _synthetic_currents(0.4, 240, True))
        _write_series(prd_path, _synthetic_currents(0.53, 240, False))
        obs_df = load_julian_disk_series(obs_path, n_value_cols=4)
        ofs_df = load_julian_disk_series(prd_path, n_value_cols=4)
        result = paired_vector(
            obs_df, ofs_df, '20260327-18:00:00', '20260328-18:00:00', logger)
        assert isinstance(result, tuple), 'synthetic series must pair'
        return result

    def test_dir_bias_column_matches_scalar_loop(self, tmp_path, logger):
        """DIR_BIAS renders exactly as the per-timestep loop rendered it."""
        _rows, paired = self._paired(tmp_path, logger)
        reference = [
            get_distance_angle(ofs, obs)
            for ofs, obs in zip(
                paired['OFS_DIR'].to_numpy(), paired['OBS_DIR'].to_numpy())
        ]
        produced = paired['DIR_BIAS'].to_numpy()
        assert [repr(value) for value in produced.tolist()] == [
            repr(value)
            for value in np.asarray(reference, dtype=float).tolist()]
        # A run with no missing directions would not exercise the NaN
        # branch that reaches the file as the literal 'nan'.
        assert np.isnan(produced).any()

    def test_int_file_bytes_match_legacy_pipeline(self, tmp_path, logger):
        """Whole currents ``.int`` file, header included, is unchanged.

        The produced side goes through ``_write_int_file``; the
        reference side is written here the way the pipeline wrote it
        before this change, header string and all.
        """
        rows, _paired = self._paired(tmp_path, logger)
        out = tmp_path / 'cu_pair.int'
        _write_int_file(str(out), 'cu', rows)

        legacy = tmp_path / 'cu_pair_legacy.int'
        with open(legacy, 'w', encoding='utf-8') as handle:
            handle.write(_VECTOR_HEADER)
            handle.write(_legacy_int_body(rows))

        assert out.read_bytes() == legacy.read_bytes()
        assert b'nan' in out.read_bytes()


def _stamps(minutes):
    """Convert integer minute offsets to a datetime64 array.

    ``None`` becomes ``NaT`` so that missing-timestamp cases can be
    written the same way as ordinary ones.
    """
    base = np.datetime64('2026-03-27T00:00')
    return np.array([
        np.datetime64('NaT') if m is None else base + np.timedelta64(int(m), 'm')
        for m in minutes
    ], dtype='datetime64[m]')


class TestExtremaPairing:
    """``_pair_extrema`` must match the exhaustive scan it replaced."""

    window = np.timedelta64(3, 'h')

    def _compare(self, m_minutes, m_amps, o_minutes, o_amps):
        """Assert bisect and scan agree on rows and claimed indices."""
        m_times = _stamps(m_minutes)
        o_times = _stamps(o_minutes)
        m_amps = np.asarray(m_amps, dtype=float)
        o_amps = np.asarray(o_amps, dtype=float)
        fast_rows, fast_idx = _pair_extrema(
            m_times, m_amps, o_times, o_amps, self.window)
        slow_rows, slow_idx = _pair_extrema_scan(
            m_times, m_amps, o_times, o_amps, self.window)
        assert fast_idx == slow_idx
        assert len(fast_rows) == len(slow_rows)
        for fast, slow in zip(fast_rows, slow_rows):
            assert fast.keys() == slow.keys()
            for key in fast:
                assert repr(fast[key]) == repr(slow[key]), key
        return fast_rows, fast_idx

    def test_randomized_sweep_with_duplicate_timestamps(self):
        """20k seeded trials over a minute range dense enough that obs
        extrema frequently share a timestamp.

        Duplicated timestamps are the case a naive left-neighbour bisect
        gets wrong: it lands on whichever copy survived earlier claims
        rather than the earliest one, pairing a different amplitude and
        writing a different BIAS into the HW/LW skill table.
        """
        rng = random.Random(237)
        for trial in range(20000):
            n_mod = rng.randint(0, 8)
            n_obs = rng.randint(0, 8)
            m_minutes = sorted(rng.randrange(0, 900, 30) for _ in range(n_mod))
            o_minutes = sorted(rng.randrange(0, 900, 30) for _ in range(n_obs))
            # Occasionally drop a timestamp on either side. Detection can
            # return an extremum whose timestamp did not parse, and the
            # two implementations must agree about ignoring it.
            if n_mod and trial % 17 == 0:
                m_minutes[rng.randrange(n_mod)] = None
            if n_obs and trial % 23 == 0:
                o_minutes[rng.randrange(n_obs)] = None
            m_amps = [round(rng.uniform(-2, 2), 3) for _ in range(n_mod)]
            o_amps = [round(rng.uniform(-2, 2), 3) for _ in range(n_obs)]
            self._compare(m_minutes, m_amps, o_minutes, o_amps)

    def test_duplicate_obs_timestamps_claim_the_earliest(self):
        """Two obs extrema at one timestamp: the first one is claimed."""
        rows, claimed = self._compare(
            [120], [1.0], [120, 120], [0.25, 0.75])
        assert claimed == {0}
        assert rows[0]['OBS'] == 0.25

    def test_second_model_peak_takes_the_duplicate_left_over(self):
        """The next model peak claims the surviving duplicate, not the
        one already taken."""
        rows, claimed = self._compare(
            [120, 150], [1.0, 1.2], [120, 120], [0.25, 0.75])
        assert claimed == {0, 1}
        assert [row['OBS'] for row in rows] == [0.25, 0.75]

    def test_equidistant_neighbours_break_toward_the_earlier(self):
        """A tie on distance resolves to the earlier observation."""
        rows, _claimed = self._compare(
            [120], [1.0], [60, 180], [0.1, 0.9])
        assert rows[0]['OBS'] == 0.1

    def test_window_boundary_is_inclusive(self):
        """An extremum exactly 3 h away still pairs; 3 h plus a minute
        does not."""
        rows, _ = self._compare([180], [1.0], [0], [0.5])
        assert len(rows) == 1
        rows, claimed = self._compare([181], [1.0], [0], [0.5])
        assert rows == []
        assert claimed == set()

    def test_no_extremum_is_claimed_twice(self):
        """Several model peaks crowded around one observation."""
        rows, claimed = self._compare(
            [120, 125, 130], [1.0, 1.1, 1.2], [126], [0.5])
        assert len(rows) == 1
        assert claimed == {0}

    def test_empty_inputs(self):
        """No model or no obs extrema yields no rows and no claims."""
        assert self._compare([], [], [60], [0.5]) == ([], set())
        assert self._compare([60], [1.0], [], []) == ([], set())

    def test_missing_obs_timestamp_is_never_claimed(self):
        """A NaT observation fails every window test, as before."""
        m_times = _stamps([120])
        o_times = np.array(
            [np.datetime64('NaT'), np.datetime64('2026-03-27T02:05')])
        fast_rows, fast_idx = _pair_extrema(
            m_times, np.array([1.0]), o_times, np.array([0.1, 0.9]),
            self.window)
        slow_rows, slow_idx = _pair_extrema_scan(
            m_times, np.array([1.0]), o_times, np.array([0.1, 0.9]),
            self.window)
        assert fast_idx == slow_idx == {1}
        assert fast_rows[0]['OBS'] == slow_rows[0]['OBS'] == 0.9

    def test_missing_model_timestamp_claims_nothing(self):
        """A NaT model extremum must not consume an observation.

        Every comparison against NaT is false, so the exhaustive scan
        found no candidates for it. An unguarded bisect lands at
        position 0 instead and its NaT distance is not greater than the
        window either, which would let a phantom extremum claim a real
        observation and add a row with a NaT DateTime and a NaN timing
        error to the HW/LW skill table.
        """
        m_times = np.array([
            np.datetime64('NaT'), np.datetime64('2026-03-27T02:00')])
        m_amps = np.array([1.0, 2.0])
        o_times = _stamps([60, 125])
        o_amps = np.array([0.1, 0.9])
        fast_rows, fast_idx = _pair_extrema(
            m_times, m_amps, o_times, o_amps, self.window)
        slow_rows, slow_idx = _pair_extrema_scan(
            m_times, m_amps, o_times, o_amps, self.window)
        assert fast_idx == slow_idx == {1}
        assert len(fast_rows) == len(slow_rows) == 1
        assert fast_rows[0]['OFS'] == 2.0
        assert fast_rows[0]['OBS'] == 0.9

    def test_unsorted_obs_falls_back_to_the_scan(self):
        """Out-of-order observations still produce the scan's answer."""
        m_times = _stamps([120, 300])
        o_times = _stamps([300, 60])
        m_amps = np.array([1.0, 2.0])
        o_amps = np.array([0.7, 0.2])
        assert _pair_extrema(
            m_times, m_amps, o_times, o_amps, self.window) == \
            _pair_extrema_scan(m_times, m_amps, o_times, o_amps, self.window)


class TestObsFrameCache:
    """The ``.obs`` parse cache must be invisible to every caller."""

    @pytest.fixture(autouse=True)
    def _clear(self):
        """Isolate each test from cache state left by its neighbours."""
        _clear_obs_frame_cache()
        yield
        _clear_obs_frame_cache()

    @staticmethod
    def _write(path: Path, count: int, offset: float = 0.0) -> None:
        rows = [
            [2461127.25 + k * 0.00417, 2026, 3, 27, 18, 6 * k,
             0.269 + k * 0.01 + offset]
            for k in range(count)
        ]
        _write_series(path, rows)

    def test_cached_frame_equals_a_direct_read(self, tmp_path):
        """A cache hit returns the same values a fresh parse would."""
        path = tmp_path / 'a_cbofs_wl_station.obs'
        self._write(path, 20)
        direct = _legacy_obs_read(path)
        first = _read_obs_frame(str(path))
        second = _read_obs_frame(str(path))
        pd.testing.assert_frame_equal(first, direct)
        pd.testing.assert_frame_equal(second, direct)

    def test_header_row_is_still_skipped(self, tmp_path):
        """Headered files must lose their header row, cached or not.

        ``series_rows_to_skip`` peeks at the first line; caching the
        parse must not cache away that step, or the header would be read
        back as a data row full of strings.
        """
        path = tmp_path / 'h_cbofs_wl_station.obs'
        self._write(path, 20)
        path.write_text(
            'Julian days, Year, Month, Day, Hours, Minutes, '
            'Water level (m)\n' + path.read_text(encoding='utf-8'),
            encoding='utf-8',
        )
        assert series_rows_to_skip(path) == 1
        for _ in range(3):
            frame = _read_obs_frame(str(path))
            pd.testing.assert_frame_equal(frame, _legacy_obs_read(path))
            assert len(frame) == 20
            assert frame[0].dtype.kind == 'f'

    def test_matches_the_uncached_read_of_the_committed_fixture(self):
        """The production CBOFS observation file parses unchanged."""
        for _ in range(3):
            pd.testing.assert_frame_equal(
                _read_obs_frame(str(_OBS_PATH)), _legacy_obs_read(_OBS_PATH))

    def test_second_read_is_served_from_the_cache(self, tmp_path):
        """The point of the cache: one parse for repeated reads."""
        path = tmp_path / 'b_cbofs_wl_station.obs'
        self._write(path, 20)
        _read_obs_frame(str(path))
        assert len(_OBS_FRAME_CACHE) == 1
        _read_obs_frame(str(path))
        assert len(_OBS_FRAME_CACHE) == 1

    def test_caller_mutation_does_not_leak_to_the_next_read(self, tmp_path):
        """Pairing renames columns and adds DateTime in place.

        Handing back the cached object instead of a copy would corrupt
        the next whichcast's pairing silently rather than crashing. The
        cache-miss and the cache-hit paths both have to hand back a
        copy, so every read below mutates what it was given and the
        next read checks that the stored frame survived intact.
        """
        path = tmp_path / 'c_cbofs_wl_station.obs'
        self._write(path, 20)

        # First read is the cache miss that populates the entry.
        miss = _read_obs_frame(str(path))
        baseline = float(miss.iloc[0, 0])
        miss['DateTime'] = pd.to_datetime('2026-03-27')
        miss.rename(columns={6: 'OBS'}, inplace=True)
        miss.iloc[0, 0] = -777.0

        for sentinel in (-999.0, -888.0):
            frame = _read_obs_frame(str(path))
            assert 'DateTime' not in frame.columns
            assert 6 in frame.columns
            assert float(frame.iloc[0, 0]) == baseline
            frame['DateTime'] = pd.to_datetime('2026-03-27')
            frame.rename(columns={6: 'OBS'}, inplace=True)
            frame.iloc[0, 0] = sentinel

        final = _read_obs_frame(str(path))
        assert 'DateTime' not in final.columns
        assert 6 in final.columns
        assert float(final.iloc[0, 0]) == baseline

    def test_rewritten_file_is_re_read(self, tmp_path):
        """New content at the same path must not serve the old parse."""
        path = tmp_path / 'd_cbofs_wl_station.obs'
        self._write(path, 20)
        before = _read_obs_frame(str(path))
        self._write(path, 40, offset=5.0)
        after = _read_obs_frame(str(path))
        assert len(before) == 20
        assert len(after) == 40
        assert after.iloc[0, 6] != before.iloc[0, 6]

    def test_clear_forces_a_re_read(self, tmp_path):
        """The explicit clear after an obs re-fetch empties the cache."""
        path = tmp_path / 'e_cbofs_wl_station.obs'
        self._write(path, 20)
        _read_obs_frame(str(path))
        _clear_obs_frame_cache()
        assert len(_OBS_FRAME_CACHE) == 0

    def test_concurrent_reads_agree(self, tmp_path):
        """Skill runs stations in a thread pool; the cache is shared."""
        paths = []
        for index in range(6):
            path = tmp_path / f'f{index}_cbofs_wl_station.obs'
            self._write(path, 30 + index)
            paths.append(str(path))
        with ThreadPoolExecutor(max_workers=6) as pool:
            frames = list(pool.map(_read_obs_frame, paths * 4))
        for offset, frame in enumerate(frames):
            assert len(frame) == 30 + (offset % 6)

    def test_missing_file_still_raises(self, tmp_path):
        """A path that vanished behaves as it did before the cache."""
        with pytest.raises(FileNotFoundError):
            _read_obs_frame(str(tmp_path / 'absent_cbofs_wl_station.obs'))

    def test_cache_is_bounded(self, tmp_path, monkeypatch):
        """Long hindcasts must not accumulate every parsed frame."""
        monkeypatch.setattr(
            get_skill_module, '_OBS_FRAME_CACHE_MAX_BYTES', 12000)
        for index in range(12):
            path = tmp_path / f'g{index}_cbofs_wl_station.obs'
            self._write(path, 60)
            _read_obs_frame(str(path))
        assert 0 < len(_OBS_FRAME_CACHE) < 12
        assert (get_skill_module._OBS_FRAME_CACHE_BYTES
                <= 12000)

    def test_over_budget_run_still_hits_on_later_casts(
            self, tmp_path, monkeypatch):
        """An over-budget working set must not degenerate to no cache.

        The real access pattern is a sweep over every station repeated
        once per whichcast. Under LRU eviction that pattern evicts
        exactly the entry the next sweep asks for first, so the hit rate
        on cast 2 and every cast after it is zero while the process
        still holds the full budget -- strictly worse than not caching,
        because the memory is spent and every miss still pays for a
        copy. Admission control keeps whatever prefix fits, and that
        prefix keeps hitting.
        """
        monkeypatch.setattr(
            get_skill_module, '_OBS_FRAME_CACHE_MAX_BYTES', 12000)
        paths = []
        for index in range(12):
            path = tmp_path / f'h{index}_cbofs_wl_station.obs'
            self._write(path, 60)
            paths.append(str(path))

        def _sweep():
            """One whichcast: read every station once, count hits."""
            hits = 0
            for path in paths:
                stat = os.stat(path)
                key = (os.path.abspath(path), stat.st_size,
                       stat.st_mtime_ns)
                hits += key in _OBS_FRAME_CACHE
                _read_obs_frame(path)
            return hits

        assert _sweep() == 0, 'nothing is cached on the first sweep'
        held = len(_OBS_FRAME_CACHE)
        assert 0 < held < 12, 'working set must exceed the budget'
        for _ in range(3):
            assert _sweep() == held
            assert len(_OBS_FRAME_CACHE) == held
        assert (get_skill_module._OBS_FRAME_CACHE_BYTES <= 12000)

    def test_budget_exhaustion_is_logged_once(
            self, tmp_path, monkeypatch, caplog):
        """The no-win case must be visible in the run log, not silent."""
        monkeypatch.setattr(
            get_skill_module, '_OBS_FRAME_CACHE_MAX_BYTES', 12000)
        with caplog.at_level(
                logging.INFO,
                logger='ofs_skill.skill_assessment.get_skill'):
            for index in range(12):
                path = tmp_path / f'i{index}_cbofs_wl_station.obs'
                self._write(path, 60)
                for _ in range(2):
                    _read_obs_frame(str(path))
        messages = [
            record.getMessage() for record in caplog.records
            if 'Observation parse cache reached' in record.getMessage()
        ]
        assert len(messages) == 1, messages
        assert 'MiB budget' in messages[0]

    def test_a_frame_larger_than_the_budget_is_not_cached(
            self, tmp_path, monkeypatch):
        """A single oversized frame must not blow past the budget."""
        monkeypatch.setattr(
            get_skill_module, '_OBS_FRAME_CACHE_MAX_BYTES', 64)
        path = tmp_path / 'j_cbofs_wl_station.obs'
        self._write(path, 60)
        frame = _read_obs_frame(str(path))
        assert len(_OBS_FRAME_CACHE) == 0
        pd.testing.assert_frame_equal(frame, _legacy_obs_read(path))
