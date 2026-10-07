"""
Tests for ``get_fcst_dates`` — the forecast_a window resolver.

Issue #110 removed the forecast_a cycle fan-out from ``create_1dplot``. That
deletion is only safe because resolving the window is a FIXPOINT: the fan-out
called ``get_fcst_dates`` a second time on an already-adjusted prop, and the
second call has to be a no-op for the two paths to agree.
``test_resolution_is_a_fixpoint`` is what licenses the deletion.

The rest pin the nearest-valid-cycle autotune (both wraparound directions) and
the input-hardening added alongside the refactor.

S3 is never reached: every test forces ``use_s3_fallback`` off.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / 'src'))

from ofs_skill.model_processing import get_fcst_cycle  # noqa: E402
from ofs_skill.model_processing.get_fcst_cycle import (  # noqa: E402
    get_fcst_dates,
    get_fcst_hours,
)

ISO = '%Y-%m-%dT%H:%M:%SZ'


class _MockLogger:
    def __init__(self):
        self.messages = []

    def _record(self, msg, *args):
        self.messages.append(str(msg) % args if args else str(msg))

    info = warning = error = debug = _record


class _StubProp:
    def __init__(self, ofs, forecast_hr, start='2025-07-15T00:00:00Z'):
        self.ofs = ofs
        self.forecast_hr = forecast_hr
        self.start_date_full = start
        self.end_date_full = None
        self.config_file = None


@pytest.fixture(autouse=True)
def no_s3(monkeypatch):
    """Force ``use_s3_fallback`` off so no test can reach a live S3 listing."""

    class _StubUtils:
        def __init__(self, *a, **k):
            pass

        def read_config_section(self, section, logger):
            return {'use_s3_fallback': 'False'}

    monkeypatch.setattr(get_fcst_cycle.utils, 'Utils', _StubUtils)


def test_autotune_wraps_forward():
    """cbofs cycles are 00/06/12/18; 23z rounds forward onto the next day's 00z."""
    prop = _StubProp('cbofs', '23z')
    start, end = get_fcst_dates(prop, _MockLogger())

    assert start == '2025-07-16T00:00:00Z'
    assert end == '2025-07-18T00:00:00Z'
    # The 'z' suffix must survive the autotune: get_model_data strips the last
    # character, so a bare '00' would become '0', miss fcstcycles and exit.
    assert prop.forecast_hr == '00z'


def test_autotune_wraps_backward():
    """ngofs2 cycles are 03/09/15/21; 23z rounds back onto the same day's 21z."""
    prop = _StubProp('ngofs2', '23z')
    start, end = get_fcst_dates(prop, _MockLogger())

    assert start == '2025-07-15T21:00:00Z'
    assert end == '2025-07-17T21:00:00Z'
    assert prop.forecast_hr == '21z'


@pytest.mark.parametrize(
    ('ofs', 'forecast_hr'),
    [('cbofs', '06z'), ('cbofs', '23z'), ('ngofs2', '23z'), ('ngofs2', '03z')],
)
def test_resolution_is_a_fixpoint(ofs, forecast_hr):
    """Resolving an already-resolved window must return it unchanged.

    The deleted ``_process_forecast_cycle`` re-derived the window on a prop
    whose dates had already been reshuffled during validation. Removing that
    second call is behavior-preserving only if the function is a fixpoint on
    its own output — including through both wraparound branches.
    """
    logger = _MockLogger()
    prop = _StubProp(ofs, forecast_hr)

    first = get_fcst_dates(prop, logger)

    # Apply create_1dplot's normalization, exactly as the caller does.
    prop.start_date_full, prop.end_date_full = first
    prop.forecast_hr = first[0].split('T')[1][0:2] + 'z'
    settled_hr = prop.forecast_hr

    second = get_fcst_dates(prop, logger)

    assert second == first
    assert prop.forecast_hr == settled_hr


@pytest.mark.parametrize(
    ('ofs', 'forecast_hr'),
    [
        ('cbofs', '06z'),
        ('gomofs', '12z'),
        ('stofs_3d_atl', '12z'),
        ('stofs_2d_glo', '06z'),
    ],
)
def test_end_date_matches_get_fcst_hours(ofs, forecast_hr):
    """The window length must stay tied to the forecast-length table."""
    start, end = get_fcst_dates(_StubProp(ofs, forecast_hr), _MockLogger())

    fcstlength, _ = get_fcst_hours(ofs)
    delta = datetime.strptime(end, ISO) - datetime.strptime(start, ISO)
    assert delta == timedelta(hours=fcstlength)


def test_missing_start_date_aborts_cleanly():
    """A missing start date must produce the friendly message, not a traceback.

    ``'T' in None`` raises TypeError, which the original ``except AttributeError``
    did not catch — so the guidance below never reached the user.
    """
    prop = _StubProp('cbofs', '06z', start=None)
    logger = _MockLogger()

    with pytest.raises(SystemExit) as excinfo:
        get_fcst_dates(prop, logger)

    assert excinfo.value.code == 1
    assert any('must specify a start date' in m for m in logger.messages)


def test_malformed_cycle_aborts_instead_of_running_00z():
    """A non-numeric cycle must abort rather than silently assess 00z."""
    prop = _StubProp('cbofs', '6pm')
    logger = _MockLogger()

    with pytest.raises(SystemExit) as excinfo:
        get_fcst_dates(prop, logger)

    assert excinfo.value.code == 1
    assert any('Invalid forecast cycle' in m for m in logger.messages)


def test_now_without_s3_still_coerces_to_00z():
    """The one legitimate coercion must survive: 'now' with S3 fallback off."""
    prop = _StubProp('cbofs', 'now')
    logger = _MockLogger()

    start, end = get_fcst_dates(prop, logger)

    assert start == '2025-07-15T00:00:00Z'
    assert prop.forecast_hr == '00z'
    assert any('now' in m and '00Z' in m for m in logger.messages)


def test_non_iso_start_date_fallback():
    """Compact ``YYYYMMDD-HH:MM:SS`` dates must keep resolving.

    ``check_model_files`` leaves prop's dates in this form on the
    missing-files path, and ``ofs_ctlfile_read`` handles it explicitly.
    """
    prop = _StubProp('cbofs', '06z', start='20250715-00:00:00')
    start, _ = get_fcst_dates(prop, _MockLogger())

    assert start == '2025-07-15T06:00:00Z'
