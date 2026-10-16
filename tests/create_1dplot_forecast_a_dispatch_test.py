"""
Regression tests for issue #110.

``create_1dplot`` used to carry a cycle fan-out for ``forecast_a``: a
``_process_forecast_cycle`` worker dispatched over a list of forecast cycles,
plus a duplicate ``_plot_variable_for_cycle`` copy of the per-variable
plotting body. forecast_a assesses exactly one cycle, so the loop only ever
had one item and the ThreadPoolExecutor branch behind ``len(...) > 1`` was
unreachable.

Worse, the dispatch was gated on ``prop.whichcast`` (singular), which
``create_1dplot`` never assigns. Its value was whatever ``check_model_files``
happened to leave behind in its ``for cast in prop.whichcasts`` loop, so which
branch a forecast_a run took depended on the ORDER of ``-ws`` and on whether
``-df`` skipped the file check entirely.

These tests pin the fan-out as removed and the guard as gone.
"""

import importlib.util
import inspect
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
CREATE_1DPLOT_PATH = REPO_ROOT / 'bin' / 'visualization' / 'create_1dplot.py'


@pytest.fixture(scope='module')
def mod():
    spec = importlib.util.spec_from_file_location(
        'create_1dplot_forecast_a_under_test', CREATE_1DPLOT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules['create_1dplot_forecast_a_under_test'] = module
    spec.loader.exec_module(module)
    return module


class _MockLogger:
    def __init__(self):
        self.messages = []

    def _record(self, msg, *args):
        self.messages.append(str(msg) % args if args else str(msg))

    info = warning = error = debug = _record


class _StubProp:
    def __init__(self, tmp_path):
        self.ofs = 'cbofs'
        self.whichcasts = ['forecast_a']
        self.whichcast = 'forecast_a'
        self.forecast_hr = '06z'
        self.start_date_full = '2026-02-16T00:00:00Z'
        self.end_date_full = '2026-02-18T00:00:00Z'
        self.ofsfiletype = 'stations'
        self.var_list = ['water_level']
        for attr in (
            'control_files_path',
            'data_skill_1d_pair_path',
            'data_model_1d_node_path',
            'visuals_1d_station_path',
        ):
            setattr(self, attr, str(tmp_path))


def test_cycle_fanout_is_gone(mod):
    """The per-cycle worker and its duplicated plotting body are removed."""
    assert not hasattr(mod, '_process_forecast_cycle')
    assert not hasattr(mod, '_plot_variable_for_cycle')


def test_plot_variable_is_module_level(mod):
    """The surviving plotting body is module level and takes an explicit logger."""
    assert inspect.isfunction(mod._plot_variable)
    assert list(inspect.signature(mod._plot_variable).parameters) == [
        'variable',
        'prop',
        'logger',
    ]


def test_create_1dplot_does_not_read_singular_whichcast(mod):
    """``create_1dplot`` must branch on the validated cast list, never on the
    singular ``prop.whichcast`` that only helper functions own.

    This is the assertion that would have caught the guard flip in 5015b2b.
    Scoped to ``create_1dplot``'s own source: helpers such as
    ``ofs_ctlfile_read`` and ``_emit_summary_barplots`` legitimately assign it.
    """
    src = inspect.getsource(mod.create_1dplot)
    assert re.search(r'prop\.whichcast\b(?!s)', src) is None
    assert 'forecast_cycles' not in src
    assert 'ThreadPoolExecutor' not in src


@pytest.mark.parametrize(
    ('variable', 'name_var', 'n_headings'),
    [
        ('water_level', 'wl', 9),
        ('water_temperature', 'temp', 9),
        ('salinity', 'salt', 9),
        ('currents', 'cu', 12),
    ],
)
def test_plot_variable_dispatches_known_variables(
    mod, monkeypatch, tmp_path, variable, name_var, n_headings
):
    calls = {}
    sentinel = object()

    def fake_ctlfile_read(prop, short_name, logger):
        calls['name_var'] = short_name
        return sentinel

    def fake_second_part(read_ofs_ctl_file, prop, var_info, logger):
        calls['var_info'] = var_info
        calls.setdefault('second_part', 0)
        calls['second_part'] += 1

    def fake_barplots(prop, var_info, logger):
        calls.setdefault('barplots', 0)
        calls['barplots'] += 1

    monkeypatch.setattr(mod, 'ofs_ctlfile_read', fake_ctlfile_read)
    monkeypatch.setattr(mod, 'create_1dplot_2nd_part', fake_second_part)
    monkeypatch.setattr(mod, '_emit_summary_barplots', fake_barplots)

    mod._plot_variable(variable, _StubProp(tmp_path), _MockLogger())

    assert calls['name_var'] == name_var
    var_info = calls['var_info']
    assert var_info[0] == variable
    assert var_info[1] == name_var
    assert len(var_info[2]) == n_headings
    assert calls['second_part'] == 1
    assert calls['barplots'] == 1
    if variable == 'currents':
        assert var_info[2][-6:] == [
            'OBS_SPD',
            'OFS_SPD',
            'BIAS_SPD',
            'OBS_DIR',
            'OFS_DIR',
            'BIAS_DIR',
        ]


def test_plot_variable_ignores_unknown_variable(mod, monkeypatch, tmp_path):
    """An unrecognized variable returns early instead of raising
    UnboundLocalError on ``name_var``. The deleted inner closure lacked this
    guard; the surviving module-level body has it.
    """
    called = []
    monkeypatch.setattr(
        mod, 'ofs_ctlfile_read', lambda *a, **k: called.append(1))

    assert mod._plot_variable('bogus', _StubProp(tmp_path), _MockLogger()) is None
    assert not called


def test_plot_variable_skips_downstream_when_ctl_missing(
    mod, monkeypatch, tmp_path
):
    """A missing model ctl file must not reach the plotting routines."""
    downstream = []
    monkeypatch.setattr(mod, 'ofs_ctlfile_read', lambda *a, **k: None)
    monkeypatch.setattr(
        mod, 'create_1dplot_2nd_part', lambda *a, **k: downstream.append('plot'))
    monkeypatch.setattr(
        mod, '_emit_summary_barplots', lambda *a, **k: downstream.append('bars'))

    mod._plot_variable('water_level', _StubProp(tmp_path), _MockLogger())

    assert not downstream
