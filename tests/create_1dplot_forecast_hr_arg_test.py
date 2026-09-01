"""
Tests for the ``-f/--Forecast_Hr`` argparse validator added for issue #110.

Before this, ``-f 06`` (missing the 'z') reached ``get_fcst_dates`` and died on
an UnboundLocalError, ``-f 99z``/``-f 24z`` raised a raw strptime ValueError,
and ``-f 6pm`` was silently coerced to a full 00z run under a warning that
misdescribed the cause. All four are now argparse usage errors.
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
CREATE_1DPLOT_PATH = REPO_ROOT / 'bin' / 'visualization' / 'create_1dplot.py'


@pytest.fixture(scope='module')
def mod():
    spec = importlib.util.spec_from_file_location(
        'create_1dplot_fcst_hr_under_test', CREATE_1DPLOT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules['create_1dplot_fcst_hr_under_test'] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ('value', 'expected'),
    [
        ('now', 'now'),
        ('NOW', 'now'),
        (' now ', 'now'),
        ('06z', '06z'),
        ('6Z', '06z'),
        ('6z', '06z'),
        ('0z', '00z'),
        ('23z', '23z'),
    ],
)
def test_valid_forecast_hours_are_normalized(mod, value, expected):
    assert mod._forecast_hr_arg(value) == expected


@pytest.mark.parametrize('value', ['06', '99z', '24z', '6pm', '', 'zz', 'z'])
def test_invalid_forecast_hours_are_rejected(mod, value):
    with pytest.raises(argparse.ArgumentTypeError):
        mod._forecast_hr_arg(value)
