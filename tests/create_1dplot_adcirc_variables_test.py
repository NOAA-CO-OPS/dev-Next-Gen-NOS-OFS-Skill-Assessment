"""
create_1dplot variable validation for ADCIRC (STOFS-2D-Global).

ADCIRC output has no water temperature or salinity, and its currents are
not supported by this pipeline yet. create_1dplot.restrict_adcirc_variables,
run during create_1dplot's parameter validation, keeps only water_level for
model_source 'adcirc' and leaves every other OFS's list unchanged.
"""

import importlib.util
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
CREATE_1DPLOT_PATH = REPO_ROOT / 'bin' / 'visualization' / 'create_1dplot.py'
DEFAULT_VARS = ['water_level', 'water_temperature', 'salinity', 'currents']
_LOGGER = logging.getLogger('create_1dplot_adcirc_variables_test')


@pytest.fixture(scope='module')
def create_1dplot_mod():
    """Import the create_1dplot script as a module."""
    spec = importlib.util.spec_from_file_location(
        'create_1dplot_adcirc_vars_under_test', CREATE_1DPLOT_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules['create_1dplot_adcirc_vars_under_test'] = mod
    spec.loader.exec_module(mod)
    return mod


def _warnings(caplog):
    return [r.getMessage() for r in caplog.records
            if r.levelno == logging.WARNING]


def test_stofs_2d_glo_default_list_keeps_water_level(create_1dplot_mod, caplog):
    prop = SimpleNamespace(ofs='stofs_2d_glo', var_list=list(DEFAULT_VARS))
    with caplog.at_level(logging.WARNING, logger=_LOGGER.name):
        create_1dplot_mod.restrict_adcirc_variables(prop, _LOGGER)
    assert prop.var_list == ['water_level']
    warnings = _warnings(caplog)
    assert len(warnings) == 2
    assert 'no water temperature or salinity output' in warnings[0]
    assert 'currents are not yet supported' in warnings[1]


def test_stofs_2d_glo_without_water_level_exits(create_1dplot_mod, caplog):
    prop = SimpleNamespace(ofs='stofs_2d_glo', var_list=['salinity'])
    with caplog.at_level(logging.WARNING, logger=_LOGGER.name):
        with pytest.raises(SystemExit):
            create_1dplot_mod.restrict_adcirc_variables(prop, _LOGGER)
    assert any(r.levelno == logging.ERROR
               and 'water_level is the only variable' in r.getMessage()
               for r in caplog.records)


@pytest.mark.parametrize('ofs', ['cbofs', 'stofs_3d_atl', 'ngofs2'])
def test_other_ofs_list_unchanged(create_1dplot_mod, caplog, ofs):
    prop = SimpleNamespace(ofs=ofs, var_list=list(DEFAULT_VARS))
    with caplog.at_level(logging.WARNING, logger=_LOGGER.name):
        create_1dplot_mod.restrict_adcirc_variables(prop, _LOGGER)
    assert prop.var_list == DEFAULT_VARS
    assert _warnings(caplog) == []
