"""Issue #217: RMSE labels on the skill maps must carry their unit."""
from __future__ import annotations

import logging
import os
import types

import pytest

from ofs_skill.skill_assessment.make_skill_maps import make_skill_maps


def _write_error_ranges(root) -> None:
    conf_dir = root / 'conf'
    conf_dir.mkdir(parents=True, exist_ok=True)
    (conf_dir / 'error_ranges.csv').write_text(
        'name_var,X1,X2\n'
        'salt,3.5,0.5\n'
        'temp,3,0.5\n'
        'wl,0.15,0.5\n'
        'cu,0.26,0.5\n'
        'cu_dir,22.5,0.5\n'
        'ice_conc,10,0.5\n',
        encoding='utf-8')


def _make_prop(tmp_path):
    prop = types.SimpleNamespace()
    prop.path = str(tmp_path)
    prop.ofs = 'cbofs'
    prop.whichcast = 'nowcast'
    prop.start_date_full = '2026-03-28T00:00:00Z'
    prop.end_date_full = '2026-03-29T00:00:00Z'
    prop.plotly_maps = str(tmp_path / 'maps')
    os.makedirs(prop.plotly_maps, exist_ok=True)
    return prop


def _output(n=3):
    skill_row = (
        0.12, 0.95, 0.01, 5.0, 0.0, 95.0, 'pass', 0.0, 'pass', 0.0, 'pass',
        0.0, 'pass', 0.0, 'pass', 0.0, 'pass', 0.05, 0.15,
    )
    return {
        'station_id': [f'8{i:06d}' for i in range(n)],
        'node': list(range(n)),
        'X': [-76.5 + 0.05 * i for i in range(n)],
        'Y': [38.0 + 0.10 * i for i in range(n)],
        'skill': [list(skill_row) for _ in range(n)],
    }


def _render(tmp_path, variable, name_var):
    _write_error_ranges(tmp_path)
    prop = _make_prop(tmp_path)
    make_skill_maps(
        _output(), prop, variable, name_var, logging.getLogger('test'))
    path = os.path.join(
        prop.plotly_maps,
        f'{prop.ofs}_{variable}_{prop.whichcast}_Skill_Map.html')
    assert os.path.isfile(path), path
    with open(path, encoding='utf-8') as fh:
        # plotly escapes '/' as \u002f inside the embedded JSON, which
        # would hide the 'm/s' unit from a plain substring search.
        return fh.read().replace('\\u002f', '/')


@pytest.mark.parametrize('variable,name_var,unit', [
    ('water_level', 'wl', 'meters'),
    ('water_temperature', 'temp', '°C'),
    ('currents', 'cu', 'm/s'),
    ('currents_dir', 'cu', 'degrees'),
])
def test_skill_map_labels_carry_units(tmp_path, variable, name_var, unit):
    html = _render(tmp_path, variable, name_var)
    # Colorbar titles.
    assert f'RMSE ({unit})' in html
    assert f'Mean bias ({unit})' in html
    # Hover rows (the column names themselves must stay unit-free).
    assert f'RMSE ({unit}):' in html
    assert f'Target RMSE ({unit}):' in html
    assert f'Mean bias ({unit}):' in html
    assert 'Central freq (%):' in html
    # Figure title.
    assert f'RMSE ({unit}) statistics' in html


def test_currents_direction_is_not_labeled_in_speed_units(tmp_path):
    html = _render(tmp_path, 'currents_dir', 'cu')
    assert 'RMSE (m/s)' not in html
    assert 'RMSE (degrees)' in html


def test_unmapped_variable_degrades_to_no_unit_not_its_own_name(tmp_path):
    html = _render(tmp_path, 'some_variable', 'wl')
    assert 'RMSE (some_variable)' not in html
    assert 'some_variable RMSE statistics' in html
