"""
Tests for the STOFS-2D-Global bias-correction label decision.

``stofs_2d_glo_stations_bias_corrected`` checks every cast in the
request. With the first bias-corrected cycle C1 = 2024-05-14 12Z,
nowcast times are bias-corrected after C1 - 6 h, forecast_b times
after C1, and a forecast_a run when its cycle (the start date) is at
or after C1. A request that mixes corrected and uncorrected times
raises ValueError.
"""
from datetime import datetime
from types import SimpleNamespace

import pytest

from ofs_skill.visualization import plotting_functions as pf


def _prop(whichcasts):
    return SimpleNamespace(ofs='stofs_2d_glo', ofsfiletype='stations',
                           whichcasts=whichcasts)


def _dt(text):
    return datetime.strptime(text, '%Y-%m-%d %H:%M')


class TestAllUncorrected:
    @pytest.mark.parametrize('casts', [
        ['nowcast'], ['forecast_b'], ['nowcast', 'forecast_b']])
    def test_well_before_switch(self, casts):
        assert pf.stofs_2d_glo_stations_bias_corrected(
            _prop(casts), _dt('2024-05-01 00:00'),
            _dt('2024-05-10 00:00')) is False

    def test_nowcast_ends_at_threshold(self):
        # Nowcast time 06Z May 14 comes from the 06Z cycle (uncorrected).
        assert pf.stofs_2d_glo_stations_bias_corrected(
            _prop(['nowcast']), _dt('2024-05-13 00:00'),
            _dt('2024-05-14 06:00')) is False

    def test_forecast_a_cycle_before_switch(self):
        assert pf.stofs_2d_glo_stations_bias_corrected(
            _prop(['forecast_a']), _dt('2024-05-14 06:00'),
            _dt('2024-05-21 18:00')) is False


class TestAllCorrected:
    @pytest.mark.parametrize('casts', [
        ['nowcast'], ['forecast_b'], ['nowcast', 'forecast_b']])
    def test_well_after_switch(self, casts):
        assert pf.stofs_2d_glo_stations_bias_corrected(
            _prop(casts), _dt('2024-06-01 00:00'),
            _dt('2024-06-10 00:00')) is True

    def test_forecast_a_first_corrected_cycle(self):
        assert pf.stofs_2d_glo_stations_bias_corrected(
            _prop(['forecast_a']), _dt('2024-05-14 12:00'),
            _dt('2024-05-22 00:00')) is True

    def test_comma_separated_string(self):
        assert pf.stofs_2d_glo_stations_bias_corrected(
            _prop('nowcast, forecast_b'), _dt('2024-06-01 00:00'),
            _dt('2024-06-10 00:00')) is True

    def test_falls_back_to_whichcast(self):
        prop = SimpleNamespace(whichcast='forecast_b')
        assert pf.stofs_2d_glo_stations_bias_corrected(
            prop, _dt('2024-06-01 00:00'),
            _dt('2024-06-10 00:00')) is True


class TestMixedRaises:
    @pytest.mark.parametrize('casts', [['nowcast'], ['forecast_b']])
    def test_range_spans_switch(self, casts):
        with pytest.raises(ValueError, match='would mix'):
            pf.stofs_2d_glo_stations_bias_corrected(
                _prop(casts), _dt('2024-05-10 00:00'),
                _dt('2024-05-20 00:00'))

    def test_forecast_b_starts_at_first_run(self):
        # forecast_b time 12Z May 14 comes from the 06Z cycle.
        with pytest.raises(ValueError):
            pf.stofs_2d_glo_stations_bias_corrected(
                _prop(['forecast_b']), _dt('2024-05-14 12:00'),
                _dt('2024-05-20 00:00'))

    def test_casts_disagree(self):
        # Range 07Z-12Z May 14: nowcast all corrected, forecast_b none.
        with pytest.raises(ValueError):
            pf.stofs_2d_glo_stations_bias_corrected(
                _prop(['nowcast', 'forecast_b']), _dt('2024-05-14 07:00'),
                _dt('2024-05-14 12:00'))
