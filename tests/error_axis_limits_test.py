"""Error panel y-axis stays at ±4× target unless a chunk of the series exceeds it.

Issue #228: scalar/vector 1D plots hard-coded the error time-series y-axis to
``+/- 4 * X1``, so large model−obs differences were clipped even though the
values were present in the series.

These tests cover ``error_axis_limits``: keep the default band for normal /
sparse-outlier series, and expand to the series min/max when a significant
fraction of points falls outside that band.
"""

from __future__ import annotations

import numpy as np

from ofs_skill.visualization.plotting_functions import error_axis_limits


def test_error_axis_keeps_default_when_within_bounds():
    x1 = 0.15
    errors = np.linspace(-0.4, 0.4, 100)  # all inside +/- 0.6
    y_range, ticks = error_axis_limits(errors, x1)
    assert y_range == [-0.6, 0.6]
    expected_ticks = [-4 * x1, -3 * x1, -2 * x1, -x1, 0.0, x1, 2 * x1, 3 * x1, 4 * x1]
    assert ticks == expected_ticks


def test_error_axis_keeps_default_for_few_outliers():
    x1 = 0.15
    # 2% outside default band — below 5% threshold
    errors = np.concatenate([np.zeros(98), np.array([1.0, -1.0])])
    y_range, _ticks = error_axis_limits(errors, x1)
    assert y_range == [-0.6, 0.6]


def test_error_axis_expands_when_significant_chunk_exceeds():
    x1 = 0.15
    # ~50% of points beyond +/- 0.6
    errors = np.concatenate([np.full(50, 1.2), np.full(50, 0.1)])
    y_range, ticks = error_axis_limits(errors, x1)
    assert y_range != [-0.6, 0.6]
    assert y_range[0] < 0.1
    assert y_range[1] > 1.2
    assert ticks[0] <= y_range[0]
    assert ticks[-1] >= y_range[1]


def test_error_axis_empty_errors_keeps_default():
    y_range, ticks = error_axis_limits([], 0.15)
    assert y_range == [-0.6, 0.6]
    assert 0.0 in ticks
