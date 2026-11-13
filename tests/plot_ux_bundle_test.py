"""Tests for the 1D/2D plot UX bundle: issues #119, #136 and #8.

Three independent fixes share this module because they all guard the
plotly layer against regressions that are invisible in Python and only
show up in a browser:

* #119 - the skill-map writers now name the camera-button PNG export
  after the HTML file instead of letting plotly.js fall back to
  ``newplot.png``.
* #136 - the top margin of the 1D time-series plots is sized from the
  real title height and the real legend row count, so a long title and
  a wrapped horizontal legend cannot collide.
* #8 - no live source passes plotly's ``titlefont``-family magic
  underscore keywords, which are removed in plotly v6.

Everything here is pure config/geometry: no network, no model data, no
figure rendering.
"""
from __future__ import annotations

import ast
import importlib
import os
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import plotly
import plotly.graph_objects as go

from ofs_skill.visualization.plotting_functions import (
    LEGEND_ROW_PX,
    TITLE_LEGEND_GAP_PX,
    TITLE_LINE_PX,
    TITLE_TOP_PX,
    apply_title_band,
    estimate_legend_rows,
    legend_wrap_width_px,
    title_y_container,
    top_margin_px,
)

# The package re-exports these names as functions, so reach the modules
# themselves to keep ``module.function`` unambiguous.
make_skill_maps = importlib.import_module(
    'ofs_skill.skill_assessment.make_skill_maps'
)
make_2d_skill_maps = importlib.import_module(
    'ofs_skill.skill_assessment.make_2d_skill_maps'
)

REPO_ROOT = Path(__file__).resolve().parents[1]

# The three 1D time-series writers, and the plotting functions in each
# whose figure dimensions must stay plain numbers.
ONED_MODULES = {
    'src/ofs_skill/visualization/plotting_scalar.py': ['oned_scalar_plot'],
    'src/ofs_skill/visualization/plotting_scalar_ice.py': [
        'oned_scalar_plot_ice'
    ],
    'src/ofs_skill/visualization/plotting_vector.py': [
        'oned_vector_plot1',
        'oned_vector_plot2b',
        'oned_vector_plot3',
        'oned_vector_diff_plot3',
    ],
}

# The real legend entry sets, copied from the trace names the 1D
# scalar plot adds. Station names are in the title, never the legend.
WL_ONE_CAST = [
    'Observations',
    'Model Nowcast Guidance',
    'Tidal Predictions',
    'Nowcast - Obs.',
    'Target error range',
    '2x target error range',
]
WL_THREE_CASTS = WL_ONE_CAST + [
    'Model Forecast Guidance',
    'Forecast - Obs.',
    'Model Forecast Guidance, 06z cycle',
    'Forecast 06z - Obs.',
]
TEMP_ONE_CAST = [
    'Observations',
    'Model Nowcast Guidance',
    'Nowcast - Obs.',
    'Target error range',
    '2x target error range',
]

FOUR_LINE_TITLE = 'Water Level<br>Station 8454000<br>NECOFS<br>Jan 1 - Jan 5'


def _skill_map_output() -> dict:
    """Smallest station table make_skill_maps will draw."""
    skill_row = (
        0.12, 0.98, 0.01, 1.0, 0.0,          # rmse, r, bias, bias %, dir bias
        95.0, 'pass',                        # central frequency
        0.5, 'pass', 0.5, 'pass',            # positive / negative outliers
        1.0, 'pass', 1.0, 'pass',            # max outlier durations
        0.5, 'pass',                         # worst case outlier frequency
        0.05, 0.15,                          # bias std dev, target rmse
    )
    return {
        'station_id': ['8454000', '8452660'],
        'node': ['101', '202'],
        'X': [-71.4, -71.3],
        'Y': [41.8, 41.5],
        'skill': [skill_row, skill_row],
    }


class TestSkillMapExportFilename(unittest.TestCase):
    """Issue #119: the camera button must not save ``newplot.png``."""

    def setUp(self):
        self.captured = {}

        def fake_plot(fig, **kwargs):
            self.captured['fig'] = fig
            self.captured['kwargs'] = kwargs

        patcher = mock.patch.object(plotly.offline, 'plot', fake_plot)
        patcher.start()
        self.addCleanup(patcher.stop)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp_dir = tmp.name
        self.logger = mock.MagicMock()

    def _run_skill_map(self, tmp_dir):
        prop = SimpleNamespace(
            ofs='necofs',
            whichcast='forecast_b',
            path=str(REPO_ROOT),
            start_date_full='2025-01-01T00:00:00Z',
            end_date_full='2025-01-05T00:00:00Z',
            plotly_maps=str(tmp_dir),
        )
        make_skill_maps.make_skill_maps(
            _skill_map_output(), prop, 'salinity', 'salt', self.logger
        )
        return self.captured['kwargs']

    def test_export_name_is_html_basename_without_extension(self):
        """The PNG export name mirrors the HTML file it came from."""
        kwargs = self._run_skill_map(self.tmp_dir)
        options = kwargs['config']['toImageButtonOptions']
        self.assertEqual(options['filename'], 'necofs_salinity_forecast_b_Skill_Map')
        self.assertFalse(options['filename'].endswith('.html'))
        self.assertEqual(options['format'], 'png')

    def test_export_dimensions_match_the_figure(self):
        """Exported PNG uses the same canvas size as the rendered map."""
        kwargs = self._run_skill_map(self.tmp_dir)
        options = kwargs['config']['toImageButtonOptions']
        self.assertEqual(options['height'], 650)
        self.assertEqual(options['width'], 1000)
        self.assertEqual(self.captured['fig'].layout.height, 650)
        self.assertEqual(self.captured['fig'].layout.width, 1000)

    def test_html_output_path_is_unchanged(self):
        """``filename=`` stays the HTML path, not the export name."""
        kwargs = self._run_skill_map(self.tmp_dir)
        self.assertEqual(
            os.path.basename(kwargs['filename']),
            'necofs_salinity_forecast_b_Skill_Map.html',
        )

    def test_scroll_zoom_is_preserved(self):
        """Adding the export options must not drop scrollZoom."""
        kwargs = self._run_skill_map(self.tmp_dir)
        self.assertIs(kwargs['config']['scrollZoom'], True)


class TestTwoDSkillMapExportFilename(unittest.TestCase):
    """Issue #119, 2D map writer: it passed no plot config at all."""

    def setUp(self):
        self.captured = {}

        def fake_plot(fig, **kwargs):
            self.captured['fig'] = fig
            self.captured['kwargs'] = kwargs

        patcher = mock.patch.object(plotly.offline, 'plot', fake_plot)
        patcher.start()
        self.addCleanup(patcher.stop)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp_dir = tmp.name
        self.logger = mock.MagicMock()

    def _run_2d_map(self, tmp_dir):
        import numpy as np

        lat = np.array([[41.0, 41.5], [42.0, 42.5]])
        lon = np.array([[-71.0, -70.5], [-70.0, -69.5]])
        z = np.array([
            [[1.0, 2.0], [3.0, 4.0]],
            [[1.5, 2.5], [3.5, 4.5]],
        ])
        prop1 = SimpleNamespace(
            ofs='cbofs',
            whichcast='nowcast',
            start_date_full='20250101-00:00:00',
            end_date_full='20250105-00:00:00',
            data_skill_2d_px_path=str(tmp_dir),
        )
        make_2d_skill_maps.make_2d_skill_maps(
            z, lat, lon, ['20250101-00z', '20250102-00z'],
            'rmse', 'ostia', prop1, self.logger,
        )
        return self.captured['kwargs']

    def test_config_is_passed_with_an_export_name(self):
        """A config kwarg now reaches plotly, carrying the export name."""
        kwargs = self._run_2d_map(self.tmp_dir)
        self.assertIn('config', kwargs)
        expected = os.path.splitext(os.path.basename(kwargs['filename']))[0]
        self.assertEqual(
            kwargs['config']['toImageButtonOptions']['filename'], expected
        )
        self.assertNotIn('.html', expected)

    def test_export_name_is_a_bare_basename(self):
        """No path separators leak into the download name on any OS."""
        kwargs = self._run_2d_map(self.tmp_dir)
        name = kwargs['config']['toImageButtonOptions']['filename']
        self.assertNotIn('/', name)
        self.assertNotIn('\\', name)
        self.assertNotIn(os.sep, name)

    def test_export_dimensions_match_the_figure(self):
        """A resized map must not export a letterboxed PNG."""
        kwargs = self._run_2d_map(self.tmp_dir)
        options = kwargs['config']['toImageButtonOptions']
        self.assertEqual(options['height'], self.captured['fig'].layout.height)
        self.assertEqual(options['width'], self.captured['fig'].layout.width)


class TestBandConstants(unittest.TestCase):
    """Issue #136: the band is only safe while these match the render."""

    def test_constants_match_the_font_metrics_they_model(self):
        """Retuning one of these downward silently shrinks the band.

        Pinned here rather than re-derived, because every geometry
        assertion below runs through the same formula these feed and so
        cannot notice an under-sized constant on its own.
        """
        self.assertEqual(TITLE_TOP_PX, 20)
        self.assertEqual(TITLE_LINE_PX, 19)     # 14 px x 1.3 line spacing
        self.assertEqual(TITLE_LEGEND_GAP_PX, 14)
        self.assertEqual(LEGEND_ROW_PX, 22)     # 12 px x 1.3 + 5 px item gap


class TestTopMarginGeometry(unittest.TestCase):
    """Issue #136: the top band must hold the title AND the legend."""

    def test_single_whichcast_water_level_clears_the_title(self):
        """The plain reported case: one nowcast, legend wraps to 2 rows.

        20 px anchor + 4 title rows = 96 px of title; two 22 px legend
        rows lifted 1% of the 440 px plot area (4.4 px) put the legend's
        top edge at 111.6 px, 15.6 px clear of the title.
        """
        rows = estimate_legend_rows(WL_ONE_CAST, legend_wrap_width_px(950))
        self.assertEqual(rows, 2)
        self.assertEqual(top_margin_px(4, rows, 700, 1.01), 160)

    def test_ice_figure_clears_the_title(self):
        """The 600 px ice figure overlapped by ~30 px at one whichcast.

        Same 96 px title, but the legend is lifted 3% of a 331 px plot
        area, so the band has to grow to 169 px to keep the 14 px gap.
        """
        self.assertEqual(top_margin_px(4, 2, 600, 1.03), 169)

    def test_six_line_currents_title_clears_a_wrapped_legend(self):
        """Currents titles carry depth and ADCP-type rows on top.

        20 px anchor + 6 title rows = 134 px, three legend rows = 66 px,
        plus the 14 px gap and a 4.4 px lift.
        """
        self.assertEqual(top_margin_px(6, 3, 700, 1.01), 220)

    def test_margin_does_not_over_reserve(self):
        """Buying the gap with plot area would be the wrong trade."""
        self.assertLessEqual(top_margin_px(4, 4, 700, 1.01), 210)
        self.assertLess(top_margin_px(4, 1, 700, 1.01), 150)

    def test_long_label_forces_uniform_columns(self):
        """One wide entry costs every column, so text width rules rows."""
        self.assertEqual(
            estimate_legend_rows(WL_THREE_CASTS, legend_wrap_width_px(950)), 4
        )
        self.assertEqual(
            estimate_legend_rows(TEMP_ONE_CAST, legend_wrap_width_px(950)), 1
        )

    def test_empty_legend_reserves_no_rows(self):
        """A figure with no legend entries only reserves the title band."""
        self.assertEqual(estimate_legend_rows([], 870), 0)
        self.assertEqual(top_margin_px(4, 0, 700, 1.01), 116)

    def test_title_y_is_a_fixed_pixel_offset(self):
        """title.yref is 'container', so a fixed fraction drifts."""
        self.assertAlmostEqual(title_y_container(700), 1 - 20 / 700)
        self.assertAlmostEqual(title_y_container(600), 1 - 20 / 600)
        self.assertNotAlmostEqual(title_y_container(600), 0.97)
        # Four rose rows: the old 0.975 sat 54 px down instead of 20.
        self.assertGreater(title_y_container(2160), 0.99)

    def test_legend_wrap_width_matches_plotly_anchoring(self):
        """x<0 with xanchor='left' wraps at left margin + plot width."""
        self.assertEqual(legend_wrap_width_px(950), 870)
        self.assertEqual(legend_wrap_width_px(900), 820)


class TestApplyTitleBand(unittest.TestCase):
    """The computed geometry has to survive as real layout values."""

    @staticmethod
    def _figure(labels, height=700, width=950):
        fig = go.Figure()
        for label in labels:
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name=label))
        fig.update_layout(
            height=height,
            width=width,
            margin=dict(b=100),
            title=dict(text=FOUR_LINE_TITLE, x=0.5, xanchor='center'),
            legend=dict(orientation='h', yanchor='bottom', y=1.01, x=-0.05),
        )
        return fig

    def test_band_lands_on_the_layout(self):
        """margin.t and title.y are what the geometry asked for."""
        fig = self._figure(WL_ONE_CAST)
        applied = apply_title_band(fig, FOUR_LINE_TITLE, 700, 950, 1.01)
        self.assertEqual(applied, 160)
        self.assertEqual(fig.layout.margin.t, 160)
        self.assertAlmostEqual(fig.layout.title.y, 1 - 20 / 700)

    def test_it_overrides_a_stale_band_left_on_the_figure(self):
        """A site cannot compute the band then quietly undercut it."""
        fig = self._figure(WL_ONE_CAST)
        fig.update_layout(margin_t=100, title_y=0.83)
        apply_title_band(fig, FOUR_LINE_TITLE, 700, 950, 1.01)
        self.assertEqual(fig.layout.margin.t, 160)
        self.assertAlmostEqual(fig.layout.title.y, 1 - 20 / 700)

    def test_it_leaves_the_rest_of_the_layout_alone(self):
        """Only margin.t and title.y are ours to set."""
        fig = self._figure(WL_ONE_CAST)
        apply_title_band(fig, FOUR_LINE_TITLE, 700, 950, 1.01)
        self.assertEqual(fig.layout.margin.b, 100)
        self.assertEqual(fig.layout.title.text, FOUR_LINE_TITLE)
        self.assertEqual(fig.layout.title.x, 0.5)
        self.assertEqual(fig.layout.height, 700)

    def test_the_band_measures_the_traces_on_the_figure(self):
        """Fewer, shorter entries fit one row and cost less band."""
        fig = self._figure(TEMP_ONE_CAST)
        self.assertEqual(
            apply_title_band(fig, FOUR_LINE_TITLE, 700, 950, 1.01), 138
        )

    def test_hidden_traces_do_not_buy_legend_rows(self):
        """showlegend=False entries never render, so never reserve."""
        fig = self._figure(WL_ONE_CAST)
        for trace in fig.data:
            trace.showlegend = False
        self.assertEqual(
            apply_title_band(fig, FOUR_LINE_TITLE, 700, 950, 1.01), 116
        )

    def test_the_ice_figure_gets_its_taller_band(self):
        """600 px tall with a 3% legend lift is the tightest case."""
        fig = self._figure(WL_ONE_CAST, height=600, width=900)
        self.assertEqual(
            apply_title_band(fig, FOUR_LINE_TITLE, 600, 900, 1.03), 169
        )

    def test_explicit_rows_replace_the_trace_measurement(self):
        """The currents caption shares the band instead of a legend."""
        six_line = FOUR_LINE_TITLE + '<br>10.5 m<br>Side-looking ADCP'
        fig = self._figure(WL_ONE_CAST, width=820)
        self.assertEqual(
            apply_title_band(
                fig, six_line, 700, 820, 1.05, legend_rows=1
            ),
            201,
        )

    def test_min_margin_holds_the_band_open(self):
        """The rose grid keeps its known-good 160 px band."""
        fig = self._figure([], height=2160, width=1500)
        self.assertEqual(
            apply_title_band(
                fig, FOUR_LINE_TITLE, 2160, 1500, 1.0,
                legend_rows=0, bottom_margin=90, min_margin=160,
            ),
            160,
        )
        self.assertAlmostEqual(fig.layout.title.y, 1 - 20 / 2160)


class TestTimeSeriesModulesUseTheHelper(unittest.TestCase):
    """Secondary guard: every 1D writer routes through the helper."""

    def test_no_module_sizes_its_own_band(self):
        """apply_title_band owns margin.t and title.y everywhere."""
        for rel in ONED_MODULES:
            source = (REPO_ROOT / rel).read_text(encoding='utf-8')
            with self.subTest(module=rel):
                self.assertTrue(
                    'apply_title_band(' in source,
                    f'{rel} does not size the band through the helper',
                )

    def test_no_module_sets_its_own_margin_t_or_title_y(self):
        """Those two belong to apply_title_band alone.

        A site that keeps its own literal is the #136 defect: it can
        compute the band correctly and still lay out a short one.
        """
        for rel, functions in ONED_MODULES.items():
            for func in self._target_functions(rel, functions):
                for call in ast.walk(func):
                    if not self._is_update_layout(call):
                        continue
                    with self.subTest(module=rel, func=func.name):
                        self._assert_band_is_not_set_here(call)

    @staticmethod
    def _is_update_layout(node) -> bool:
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == 'update_layout'
        )

    def _assert_band_is_not_set_here(self, call):
        """Neither the magic-underscore nor the nested-dict spelling."""
        owned = {'margin': 't', 'title': 'y'}
        for keyword in call.keywords:
            self.assertNotIn(
                keyword.arg, ('margin_t', 'title_y'),
                f'{keyword.arg} is apply_title_band\'s to set',
            )
            inner = owned.get(keyword.arg)
            if inner is None:
                continue
            self.assertNotIn(
                inner, self._dict_keys(keyword.value),
                f'{keyword.arg}.{inner} is apply_title_band\'s to set',
            )

    @staticmethod
    def _dict_keys(node) -> list:
        """Keys of ``dict(a=1)`` or ``{'a': 1}``; empty for anything else."""
        if isinstance(node, ast.Dict):
            return [
                k.value for k in node.keys
                if isinstance(k, ast.Constant)
            ]
        if isinstance(node, ast.Call) and getattr(node.func, 'id', '') == 'dict':
            return [k.arg for k in node.keywords]
        return []

    def _target_functions(self, rel, names):
        """The named function definitions in one module."""
        tree = ast.parse((REPO_ROOT / rel).read_text(encoding='utf-8'))
        return [
            node for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name in names
        ]

    def test_figure_dimensions_are_plain_numbers(self):
        """A trailing comma makes ``figwidth=820,`` a tuple.

        plotly rejects a tuple width, so the whole writer raised and
        produced no HTML - and the caller swallows the exception, so the
        only symptom was a missing plot.
        """
        for rel, functions in ONED_MODULES.items():
            for func in self._target_functions(rel, functions):
                for assign in ast.walk(func):
                    if not isinstance(assign, ast.Assign):
                        continue
                    names = [
                        t.id for t in assign.targets if isinstance(t, ast.Name)
                    ]
                    for name in names:
                        if name not in ('figwidth', 'figheight'):
                            continue
                        with self.subTest(module=rel, func=func.name, var=name):
                            self._assert_layout_accepts(name, assign.value)

    def _assert_layout_accepts(self, name, value_node):
        """A literal dimension must be one plotly can put on a layout."""
        self.assertNotIsInstance(
            value_node, ast.Tuple, f'{name} is a tuple, not a number'
        )
        try:
            literal = ast.literal_eval(value_node)
        except ValueError:
            return  # computed at runtime from the row count
        go.Figure().update_layout(**{
            'width' if name == 'figwidth' else 'height': literal
        })


class TestNoDeprecatedTitleFontKeywords(unittest.TestCase):
    """Issue #8: plotly v6 removes the ``titlefont`` magic underscores."""

    # Matches titlefont / titleposition / titleside / titleoffset used as
    # a keyword (``titlefont_family=``), as the tail of a prefixed magic
    # underscore (``xaxis_titlefont_size=``), as an attribute
    # (``layout.titlefont``) or as a dict key (``'titleside':``). Only a
    # preceding letter or digit rules a match out, so the underscore and
    # dotted spellings are covered; the forms plotly v6 keeps
    # (``title_font=``) never match, since the alternation cannot span
    # the underscore after ``title``.
    PATTERN = re.compile(
        r'(?<![A-Za-z0-9])title(?:font|position|side|offset)(?![a-z])'
    )

    def test_no_live_source_uses_them(self):
        """Every live occurrence under src/ and bin/ must be gone."""
        offenders = []
        for root in ('src', 'bin'):
            for path in sorted((REPO_ROOT / root).rglob('*.py')):
                for lineno, line in enumerate(
                    path.read_text(encoding='utf-8').splitlines(), start=1
                ):
                    # Prose may name the keyword this migration removed;
                    # only executable text is a defect.
                    code = line.split('#', 1)[0]
                    if self.PATTERN.search(code):
                        rel = path.relative_to(REPO_ROOT).as_posix()
                        offenders.append(f'{rel}:{lineno}: {line.strip()}')
        self.assertEqual(offenders, [], 'plotly v6 drops these keywords:\n' + '\n'.join(offenders))

    def test_guard_still_catches_the_original_defect(self):
        """The pattern flags the removed forms and spares the kept ones."""
        self.assertTrue(self.PATTERN.search("titlefont_family='Open Sans',"))
        self.assertTrue(self.PATTERN.search("'titleside': 'top',"))
        self.assertTrue(
            self.PATTERN.search('fig.layout.titlefont = dict(size=12)')
        )
        self.assertTrue(
            self.PATTERN.search('fig.update_layout(xaxis_titlefont_size=12)')
        )
        self.assertTrue(
            self.PATTERN.search("fig.update_layout(colorbar_titleside='right')")
        )
        self.assertIsNone(self.PATTERN.search("title_font_family='Open Sans',"))
        self.assertIsNone(self.PATTERN.search("title_font=dict(size=16),"))
        self.assertIsNone(self.PATTERN.search('subtitlefont_size=12,'))


if __name__ == '__main__':
    unittest.main()
