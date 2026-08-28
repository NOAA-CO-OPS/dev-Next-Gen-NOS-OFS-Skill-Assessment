"""
Created on Mon Jul  6 13:58:20 2026

@author: PWL
"""

import argparse
import glob
import logging.config
import math
import os
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

#from create_1dplot import create_1dplot
#from get_shapefile_intersection import get_shapefile_intersection
from plotly.subplots import make_subplots

from ofs_skill.model_processing import check_model_files, model_properties
from ofs_skill.obs_retrieval import utils


def generate_comparisons(ofs1, ofs2, overlap_csv, var_selection, home_path, datum, logger):
    """Ingests paired datasets for overlapping stations, creates interactive Plotly time series, and paginated Matplotlib scatter grids."""

    # Helper to resolve target error range robustly
    def fetch_error_range(short_var, base_path):
        class MockProp:
            def __init__(self, path):
                self.path = path

        try:
            from ofs_skill.visualization.plotting_functions import get_error_range
            return get_error_range(short_var, MockProp(base_path), logger)[0]
        except ImportError:
            pass

        try:
            from plotting_functions import get_error_range
            return get_error_range(short_var, MockProp(base_path), logger)[0]
        except ImportError:
            pass

        logger.warning('Could not import get_error_range. Using direct CSV fallback for target error.')
        config_path = os.path.join(base_path, 'conf', 'error_ranges.csv')
        defaults = {'salt': 3.5, 'temp': 3.0, 'wl': 0.15, 'cu': 0.26, 'ice_conc': 10.0}

        if os.path.exists(config_path):
            try:
                df_err = pd.read_csv(config_path)
                match = df_err[df_err['name_var'] == short_var]
                if not match.empty:
                    return float(match.iloc[0]['X1'])
            except Exception as e:
                logger.warning(f'Error reading {config_path}: {e}')

        return defaults.get(short_var, 0)

    logger.info('--- Starting Comparison Plotting ---')

    try:
        inventory = pd.read_csv(overlap_csv)
        station_col = 'ID' if 'ID' in inventory.columns else inventory.columns[0]
        overlap_stations = inventory[station_col].astype(str).unique().tolist()
    except Exception as e:
        logger.error(f'Failed to read inventory CSV: {e}')
        return

    pair_dir = os.path.join(home_path, 'data', 'skill', '1d_pair')
    visual_dir = os.path.join(home_path, 'data', 'visual', 'comparisons')
    os.makedirs(visual_dir, exist_ok=True)

    vars_to_process = var_selection.split(',')

    for var in vars_to_process:
        var = var.strip()
        display_var = var.replace('_', ' ').title()

        var_map = {'water_level': 'wl', 'water_temperature': 'temp', 'salinity': 'salt', 'currents': 'cu'}
        short_var = var_map.get(var, var)

        if short_var == 'wl':
            y_title = f'Water Level<br>(<i>meters {datum}</i>)'
            unit = 'm'
        elif short_var == 'temp':
            y_title = 'Water Temperature<br>(<i>\u00b0C</i>)'
            unit = '\u00b0C'
        elif short_var == 'cu':
            y_title = 'Current Speed<br>(<i>knots</i>)'
            unit = 'knots'
        elif short_var == 'salt':
            y_title = 'Salinity<br>(<i>PSU</i>)'
            unit = 'PSU'
        else:
            y_title = display_var
            unit = ''

        if short_var == 'cu':
            col_names = ['Julian', 'year', 'month', 'day', 'hour', 'minute', 'OBS_SPD', 'OFS_SPD', 'BIAS_SPD', 'OBS_DIR', 'OFS_DIR', 'BIAS_DIR']
        else:
            col_names = ['Julian', 'year', 'month', 'day', 'hour', 'minute', 'OBS', 'OFS', 'BIAS']

        X1 = fetch_error_range(short_var, home_path)
        if short_var == 'cu':
            X1 *= 1.943844

        # Dictionary to store valid dataframes for the scatter grid phase
        station_data_map = {}

        for station in overlap_stations:
            logger.info(f'Processing comparison for station {station}, variable {short_var}')

            ofs1_pattern = os.path.join(pair_dir, f'{ofs1}_{short_var}_{station}_*_pair.int')
            ofs2_pattern = os.path.join(pair_dir, f'{ofs2}_{short_var}_{station}_*_pair.int')

            ofs1_files = glob.glob(ofs1_pattern)
            ofs2_files = glob.glob(ofs2_pattern)

            if not ofs1_files or not ofs2_files:
                logger.warning(f'Missing pair files for station {station}. Skipping.')
                continue

            try:
                df1 = pd.read_csv(ofs1_files[0], sep=r'\s+', names=col_names, header=0)
                df2 = pd.read_csv(ofs2_files[0], sep=r'\s+', names=col_names, header=0)

                df1['DateTime'] = pd.to_datetime(df1[['year', 'month', 'day', 'hour', 'minute']])
                df2['DateTime'] = pd.to_datetime(df2[['year', 'month', 'day', 'hour', 'minute']])

                if short_var == 'cu':
                    df1['OBS'] = df1['OBS_SPD'] * 1.943844
                    df1['OFS'] = df1['OFS_SPD'] * 1.943844
                    df2['OBS'] = df2['OBS_SPD'] * 1.943844
                    df2['OFS'] = df2['OFS_SPD'] * 1.943844

                merged = pd.merge(
                    df1[['DateTime', 'OBS', 'OFS']],
                    df2[['DateTime', 'OFS']],
                    on='DateTime',
                    suffixes=(f'_{ofs1}', f'_{ofs2}')
                )

                if merged.empty:
                    logger.warning(f'No overlapping timeframe for station {station}. Skipping.')
                    continue

                ts_hover = f'<b>Time:</b> %{{x|%m/%d/%Y %H:%M}}<br><b>%{{data.name}}:</b> %{{y:.2f}} {unit}<extra></extra>'
                err_hover = f'<b>Time:</b> %{{x|%m/%d/%Y %H:%M}}<br><b>%{{data.name}}:</b> %{{y:.2f}} {unit}<extra></extra>'

                merged[f'Error_{ofs1}'] = merged[f'OFS_{ofs1}'] - merged['OBS']
                merged[f'Error_{ofs2}'] = merged[f'OFS_{ofs2}'] - merged['OBS']

                # Store for scatter plots later
                station_data_map[station] = merged

                # =========================================================
                # 1. TIME SERIES & ERROR PLOT (Plotly HTML - per station)
                # =========================================================
                fig_ts = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.7, 0.3])

                fig_ts.add_trace(go.Scatter(x=merged['DateTime'], y=merged['OBS'], name='Observation', mode='lines', hovertemplate=ts_hover, line=dict(color='red', width=2)), row=1, col=1)
                fig_ts.add_trace(go.Scatter(x=merged['DateTime'], y=merged[f'OFS_{ofs1}'], name=ofs1.upper(), mode='lines', hovertemplate=ts_hover, line=dict(color='#d55e00', width=1.5), opacity=0.8, legendgroup=ofs1), row=1, col=1)
                fig_ts.add_trace(go.Scatter(x=merged['DateTime'], y=merged[f'OFS_{ofs2}'], name=ofs2.upper(), mode='lines', hovertemplate=ts_hover, line=dict(color='#0072b2', width=1.5), opacity=0.8, legendgroup=ofs2), row=1, col=1)

                min_dt = merged['DateTime'].min()
                max_dt = merged['DateTime'].max()

                if X1 > 0:
                    X2 = X1 * 2
                    # List the target error before the 2x target error in the legend.
                    fig_ts.add_trace(go.Scatter(x=[min_dt, max_dt, max_dt, min_dt], y=[X1, X1, -X1, -X1], fill='toself', fillcolor='rgba(255, 165, 0, 0.3)', line=dict(color='rgba(255,255,255,0)'), name=f'Target Error (\u00B1{X1:.2f} {unit})', hoverinfo='skip'), row=2, col=1)
                    fig_ts.add_trace(go.Scatter(x=[min_dt, max_dt, max_dt, min_dt], y=[X2, X2, -X2, -X2], fill='toself', fillcolor='rgba(255, 0, 0, 0.15)', line=dict(color='rgba(255,255,255,0)'), name=f'2x Target Error (\u00B1{X2:.2f} {unit})', hoverinfo='skip'), row=2, col=1)

                fig_ts.add_trace(go.Scatter(x=[min_dt, max_dt], y=[0, 0], mode='lines', name='Zero Error', showlegend=False, hoverinfo='skip', line=dict(color='black', dash='dash', width=1)), row=2, col=1)
                fig_ts.add_trace(go.Scatter(x=merged['DateTime'], y=merged[f'Error_{ofs1}'], name=f'{ofs1.upper()} Error', mode='lines', hovertemplate=err_hover, line=dict(color='#d55e00', width=1.5), opacity=0.8, showlegend=False, legendgroup=ofs1), row=2, col=1)
                fig_ts.add_trace(go.Scatter(x=merged['DateTime'], y=merged[f'Error_{ofs2}'], name=f'{ofs2.upper()} Error', mode='lines', hovertemplate=err_hover, line=dict(color='#0072b2', width=1.5), opacity=0.8, showlegend=False, legendgroup=ofs2), row=2, col=1)

                fig_ts.update_layout(
                    title=dict(text=f'<b>Time Series Comparison: {station} - {display_var}</b>', font=dict(size=18, color='black', family='Open Sans'), y=0.98, x=0.5, xanchor='center', yanchor='top'),
                    template='plotly_white', hovermode='x unified', hoverlabel=dict(bgcolor='white', bordercolor='#cccccc', font=dict(family='Open Sans', size=13, color='#333333')),
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0, font=dict(size=16, color='black'), itemclick='toggle', itemdoubleclick=False),
                    margin=dict(t=120, b=120), height=780, width=900
                )
                fig_ts.update_xaxes(mirror=True, ticks='inside', showline=True, linecolor='black', linewidth=1, showspikes=True, spikemode='across', spikesnap='cursor', showgrid=True, tickfont=dict(family='Open Sans', color='black', size=14), minor=dict(ticklen=4, tickcolor='black', ticks='inside', showgrid=False), tickformat='%H:%M<br>%m/%d', hoverformat='%b %d, %Y, %H:%M UTC')
                fig_ts.update_xaxes(title_text='<br>Time (UTC)', titlefont=dict(family='Open Sans', color='black', size=18), rangeslider=dict(visible=True, thickness=0.06, bordercolor='black', borderwidth=1), row=2, col=1)
                fig_ts.update_yaxes(title_text=y_title, titlefont=dict(family='Open Sans', color='black', size=17), mirror=True, ticks='inside', showline=True, linecolor='black', linewidth=1, tickfont=dict(family='Open Sans', color='black', size=14), minor=dict(ticklen=4, tickcolor='black', ticks='inside', showgrid=False), zeroline=(short_var == 'wl'), zerolinewidth=1, zerolinecolor='black', row=1, col=1)
                fig_ts.update_yaxes(title_text=f'Error ({unit})' if unit else 'Error', titlefont=dict(family='Open Sans', color='black', size=17), mirror=True, ticks='inside', showline=True, linecolor='black', linewidth=1, tickfont=dict(family='Open Sans', color='black', size=14), minor=dict(ticklen=4, tickcolor='black', ticks='inside', showgrid=False), zeroline=False, row=2, col=1)

                ts_out = os.path.join(visual_dir, f'{ofs1}_vs_{ofs2}_{short_var}_{station}_timeseries.html')
                fig_ts.write_html(ts_out)

            except Exception as e:
                logger.error(f'Error plotting TS for station {station}: {e}')

        # =========================================================
        # 2. SCATTER PLOT GRID (Matplotlib - Paginated)
        # =========================================================
        max_plots_per_page = 12
        stations_with_data = list(station_data_map.keys())

        for batch_idx in range(0, len(stations_with_data), max_plots_per_page):
            batch_stations = stations_with_data[batch_idx : batch_idx + max_plots_per_page]
            num_stations = len(batch_stations)

            cols = min(3, num_stations)
            rows = math.ceil(num_stations / cols)

            fig_width = cols * 4.5
            fig_height = rows * 4.5

            fig_scat_all, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(fig_width, fig_height))
            if num_stations > 0:
                axes = np.atleast_1d(axes).flatten()

            plot_idx = 0

            for station in batch_stations:
                merged = station_data_map[station]
                ax = axes[plot_idx]

                min_val = min(merged['OBS'].min(), merged[f'OFS_{ofs1}'].min(), merged[f'OFS_{ofs2}'].min())
                max_val = max(merged['OBS'].max(), merged[f'OFS_{ofs1}'].max(), merged[f'OFS_{ofs2}'].max())

                if pd.notna(min_val) and pd.notna(max_val):
                    buffer = (max_val - min_val) * 0.05 if (max_val - min_val) != 0 else 0.1
                    ax_min, ax_max = min_val - buffer, max_val + buffer

                    if X1 > 0:
                        X2 = X1 * 2
                        ax.fill_between([ax_min, ax_max], [ax_min - X2, ax_max - X2], [ax_min + X2, ax_max + X2],
                                        color='red', alpha=0.15, edgecolor='none', label=f'2x Target Error (\u00B1{X2:.2f} {unit})')

                        ax.fill_between([ax_min, ax_max], [ax_min - X1, ax_max - X1], [ax_min + X1, ax_max + X1],
                                        color='orange', alpha=0.3, edgecolor='none', label=f'Target Error (\u00B1{X1:.2f} {unit})')

                    ax.plot([ax_min, ax_max], [ax_min, ax_max], 'k--', label='1:1 Line')
                    ax.set_xlim(ax_min, ax_max)
                    ax.set_ylim(ax_min, ax_max)

                ax.scatter(merged['OBS'], merged[f'OFS_{ofs1}'], color='#d55e00', s=20, alpha=0.6, label=ofs1.upper())
                ax.scatter(merged['OBS'], merged[f'OFS_{ofs2}'], color='#0072b2', s=20, alpha=0.6, label=ofs2.upper())

                ax.set_title(f'Station: {station}', fontsize=13, pad=8)
                ax.set_xlabel(f'Observation ({unit})' if unit else 'Observation', fontsize=11)
                ax.set_ylabel(f'Model ({unit})' if unit else 'Model', fontsize=11)
                ax.set_aspect('equal', adjustable='box')
                ax.grid(True, linestyle='--', alpha=0.6)

                plot_idx += 1

            # Clean up empty subplots
            for j in range(plot_idx, len(axes)):
                fig_scat_all.delaxes(axes[j])

            # Restrict rect so title and legend have dedicated space at the top without overlapping plots
            plt.tight_layout(rect=[0, 0, 1, 0.90])

            # Deduplicate labels and add unified legend
            handles, labels = axes[0].get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            fig_scat_all.legend(by_label.values(), by_label.keys(), loc='lower center', bbox_to_anchor=(0.5, 0.90), ncol=5, frameon=False, fontsize=12)

            page_text = f' (Page {batch_idx // max_plots_per_page + 1})' if len(stations_with_data) > max_plots_per_page else ''
            fig_scat_all.suptitle(f'Scatter Comparisons: {display_var}{page_text}', fontsize=16, y=0.98)

            page_suffix = f'_page{batch_idx // max_plots_per_page + 1}' if len(stations_with_data) > max_plots_per_page else ''
            scat_out_all = os.path.join(visual_dir, f'{ofs1}_vs_{ofs2}_{short_var}_all_scatter{page_suffix}.png')

            fig_scat_all.savefig(scat_out_all, bbox_inches='tight', dpi=150)
            logger.info(f'Saved aggregated Matplotlib scatter grid to {scat_out_all}')

            plt.close(fig_scat_all)

def generate_stat_comparisons(ofs1, ofs2, home_path, logger, make_bar_plots=False):
    """Reads the generated skill stat CSVs and plots interactive 1-to-1
    scatters with bounded target thresholds. Grouped station-by-station
    bar plots are only produced when ``make_bar_plots`` is True."""
    import os

    import pandas as pd
    import plotly.graph_objects as go

    # Helper to resolve target error range robustly
    def fetch_error_range(short_var, base_path):
        class MockProp:
            def __init__(self, path):
                self.path = path
        try:
            from ofs_skill.visualization.plotting_functions import get_error_range
            return get_error_range(short_var, MockProp(base_path), logger)[0]
        except ImportError:
            pass

        try:
            from plotting_functions import get_error_range
            return get_error_range(short_var, MockProp(base_path), logger)[0]
        except ImportError:
            pass

        logger.warning('Could not import get_error_range. Using direct CSV fallback for target error.')
        config_path = os.path.join(base_path, 'conf', 'error_ranges.csv')
        defaults = {'salt': 3.5, 'temp': 3.0, 'wl': 0.15, 'cu': 0.26, 'ice_conc': 10.0}

        if os.path.exists(config_path):
            try:
                df_err = pd.read_csv(config_path)
                match = df_err[df_err['name_var'] == short_var]
                if not match.empty:
                    return float(match.iloc[0]['X1'])
            except Exception as e:
                logger.warning(f'Error reading {config_path}: {e}')

        return defaults.get(short_var, 0)

    logger.info('--- Starting Stats Comparison Plotting (Plotly) ---')

    stats_dir = os.path.join(home_path, 'data', 'skill', 'stats')
    vis_dir = os.path.join(home_path, 'data', 'visual', 'comparisons')
    os.makedirs(vis_dir, exist_ok=True)

    ofs1_file = os.path.join(stats_dir, f'skill_{ofs1}_all_stations.csv')
    ofs2_file = os.path.join(stats_dir, f'skill_{ofs2}_all_stations.csv')

    if not os.path.exists(ofs1_file):
        ofs1_file = os.path.join(home_path, f'skill_{ofs1}_all_stations.csv')
    if not os.path.exists(ofs2_file):
        ofs2_file = os.path.join(home_path, f'skill_{ofs2}_all_stations.csv')

    if not os.path.exists(ofs1_file) or not os.path.exists(ofs2_file):
        logger.warning(f'Stats files not found. Searched {stats_dir} and {home_path}. Skipping stats comparison.')
        return

    try:
        df1 = pd.read_csv(ofs1_file)
        df2 = pd.read_csv(ofs2_file)

        df1['ID'] = df1['ID'].astype(str)
        df2['ID'] = df2['ID'].astype(str)

        merged = pd.merge(df1, df2, on=['ID', 'variable'], suffixes=(f'_{ofs1}', f'_{ofs2}'))

        if merged.empty:
            logger.warning('No overlapping stations found in the stats files.')
            return

        # Map internal column names to display names
        stats_to_plot = {
            'rmse': 'RMSE',
            'bias': 'Bias',
            'central_freq': 'Central Frequency'
        }

        for var in merged['variable'].unique():
            var_data = merged[merged['variable'] == var].reset_index(drop=True)
            display_var = var.replace('_', ' ').title()

            # Detect the base variable to fetch the correct error range (ignores "high tide" and "low tide")
            var_lower = var.strip().lower()
            if 'water level' in var_lower or var_lower == 'wl':
                base_var = 'wl'
            elif 'temperature' in var_lower or var_lower == 'temp':
                base_var = 'temp'
            elif 'salinity' in var_lower or var_lower == 'salt':
                base_var = 'salt'
            elif 'current' in var_lower or var_lower == 'cu':
                base_var = 'cu'
            else:
                base_var = var.replace(' ', '_').lower()

            file_var = var.replace(' ', '_').lower()

            # Fetch target error range for thresholds
            X1 = fetch_error_range(base_var, home_path)

            for stat_key, stat_display in stats_to_plot.items():
                stat1 = f'{stat_key}_{ofs1}'
                stat2 = f'{stat_key}_{ofs2}'

                if stat1 not in var_data.columns or stat2 not in var_data.columns:
                    continue
                if var_data[stat1].isna().all() and var_data[stat2].isna().all():
                    continue

                bar_hover = f'<b>Station ID:</b> %{{x}}<br><b>Model:</b> %{{data.name}}<br><b>{stat_display}:</b> %{{y:.3f}}<extra></extra>'

                # --- 1. Grouped Bar Plot (Plotly) - optional ---
                if make_bar_plots:
                    fig_bar = go.Figure()
                    fig_bar.add_trace(go.Bar(x=var_data['ID'], y=var_data[stat1], name=ofs1.upper(), hovertemplate=bar_hover, marker_color='#d55e00'))
                    fig_bar.add_trace(go.Bar(x=var_data['ID'], y=var_data[stat2], name=ofs2.upper(), hovertemplate=bar_hover, marker_color='#0072b2'))

                    # Add Threshold Lines to Bar Plot (Solid)
                    if stat_key == 'central_freq':
                        fig_bar.add_hline(y=90, line_dash='solid', line_color='red',
                                          annotation_text='90% Target', annotation_position='top left',
                                          annotation_font=dict(color='red', size=13))
                    elif stat_key == 'rmse' and X1 > 0:
                        fig_bar.add_hline(y=X1, line_dash='solid', line_color='red',
                                          annotation_text=f'Target Error ({X1:.2f})', annotation_position='top left',
                                          annotation_font=dict(color='red', size=13))
                    elif stat_key == 'bias' and X1 > 0:
                        fig_bar.add_hline(y=X1, line_dash='solid', line_color='red',
                                          annotation_text=f'+Target Error (+{X1:.2f})', annotation_position='top left',
                                          annotation_font=dict(color='red', size=13))
                        fig_bar.add_hline(y=-X1, line_dash='solid', line_color='red',
                                          annotation_text=f'-Target Error (-{X1:.2f})', annotation_position='bottom left',
                                          annotation_font=dict(color='red', size=13))

                    fig_bar.update_layout(
                        barmode='group',
                        title=dict(
                            text=f'<b>Station-by-Station {stat_display} Comparison: {display_var}</b>',
                            font=dict(size=14, color='black', family='Open Sans'),
                            y=0.97, x=0.5, xanchor='center', yanchor='top',
                        ),
                        template='plotly_white', hovermode='x unified',
                        hoverlabel=dict(bgcolor='white', bordercolor='#cccccc', font=dict(family='Open Sans', size=13, color='#333333')),
                        legend=dict(
                            orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0,
                            font=dict(size=16, color='black'),
                            itemclick=False, itemdoubleclick=False
                        ),
                        margin=dict(t=100, b=100), height=550, width=1000
                    )
                    fig_bar.update_xaxes(
                        title_text='Station ID',
                        titlefont=dict(family='Open Sans', color='black', size=18),
                        mirror=True, ticks='inside', showline=True, linecolor='black', linewidth=1, tickangle=45,
                        tickfont=dict(family='Open Sans', color='black', size=14),
                        minor=dict(ticklen=4, tickcolor='black', ticks='inside', showgrid=False)
                    )

                    # Update Y-Axes (Includes explicit zeroline injection for Bias)
                    fig_bar.update_yaxes(
                        title_text=stat_display,
                        titlefont=dict(family='Open Sans', color='black', size=17),
                        mirror=True, ticks='inside', showline=True, linecolor='black', linewidth=1, showgrid=True,
                        tickfont=dict(family='Open Sans', color='black', size=14),
                        minor=dict(ticklen=4, tickcolor='black', ticks='inside', showgrid=False),
                        zeroline=(stat_key == 'bias'), zerolinewidth=1, zerolinecolor='black'
                    )

                    out_file_cat = os.path.join(vis_dir, f'{ofs1}_vs_{ofs2}_{file_var}_{stat_key}_stations.html')
                    fig_bar.write_html(out_file_cat)

                # --- 2. 1-to-1 Scatter (Plotly) ---
                fig_stat_scat = go.Figure()

                stat_scat_hover = f'<b>Station ID:</b> %{{customdata}}<br><b>{ofs1.upper()}:</b> %{{x:.3f}}<br><b>{ofs2.upper()}:</b> %{{y:.3f}}<extra></extra>'

                fig_stat_scat.add_trace(go.Scatter(
                    x=var_data[stat1], y=var_data[stat2], mode='markers',
                    customdata=var_data['ID'], hovertemplate=stat_scat_hover,
                    name='Stations',
                    showlegend=True,
                    marker=dict(size=10, color='#009E73', opacity=0.7, line=dict(color='black', width=1))
                ))

                min_val = min(var_data[stat1].min(), var_data[stat2].min())
                max_val = max(var_data[stat1].max(), var_data[stat2].max())

                # Check thresholds to ensure axes encompass the threshold lines
                if stat_key == 'central_freq':
                    min_val = min(min_val, 85) # Provide buffer below 90
                    max_val = max(max_val, 100)
                elif stat_key == 'rmse' and X1 > 0:
                    # Ensure the 3x band is visible on both axes
                    max_val = max(max_val, X1 * 3 * 1.1)
                    min_val = min(min_val, 0)
                elif stat_key == 'bias' and X1 > 0:
                    min_val = min(min_val, -X1 * 3 * 1.1)
                    max_val = max(max_val, X1 * 3 * 1.1)

                axis_range = None
                if pd.notna(min_val) and pd.notna(max_val):
                    buffer = (max_val - min_val) * 0.1 if (max_val - min_val) != 0 else 0.1
                    axis_range = [min_val - buffer, max_val + buffer]

                    # Add 1:1 line
                    fig_stat_scat.add_trace(go.Scatter(
                        x=axis_range, y=axis_range,
                        mode='lines', name='1:1 Line', hoverinfo='skip', showlegend=False, line=dict(color='black', dash='dash',
                                                                                                     width=1)
                    ))

                    # Add Bounded Threshold Lines. For RMSE and central
                    # frequency the horizontal segment is drawn all the way
                    # to the y-axis (x=axis_range[0]) so the target line is
                    # anchored to the axis rather than floating in the plot.
                    if stat_key == 'central_freq':
                        fig_stat_scat.add_trace(go.Scatter(
                            x=[axis_range[0]-10, 90, 90], y=[90, 90, axis_range[0]],
                            mode='lines', name='90% Target', showlegend=True, hoverinfo='skip',
                            line=dict(color='red', width=1.5, dash='solid')
                        ))
                    elif stat_key == 'rmse' and X1 > 0:
                        # Draw target, 2x and 3x error thresholds as nested
                        # L-shaped red lines that reach the y-axis.
                        for mult, dash, color in ((1, 'solid', 'gold'), (2, 'solid', 'orange'), (3, 'solid', 'red')):
                            thr = X1 * mult
                            fig_stat_scat.add_trace(go.Scatter(
                                x=[axis_range[0]-10, thr, thr], y=[thr, thr, axis_range[0]-10],
                                mode='lines',
                                name=f'{mult}x Target Error' if mult > 1 else 'Target Error',
                                showlegend=True, hoverinfo='skip',
                                line=dict(color=color, width=1.5, dash=dash)
                            ))
                    elif stat_key == 'bias' and X1 > 0:
                        # Draw target, 2x and 3x error boxes centred on zero.
                        for mult, dash, color in ((1, 'solid', 'gold'), (2, 'solid', 'orange'), (3, 'solid', 'red')):
                            thr = X1 * mult
                            fig_stat_scat.add_trace(go.Scatter(
                                x=[-thr, thr, thr, -thr, -thr], y=[-thr, -thr, thr, thr, -thr],
                                mode='lines',
                                name=f'{mult}x Target Error' if mult > 1 else 'Target Error',
                                showlegend=True, hoverinfo='skip',
                                line=dict(color=color, width=1.5, dash=dash)
                            ))
                if stat_key != 'bias':
                    # Top-Left Annotation
                    fig_stat_scat.add_annotation(
                        text=f'Higher {stat_display} for {ofs2.upper()}',
                        xref='paper', yref='paper', x=0.02, y=0.98,
                        xanchor='left', yanchor='top', showarrow=False,
                        font=dict(family='Open Sans', size=15, color='black'),
                        bgcolor='rgba(255, 255, 255, 0.8)', borderwidth=0
                    )

                    # Bottom-Right Annotation
                    fig_stat_scat.add_annotation(
                        text=f'Higher {stat_display} for {ofs1.upper()}',
                        xref='paper', yref='paper', x=0.98, y=0.02,
                        xanchor='right', yanchor='bottom', showarrow=False,
                        font=dict(family='Open Sans', size=15, color='black'),
                        bgcolor='rgba(255, 255, 255, 0.8)', borderwidth=0
                    )
                elif stat_key == 'bias':
                    anno_size = 12
                    # Top-Left Annotation
                    fig_stat_scat.add_annotation(
                        text=f'<i>{ofs2.upper()} overprediction,<br>{ofs1.upper()} underprediction</i>',
                        xref='paper', yref='paper', x=0.02, y=0.98,
                        xanchor='left', yanchor='top', showarrow=False,
                        font=dict(family='Open Sans', size=anno_size, color='black'),
                        bgcolor='rgba(255, 255, 255, 0.8)', borderwidth=0
                    )
                    # Bottom-Right Annotation
                    fig_stat_scat.add_annotation(
                        text=f'<i>{ofs2.upper()} underprediction,<br>{ofs1.upper()} overprediction</i>',
                        xref='paper', yref='paper', x=0.98, y=0.02,
                        xanchor='right', yanchor='bottom', showarrow=False,
                        font=dict(family='Open Sans', size=anno_size, color='black'),
                        bgcolor='rgba(255, 255, 255, 0.8)', borderwidth=0
                    )
                    # Bottom-Left Annotation
                    fig_stat_scat.add_annotation(
                        text=f'<i>{ofs2.upper()} underprediction,<br>{ofs1.upper()} underprediction</i>',
                        xref='paper', yref='paper', x=0.02, y=0.02,
                        xanchor='left', yanchor='bottom', showarrow=False,
                        font=dict(family='Open Sans', size=anno_size, color='black'),
                        bgcolor='rgba(255, 255, 255, 0.8)', borderwidth=0
                    )
                    # Top-Right Annotation
                    fig_stat_scat.add_annotation(
                        text=f'<i>{ofs2.upper()} overprediction,<br>{ofs1.upper()} overprediction</i>',
                        xref='paper', yref='paper', x=0.98, y=0.98,
                        xanchor='right', yanchor='top', showarrow=False,
                        font=dict(family='Open Sans', size=anno_size, color='black'),
                        bgcolor='rgba(255, 255, 255, 0.8)', borderwidth=0
                    )

                fig_stat_scat.update_layout(
                    title=dict(
                        text=f'<b>{ofs1.upper()} vs {ofs2.upper()} {stat_display}: {display_var}</b>',
                        font=dict(size=18, color='black', family='Open Sans'),
                        y=0.97, x=0.5, xanchor='center', yanchor='top',
                    ),
                    template='plotly_white', hovermode='closest',
                    hoverlabel=dict(bgcolor='white', bordercolor='#cccccc', font=dict(family='Open Sans', size=13, color='#333333')),
                    legend=dict(
                        orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0,
                        font=dict(size=16, color='black'),
                        itemclick=False, itemdoubleclick=False
                    ),
                    margin=dict(t=100, b=100), height=650, width=650
                )

                fig_stat_scat.update_xaxes(
                    title_text=f'{ofs1.upper()} {stat_display}', range=axis_range,
                    titlefont=dict(family='Open Sans', color='black', size=18),
                    mirror=True, ticks='inside', showline=True, linecolor='black', linewidth=1, showgrid=True,
                    tickfont=dict(family='Open Sans', color='black', size=14),
                    minor=dict(ticklen=4, tickcolor='black', ticks='inside', showgrid=False),
                    zeroline=(stat_key == 'bias'), zerolinewidth=1, zerolinecolor='black'
                )
                fig_stat_scat.update_yaxes(
                    title_text=f'{ofs2.upper()} {stat_display}', range=axis_range,
                    titlefont=dict(family='Open Sans', color='black', size=18),
                    mirror=True, ticks='inside', showline=True, linecolor='black', linewidth=1, showgrid=True,
                    tickfont=dict(family='Open Sans', color='black', size=14),
                    minor=dict(ticklen=4, tickcolor='black', ticks='inside', showgrid=False),
                    scaleanchor='x', scaleratio=1,
                    zeroline=(stat_key == 'bias'), zerolinewidth=1, zerolinecolor='black'
                )

                out_file_scat = os.path.join(vis_dir, f'{ofs1}_vs_{ofs2}_{file_var}_{stat_key}_scatter.html')
                fig_stat_scat.write_html(out_file_scat)

        logger.info(f'Saved stats comparisons to {vis_dir}')
    except Exception as e:
        logger.error(f'Error generating stat comparisons: {e}')

def setup_logger(home_path, config_file_arg):
    """Sets up the logger by reading conf/logging.conf."""
    try:
        config_file = utils.Utils(config_file_arg).get_config_file()
    except FileNotFoundError as err:
        print(f'CRITICAL ERROR: {err}')
        sys.exit(-1)

    log_config_file = os.path.join(home_path, 'conf', 'logging.conf')

    if not os.path.isfile(log_config_file):
        print(f'CRITICAL ERROR: Log config file not found at {log_config_file}')
        sys.exit(-1)

    if not os.path.isfile(config_file):
        print(f'CRITICAL ERROR: Config file not found at {config_file}')
        sys.exit(-1)

    logging.config.fileConfig(log_config_file)
    logger = logging.getLogger('root')
    logger.info('Using config %s', config_file)
    logger.info('Using log config %s', log_config_file)

    return logger

def run_skill_assessment(ofs_name, args, logger, create_1dplot):
    """Configures and runs create_1dplot for a given OFS."""
    logger.info(f'--- Running 1D Plot Assessment for {ofs_name.upper()} ---')
    prop = model_properties.ModelProperties()
    prop.ofs = ofs_name.lower()
    prop.path = args.home_path
    prop.start_date_full = args.start_date
    prop.end_date_full = args.end_date
    prop.whichcasts = args.whichcasts
    prop.datum = args.datum
    prop.ofsfiletype = args.filetype
    prop.stationowner = args.station_owner
    prop.horizonskill = False
    prop.forecast_hr = 'now'
    prop.var_list = args.var_selection
    prop.aux_vars = ''
    prop.filecheck = False # Set to false to avoid duplicating the check
    prop.config_file = args.config
    prop.user_input_location = False

    # Run the assessment
    create_1dplot(prop, logger)

def setup_overlap_inventories(ofs1, ofs2, home_path, logger):
    """Copies the overlap inventory so each OFS reads it as its own.

    Returns a tuple of ``(overlap_csv, backups)`` where ``backups`` is a
    list of ``(backup_csv, target_csv)`` pairs that must be restored once
    the comparison run has finished (see ``restore_inventories``).
    """
    control_dir = os.path.join(home_path, 'control_files')
    overlap_name = f'{ofs1}_{ofs2}_overlap'
    overlap_csv = os.path.join(control_dir, f'inventory_all_{overlap_name}.csv')

    if not os.path.exists(overlap_csv):
        logger.error(f'Overlap inventory not found at {overlap_csv}. Cannot proceed.')
        sys.exit(-1)

    ofs1_csv = os.path.join(control_dir, f'inventory_all_{ofs1}.csv')
    ofs2_csv = os.path.join(control_dir, f'inventory_all_{ofs2}.csv')

    backups = []
    for target_csv in [ofs1_csv, ofs2_csv]:
        if os.path.exists(target_csv):
            backup_csv = target_csv + '.bak'
            logger.info(f'Backing up existing inventory: {target_csv} -> {backup_csv}')
            shutil.copy2(target_csv, backup_csv)
            backups.append((backup_csv, target_csv))

    logger.info(f'Restricting {ofs1} and {ofs2} to overlapping stations only...')
    shutil.copy2(overlap_csv, ofs1_csv)
    shutil.copy2(overlap_csv, ofs2_csv)

    return overlap_csv, backups


def restore_inventories(backups, logger):
    """Restores the original per-OFS inventories from their backups."""
    for backup_csv, target_csv in backups:
        if os.path.exists(backup_csv):
            logger.info(f'Restoring original inventory: {backup_csv} -> {target_csv}')
            shutil.move(backup_csv, target_csv)


def main(args):
    """Run shapefile intersection, restricted assessment, and comparisons."""
    ofs1, ofs2 = args.ofs1.lower(), args.ofs2.lower()

    vis_dir = os.path.join(args.home_path, 'bin', 'visualization')
    utils_dir = os.path.join(args.home_path, 'bin', 'utils')

    for dynamic_dir in [vis_dir, utils_dir]:
        if not os.path.exists(dynamic_dir):
            print(f"CRITICAL ERROR: Could not find 'bin' directory at {dynamic_dir}")
            sys.exit(-1)
        if dynamic_dir not in sys.path:
            sys.path.insert(0, dynamic_dir)

    from create_1dplot import create_1dplot
    from get_shapefile_intersection import get_shapefile_intersection

    # loggin'
    logger = setup_logger(args.home_path, args.config)

    logger.info('=== Pre-checking Model Files ===')
    for ofs_name in [ofs1, ofs2]:
        pre_prop = model_properties.ModelProperties()
        pre_prop.ofs = ofs_name
        pre_prop.path = args.home_path
        pre_prop.start_date_full = args.start_date
        pre_prop.end_date_full = args.end_date
        pre_prop.whichcasts = args.whichcasts.split(',') # check_model_files expects a list
        pre_prop.ofsfiletype = args.filetype
        pre_prop.config_file = args.config

        logger.info(f'Verifying files for {ofs_name.upper()}...')
        check_model_files(pre_prop, logger)
        logger.info(f'Model file check was successful for {ofs_name.upper()}.')

    # Do shapefile Intersection
    logger.info('=== Computing Shapefile Intersection ===')
    get_shapefile_intersection(
        shp1=ofs1, shp2=ofs2, home_path=args.home_path,
        stationowner=args.station_owner, logger=logger,
    )

    # Make two identical inventories, one for each OFS
    logger.info('=== Applying Overlap Restrictions ===')
    overlap_csv, backups = setup_overlap_inventories(
        ofs1, ofs2, args.home_path, logger,
    )

    try:
        # Run create_1dplot for restricted domains
        logger.info('=== Running Assessment on Overlapping Stations ===')
        run_skill_assessment(ofs1, args, logger, create_1dplot)
        run_skill_assessment(ofs2, args, logger, create_1dplot)

        # generate inter-model comparisons
        logger.info('=== Generating Data Comparisons ===')
        generate_comparisons(
            ofs1, ofs2, overlap_csv, args.var_selection,
            args.home_path, args.datum, logger,
        )

        # generate scatter plots, etc.
        logger.info('=== Generating Stats Comparisons ===')
        generate_stat_comparisons(
            ofs1, ofs2, args.home_path, logger,
            make_bar_plots=args.make_bar_plots,
        )
    finally:
        # Always restore the genuine per-OFS inventories, even on failure,
        # so subsequent (non-comparison) runs are not left with the
        # overlap-only subset.
        restore_inventories(backups, logger)

    logger.info('Program Complete!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run shapefile intersection, restricted skill assessment, and comparisons.')
    parser.add_argument('-o1', '--ofs1', required=True, help='First OFS to overlap')
    parser.add_argument('-o2', '--ofs2', required=True, help='Second OFS to overlap')
    parser.add_argument('-p', '--home_path', required=True, help='Path to package installation')
    parser.add_argument('-s', '--start_date', required=True, help='Assessment start date')
    parser.add_argument('-e', '--end_date', required=True, help='Assessment end date')
    parser.add_argument('-vs', '--var_selection', default='water_level', help='Variables to assess')
    parser.add_argument('-ws', '--whichcasts', default='nowcast', help='Whichcasts to assess')
    parser.add_argument('-so', '--station_owner', default='co-ops,ndbc,usgs,chs', help='Station providers')
    parser.add_argument('-d', '--datum', default='MLLW', help='Datum')
    parser.add_argument('-t', '--filetype', default='stations', help='OFS filetype')
    parser.add_argument('-c', '--config', help='Path to config file')
    parser.add_argument(
        '-b', '--make_bar_plots', action='store_true',
        help='Also generate station-by-station grouped bar plots for each '
             'skill statistic (off by default; scatter plots are always made).',
    )

    main(parser.parse_args())
