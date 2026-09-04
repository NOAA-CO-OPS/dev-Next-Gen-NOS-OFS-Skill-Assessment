"""
OFS Skill Assessment Comparison Tool

This script compares the performance of two Operational Forecast Systems (OFS)
by evaluating them on a shared set of overlapping stations. The workflow includes:
1. Identifying overlapping stations via shapefile intersection.
2. Isolating the assessment to only those overlapping stations by temporarily
   modifying the station inventory files (with fail-safe in-memory backups).
3. Running 1D skill assessments (`create_1dplot`) for both models.
4. Generating comparative Plotly time series for paired variables.
5. Generating Plotly statistical comparisons (RMSE, Bias, Central Frequency) across all stations,
   including interactive Map views with dropdowns.

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

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ofs_skill.model_processing import model_properties


def fetch_error_range(short_var, base_path, logger):
    """Helper to resolve target error range robustly."""
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

    logger.warning('Could not import get_error_range. Using direct CSV fallback.')
    config_path = os.path.join(base_path, 'conf', 'error_ranges.csv')
    defaults = {
        'salt': 3.5, 'temp': 3.0, 'wl': 0.15, 'cu': 0.26,
        'ice_conc': 10.0, 'cu_dir': 22.5
    }

    if os.path.exists(config_path):
        try:
            df_err = pd.read_csv(config_path)
            match = df_err[df_err['name_var'] == short_var]
            if not match.empty:
                return float(match.iloc[0]['X1'])
        except Exception as e:
            logger.warning(f'Error reading {config_path}: {e}')

    return defaults.get(short_var, 0)


def generate_comparisons(ofs1, ofs2, overlap_csv, var_selection, whichcasts,
                         home_path, datum, logger):
    """Ingests paired datasets for overlapping stations and creates interactive Plotly time series."""
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
    casts_to_process = [c.strip() for c in whichcasts.split(',')]

    for var in vars_to_process:
        var = var.strip()
        display_var = var.replace('_', ' ').title()
        var_map = {'water_level': 'wl', 'water_temperature': 'temp', 'salinity': 'salt', 'currents': 'cu'}
        short_var = var_map.get(var, var)

        if short_var == 'wl':
            y_title = f'Water Level ({datum})'
            unit = 'm'
        elif short_var == 'temp':
            y_title = 'Water Temperature (C)'
            unit = '\u00b0C'
        elif short_var == 'cu':
            y_title = 'Current Speed (knots)'
            unit = 'knots'
        elif short_var == 'salt':
            y_title = 'Salinity (PSU)'
            unit = 'PSU'
        else:
            y_title = display_var
            unit = ''

        col_names = (
            ['Julian', 'year', 'month', 'day', 'hour', 'minute', 'OBS_SPD', 'OFS_SPD', 'BIAS_SPD', 'OBS_DIR', 'OFS_DIR', 'BIAS_DIR']
            if short_var == 'cu' else
            ['Julian', 'year', 'month', 'day', 'hour', 'minute', 'OBS', 'OFS', 'BIAS']
        )

        target_err = fetch_error_range(short_var, home_path, logger)
        if short_var == 'cu':
            target_err *= 1.943844

        for cast in casts_to_process:
            cast_file = cast.lower()
            cast_title = cast_file.replace('_b', '')

            for station in overlap_stations:
                depth_bins = ['']
                if short_var == 'cu':
                    found_ofs1 = glob.glob(os.path.join(pair_dir, f'{ofs1}_{short_var}_{station}_*_{cast}_*_pair.int'))
                    if not found_ofs1:
                        continue
                    depth_bins = list({os.path.basename(f)[len(f'{ofs1}_{short_var}_{station}_'):].split(f'_{cast}_')[0].split('_')[0] for f in found_ofs1})

                for depth_bin in depth_bins:
                    station_key = f'{station}_{depth_bin}' if depth_bin else station
                    ofs1_files = glob.glob(os.path.join(pair_dir, f'{ofs1}_{short_var}_{station_key}*_{cast}_*_pair.int'))
                    ofs2_files = glob.glob(os.path.join(pair_dir, f'{ofs2}_{short_var}_{station_key}*_{cast}_*_pair.int'))

                    if not ofs1_files or not ofs2_files:
                        continue

                    try:
                        df1 = pd.read_csv(ofs1_files[0], sep=r'\s+', names=col_names, header=0)
                        df2 = pd.read_csv(ofs2_files[0], sep=r'\s+', names=col_names, header=0)

                        df1['DateTime'] = pd.to_datetime(df1[['year', 'month', 'day', 'hour', 'minute']])
                        df2['DateTime'] = pd.to_datetime(df2[['year', 'month', 'day', 'hour', 'minute']])

                        if short_var == 'cu':
                            df1['OBS'], df1['OFS'] = df1['OBS_SPD'] * 1.943844, df1['OFS_SPD'] * 1.943844
                            df2['OBS'], df2['OFS'] = df2['OBS_SPD'] * 1.943844, df2['OFS_SPD'] * 1.943844

                        merged = pd.merge(df1[['DateTime', 'OBS', 'OFS']], df2[['DateTime', 'OFS']], on='DateTime', suffixes=(f'_{ofs1}', f'_{ofs2}'))
                        if merged.empty:
                            continue

                        merged[f'Error_{ofs1}'] = merged[f'OFS_{ofs1}'] - merged['OBS']
                        merged[f'Error_{ofs2}'] = merged[f'OFS_{ofs2}'] - merged['OBS']

                        ts_hover = f'<b>Time:</b> %{{x|%m/%d/%Y %H:%M}}<br><b>%{{data.name}}:</b> %{{y:.2f}} {unit}<extra></extra>'
                        fig_ts = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])

                        fig_ts.add_trace(go.Scatter(x=merged['DateTime'], y=merged['OBS'], name='Observation', hovertemplate=ts_hover, line=dict(color='red')), row=1, col=1)
                        fig_ts.add_trace(go.Scatter(x=merged['DateTime'], y=merged[f'OFS_{ofs1}'], name=ofs1.upper(), hovertemplate=ts_hover, line=dict(color='#d55e00')), row=1, col=1)
                        fig_ts.add_trace(go.Scatter(x=merged['DateTime'], y=merged[f'OFS_{ofs2}'], name=ofs2.upper(), hovertemplate=ts_hover, line=dict(color='#0072b2')), row=1, col=1)

                        if target_err > 0:
                            fig_ts.add_trace(go.Scatter(x=[merged['DateTime'].min(), merged['DateTime'].max(), merged['DateTime'].max(), merged['DateTime'].min()], y=[target_err, target_err, -target_err, -target_err], fill='toself', fillcolor='rgba(255, 165, 0, 0.3)', line=dict(color='rgba(255,255,255,0)'), name='Target Error', hoverinfo='skip'), row=2, col=1)

                        fig_ts.add_trace(go.Scatter(x=merged['DateTime'], y=merged[f'Error_{ofs1}'], name=f'{ofs1.upper()} Error', hovertemplate=ts_hover, line=dict(color='#d55e00')), row=2, col=1)
                        fig_ts.add_trace(go.Scatter(x=merged['DateTime'], y=merged[f'Error_{ofs2}'], name=f'{ofs2.upper()} Error', hovertemplate=ts_hover, line=dict(color='#0072b2')), row=2, col=1)

                        fig_ts.update_layout(title=dict(text=f'Time Series: {station_key} - {display_var} ({cast_title})'), template='plotly_white')
                        fig_ts.update_yaxes(title_text=y_title, row=1, col=1)
                        fig_ts.update_yaxes(title_text='Error', row=2, col=1)

                        fig_ts.write_html(os.path.join(visual_dir, f'{ofs1}_vs_{ofs2}_{short_var}_{station_key}_{cast_file}_timeseries.html'))

                    except Exception as e:
                        logger.error(f'Error plotting TS for {station_key}: {e}')


def generate_stat_comparisons(ofs1, ofs2, var_selection, whichcasts, home_path,
                              start_date, end_date, logger, make_bar_plots=False):
    """Reads skill stat CSVs and plots interactive 1-to-1 scatters with bounded thresholds."""
    start_str = start_date.replace('T', ' ').replace('Z', '')
    end_str = end_date.replace('T', ' ').replace('Z', '')

    variable_keywords = {'water_level_hw': 'Water Level high tide', 'water_level': 'Water Level', 'currents': 'Current speed'}
    cast_keywords = {'nowcast': 'Nowcast', 'forecast_b': 'Forecast (B)', 'hindcast': 'Hindcast'}

    stats_dir = os.path.join(home_path, 'data', 'skill', 'stats')
    vis_dir = os.path.join(home_path, 'data', 'visual', 'comparisons')
    os.makedirs(vis_dir, exist_ok=True)

    ofs1_file, ofs2_file = os.path.join(stats_dir, f'skill_{ofs1}_all_stations.csv'), os.path.join(stats_dir, f'skill_{ofs2}_all_stations.csv')
    if not os.path.exists(ofs1_file) or not os.path.exists(ofs2_file):
        return

    try:
        merged = pd.merge(pd.read_csv(ofs1_file).astype({'ID': str}), pd.read_csv(ofs2_file).astype({'ID': str}), on=['ID', 'variable', 'type'], suffixes=(f'_{ofs1}', f'_{ofs2}'))
        if merged.empty:
            return

        stats_to_plot = {'rmse': 'RMSE', 'bias': 'Bias', 'central_freq': 'Central Frequency'}
        vars_to_process = list(dict.fromkeys(var_selection.split(',')))
        casts_to_process = [c.strip() for c in whichcasts.split(',')]

        for var in vars_to_process:
            csv_var = variable_keywords.get(var, var)
            for cast in casts_to_process:
                csv_cast = cast_keywords.get(cast, cast)
                var_data = merged[(merged['variable'] == csv_var) & (merged['type'] == csv_cast)].reset_index(drop=True)
                if var_data.empty:
                    continue

                display_var = var.replace('_', ' ').title()
                base_var = 'wl' if 'water_level' in var else var
                target_err = fetch_error_range(base_var, home_path, logger)
                lon_col, lat_col = f'X_{ofs1}' if f'X_{ofs1}' in var_data.columns else 'X', f'Y_{ofs1}' if f'Y_{ofs1}' in var_data.columns else 'Y'

                fig_map = go.Figure()
                stat_traces_map, map_trace_idx = {k: [] for k in stats_to_plot}, 0

                if not var_data[lon_col].dropna().empty:
                    mean_lon, mean_lat = var_data[lon_col].dropna().mean(), var_data[lat_col].dropna().mean()
                    lon_diff, lat_diff = var_data[lon_col].max() - var_data[lon_col].min(), var_data[lat_col].max() - var_data[lat_col].min()
                    zoom_level = min(math.log2(360 / lon_diff) if lon_diff > 0 else 15.0, math.log2(180 / lat_diff) if lat_diff > 0 else 15.0) - 0.25
                else:
                    mean_lon, mean_lat, zoom_level = -95, 38, 4

                for stat_key, stat_display in stats_to_plot.items():
                    stat1, stat2 = f'{stat_key}_{ofs1}', f'{stat_key}_{ofs2}'
                    if stat1 not in var_data.columns or var_data[stat1].isna().all():
                        continue

                    fig_stat_scat = go.Figure()
                    pass_1 = var_data[stat1] <= target_err if stat_key != 'central_freq' else var_data[stat1] >= 90
                    pass_2 = var_data[stat2] <= target_err if stat_key != 'central_freq' else var_data[stat2] >= 90

                    cat_masks = {
                        'Both Pass': (pass_1 & pass_2, '#009E73'),
                        f'{ofs1.upper()} Fails, {ofs2.upper()} Passes': (~pass_1 & pass_2, '#56B4E9'),
                        f'{ofs2.upper()} Fails, {ofs1.upper()} Passes': (pass_1 & ~pass_2, '#E69F00'),
                        'Both Fail': (~pass_1 & ~pass_2, '#D55E00')
                    }

                    for label, (mask, color) in cat_masks.items():
                        if not mask.any():
                            continue
                        subset = var_data[mask]
                        fig_stat_scat.add_trace(go.Scatter(x=subset[stat1], y=subset[stat2], mode='markers', name=label, marker=dict(color=color)))

                        if lon_col in subset.columns:
                            fig_map.add_trace(go.Scattermap(lon=subset[lon_col], lat=subset[lat_col], mode='markers', name=label, marker=dict(color=color)))
                            stat_traces_map[stat_key].append(map_trace_idx)
                            map_trace_idx += 1

                    fig_stat_scat.write_html(os.path.join(vis_dir, f'{ofs1}_vs_{ofs2}_{base_var}_{cast}_scatter.html'))

                if map_trace_idx > 0:
                    fig_map.update_layout(
                        title=f'{ofs1.upper()} vs {ofs2.upper()} Map: {display_var} ({start_str} to {end_str})',
                        map=dict(center=dict(lat=mean_lat, lon=mean_lon), zoom=zoom_level)
                    )
                    fig_map.write_html(os.path.join(vis_dir, f'{ofs1}_vs_{ofs2}_{base_var}_{cast}_map.html'))

    except Exception as e:
        logger.error(f'Error generating stat comparisons: {e}')


def setup_logger(home_path, config_file_arg):
    """Sets up the logger by reading conf/logging.conf."""
    # try:
    #     config_file = utils.Utils(config_file_arg).get_config_file()
    # except FileNotFoundError:
    #     sys.exit(-1)

    log_config_file = os.path.join(home_path, 'conf', 'logging.conf')
    if not os.path.isfile(log_config_file):
        sys.exit(-1)

    logging.config.fileConfig(log_config_file)
    logger = logging.getLogger('root')
    return logger


def run_skill_assessment(ofs_name, args, logger, create_1dplot):
    """Configures and runs create_1dplot for a given OFS."""
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
    prop.config_file = args.config
    create_1dplot(prop, logger)


def setup_overlap_inventories(ofs1, ofs2, args, logger):
    """Copies the overlap inventory and stores originals in memory."""
    from ofs_skill.utils import cache_manifest

    control_dir = os.path.join(args.home_path, 'control_files')
    overlap_csv = os.path.join(control_dir, f'inventory_all_{ofs1}_{ofs2}_overlap.csv')
    ofs1_csv, ofs2_csv = os.path.join(control_dir, f'inventory_all_{ofs1}.csv'), os.path.join(control_dir, f'inventory_all_{ofs2}.csv')

    backups = []
    for ofs, target_csv in [(ofs1, ofs1_csv), (ofs2, ofs2_csv)]:
        if os.path.exists(target_csv):
            with open(target_csv) as f:
                backups.append((f.read(), target_csv, cache_manifest.inventory_signature(ofs, args.start_date, args.end_date, args.station_owner.split(',')), True))
        else:
            backups.append((None, target_csv, None, False))

    shutil.copy2(overlap_csv, ofs1_csv)
    shutil.copy2(overlap_csv, ofs2_csv)
    return overlap_csv, backups


def restore_inventories(backups, home_path, logger):
    """Restores the original per-OFS inventories from memory or deletes temporary ones."""
    from ofs_skill.utils import cache_manifest
    for content, target_csv, orig_sig, existed in backups:
        if existed:
            with open(target_csv, 'w') as f:
                f.write(content)
            cache_manifest.record_artifact(target_csv, orig_sig, home_path, logger)
        elif os.path.exists(target_csv):
            os.remove(target_csv)


def main(args):
    """Run shapefile intersection, restricted assessment, and comparisons."""
    ofs1, ofs2 = args.ofs1.lower(), args.ofs2.lower()
    for dynamic_dir in [os.path.join(args.home_path, 'bin', 'visualization'), os.path.join(args.home_path, 'bin', 'utils')]:
        if dynamic_dir not in sys.path:
            sys.path.insert(0, dynamic_dir)

    from create_1dplot import create_1dplot
    from get_shapefile_intersection import get_shapefile_intersection

    logger = setup_logger(args.home_path, args.config)
    get_shapefile_intersection(shp1=ofs1, shp2=ofs2, home_path=args.home_path, stationowner=args.station_owner, logger=logger)
    overlap_csv, backups = setup_overlap_inventories(ofs1, ofs2, args, logger)

    try:
        run_skill_assessment(ofs1, args, logger, create_1dplot)
        run_skill_assessment(ofs2, args, logger, create_1dplot)
        generate_comparisons(ofs1, ofs2, overlap_csv, args.var_selection, args.whichcasts, args.home_path, args.datum, logger)
        generate_stat_comparisons(ofs1, ofs2, args.var_selection, args.whichcasts, args.home_path, args.start_date, args.end_date, logger, args.make_bar_plots)
    finally:
        restore_inventories(backups, args.home_path, logger)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run shapefile intersection and comparisons.')
    parser.add_argument('-o1', '--ofs1', required=True)
    parser.add_argument('-o2', '--ofs2', required=True)
    parser.add_argument('-p', '--home_path', required=True)
    parser.add_argument('-s', '--start_date', required=True)
    parser.add_argument('-e', '--end_date', required=True)
    parser.add_argument('-vs', '--var_selection', default='water_level,salinity,water_temperature,currents')
    parser.add_argument('-ws', '--whichcasts', default='nowcast')
    parser.add_argument('-so', '--station_owner', default='co-ops,ndbc,usgs,chs')
    parser.add_argument('-d', '--datum', default='MLLW')
    parser.add_argument('-t', '--filetype', default='stations')
    parser.add_argument('-c', '--config')
    parser.add_argument('-b', '--make_bar_plots', action='store_true')

    main(parser.parse_args())
