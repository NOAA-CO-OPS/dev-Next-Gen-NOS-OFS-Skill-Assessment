"""
Standalone script to plot model time series from .prd files using Plotly.
Combines nowcast and forecast runs onto a single plot per station,
with a vertical separator if the runs are sequential, and includes
observation data from .int files (preferring .int over .prd when both exist).
"""

import argparse
import difflib
import glob
import logging
import logging.config
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import from ofs_skill package
from ofs_skill.obs_retrieval import utils
from ofs_skill.obs_retrieval.get_station_tidal_data import get_station_tidal_data
from ofs_skill.obs_retrieval.station_ctl_file_extract import station_ctl_file_extract


def get_pcd_value(ctl_file, target_id, target_name, logger):
    """
    Extracts the Principal Current Direction (PCD) value from the control file
    using exact ID matching or closest (fuzzy) string matching for station names.
    """
    if not os.path.exists(ctl_file):
        # Fallback: check if the control file is in the current working directory
        fallback_file = os.path.basename(ctl_file)
        if os.path.exists(fallback_file):
            ctl_file = fallback_file
        else:
            logger.warning(f'Could not find control file to read PCD: {ctl_file}')
            return None

    try:
        name_to_pcd = {}
        id_to_pcd = {}

        with open(ctl_file) as file:
            for line in file:
                if line.startswith('ST='):
                    parts = line.split("'")
                    if len(parts) >= 5:
                        st_name = parts[1].strip().lower()
                        st_abbr = parts[3].strip().lower()
                        numeric_data = parts[4].split()
                        if len(numeric_data) >= 3:
                            pcd = float(numeric_data[2])
                            name_to_pcd[st_name] = pcd
                            id_to_pcd[st_abbr] = pcd

        target_id_clean = target_id.strip().lower() if target_id else ''
        target_name_clean = target_name.strip().lower() if target_name else ''

        # 1. Exact match on ID
        if target_id_clean and target_id_clean in id_to_pcd:
            return id_to_pcd[target_id_clean]

        # 2. Substring match on ID
        for abbr, pcd in id_to_pcd.items():
            if target_id_clean and (target_id_clean in abbr or abbr in target_id_clean):
                return pcd

        # 3. Fuzzy match on Station Name
        if target_name_clean and name_to_pcd:
            # Find the closest match above a 60% similarity threshold
            matches = difflib.get_close_matches(target_name_clean, name_to_pcd.keys(), n=1, cutoff=0.6)
            if matches:
                best_match = matches[0]
                if best_match != target_name_clean:
                    logger.info(f"Using fuzzy match '{best_match}' for target station '{target_name_clean}'.")
                return name_to_pcd[best_match]

    except Exception as e:
        logger.warning(f'Error parsing {ctl_file} for PCD: {e}')

    return None


def get_plot_title(plotinfo):
    """Formats the plot title using the metadata extracted from the filename."""
    start_str = datetime.strftime(plotinfo['start'], '%Y/%m/%d %H:%M:%S')
    end_str = datetime.strftime(plotinfo['end'], '%Y/%m/%d %H:%M:%S')

    title = (f'<b>NOAA/National Ocean Service, {plotinfo["ofs"].upper()} <br>'
             f'Station Name:&nbsp;{plotinfo["station_name"]} &nbsp;&nbsp;&nbsp;'
             f'Station ID:&nbsp;{plotinfo["station_id"]}')

    if plotinfo.get('pcd_value') is not None:
        title += f'<br>Along-Channel Direction:&nbsp;{plotinfo["pcd_value"]}\u00b0 (Positive = Flood, Negative = Ebb)'

    title += (f'<br>From:&nbsp;{start_str}'
              f'&nbsp;&nbsp;&nbsp;To:&nbsp;{end_str}</b>')

    return title


def get_variable_names(name_var, datum):
    """Maps short variable names to full plot titles and save names."""
    if name_var == 'wl':
        plot_name = f'Water Level (<i>feet {datum}</i>)'
        save_name = 'water_level'
        unit = ' ft'
    elif name_var in ('temp', 'water_temperature'):
        plot_name = 'Water Temperature (<i>\u00b0C</i>)'
        save_name = 'water_temperature'
        unit = ' \u00b0C'
    elif name_var in ('salt', 'salinity'):
        plot_name = 'Salinity (<i>PSU</i>)'
        save_name = 'salinity'
        unit = ' PSU'
    elif name_var in ('cu', 'currents'):
        plot_name = 'Current Speed (<i>knots</i>)'
        save_name = 'currents'
        unit = ' knots'
    elif name_var == 'wind':
        plot_name = 'Wind Speed & Direction<br>(<i>m/s & deg</i>)'
        save_name = 'wind'
        unit = ' m/s'
    else:
        plot_name = f'{name_var.capitalize()}'
        save_name = name_var
        unit = ''
    return plot_name, save_name, unit


def get_trace_styling(whichcast, trace_type):
    """Assigns distinct colors and line dashes to differentiate nowcasts from forecasts."""
    is_forecast = 'forecast' in whichcast.lower()

    if is_forecast:
        color = 'green'
        dash = 'dash'
    else:
        color = 'black'
        dash = 'dashdot'

    return color, dash


def main(logger, _conf=None, inventory_file=None, variable=None, ofs_filter=None,
         datum=None):
    """This function plots combined time series from .prd and .int files"""

    # Directories from conf file
    dir_params = utils.Utils(_conf).read_config_section('directories', logger)
    home_path = dir_params['home']

    # Logger setup
    if logger is None:
        config_file = utils.Utils(_conf).get_config_file()
        log_config_file = 'conf/logging.conf'
        log_config_file = os.path.join(Path(home_path), log_config_file)

        if not os.path.isfile(log_config_file):
            print(f'Log config not found at {log_config_file}. Exiting.')
            sys.exit(-1)

        logging.config.fileConfig(log_config_file)
        logger = logging.getLogger('root')
        logger.info('Using config %s', config_file)

    logger.info('--- Starting Visualization Process ---')
    if variable:
        logger.info(f"Filtering plots for variable: '{variable}'")
    if ofs_filter:
        logger.info(f"Filtering plots for OFS: '{ofs_filter}'")

    prd_folder = os.path.join(
        home_path, dir_params['data_dir'], dir_params['model_dir'],
        dir_params['1d_node_dir']
    )
    os.makedirs(prd_folder, exist_ok=True)
    int_folder = os.path.join(
        home_path, dir_params['data_dir'], dir_params['skill_dir'],
        dir_params['1d_pair_dir']
    )
    os.makedirs(int_folder, exist_ok=True)
    ctl_folder = os.path.join(
        home_path, dir_params['control_files_dir']
    )
    os.makedirs(ctl_folder, exist_ok=True)
    save_path = os.path.join(prd_folder, 'prd_plots')
    os.makedirs(save_path, exist_ok=True)

    # Search for all PRD and INT files
    search_pattern_prd = os.path.join(prd_folder, '*.prd')
    search_pattern_int = os.path.join(int_folder, '*.int')
    all_files = glob.glob(search_pattern_prd) + glob.glob(search_pattern_int)

    if not all_files:
        logger.info('No .prd or .int output files found! No plots were made.')
        return

    # Group files by (station_id, ofs, var_name, node)
    grouped_files = {}
    for file_path in all_files:
        file = os.path.basename(file_path)
        parts = file.split('_')

        # Protect against short filenames
        if len(parts) < 5:
            continue

        # Find where the cast starts ('forecast' or 'nowcast') to anchor our parsing
        cast_idx = -1
        for i, part in enumerate(parts):
            if part in ['forecast', 'nowcast']:
                cast_idx = i
                break

        if cast_idx == -1:
            continue

        # Parse based on file type robustly handling underscores in station IDs
        if file.endswith('.prd'):
            if cast_idx < 4:
                continue

            node = parts[cast_idx - 1]
            var_name = parts[cast_idx - 2]
            ofs = parts[cast_idx - 3]
            station_id = '_'.join(parts[0:cast_idx - 3])

            whichcast = parts[cast_idx]
            if whichcast == 'forecast' and len(parts) > cast_idx + 1 and parts[cast_idx + 1] in ['a', 'b']:
                whichcast = f'forecast_{parts[cast_idx + 1]}'

        elif file.endswith('.int'):
            if cast_idx < 3:
                continue

            ofs = parts[0]
            var_name = parts[1]
            node = parts[cast_idx - 1]
            station_id = '_'.join(parts[2:cast_idx - 1])

            whichcast = parts[cast_idx]
            if whichcast == 'forecast' and len(parts) > cast_idx + 1 and parts[cast_idx + 1] in ['a', 'b']:
                whichcast = f'forecast_{parts[cast_idx + 1]}'
        else:
            continue

        # If a variable filter is applied, skip files that don't match
        if variable and var_name != variable:
            continue

        # If an OFS filter is applied, skip files that don't match
        if ofs_filter and ofs.lower() != ofs_filter.lower():
            continue

        group_key = (station_id, ofs, var_name, node)
        if group_key not in grouped_files:
            grouped_files[group_key] = []
        grouped_files[group_key].append({'path': file_path, 'whichcast': whichcast})

    if not grouped_files:
        logger.info('No files matched the applied filters. No plots were made.')
        return

    logger.info(f'Found {sum(len(v) for v in grouped_files.values())} matching files. Grouping by station and variable...')

    # Assign datum -- check for datum input argument first, then check for ctl file
    if datum:
        logger.info('Datum input argument found!')
    else:
        logger.info('No datum input argument found. Reading datum from control file...')
        try:
            read_station_ctl_file = station_ctl_file_extract(
                r'' + ctl_folder + '/' + ofs + '_' + var_name + '_station.ctl')
            logger.info('Station ctl file (%s_%s_station.ctl) found.', ofs_filter, var_name)
            datum = read_station_ctl_file[1][0][-1].upper()
        except (FileNotFoundError, IndexError, TypeError):
            logger.error('Station ctl file not found. No datum found, so tide predictions will not be plotted.')

    # Cache for inventory lookup
    inv_cache = {}

    # Iterate over each distinct Station+Variable group
    for (station_id, ofs, var_name, node), files_list in grouped_files.items():
        try:
            plotinfo = {
                'station_id': station_id,
                'ofs': ofs,
                'var_name': var_name,
                'node': node,
                'station_name': 'Unknown'
            }

            plotinfo['plot_name'], plotinfo['save_name'], plotinfo['unit'] = \
                get_variable_names(plotinfo['var_name'], datum)

            # --- Deduplicate: Prefer .int over .prd for the same run ---
            deduped_files = []
            whichcasts = {f['whichcast'] for f in files_list}
            for wc in whichcasts:
                wc_files = [f for f in files_list if f['whichcast'] == wc]
                int_files = [f for f in wc_files if f['path'].endswith('.int')]
                if int_files:
                    deduped_files.extend(int_files)
                else:
                    deduped_files.extend(wc_files)
            files_list = deduped_files

            # --- STATION NAME LOOKUP LOGIC ---
            if inventory_file and os.path.exists(inventory_file):
                if 'manual' not in inv_cache:
                    inv_cache['manual'] = pd.read_csv(inventory_file)
                    if 'ID' in inv_cache['manual'].columns:
                        inv_cache['manual']['ID'] = inv_cache['manual']['ID'].astype(str)
                inv_df = inv_cache['manual']
            else:
                if ofs not in inv_cache:
                    control_dir = os.path.join(home_path, dir_params.get('control_files_dir', ''))
                    paths_to_check = [
                        os.path.join(control_dir, f'inventory_all_{ofs}.csv'),
                        f'inventory_all_{ofs}.csv'
                    ]
                    inv_df_loaded = None
                    for p in paths_to_check:
                        if os.path.exists(p):
                            inv_df_loaded = pd.read_csv(p)
                            if 'ID' in inv_df_loaded.columns:
                                inv_df_loaded['ID'] = inv_df_loaded['ID'].astype(str)
                            break
                    inv_cache[ofs] = inv_df_loaded
                inv_df = inv_cache[ofs]

            if inv_df is not None and 'ID' in inv_df.columns and 'Name' in inv_df.columns:
                match = inv_df[inv_df['ID'].str.contains(station_id.split('_')[0], case=False, na=False)]
                if not match.empty:
                    plotinfo['station_name'] = match.iloc[0]['Name']
            # ----------------------------------

            # --- PCD LOOKUP LOGIC FOR ALONG-CHANNEL ---
            pcd_value = None
            if plotinfo['var_name'] in ('cu', 'currents'):
                # Force 'cu' naming convention for control file lookup just in case 'currents' was passed
                search_var = 'cu'
                ctl_file_path = os.path.join(ctl_folder, f'plot_timeseries_{search_var}_{ofs}.ctl')

                pcd_value = get_pcd_value(ctl_file_path, plotinfo['station_id'], plotinfo['station_name'], logger)

                if pcd_value is not None:
                    logger.info(f'Found PCD value {pcd_value}\u00b0 for station {station_id}. Computing along-channel velocity.')
                    plotinfo['plot_name'] = 'Along-Channel Velocity (<i>knots</i>)'
                    plotinfo['pcd_value'] = pcd_value  # Passed down to the title formatter
                else:
                    logger.warning(f"Missing PCD mapping for '{station_id}' / '{plotinfo['station_name']}'. Reverting to absolute speed.")

            # Setup layout rules (Cu and scalar plots are 1 row)
            if plotinfo['var_name'] in ('cu', 'currents', 'wind'):
                nrows = 1
                sharexaxis = False
            else:
                nrows = 1
                sharexaxis = False

            fig = make_subplots(rows=nrows, cols=1, shared_xaxes=sharexaxis)

            global_starts = []
            global_ends = []
            time_bounds = {}
            obs_legend_added = False

            parsed_data = []

            # ==============================================================
            # PASS 1: Extract, Process, and Collect all Data for this Station
            # ==============================================================
            for file_info in files_list:
                file_path = file_info['path']
                whichcast = file_info['whichcast']
                is_int = file_path.endswith('.int')

                # Format the display name to updated values
                if 'forecast' in whichcast.lower():
                    display_cast = 'Forecast Guidance'
                elif 'nowcast' in whichcast.lower():
                    display_cast = 'Nowcast Guidance'
                else:
                    display_cast = whichcast.capitalize()

                if is_int:
                    try:
                        # Scan file to bypass text headers to reach the 'YEAR' column row
                        skip_rows = 0
                        with open(file_path) as f:
                            for i, line in enumerate(f):
                                if 'year' in line.lower() or 'date' in line.lower():
                                    skip_rows = i
                                    break

                        df = pd.read_csv(file_path, sep=r'\s+', skiprows=skip_rows)
                        df.columns = df.columns.str.strip().str.upper()

                        if not all(col in df.columns for col in ['YEAR', 'MONTH', 'DAY', 'HOUR', 'MINUTE']):
                            logger.warning(f'Missing expected time columns in .int file: {os.path.basename(file_path)}')
                            continue

                        # Ensure numeric types for time
                        for c in ['YEAR', 'MONTH', 'DAY', 'HOUR', 'MINUTE']:
                            df[c] = pd.to_numeric(df[c], errors='coerce')

                        # === DYNAMIC OBSERVATION MAPPING ===
                        for col in ['VAL_OB', 'VAL_OB_SPD', 'OB_SPD', 'SPEED_OB']:
                            if col in df.columns:
                                df['OBS'] = pd.to_numeric(df[col], errors='coerce')
                                break

                        if 'OBS' not in df.columns:
                            u_col = next((c for c in df.columns if c in ['VAL_OB_U', 'OB_U', 'U_OB']), None)
                            v_col = next((c for c in df.columns if c in ['VAL_OB_V', 'OB_V', 'V_OB']), None)
                            if u_col and v_col:
                                u = pd.to_numeric(df[u_col], errors='coerce')
                                v = pd.to_numeric(df[v_col], errors='coerce')
                                df['OBS'] = np.sqrt(u**2 + v**2)
                                df['OBS_DIR'] = np.mod(90 - np.degrees(np.arctan2(v, u)), 360)

                        # === DYNAMIC MODEL MAPPING ===
                        for col in ['VAL_MODEL', 'VAL_MODEL_SPD', 'MOD_SPD', 'VAL_MOD', 'VAL_MOD_SPD', 'MODEL_SPD', 'SPEED_MODEL']:
                            if col in df.columns:
                                df['OFS'] = pd.to_numeric(df[col], errors='coerce')
                                break

                        if 'OFS' not in df.columns:
                            u_col = next((c for c in df.columns if c in ['VAL_MODEL_U', 'MOD_U', 'VAL_MOD_U', 'U_MOD', 'MODEL_U']), None)
                            v_col = next((c for c in df.columns if c in ['VAL_MODEL_V', 'MOD_V', 'VAL_MOD_V', 'V_MOD', 'MODEL_V']), None)
                            if u_col and v_col:
                                u = pd.to_numeric(df[u_col], errors='coerce')
                                v = pd.to_numeric(df[v_col], errors='coerce')
                                df['OFS'] = np.sqrt(u**2 + v**2)
                                df['OFS_DIR'] = np.mod(90 - np.degrees(np.arctan2(v, u)), 360)

                        # Grab Directions if present and not already derived
                        for col in ['VAL_OB_DIR', 'DIR_OB', 'OB_DIR']:
                            if col in df.columns and 'OBS_DIR' not in df.columns:
                                df['OBS_DIR'] = pd.to_numeric(df[col], errors='coerce')
                                break
                        for col in ['VAL_MODEL_DIR', 'DIR_MODEL', 'MODEL_DIR']:
                            if col in df.columns and 'OFS_DIR' not in df.columns:
                                df['OFS_DIR'] = pd.to_numeric(df[col], errors='coerce')
                                break

                        df = df.dropna(subset=['YEAR', 'MONTH', 'DAY'])
                        df['DateTime'] = pd.to_datetime(
                            dict(year=df['YEAR'], month=df['MONTH'], day=df['DAY'], hour=df['HOUR'], minute=df['MINUTE'])
                        )

                        if 'OFS' not in df.columns and 'OBS' not in df.columns:
                            logger.warning(f'Could not extract Observation or Model runs from {os.path.basename(file_path)}. Found columns: {list(df.columns)}')
                            continue

                    except Exception as e:
                        logger.warning(f'Error parsing .int file {os.path.basename(file_path)}: {e}')
                        continue
                else:
                    # Parse standard .prd file
                    try:
                        df = pd.read_csv(file_path, sep=r'\s+', header=None)
                    except pd.errors.EmptyDataError:
                        logger.warning(f'File {os.path.basename(file_path)} is empty! Skipping trace...')
                        continue

                    df['DateTime'] = pd.to_datetime(
                        dict(year=df[1], month=df[2], day=df[3], hour=df[4], minute=df[5])
                    )

                    # PRD files for Currents output U and V natively. Calculate speed!
                    if plotinfo['var_name'] in ('cu', 'currents'):
                        u = pd.to_numeric(df[6], errors='coerce')
                        v = pd.to_numeric(df[7], errors='coerce')
                        df['OFS'] = np.sqrt(u**2 + v**2)
                        df['OFS_DIR'] = np.mod(90 - np.degrees(np.arctan2(v, u)), 360)
                    else:
                        df = df.rename(columns={6: 'OFS'})
                        if plotinfo['var_name'] == 'wind':
                            df = df.rename(columns={7: 'OFS_DIR'})

                # Mask out negative NOAA missing value flags (-99.9, -9999.0) so they don't break the axes scale
                if 'OBS' in df.columns:
                    df.loc[df['OBS'] < -90, 'OBS'] = np.nan
                if 'OFS' in df.columns:
                    df.loc[df['OFS'] < -90, 'OFS'] = np.nan

                # --- Apply Unit Conversions & Along-Channel Projections ---
                if plotinfo['var_name'] in ('cu', 'currents'):
                    if 'OBS' in df.columns:
                        df['OBS'] = pd.to_numeric(df['OBS'], errors='coerce') * 1.943844
                    if 'OFS' in df.columns:
                        df['OFS'] = pd.to_numeric(df['OFS'], errors='coerce') * 1.943844

                    # Convert to along-channel direction if PCD exists
                    if pcd_value is not None:
                        if 'OBS' in df.columns and 'OBS_DIR' in df.columns:
                            df['OBS'] = df['OBS'] * np.cos(np.radians(df['OBS_DIR'] - pcd_value))
                        if 'OFS' in df.columns and 'OFS_DIR' in df.columns:
                            df['OFS'] = df['OFS'] * np.cos(np.radians(df['OFS_DIR'] - pcd_value))

                elif plotinfo['var_name'] == 'wl':
                    if 'OBS' in df.columns:
                        df['OBS'] = pd.to_numeric(df['OBS'], errors='coerce') * 3.28084
                    if 'OFS' in df.columns:
                        df['OFS'] = pd.to_numeric(df['OFS'], errors='coerce') * 3.28084

                # Ensure dataframe has data remaining
                if df.empty:
                    continue

                start_time = df['DateTime'].iloc[0]
                end_time = df['DateTime'].iloc[-1]

                global_starts.append(start_time)
                global_ends.append(end_time)
                time_bounds[whichcast] = {'start': start_time, 'end': end_time}

                # Store the processed dataframe for Pass 2 plotting
                parsed_data.append({
                    'df': df,
                    'whichcast': whichcast,
                    'display_cast': display_cast
                })

            if not global_starts:
                continue # Skip if all files were empty

            plotinfo['start'] = min(global_starts)
            plotinfo['end'] = max(global_ends)

            # ==============================================================
            # PASS 2: Plot the Data
            # ==============================================================
            for data in parsed_data:
                df = data['df']
                whichcast = data['whichcast']
                display_cast = data['display_cast']

                # === 1. Plot Observations (if they exist) ===
                if 'OBS' in df.columns:
                    if plotinfo['var_name'] in ('cu', 'currents') and pcd_value is not None:
                        hovertemplate_obs = f'Obs Velocity: %{{y:.2f}}{plotinfo["unit"]}<extra></extra>'
                    else:
                        hovertemplate_obs = f'Observations: %{{y:.2f}}{plotinfo["unit"]}<extra></extra>'

                    fig.add_trace(
                        go.Scatter(
                            x=df['DateTime'],
                            y=df['OBS'],
                            name='Observations',
                            legendgroup='Observations',
                            showlegend=not obs_legend_added,
                            hovertemplate=hovertemplate_obs,
                            mode='lines',
                            opacity=1,
                            line=dict(color='red', width=2),
                        ), 1, 1,
                    )

                    if nrows == 2 and 'OBS_DIR' in df.columns:
                        fig.add_trace(
                            go.Scatter(
                                x=df['DateTime'],
                                y=df['OBS_DIR'],
                                name='Observations',
                                legendgroup='Observations',
                                showlegend=False,
                                hovertemplate='Obs Direction: %{y:.1f}\u00b0<extra></extra>',
                                mode='lines',
                                opacity=1,
                                line=dict(color='red', width=2),
                            ), 2, 1,
                        )
                    obs_legend_added = True

                # === 2. Plot Model Runs ===
                if 'OFS' in df.columns:
                    customdata = None
                    hovertemplate_model = f'{display_cast}: %{{y:.2f}}{plotinfo["unit"]}<extra></extra>'

                    if plotinfo['var_name'] == 'wind' and 'OFS_DIR' in df.columns:
                        customdata = df['OFS_DIR']
                        hovertemplate_model = f'{display_cast} Speed: %{{y:.2f}} m/s<br>{display_cast} Direction: %{{customdata:.1f}}\u00b0<extra></extra>'
                    elif plotinfo['var_name'] in ('cu', 'currents'):
                        if pcd_value is not None:
                            hovertemplate_model = f'{display_cast} Velocity: %{{y:.2f}} knots<br><extra></extra>'
                        else:
                            hovertemplate_model = f'{display_cast} Speed: %{{y:.2f}} knots<br><extra></extra>'

                    c1, d1 = get_trace_styling(whichcast, 'primary')

                    # Force solid lines purely for the wind charts
                    if plotinfo['var_name'] == 'wind':
                        d1 = 'solid'

                    # Primary Line (Speed/WL/Temp/Salt)
                    fig.add_trace(
                        go.Scatter(
                            x=df['DateTime'],
                            y=df['OFS'],
                            name=display_cast,
                            legendgroup=display_cast,
                            showlegend=True,
                            customdata=customdata,
                            hovertemplate=hovertemplate_model,
                            mode='lines',
                            opacity=1,
                            connectgaps=False,
                            line=dict(color=c1, width=1.5, dash=d1),
                        ), 1, 1,
                    )

                    # Secondary Line (Currents Dir)
                    if nrows == 2 and 'OFS_DIR' in df.columns:
                        c2, d2 = get_trace_styling(whichcast, 'secondary')
                        fig.add_trace(
                            go.Scatter(
                                x=df['DateTime'],
                                y=df['OFS_DIR'],
                                name=display_cast,
                                legendgroup=display_cast,
                                showlegend=False,
                                hovertemplate=f'{display_cast} Direction: %{{y:.1f}}\u00b0<extra></extra>',
                                mode='lines',
                                opacity=1,
                                connectgaps=False,
                                line=dict(color=c2, width=1.5, dash=d2),
                            ), 2, 1,
                        )

                    # === Wind Arrows (True Quiver Layout Annotations) ===
                    if plotinfo['var_name'] == 'wind' and 'OFS_DIR' in df.columns:

                        # Filter to only grab data points strictly at the top of the hour (00 minutes)
                        subset = df[df['DateTime'].dt.minute == 0].dropna(subset=['OFS', 'OFS_DIR'])

                        # Adds a physical dot marker exactly where the arrow sprouts from the line
                        fig.add_trace(
                            go.Scatter(
                                x=subset['DateTime'],
                                y=subset['OFS'],
                                mode='markers',
                                showlegend=False,
                                hoverinfo='skip',  # Keeps the hover-box clean
                                marker=dict(size=6, color=c1, line=dict(width=1, color='white'))
                            ), 1, 1
                        )

                        for _, row in subset.iterrows():
                            angle_rad = np.radians(row['OFS_DIR'])
                            length = 18  # Reduced length for shorter arrows

                            fig.add_annotation(
                                x=row['DateTime'],
                                y=row['OFS'],
                                ax=length * np.sin(angle_rad),
                                ay=-length * np.cos(angle_rad),
                                xref='x', yref='y',
                                axref='pixel', ayref='pixel',
                                showarrow=True,
                                arrowhead=0,
                                startarrowhead=2,
                                startarrowsize=1.5,
                                arrowwidth=1.0,
                                arrowcolor='blue'
                            )

            # ==============================================================
            # PASS 3: Tides (Water Level Only)
            # ==============================================================
            # Add tidal predictions for water level plots excluding Great Lakes (ofs starting with 'l')
            if plotinfo['var_name'] == 'wl' and plotinfo['ofs'][0].lower() != 'l' and datum:
                # Create a minimal Mock property object so get_station_tidal_data can parse it standalone
                class MockProp:
                    def __init__(self, ofs_name):
                        self.ofs = ofs_name
                        self.datum = datum
                        self.time_zone = 'GMT'
                        self.control_files_path = ctl_folder

                try:
                    start_dt = plotinfo['start'].to_pydatetime()
                    end_dt = plotinfo['end'].to_pydatetime()
                    tidal_data, tidal_info = get_station_tidal_data(
                        start_dt, end_dt, MockProp(plotinfo['ofs']), plotinfo['station_id'], logger
                    )

                    if tidal_data is not None and not tidal_data.empty:
                        tidal_data['TIDE'] = pd.to_numeric(tidal_data['TIDE'], errors='coerce') * 3.28084
                        if tidal_info.get('tidal_station_name'):
                            source_text = f'CO-OPS Station {tidal_info["tidal_station_id"]} ({tidal_info["tidal_station_name"]})'
                        else:
                            source_text = f'CO-OPS Station {tidal_info["tidal_station_id"]}'

                        distance = tidal_info.get('tidal_station_distance')
                        distance_text = f'<br>Distance: {distance:.1f} km' if distance and distance > 0 else ''

                        used_datum = tidal_info.get('used_datum', 'MSL')
                        datum_text = f'<br>Datum: {used_datum}'

                        hover_text = (
                            f'Tidal Prediction: %{{y:.2f}}{plotinfo["unit"]}'
                            f'<br><i>Source: {source_text}{distance_text}{datum_text}</i><extra></extra>'
                        )

                        fig.add_trace(
                            go.Scatter(
                                x=tidal_data['DateTime'],
                                y=tidal_data['TIDE'],
                                name='Tidal Predictions',
                                hovertemplate=hover_text,
                                mode='lines',
                                opacity=0.7,
                                line=dict(color='purple', width=1.5, dash='dot'),
                                legendgroup='tide',
                            ), 1, 1
                        )
                        logger.info(f"Tidal predictions successfully added to plot for station {plotinfo['station_id']}.")
                except Exception as ex:
                    logger.warning(f"Could not retrieve tidal predictions for station {plotinfo['station_id']}: {ex}")

            # --- Layout Configuration ---
            # Set dynamic margins and annotation positions based on plot type
            is_currents = plotinfo['var_name'] in ('cu', 'currents')
            top_margin = 130 if is_currents else 100
            title_y = 0.98 if is_currents else 0.97
            annotation_y = -0.42 if is_currents else -0.36
            bottom_margin = 180 if is_currents else 160

            # Check if nowcast and forecast_a are sequential and not overlapping
            if 'nowcast' in time_bounds and 'forecast_a' in time_bounds:
                nc_end = time_bounds['nowcast']['end']
                fc_start = time_bounds['forecast_a']['start']

                if nc_end <= fc_start:
                    # Vertical separator line
                    fig.add_shape(
                        type='line',
                        x0=fc_start, y0=0,
                        x1=fc_start, y1=1,
                        yref='paper',
                        line=dict(color='black', width=2, dash='solid'),
                        opacity=1.0
                    )
                    fig.add_annotation(
                        x=fc_start, y=1,
                        yref='paper',
                        text='<i>Forecast</i> ➡️',
                        showarrow=False,
                        xanchor='left',
                        yanchor='top',
                        xshift=8,                               # Spacer to the right
                        font=dict(color='grey', size=14)        # Increased size, set to grey
                    )
                    fig.add_annotation(
                        x=fc_start, y=1,
                        yref='paper',
                        text='⬅️ <i>Nowcast</i>',
                        showarrow=False,
                        xanchor='right',
                        yanchor='top',
                        xshift=-8,                              # Spacer to the left
                        font=dict(color='grey', size=14)        # Increased size, set to grey
                    )

            # Range Slider Instructions (Below Slider)
            fig.add_annotation(
                text='<i>Click and drag the edges of the slider to adjust the time range.</i>',
                xref='paper', yref='paper',
                x=0.5, y=annotation_y,  # Dynamically positioned based on variable
                yanchor='top',
                showarrow=False,
                font=dict(family='Open Sans', color='#555555', size=13)
            )

            fig.update_layout(
                title=dict(
                    text=get_plot_title(plotinfo),
                    font=dict(size=14, color='black', family='Open Sans'),
                    y=title_y, x=0.5, xanchor='center', yanchor='top',
                ),
                transition_ordering='traces first', dragmode='zoom',
                height=550, width=900,
                template='plotly_white',
                # Dynamically set top/bottom margin
                margin=dict(t=top_margin, b=bottom_margin),
                legend=dict(
                    orientation='h',
                    yanchor='bottom',
                    y=1.02,
                    xanchor='left',
                    x=0,
                    font=dict(size=16, color='black'),
                    itemclick=False,         # Disable toggling via click
                    itemdoubleclick=False    # Disable isolation via double-click
                ),

                # Clean, High-Fidelity Professional Hover Styling
                hovermode='x unified',
                hoverlabel=dict(
                    bgcolor='white',
                    bordercolor='#cccccc',
                    font=dict(family='Open Sans', size=13, color='#333333'),
                    namelength=-1
                )
            )

            # Axis Configuration
            fig.update_xaxes(
                title_text='<br>Time (UTC)',
                titlefont=dict(family='Open Sans', color='black', size=18),    # Increased to 17
                mirror=True, ticks='inside', showline=True, linecolor='black', linewidth=1,
                showspikes=True, spikemode='across', spikesnap='cursor', showgrid=True,
                tickfont=dict(family='Open Sans', color='black', size=14),
                minor=dict(ticklen=4, tickcolor='black', ticks='inside', showgrid=False),
                #dtick=43200000,                  # 12 hours in milliseconds
                tickformat='%H:%M<br>%m/%d',     # Formats as HH:MM [line break] MM/DD
                tickangle=0,

                # Strips raw timestamp strings and formats nicely at the top of the unified hover box
                hoverformat='%b %d, %Y, %H:%M UTC',

                # Added Range Slider
                rangeslider=dict(
                    visible=True,
                    thickness=0.08,
                    bordercolor='black',
                    borderwidth=1
                )
            )

            if nrows == 2:
                fig.update_yaxes(
                    mirror=True, ticks='inside', showline=True, linecolor='black', linewidth=1,
                    title_text=plotinfo['plot_name'][1],
                    titlefont=dict(family='Open Sans', color='black', size=18),  # Increased to 17
                    tickfont=dict(family='Open Sans', color='black', size=14),
                    minor=dict(ticklen=4, tickcolor='black', ticks='inside', showgrid=False),
                    zeroline=(plotinfo['var_name'] == 'wl'),
                    zerolinewidth=1,
                    zerolinecolor='black',
                    #rangemode='tozero' if plotinfo['var_name'] != 'wl' else 'normal',
                    row=2, col=1,
                )
                plotinfo['plot_name'] = plotinfo['plot_name'][0]

            fig.update_yaxes(
                mirror=True, ticks='inside', showline=True, linecolor='black', linewidth=1,
                title_text=f'{plotinfo["plot_name"]}',
                titlefont=dict(family='Open Sans', color='black', size=17),  # Increased to 17
                tickfont=dict(family='Open Sans', color='black', size=14),
                minor=dict(ticklen=4, tickcolor='black', ticks='inside', showgrid=False),
                zeroline=(plotinfo['var_name'] == 'wl'),
                zerolinewidth=1,
                zerolinecolor='black',
                #rangemode='tozero' if plotinfo['var_name'] != 'wl' else 'normal',
                row=1, col=1,
            )

            # Save combined file
            out_name = f'{plotinfo["ofs"]}_{plotinfo["station_id"]}_{plotinfo["save_name"]}_combined_modelseries.html'
            out_file = os.path.join(save_path, out_name)
            fig.write_html(out_file)

            logger.info('Completed combined %s plot for %s!', plotinfo['var_name'], plotinfo['station_id'])

        except Exception as ex:
            logger.error('Caught exception plotting combined group for %s: %s', station_id, ex)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='python plot_model_timeseries.py',
        usage='%(prog)s',
        description='Plot model time series from .prd or .int files',
    )
    parser.add_argument(
        '-c',
        '--config',
        help='Path to configuration file (default: conf/ofs_dps.conf)'
    )
    parser.add_argument(
        '-i',
        '--inventory',
        help='Path to station inventory CSV file (e.g., inventory_all_cbofs.csv)',
        default=None
    )
    parser.add_argument(
        '-vs',
        '--variable',
        help='Specify a single variable to plot (e.g., wl, temp, salt, cu, wind). Defaults to all.',
        default=None
    )
    parser.add_argument(
        '-o',
        '--ofs',
        help='Specify a single OFS to plot (e.g., cbofs, dbofs). Defaults to all.',
        default=None
    )
    parser.add_argument(
        '-d', '--datum',
        required=False,
        default=None,
        help="datum options: 'MHW', 'MHHW' \
        'MLW', 'MLLW', 'NAVD88', 'XGEOID20B', 'IGLD85', 'LWD'")

    args = parser.parse_args()

    main(None, _conf=args.config, inventory_file=args.inventory, variable=args.variable, ofs_filter=args.ofs,
         datum=args.datum)
