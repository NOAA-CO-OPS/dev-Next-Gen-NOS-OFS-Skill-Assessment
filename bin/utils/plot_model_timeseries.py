"""
Standalone script to plot model time series from .prd files using Plotly.
Combines nowcast and forecast runs onto a single plot per station,
with a vertical separator if the runs are sequential, and includes
observation data from .int files (preferring .int over .prd when both exist).
"""

import argparse
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


def get_plot_title(plotinfo):
    """Formats the plot title using the metadata extracted from the filename."""
    start_str = datetime.strftime(plotinfo['start'], '%Y/%m/%d %H:%M:%S')
    end_str = datetime.strftime(plotinfo['end'], '%Y/%m/%d %H:%M:%S')

    return (f'<b>NOAA/National Ocean Service, {plotinfo["ofs"].upper()} <br>'
            f'Station Name:&nbsp;{plotinfo["station_name"]} &nbsp;&nbsp;&nbsp;'
            f'Station ID:&nbsp;{plotinfo["station_id"]}'
            f'<br>From:&nbsp;{start_str}'
            f'&nbsp;&nbsp;&nbsp;To:&nbsp;{end_str}<b>')


def get_variable_names(name_var):
    """Maps short variable names to full plot titles and save names."""
    if name_var == 'wl':
        plot_name = 'Water Level (<i>meters</i>)'
        save_name = 'water_level'
        unit = ' m'
    elif name_var in ('temp', 'water_temperature'):
        plot_name = 'Water Temperature (<i>\u00b0C</i>)'
        save_name = 'water_temperature'
        unit = ' \u00b0C'
    elif name_var in ('salt', 'salinity'):
        plot_name = 'Salinity (<i>PSU</i>)'
        save_name = 'salinity'
        unit = ' PSU'
    elif name_var in ('cu', 'currents'):
        plot_name = ['Current speed<br>(<i>meters/second</i>)',
                     'Current direction<br>(<i>0-360 deg.</i>)']
        save_name = 'currents'
        unit = ' m/s'
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


def main(logger, _conf=None, inventory_file=None, variable=None, ofs_filter=None):
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
    int_folder = os.path.join(
        home_path, dir_params['data_dir'], dir_params['skill_dir'],
        dir_params['1d_pair_dir']
    )
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

        # Parse based on file type
        if file.endswith('.prd'):
            station_id, ofs, var_name, node = parts[0], parts[1], parts[2], parts[3]
            whichcast = parts[4]
            if whichcast == 'forecast' and len(parts) > 5 and parts[5] in ['a', 'b']:
                whichcast = f'forecast_{parts[5]}'

        elif file.endswith('.int'):
            # .int format: ofs_var_stationID_node_whichcast...
            ofs, var_name, station_id, node = parts[0], parts[1], parts[2], parts[3]
            whichcast = parts[4]
            if whichcast == 'forecast' and len(parts) > 5 and parts[5] in ['a', 'b']:
                whichcast = f'forecast_{parts[5]}'
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

            plotinfo['plot_name'], plotinfo['save_name'], plotinfo['unit'] = get_variable_names(plotinfo['var_name'])

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
                match = inv_df[inv_df['ID'] == station_id]
                if not match.empty:
                    plotinfo['station_name'] = match.iloc[0]['Name']
            # ----------------------------------

            # Setup layout rules based on variable
            if plotinfo['var_name'] in ('cu', 'currents'):
                nrows = 2
                sharexaxis = True
            elif plotinfo['var_name'] == 'wind':
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

            # Plot each file in this group
            for file_info in files_list:
                file_path = file_info['path']
                whichcast = file_info['whichcast']
                is_int = file_path.endswith('.int')

                # Format the display name (e.g. "Forecast" instead of "forecast_a")
                if 'forecast' in whichcast.lower():
                    display_cast = 'Forecast'
                elif 'nowcast' in whichcast.lower():
                    display_cast = 'Nowcast'
                else:
                    display_cast = whichcast.capitalize()

                if is_int:
                    try:
                        df = pd.read_csv(file_path, sep=r'\s+', header=0)
                        df.columns = df.columns.str.strip()

                        # Fallback if header=0 missed it
                        if 'YEAR' not in df.columns:
                            df = pd.read_csv(file_path, sep=r'\s+', header=None)
                            if df.iloc[0].astype(str).str.contains('YEAR').any():
                                df.columns = df.iloc[0].astype(str).str.strip()
                                df = df[1:]

                        if not all(col in df.columns for col in ['YEAR', 'MONTH', 'DAY', 'HOUR', 'MINUTE']):
                            logger.warning(f'Missing expected time columns in .int file: {os.path.basename(file_path)}')
                            continue

                        # Ensure numeric types
                        for c in ['YEAR', 'MONTH', 'DAY', 'HOUR', 'MINUTE']:
                            df[c] = pd.to_numeric(df[c], errors='coerce')

                        # Map Observation columns
                        if 'VAL_OB' in df.columns:
                            df['OBS'] = pd.to_numeric(df['VAL_OB'], errors='coerce')
                        if 'VAL_OB_DIR' in df.columns:
                            df['OBS_DIR'] = pd.to_numeric(df['VAL_OB_DIR'], errors='coerce')

                        # Map Model columns
                        if 'VAL_MODEL' in df.columns:
                            df['OFS'] = pd.to_numeric(df['VAL_MODEL'], errors='coerce')
                        if 'VAL_MODEL_DIR' in df.columns:
                            df['OFS_DIR'] = pd.to_numeric(df['VAL_MODEL_DIR'], errors='coerce')

                        df = df.dropna(subset=['YEAR', 'MONTH', 'DAY'])
                        df['DateTime'] = pd.to_datetime(
                            dict(year=df['YEAR'], month=df['MONTH'], day=df['DAY'], hour=df['HOUR'], minute=df['MINUTE'])
                        )
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
                    df = df.rename(columns={6: 'OFS'})
                    if plotinfo['var_name'] in ('cu', 'currents', 'wind'):
                        df = df.rename(columns={7: 'OFS_DIR'})

                start_time = df['DateTime'].iloc[0]
                end_time = df['DateTime'].iloc[-1]

                global_starts.append(start_time)
                global_ends.append(end_time)
                time_bounds[whichcast] = {'start': start_time, 'end': end_time}

                # === 1. Plot Observations (if they exist) ===
                if 'OBS' in df.columns:
                    hovertemplate_obs = f'Observation: %{{y:.2f}}{plotinfo["unit"]}<extra></extra>'
                    fig.add_trace(
                        go.Scatter(  # Switched to regular Scatter to support range slider
                            x=df['DateTime'],
                            y=df['OBS'],
                            name='Observation',
                            legendgroup='Observation',
                            showlegend=not obs_legend_added,
                            hovertemplate=hovertemplate_obs,
                            mode='lines',
                            opacity=1,
                            line=dict(color='red', width=2),
                        ), 1, 1,
                    )

                    if nrows == 2 and 'OBS_DIR' in df.columns:
                        fig.add_trace(
                            go.Scatter(  # Switched to regular Scatter to support range slider
                                x=df['DateTime'],
                                y=df['OBS_DIR'],
                                name='Observation',
                                legendgroup='Observation',
                                showlegend=False,
                                hovertemplate='Obs Direction: %{y:.1f}\u00b0<extra></extra>',
                                mode='lines',
                                opacity=1,
                                line=dict(color='red', width=2),
                            ), 2, 1,
                        )
                    obs_legend_added = True

                # === 2. Plot Model Runs (from either PRD or INT) ===
                if 'OFS' in df.columns:
                    customdata = None
                    hovertemplate_model = f'{display_cast}: %{{y:.2f}}{plotinfo["unit"]}<extra></extra>'

                    if plotinfo['var_name'] == 'wind' and 'OFS_DIR' in df.columns:
                        customdata = df['OFS_DIR']
                        hovertemplate_model = f'{display_cast} Speed: %{{y:.2f}} m/s<br>{display_cast} Direction: %{{customdata:.1f}}\u00b0<extra></extra>'
                    elif plotinfo['var_name'] in ('cu', 'currents') and 'OFS_DIR' in df.columns:
                        hovertemplate_model = f'{display_cast} Speed: %{{y:.2f}} m/s<br><extra></extra>'

                    c1, d1 = get_trace_styling(whichcast, 'primary')

                    # Primary Line (Speed/WL/Temp/Salt)
                    fig.add_trace(
                        go.Scatter(  # Switched to regular Scatter to support range slider
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
                            go.Scatter(  # Switched to regular Scatter to support range slider
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
                        arrow_step = max(1, len(df) // 40)
                        marker_color = 'green' if 'forecast' in whichcast else 'black'

                        subset = df.iloc[::arrow_step]
                        for _, row in subset.iterrows():
                            angle_rad = np.radians(row['OFS_DIR'])
                            length = 18  # Length of the arrow shaft in pixels

                            fig.add_annotation(
                                x=row['DateTime'],
                                y=row['OFS'],
                                ax=-length * np.sin(angle_rad),
                                ay=length * np.cos(angle_rad),
                                xref='x', yref='y',
                                axref='pixel', ayref='pixel',
                                showarrow=True,
                                arrowhead=2,
                                arrowsize=1.4,
                                arrowwidth=1.5,
                                arrowcolor=marker_color
                            )

            # --- Layout Configuration ---
            if not global_starts:
                continue # Skip if all files were empty

            plotinfo['start'] = min(global_starts)
            plotinfo['end'] = max(global_ends)

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
                        text='Forecast ➡️',
                        showarrow=False,
                        xanchor='left',
                        yanchor='top',
                        xshift=8,                               # Spacer to the right
                        font=dict(color='black', size=14)       # Increased size
                    )
                    fig.add_annotation(
                        x=fc_start, y=1,
                        yref='paper',
                        text='⬅️ Nowcast',
                        showarrow=False,
                        xanchor='right',
                        yanchor='top',
                        xshift=-8,                              # Spacer to the left
                        font=dict(color='black', size=14)       # Increased size
                    )

            # Range Slider Instructions (Below Slider)
            fig.add_annotation(
                text='<i>Click and drag the edges of the slider to adjust the time range.</i>',
                xref='paper', yref='paper',
                x=0.5, y=-0.36,  # Placed perfectly below the slider and title
                yanchor='top',
                showarrow=False,
                font=dict(family='Open Sans', color='#555555', size=13)
            )

            fig.update_layout(
                title=dict(
                    text=get_plot_title(plotinfo),
                    font=dict(size=14, color='black', family='Open Sans'),
                    y=0.97, x=0.5, xanchor='center', yanchor='top',
                ),
                transition_ordering='traces first', dragmode='zoom',
                height=550, width=900,
                template='plotly_white',
                margin=dict(t=100, b=160),  # Increased bottom margin heavily to fit both title and annotation
                legend=dict(
                    orientation='h',
                    yanchor='bottom',
                    y=1.02,
                    xanchor='left',
                    x=0,
                    font=dict(size=16, color='black')  # Increased legend font to 16
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
                title_text='Time (UTC)',                                       # Reverted back to traditional X-Axis title
                titlefont=dict(family='Open Sans', color='black', size=17),    # Size 16 font
                mirror=True, ticks='inside', showline=True, linecolor='black', linewidth=1,
                showspikes=True, spikemode='across', spikesnap='cursor', showgrid=True,
                tickfont=dict(family='Open Sans', color='black', size=14),
                dtick=43200000,                  # 12 hours in milliseconds
                tickformat='%H:%M<br>%m/%d',     # Formats as HH:MM [line break] MM/DD

                # Strips raw timestamp strings and formats nicely at the top of the unified hover box
                hoverformat='%b %d, %Y, %H:%M UTC',

                # Added Range Slider
                rangeslider=dict(
                    visible=True,
                    thickness=0.08,  # Keeps it slim and professional
                    bordercolor='black',
                    borderwidth=1
                )
            )

            if nrows == 2:
                fig.update_yaxes(
                    mirror=True, ticks='inside', showline=True, linecolor='black', linewidth=1,
                    title_text=plotinfo['plot_name'][1],
                    titlefont=dict(family='Open Sans', color='black', size=17),  # Increased to 16
                    tickfont=dict(family='Open Sans', color='black', size=14),
                    row=2, col=1,
                )
                plotinfo['plot_name'] = plotinfo['plot_name'][0]

            fig.update_yaxes(
                mirror=True, ticks='inside', showline=True, linecolor='black', linewidth=1,
                title_text=f'{plotinfo["plot_name"]}',
                titlefont=dict(family='Open Sans', color='black', size=16),  # Increased to 16
                tickfont=dict(family='Open Sans', color='black', size=14),
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
        description='Plot model time series from .prd files',
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
        '-v',
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

    args = parser.parse_args()

    main(None, _conf=args.config, inventory_file=args.inventory, variable=args.variable, ofs_filter=args.ofs)
