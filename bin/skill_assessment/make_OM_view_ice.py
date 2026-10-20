"""
Created on Thu Sep 18 08:50:00 2025

This script aggregates skill assessment output from all OFS, and makes overview
plots and tables. More specifically,
    1) .int files for all OFS and all variables are collected to make overview
    scorecard plots;
    2) CSV skill tables for all OFS and all variables are combined to one
    master plotly table & summary central frequency plot;
    3) plotly html plots for all OFS and all variables are converted to static
    .png images, and collected into a PDF.


@author: PWL
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import os
import sys
from datetime import datetime

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from mpl_toolkits.basemap import Basemap
from PIL import Image
from plotly.subplots import make_subplots
from scipy.signal import savgol_filter

from ofs_skill.model_processing import model_properties
from ofs_skill.obs_retrieval import utils


def parameter_validation(argu_list, logger):
    """ Parameter validation """

    path, whichcast, ofs_extents_path = (
        str(argu_list[0]),
        str(argu_list[1]),
        str(argu_list[2]),
    )

    # path validation
    if not os.path.exists(ofs_extents_path):
        error_message = f"""ofs_extents/ folder is not found. Please
        check path - {path}. Abort!"""
        logger.error(error_message)
        sys.exit(-1)

    # whichcast validation
    if (
        'nowcast' not in whichcast and
        'forecast_b' not in whichcast and
        'forecast_a' not in whichcast and
        'all' not in whichcast
    ):
        error_message = f'Incorrect whichcast: {whichcast}! Exiting.'
        logger.error(error_message)
        sys.exit(-1)

def export_kpi_json(df_ofs, prop, cast, logger):
    """
    Calculates area-weighted system-wide KPI means and exports to a JSON file.
    """
    logger.info(f'Exporting system KPIs for {cast}...')

    # Approximate surface area weights for each GLOFS domain
    area_weights = {
        'lmhofs': 0.48,
        'lsofs': 0.34,
        'leofs': 0.10,
        'loofs': 0.08
    }

    # Map the weights to the dataframe based on the OFS column
    df_ofs['weight'] = df_ofs['OFS'].str.lower().map(area_weights)

    # Helper to calculate weighted average and handle NaNs
    def get_weighted_avg(df, col, weight_col='weight'):
        clean_df = df.dropna(subset=[col, weight_col])
        if not clean_df.empty:
            return np.average(clean_df[col], weights=clean_df[weight_col])
        return 0

    # Calculate all system-wide weighted averages
    system_csi = get_weighted_avg(df_ofs, 'csi_all')
    system_rmse = get_weighted_avg(df_ofs, 'rmse_all')
    system_obs = get_weighted_avg(df_ofs, 'obs_meanicecover')
    system_mod = get_weighted_avg(df_ofs, 'mod_meanicecover')
    system_clim = get_weighted_avg(df_ofs, 'icecover_hist')

    # Format the data into a dictionary
    kpi_data = {
        'csi': f'{system_csi:.2f}',
        'rmse': f'{system_rmse:.1f}%',
        'obs_ice': f'{system_obs:.1f}%',
        'mod_ice': f'{system_mod:.1f}%',
        'clim_ice': f'{system_clim:.1f}%'
    }

    # Save to JSON in the visuals folder
    output_file = os.path.join(prop.om_files, f'kpis_{cast}.json')
    with open(output_file, 'w') as f:
        json.dump(kpi_data, f)

    logger.info(f'Wrote system KPIs to {output_file}')

def make_overview_map(df1, df2, prop, cast, logger):
    '''
    Creates continuous geographic Matplotlib maps of the Great Lakes showing
    Observed, Modeled, RMSE, CSI Hits, and Mean Error using pcolor by pulling from the exported JSON map files.
    '''
    map_configs = [
        {'var': 'obs', 'suffix': 'mean_iceconc_obs.json', 'title': 'Mean NIC Ice Conc. (%)', 'cmap': 'Blues_r', 'vmin': 0, 'vmax': 100},
        {'var': 'mod', 'suffix': 'mean_iceconc_mod.json', 'title': 'Mean OFS Ice Conc. (%)', 'cmap': 'Purples_r', 'vmin': 0, 'vmax': 100},
        {'var': 'clim', 'suffix': 'mean_iceconc_clim.json', 'title': 'Mean Climatology (%)', 'cmap': 'Greens_r', 'vmin': 0, 'vmax': 100},
        {'var': 'rmse', 'suffix': 'rmse.json', 'title': 'RMSE (%)', 'cmap': 'plasma', 'vmin': 0, 'vmax': 100},
        {'var': 'csi', 'suffix': 'csi_hit.json', 'title': 'CSI Hit Rate (%)', 'cmap': 'brg', 'vmin': 0, 'vmax': 100},
        {'var': 'diff', 'suffix': 'mean_error.json', 'title': 'Mean Error (%)', 'cmap': 'Spectral', 'vmin': -100, 'vmax': 100}
    ]

    ll_lon, ll_lat = -92.5, 41.0
    ur_lon, ur_lat = -75.5, 49.0

    cities = {
        'Chicago': (-87.6298, 41.8781),
        'Detroit': (-83.0458, 42.3314),
        'Toronto': (-79.3470, 43.6511),
        'Cleveland': (-81.6944, 41.4993),
        'Buffalo': (-78.8784, 42.8864),
        'Duluth': (-92.1005, 46.7867),
        'Sault Ste. Marie': (-84.3451, 46.4953),
        'Milwaukee': (-87.9065, 43.0389),
        'Green Bay': (-88.0198, 44.5133),
        'Rochester': (-77.6109, 43.1566),
        'Erie': (-80.0851, 42.1292),
        'Marquette': (-87.3950, 46.5436),
        'Toledo': (-83.5552, 41.6528),
    }

    for config in map_configs:
        logger.info(f"Plotting {config['title']} continuous map for all lakes...")

        fig, ax = plt.subplots(figsize=(10, 8))

        # Use a new variable for the title instead of overwriting 'cast'
        title_cast = cast.strip('_b').capitalize() if config['var'] != 'obs' else ''
        if getattr(prop, 'start_date_scorecard', '') == getattr(prop, 'end_date_scorecard', ''):
            title_date = f"{getattr(prop, 'start_date_scorecard', '')}"
        else:
            title_date = f"{getattr(prop, 'start_date_scorecard', '')} - {getattr(prop, 'end_date_scorecard', '')}"
        # Build the title string safely
        ax.set_title(f'{title_cast}\n{title_date}',
                     fontsize=16, pad=15)

        # High resolution to draw detailed peninsulas and islands
        m = Basemap(projection='merc', resolution='h', area_thresh=1000,
                    llcrnrlon=ll_lon, llcrnrlat=ll_lat,
                    urcrnrlon=ur_lon, urcrnrlat=ur_lat, ax=ax)

        # Base land and ocean
        m.drawmapboundary(fill_color='lightblue')
        m.fillcontinents(color='lightgrey', lake_color='lightblue', zorder=1)

        data_plotted = False
        pcm = None

        for ofs in df1['ofs']:
            json_pattern = os.path.join(prop.visuals_json_maps_ice_path, f"{ofs}_{cast}_*--*_map_{config['suffix']}")
            matching_files = glob.glob(json_pattern)

            if matching_files:
                json_file = matching_files[0]
                try:
                    with open(json_file) as f:
                        data = json.load(f)

                    keys = list(data.keys())
                    lat_key = next((k for k in keys if k.lower() in ['lat', 'lats', 'latitude', 'latitudes']), None)
                    lon_key = next((k for k in keys if k.lower() in ['lon', 'lons', 'longitude', 'longitudes']), None)
                    val_keys = [k for k in keys if k not in [lat_key, lon_key]]

                    if not lat_key or not lon_key or not val_keys:
                        logger.warning(f'Could not parse keys in {json_file}. Skipping {ofs.upper()}.')
                        continue

                    val_key = val_keys[0]

                    lats_2d = np.array(data[lat_key])
                    lons_2d = np.array(data[lon_key])
                    vals_2d = np.array(data[val_key], dtype=float)

                    x, y = m(lons_2d, lats_2d)

                    # Plot data grid above base land (zorder 2)
                    pcm = m.pcolormesh(x, y, vals_2d, cmap=config['cmap'],
                                       vmin=config['vmin'], vmax=config['vmax'], shading='auto', zorder=2)
                    data_plotted = True

                except Exception as e:
                    logger.error(f'Error reading JSON for {ofs.upper()}: {e}')
            else:
                logger.warning(f"Grid JSON file not found matching {json_pattern}. Skipping {ofs.upper()} in {config['var']} map.")

        if not data_plotted:
            logger.warning(f"No continuous grid JSON data found for {config['var']} map.")
            plt.close(fig)
            continue

        # Draw boundaries and coastlines ON TOP of data to clearly outline land features
        m.drawcountries(linewidth=0.5, zorder=3, color='grey')
        m.drawstates(linewidth=0.5, zorder=3, color='grey')
        m.drawcoastlines(linewidth=0.5, zorder=4)

        m.drawparallels(np.arange(40., 50., 2.), labels=[1,0,0,0], fontsize=10, linewidth=0.25, zorder=5)
        m.drawmeridians(np.arange(-95., -70., 5.), labels=[0,0,0,1], fontsize=10, linewidth=0.25, zorder=5)

        for city, (lon, lat) in cities.items():
            cx, cy = m(lon, lat)
            m.plot(cx, cy, 'ko', markersize=3, zorder=6)
            if city in ['Chicago', 'Duluth', 'Detroit', 'Milwaukee', 'Green Bay', 'Marquette', 'Toledo', 'Toronto']:
                x_offset = -20000
                h_align = 'right'
            else:
                x_offset = 20000
                h_align = 'left'
            ax.text(cx + x_offset, cy, city, fontsize=7, zorder=6, va='center', ha=h_align)

        if pcm:
            cbar = fig.colorbar(pcm, ax=ax, shrink=0.6, aspect=30, pad=0.04)
            cbar.set_label(config['title'], fontsize=12)

        output_file = os.path.join(prop.om_files, f"map_ice_{cast}_continuous_{config['var']}.png")
        fig.savefig(output_file, dpi=600, bbox_inches='tight')
        plt.close(fig)
        logger.info(f'Wrote continuous map to {output_file}')


def make_daily_error_gif(df1, prop, cast, logger):
    '''
    Creates an animated GIF of daily error (Model - Obs) maps,
    only if the total time range is 10 days or less.
    '''
    logger.info(f'Creating daily error GIF for {cast}...')

    json_pattern = os.path.join(prop.visuals_json_maps_ice_path, f'*_{cast}_daily_diff.json')
    daily_files = glob.glob(json_pattern)

    if not daily_files:
        logger.warning(f'No daily diff JSON files found for {cast}. Ensure makejson=True in make_ice_map.py. Skipping GIF.')
        return

    dates = set()
    for f in daily_files:
        basename = os.path.basename(f)
        front = basename.split(f'_{cast}_daily_diff.json')[0]
        parts = front.split('-')
        if len(parts) >= 5:
            date_str = '-'.join(parts[:4])
            dates.add(date_str)

    try:
        sorted_dates = sorted(list(dates), key=lambda x: datetime.strptime(x, '%m-%d-%Y-%HZ'))
    except ValueError:
        logger.warning('Could not reliably parse dates from JSON filenames. Skipping GIF.')
        return

    if not sorted_dates:
        logger.warning('No parsed dates found. Skipping GIF.')
        return

    # Check if range is > 10 days
    first_dt = datetime.strptime(sorted_dates[0], '%m-%d-%Y-%HZ')
    last_dt = datetime.strptime(sorted_dates[-1], '%m-%d-%Y-%HZ')
    delta = last_dt - first_dt

    if delta.days > 10:
        logger.info(f'Time range is {delta.days} days (>10). Skipping GIF generation.')
        return

    ll_lon, ll_lat = -92.5, 41.0
    ur_lon, ur_lat = -75.5, 49.0

    cities = {
        'Chicago': (-87.6298, 41.8781),
        'Detroit': (-83.0458, 42.3314),
        'Toronto': (-79.3470, 43.6511),
        'Cleveland': (-81.6944, 41.4993),
        'Buffalo': (-78.8784, 42.8864),
        'Duluth': (-92.1005, 46.7867),
        'Sault Ste. Marie': (-84.3451, 46.4953),
        'Milwaukee': (-87.9065, 43.0389),
        'Green Bay': (-88.0198, 44.5133),
        'Rochester': (-77.6109, 43.1566),
        'Erie': (-80.0851, 42.1292),
        'Marquette': (-87.3950, 46.5436),
        'Toledo': (-83.5552, 41.6528),
    }

    frames = []

    for date_str in sorted_dates:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_title(f"{cast.strip('_b').upper()} daily error\n{date_str}",
                     fontsize=16, pad=15)

        # High resolution map base
        m = Basemap(projection='merc', resolution='h', area_thresh=1000,
                    llcrnrlon=ll_lon, llcrnrlat=ll_lat,
                    urcrnrlon=ur_lon, urcrnrlat=ur_lat, ax=ax)

        m.drawmapboundary(fill_color='lightblue')
        m.fillcontinents(color='lightgrey', lake_color='lightblue', zorder=1)

        data_plotted = False
        pcm = None

        for ofs in df1['ofs']:
            json_file = os.path.join(prop.visuals_json_maps_ice_path, f'{date_str}-{ofs}_{cast}_daily_diff.json')
            if os.path.exists(json_file):
                try:
                    with open(json_file) as f:
                        data = json.load(f)

                    keys = list(data.keys())
                    lat_key = next((k for k in keys if k.lower() in ['lat', 'lats', 'latitude', 'latitudes']), None)
                    lon_key = next((k for k in keys if k.lower() in ['lon', 'lons', 'longitude', 'longitudes']), None)
                    val_keys = [k for k in keys if k not in [lat_key, lon_key]]

                    if not lat_key or not lon_key or not val_keys:
                        continue

                    val_key = val_keys[0]
                    lats_2d = np.array(data[lat_key])
                    lons_2d = np.array(data[lon_key])
                    vals_2d = np.array(data[val_key], dtype=float)

                    x, y = m(lons_2d, lats_2d)

                    pcm = m.pcolormesh(x, y, vals_2d, cmap='seismic', vmin=-100, vmax=100, shading='auto', zorder=2)
                    data_plotted = True
                except Exception as e:
                    logger.error(f'Error reading JSON {json_file}: {e}')

        if not data_plotted:
            plt.close(fig)
            continue

        # Draw borders and coastlines on top of data
        m.drawcountries(linewidth=0.5, zorder=3, color='grey')
        m.drawstates(linewidth=0.5, zorder=3, color='grey')
        m.drawcoastlines(linewidth=0.5, zorder=4)

        for city, (lon, lat) in cities.items():
            cx, cy = m(lon, lat)
            m.plot(cx, cy, 'ko', markersize=3, zorder=5)
            if city in ['Chicago', 'Duluth', 'Detroit', 'Milwaukee', 'Green Bay', 'Marquette', 'Toledo']:
                x_offset = -20000
                h_align = 'right'
            else:
                x_offset = 20000
                h_align = 'left'
            ax.text(cx + x_offset, cy, city, fontsize=7, zorder=5, va='center', ha=h_align)

        cbar = fig.colorbar(pcm, ax=ax, shrink=0.6, aspect=30, pad=0.04)
        cbar.set_label('Mean Error (%)', fontsize=12)

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        frames.append(Image.open(buf))
        plt.close(fig)

    if frames:
        output_gif = os.path.join(prop.om_files, f'animated_ice_{cast}_daily_diff.gif')
        frames[0].save(output_gif, save_all=True, append_images=frames[1:], duration=600, loop=0)
        logger.info(f'Wrote daily error animated GIF to {output_gif}')
    else:
        logger.warning('No frames generated for GIF.')


def list_ofs():
    '''
    Returns a list of all OFS, sorted alphabetically
    '''
    return ['leofs', 'lmhofs', 'loofs', 'lsofs']


def is_df_nans(df):
    '''
    Checks if stats dataframe is full of nans, which happens if there are
    no .int files available to collect.
    '''
    if df is None or df.empty or 'ofs' not in df.columns:
        return True
    df_indexed = df.set_index('ofs')
    return df_indexed.isna().all().all()


def make_bar_plots(df1, df2, prop, cast, logger):
    '''
    Takes a pandas dataframe, prop, cast (nowcast or forecast_b) and writes
    a summary scorecard plotly plot that includes all OFS and all variables
    '''
    col_labels = df1['ofs']
    col_labels = [item.upper() for item in col_labels]
    # Title
    titlestr = 'GLOFS ' + cast.rstrip('_b') + ' ice skill overview,<br>' + \
        getattr(prop, 'start_date_scorecard', '') + ' - ' + getattr(prop, 'end_date_scorecard', '')
    # Marker (OFS) colors
    ofs_colors = ['#e377c2','#17becf','#ff7f0e','#9467bd']
    # Outline (stat) colors
    csi_stat_colors = ['rgba(' +
                       str(mcolors.to_rgb('lightseagreen'))
                       .strip('()') + ', ' + str(1)+')','rgba(' +
                       str(mcolors.to_rgb('purple'))
                       .strip('()') + ', ' + str(0.75)+')','rgba(' +
                       str(mcolors.to_rgb('coral'))
                       .strip('()') + ', ' + str(0.75)+')']
    rmse_stat_colors = ['rgba(' +
                        str(mcolors.to_rgb('grey'))
                        .strip('()') + ', ' + str(0.75) + ')']
    # Make figure
    fig = make_subplots(
        rows=2, cols=1, vertical_spacing=0.06,
        shared_xaxes=True,
    )
    stat_list = df1.columns[1:]
    for i,stat in enumerate(stat_list):
        fig.add_trace(
            go.Bar(
                x=col_labels,
                y=df1[stat],
                name=stat,
                width=0.5,
                marker_color=csi_stat_colors[i],
                marker_line_color=ofs_colors,
                marker_line_width=0,
                textposition='outside',
            ),
            row=1, col=1,
        )
    stat_list = ['RMSE, all']
    for i,stat in enumerate(stat_list):
        fig.add_trace(
            go.Bar(
                x=col_labels,
                y=df2[stat],
                name=stat,
                width=0.5,
                marker_color=rmse_stat_colors[i],
                marker_line_color=ofs_colors,
                marker_line_width=0,
                textposition='outside',
            ),
            row=2, col=1,
        )

    # # Update layout for shared color axis and overall figure properties
    figwidth = 600
    figheight = figwidth*1
    fig.update_layout(
        bargap=0.2,
        barmode='stack',
        title_text=titlestr,
        title_x=0.5,
        title_y=0.96,
        title_font=dict(size=24, color='black', family='Open Sans'),
        paper_bgcolor='white',
        plot_bgcolor='white',
        margin={'l': 50, 'r': 50, 'b': 100, 't': 100},
        width=figwidth, height=figheight,
        autosize=False,
    )
    fig.update_yaxes(
        tickfont={
            'size': 14, 'color': 'black', 'family': 'Open Sans',
        }, title_text='CSI',
        title_font={'size': 20, 'color': 'black', 'family': 'Open Sans'},
        #scaleanchor='x',
        showline=True,
        linewidth=1,
        linecolor='black',
        mirror=True,
        range=[0,1],
        row=1, col=1,
    )
    fig.update_yaxes(
        tickfont={
            'size': 14, 'color': 'black', 'family': 'Open Sans',
        }, title_text='Ice conc. RMSE (%)',
        title_font={'size': 20, 'color': 'black', 'family': 'Open Sans'},
        #scaleanchor='x',
        showline=True,
        linewidth=1,
        linecolor='black',
        mirror=True,
        range=[0,int(np.ceil(np.nanmax(df2['RMSE, all'].values)/10))*10],
        row=2, col=1,
    )
    fig.update_xaxes(
        showline=True,
        linewidth=1,
        linecolor='black',
        mirror=True,
        row=1, col=1,
    )
    fig.update_xaxes(
        tickangle=45,
        tickfont={
            'size': 14, 'color': 'black', 'family': 'Open Sans',
        },
        title_text='OFS',
        title_font={'size': 20, 'color': 'black', 'family': 'Open Sans'},
        showline=True,
        linewidth=1,
        linecolor='black',
        mirror=True,
        row=2, col=1,
    )

    # Write to file
    # prop.stat = prop.stat.rstrip('*')
    output_file = (
        f'{prop.om_files}/bars_ice_{cast}_'
        f'all_GLOFS'
        )
    fig_config = {
    'toImageButtonOptions': {
        'format': 'png',
        'filename': output_file.split('/')[-1],
        'height': figheight,
        'width': figwidth,
        'scale': 1
        }
    }
    logger.debug(f'Writing file: {output_file}')
    fig.write_html(output_file+'.html',config=fig_config,auto_open=False)
    logger.debug(f'Finished writing file: {output_file}')
    logger.info('Wrote bar plot to file for %s', cast)


def make_scatter_plot(prop, df, cast, logger):
    nrows = 2
    ncols = 2

    # Match the colors used in the time series plot
    ofs_list = list_ofs()
    ofs_colors = ['#e377c2', '#17becf', '#ff7f0e', '#9467bd']

    # Format subplot titles with HTML color tags
    colored_titles = [f"<span style='color:{ofs_colors[i]}'><b>{ofs.upper()}</b></span>" for i, ofs in enumerate(ofs_list)]

    fig = make_subplots(
        rows=nrows, cols=ncols, vertical_spacing=0.11,
        horizontal_spacing=0.1,
        subplot_titles=colored_titles
    )

    figtitle = 'GLOFS ' + cast.strip('_b') + '<br>' + \
        getattr(prop, 'start_date_scorecard', '') + ' - ' + getattr(prop, 'end_date_scorecard', '')

    rownum = [1, 1, 2, 2]
    colnum = [1, 2, 1, 2]
    figheight = 600
    figwidth = figheight * 1.05

    # Calculate global max for identical axes across all subplots
    global_max = df[['mod_meanicecover', 'obs_meanicecover']].max().max()
    axmax = 10 * (np.ceil(global_max / 10))
    if axmax > 100 or pd.isna(axmax):
        axmax = 100

    tick_interval = axmax / 5 if axmax > 0 else 20
    colorbar_added = False
    # Grab the start date and slice the MM/DD characters
    start_date_str = getattr(prop, 'start_date_scorecard', '11/01/YYYY')
    cbar_date = start_date_str[:5]
    for i, ofs in enumerate(ofs_list):
        df_filt = df[df['OFS'] == ofs]
        color_scale = np.linspace(0, len(df_filt)-1, len(df_filt)) if not df_filt.empty else []

        if df_filt.empty:
            fig.add_annotation(
                    text=f'No data for {ofs.upper()}!',
                    xref='x domain', yref='y domain',
                    font=dict(size=14, color='red'),
                    x=0.5, y=0.5,
                    showarrow=False,
                    row=rownum[i], col=colnum[i],
            )
            fig.update_yaxes(range=[0, axmax], dtick=tick_interval, row=rownum[i], col=colnum[i])
            fig.update_xaxes(range=[0, axmax], dtick=tick_interval, row=rownum[i], col=colnum[i])
            continue

        show_cbar = False
        if not colorbar_added:
            show_cbar = True
            colorbar_added = True

        fig.add_trace(
            go.Scatter(
                x=df_filt['obs_meanicecover'],
                y=df_filt['mod_meanicecover'],
                name=ofs.upper(),
                mode='markers',
                showlegend=False,
                marker=dict(
                    color=color_scale,
                    colorscale='viridis',
                    showscale=show_cbar,
                    colorbar=dict(
                        title=f'Days since<br>{cbar_date}',
                        lenmode='fraction',
                        len=0.5,
                        thickness=10,
                        thicknessmode='pixels',
                        tickfont=dict(size=12, color='black'),
                    ) if show_cbar else None,
                    line=dict(
                        color='black',
                        width=0.25
                    )
                ),
                hovertemplate='GLSEA: %{x:.2f}<br>OFS: %{y:.2f}<extra></extra>',
            ), row=rownum[i], col=colnum[i]
        )

        fig.add_trace(
            go.Scatter(
                x=[0, axmax],
                y=[0, axmax],
                mode='lines',
                showlegend=False,
                line=dict(color='black', width=1)
            ), row=rownum[i], col=colnum[i]
        )

        if rownum[i] == 1 and colnum[i] == 1:
            fig.add_annotation(
                    text='Model<br>over-icing',
                    xref='x domain', yref='y domain',
                    font=dict(size=12, color='black'),
                    x=0, y=1,
                    showarrow=False,
                    row=1, col=1,
            )
            fig.add_annotation(
                    text='Model<br>under-icing',
                    xref='x domain', yref='y domain',
                    font=dict(size=12, color='black'),
                    x=1, y=0,
                    showarrow=False,
                    row=1, col=1,
            )

        # Notice we removed title_text here so we can use global labels
        fig.update_yaxes(
            range=[0, axmax],
            dtick=tick_interval,
            tickfont=dict(size=14, color='black'),
            row=rownum[i], col=colnum[i]
        )
        fig.update_xaxes(
            range=[0, axmax],
            dtick=tick_interval,
            tickfont=dict(size=14, color='black'),
            row=rownum[i], col=colnum[i]
        )

    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
    fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)

    # Add Global Centered Axis Labels using paper coordinates
    fig.add_annotation(
        x=0.5, y=-0.12, xref='paper', yref='paper',
        text='GLSEA daily mean ice conc. (%)',
        showarrow=False, font=dict(size=18, color='black'),
        xanchor='center', yanchor='top'
    )
    fig.add_annotation(
        x=-0.1, y=0.5, xref='paper', yref='paper',
        text='OFS daily mean ice conc. (%)',
        showarrow=False, font=dict(size=18, color='black'),
        textangle=-90, xanchor='right', yanchor='middle'
    )

    fig.update_layout(
        title=dict(
            text=figtitle,
            font=dict(size=20, color='black'),
            y=0.95,
            x=0.5,
            xanchor='center',
            yanchor='top',
        ),
        legend_tracegroupgap=130,
        transition_ordering='traces first',
        dragmode='zoom',
        height=figheight,
        width=figwidth,
        autosize=False,
        # Force 1:1 aspect ratio on all subplots
        yaxis=dict(scaleanchor='x', scaleratio=1),
        yaxis2=dict(scaleanchor='x2', scaleratio=1),
        yaxis3=dict(scaleanchor='x3', scaleratio=1),
        yaxis4=dict(scaleanchor='x4', scaleratio=1),
        template='plotly_white',
        margin=dict(t=100, b=80, l=80, r=50), # Expanded bottom/left margins for global labels
    )

    # Write to file
    output_file = (
        f'{prop.om_files}/scatter_ice_{cast}_'
        f'all_GLOFS'
    )
    fig_config = {
        'toImageButtonOptions': {
            'format': 'png',
            'filename': output_file.split('/')[-1],
            'height': figheight,
            'width': figwidth,
            'scale': 1
        }
    }
    logger.debug(f'Writing file: {output_file}')
    fig.write_html(output_file+'.html', config=fig_config, auto_open=False)
    logger.debug(f'Finished writing file: {output_file}')
    logger.info('Wrote GLOFS scatter plot to file for %s', cast)

def make_summary_table(df_ofs, prop, cast, logger):
    """
    Creates an HTML summary table of ice cover and skill statistics
    averaged over the time period for each Great Lake.
    """
    logger.info(f'Generating summary table for {cast}...')

    # Calculate means grouped by OFS
    summary_df = df_ofs.groupby('OFS').agg(
        Obs_Ice_Mean=('obs_meanicecover', 'mean'),
        Mod_Ice_Mean=('mod_meanicecover', 'mean'),
        Clim_Ice_Mean=('icecover_hist', 'mean'), # <-- Added climatology column
        RMSE_All=('rmse_all', 'mean'),
        CSI=('csi_all', 'mean')
    ).reset_index()

    # Format OFS names to uppercase
    summary_df['OFS'] = summary_df['OFS'].str.upper()

    # Format numeric columns
    summary_df['Obs_Ice_Mean'] = summary_df['Obs_Ice_Mean'].map('{:.2f}%'.format)
    summary_df['Mod_Ice_Mean'] = summary_df['Mod_Ice_Mean'].map('{:.2f}%'.format)
    summary_df['Clim_Ice_Mean'] = summary_df['Clim_Ice_Mean'].map('{:.2f}%'.format)
    summary_df['RMSE_All'] = summary_df['RMSE_All'].map('{:.2f}%'.format)
    summary_df['CSI'] = (summary_df['CSI'] * 100).map('{:.1f}'.format)

    # Rename columns for display
    summary_df.rename(columns={
        'Obs_Ice_Mean': 'NIC Ice Conc.',
        'Mod_Ice_Mean': 'OFS Ice Conc.',
        'Clim_Ice_Mean': 'Climatology Mean',
        'RMSE_All': 'RMSE'
    }, inplace=True)

    # Use Plotly to generate a styled HTML table
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=list(summary_df.columns),
            fill_color='#6C4786',
            font=dict(color='white', size=16, family='Open Sans'),
            align='center',
            height=40
        ),
        cells=dict(
            values=[summary_df[col] for col in summary_df.columns],
            fill_color='#f4f7f6',
            font=dict(color='#333', size=13, family='Open Sans'),
            align='center',
            height=30
        )
    )])

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=200,
        paper_bgcolor='white'
    )

    output_file = os.path.join(prop.om_files, f'summary_table_ice_{cast}_all_GLOFS.html')
    fig.write_html(output_file, auto_open=False)
    logger.info(f'Wrote summary table to {output_file}')

def make_summary_series(prop, df, cast, logger):
    '''
    '''
    nrows = 2
    ncols = 2
    # Do stats time series
    fig = make_subplots(
        rows=nrows, cols=ncols, vertical_spacing=0.075,
        subplot_titles='',
        shared_xaxes=True,
    )
    figtitle = 'GLOFS ' + cast.strip('_b') + ' ice skill summary, ' + \
        getattr(prop, 'start_date_scorecard', '') + ' - ' + getattr(prop, 'end_date_scorecard', '')
    #showlegend = [True, False]
    ofs_colors = ['#e377c2','#17becf','#ff7f0e','#9467bd']
    list_stats = ['obs_meanicecover',
                  'mod_meanicecover',
                  'csi_all',
                  'rmse_all']
    stats_names = ['GLSEA mean <br>ice conc. (%)',
                   'OFS mean <br>ice conc. (%)',
                   'CSI (smoothed)',
                   'RMSE ice conc. (%)']
    rownum = [1,1,2,2]
    colnum = [1,2,1,2]
    ymin = [0,0,0,0]
    ymax = [100,100,1,100]
    figheight=500
    figwidth=figheight*2.5
    showlegend = [True, False, False, False]
    for i,ofs in enumerate(list_ofs()):
        # Filter df by ofs
        df_filt = df[df['OFS'] == ofs]
        if df_filt.empty:
            continue
        for j,stat in enumerate(list_stats):
            if stat == 'csi_all':
                # Do low-pass filter
                window_length = 7  # Must be an odd number
                if len(df_filt['csi_all']) < 10:
                    window_length = 3
                polyorder = 2
                s = pd.Series(df_filt['csi_all'])
                interpolated_s = s.bfill().ffill().interpolate()  # Fill NaNs
                if not interpolated_s.isna().all():
                    smoothed_y = savgol_filter(interpolated_s,
                                               window_length,
                                               polyorder)
                    # Reinsert NaNs
                    smoothed_y[np.argwhere(np.isnan(df_filt['csi_all']))] = np.nan
                    # Check for values > 1. Max value is 1!
                    smoothed_y[smoothed_y > 1] = 1
                    df_filt['csi_all'] = smoothed_y
                else:
                    df_filt['csi_all'] = interpolated_s
            # Done with low-pass filter
            # Start figs
            fig.add_trace(
                go.Scatter(
                    x=df_filt['time_all_dt'],
                    y=df_filt[stat],
                    name=ofs.upper(),
                    hovertemplate='%{y:.2f}',
                    mode='lines',
                    #legendgroup='1',
                    showlegend=showlegend[j],
                    line=dict(color=ofs_colors[i],
                         width=2,
                         # dash='dash',
                    ),
                ), row=rownum[j], col=colnum[j],
            )
            fig.update_yaxes(
                title_text=stats_names[j],
                title_font=dict(size=16, color='black'),
                range=[ymin[j], ymax[j]],
                row=rownum[j], col=colnum[j],
            )
    fig.update_yaxes(
        showline=True, linewidth=1, linecolor='black',
        mirror=True,
    )
    fig.update_xaxes(
        showline=True,
        linewidth=1,
        linecolor='black',
        mirror=True,
        tickfont=dict(size=16, color='black'),
        tickangle=45
    )
    fig.update_layout(
        title=dict(
            text=figtitle,
            font=dict(size=20, color='black'),
            y=0.97,  # new
            x=0.5, xanchor='center', yanchor='top',
        ),
        yaxis1=dict(tickfont=dict(size=16, color='black')),
        yaxis2=dict(tickfont=dict(size=16, color='black')),
        yaxis3=dict(tickfont=dict(size=16, color='black')),
        yaxis4=dict(tickfont=dict(size=16, color='black')),
        legend_tracegroupgap=130,
        transition_ordering='traces first',
        dragmode='zoom',
        hovermode='x unified',
        height=figheight,
        width=figwidth,
        xaxis_tickangle=-45,
        template='plotly_white',
        margin=dict(t=50, b=50),
        legend=dict(
            font=dict(size=16, color='black'),
            bgcolor='rgba(0,0,0,0)',
        ),
    )
    # Write to file
    output_file = (
        f'{prop.om_files}/series_ice_{cast}_'
        f'all_GLOFS'
        )
    fig_config = {
    'toImageButtonOptions': {
        'format': 'png',
        'filename': output_file.split('/')[-1],
        'height': figheight,
        'width': figwidth,
        'scale': 1
        }
    }
    logger.debug(f'Writing file: {output_file}')
    fig.write_html(output_file+'.html',config=fig_config,auto_open=False)
    logger.debug(f'Finished writing file: {output_file}')
    logger.info('Wrote scorecard/flag plot to file for %s', cast)


def collect_stats_files(prop, ofs, var, cast, logger):
    '''
    Finds and loads all available .csv stat files for an OFS, then combines them
    into one big pandas dataframe. Calculates RMSE and CSI metrics for
    entire OFS, and returns those stats in pandas dataframe.
    '''

    # Get all file names for var and ofs
    stat_file = [
        file for file in
        os.listdir(prop.data_skill_stats_path)
        if ofs in file and cast in file and 'icestatseries' in file
    ]
    try:  # Check to see if any files were found
        stat_file[0]
    except IndexError:
        logger.warning(
            'No files for %s in %s, moving to next '
            'variable.', var, ofs,
        )
        # Return 5 nans for 5 non-existent stats
        return None
        #return [np.nan, np.nan, np.nan, np.nan, np.nan]

    # If files exist, keep going
    # read CSV
    stats = pd.read_csv(
        r'' + f'{prop.data_skill_stats_path}/'
        f'{stat_file[0]}',)
    stats['DateTime'] = pd.to_datetime(stats['time_all_dt'],)

    return stats
    #return [csi,fa,miss,rmse,rmse_ice]


def make_OM_view_ice(prop, logger):
    '''
    Top-level function that calls (directly or indirectly) all other functions.
    '''

    # Specify defaults (can be overridden with command line options)
    if logger is None:
        logger = utils.init_root_logger(prop.path)

    logger.info('--- Starting O&M ice dashboard processing ---')

    # Parameter validation & paths
    _conf = getattr(prop, 'config_file', None)
    dir_params = utils.Utils(_conf).read_config_section('directories', logger)
    ofs_extents_path = utils.resolve_asset_path(prop.path, dir_params['ofs_extents_dir'])
    argu_list = (
        prop.path,
        prop.whichcasts,
        ofs_extents_path,
    )
    parameter_validation(argu_list, logger)
    logger.info('Parameter validation complete!')

    # Path to .int files
    prop.data_skill_1d_pair_path = os.path.join(
        prop.path, dir_params['data_dir'], dir_params['skill_dir'],
        dir_params['1d_ice_pair_dir'],
    )
    os.makedirs(prop.data_skill_1d_pair_path, exist_ok=True)
    # Path to CSV tables
    prop.data_skill_stats_path = os.path.join(
        prop.path, dir_params['data_dir'], dir_params['skill_dir'],
        dir_params['stats_dir'], dir_params['stats_ice_dir'],
    )
    os.makedirs(prop.data_skill_stats_path, exist_ok=True)
    # Path to save O&M files
    prop.om_files = os.path.join(
        prop.path, dir_params['data_dir'], dir_params['visual_dir'],
        dir_params['om_dir_ice'],
    )
    os.makedirs(prop.om_files, exist_ok=True)

    # Path to read JSON maps
    prop.visuals_json_maps_ice_path = os.path.join(
        prop.path, dir_params['data_dir'], dir_params['visual_dir'],
        dir_params['visual_ice_dir'], dir_params['visual_ice_json_maps'],
    )

    # Reformat whichcasts argument
    prop.whichcasts = prop.whichcasts.replace('[', '')
    prop.whichcasts = prop.whichcasts.replace(']', '')
    prop.whichcasts = prop.whichcasts.split(',')

    # Next loop through each OFS, each variable, load control files,
    # loop through stations in each control file, and collect .int files,
    # csv tables, and plots.
    for cast in prop.whichcasts:
        logger.info('Starting file collection for %s!', cast)
        # Set up blank array for each whichcast to hold stats
        ofs_stats = {
            'Hits': [],
            'False alarms': [],
            'Misses': [],
            'RMSE, all': [],
            'RMSE, ice': [],
        }
        # Set up blank DataFrame for each whichcast to append time series
        df_ofs = pd.DataFrame()

        # 1. Collect all stats first
        collected_stats = {}
        for ofs in list_ofs():
            logger.info(
                'Collecting station output files for %s...',
                ofs.upper(),
            )
            stats = collect_stats_files(prop, ofs, 'ice_conc', cast, logger)
            if stats is not None and not stats.empty:
                collected_stats[ofs] = stats

        if not collected_stats:
            logger.warning(f'No valid data collected for {cast}. Skipping plots.')
            continue

        # 2. Find intersection of dates to safeguard identical date ranges
        common_dates = None
        for ofs, stats in collected_stats.items():
            dates = set(stats['time_all_dt'].astype(str))
            if common_dates is None:
                common_dates = dates
            else:
                common_dates = common_dates.intersection(dates)

        if not common_dates:
            logger.warning(f'No overlapping dates found across OFS for {cast}. Skipping plots.')
            continue

        logger.info(f'Found {len(common_dates)} common dates across all available lakes.')

        # 3. Process each OFS using only the common dates
        for ofs, stats in collected_stats.items():

            # Apply the date filter
            stats = stats[stats['time_all_dt'].astype(str).isin(common_dates)].copy()

            stats_dict = {
                'Hits': [],
                'False alarms': [],
                'Misses': [],
                'RMSE, all': [],
                'RMSE, ice': [],
            }

            stats['OFS'] = ofs
            df_ofs = pd.concat([df_ofs, stats], axis=0, ignore_index=True)

            # Do some OFS-wide stats
            # But first, filter out no ice
            stats['csi_sum'] = stats['csi_all'] + stats['csi_misses'] + \
                stats['csi_falsealarms']
            stats.loc[stats['csi_sum'] == 0, ['csi_all','csi_falsealarms',
                                              'csi_misses']] = np.nan
            # Get Means for bar plots
            stats_dict['Hits'].append(stats[
                'csi_all'].mean())
            stats_dict['False alarms'].append(stats[
                'csi_falsealarms'].mean())
            stats_dict['Misses'].append(stats[
                'csi_misses'].mean())
            stats_dict['RMSE, all'].append(stats[stats['rmse_all']>0]['rmse_all'].mean())
            stats_dict['RMSE, ice'].append(stats[
                'rmse_either'].mean())

            # Make master array for means/bar plots.
            # Each row is an OFS, columns are csi, false alarms, misses,
            # RMSE, and RMSE ice only
            for stat in list(stats_dict.keys()):
                if len(stats_dict[stat]) == 0:
                    stats_dict[stat] = [np.nan]
                row_data = {
                    'ofs': ofs,
                    stat: stats_dict[stat][0],
                }
                ofs_stats[stat].append(row_data)

        if df_ofs.empty:
            logger.warning(f'No valid data collected for {cast}. Skipping plots.')
            continue

        # Get date range for table & scorecard plots below
        try:
            prop.start_date_scorecard = datetime.strftime(
                df_ofs['DateTime'].min(),'%m/%d/%Y'
                )
            prop.end_date_scorecard = datetime.strftime(
                df_ofs['DateTime'].max(),'%m/%d/%Y'
                )
        except Exception as e:
            logger.error(f'Error formatting dates: {e}')
            continue

        # Make time series & scatter plot
        try:
            make_summary_series(prop, df_ofs, cast, logger)
            make_scatter_plot(prop, df_ofs, cast, logger)
        except Exception as ex:
            logger.error('Exception caught in make_summary_series or scatter_plot: %s', ex)

        # Dataframe for each panel in bar subplots
        df1 = pd.concat([pd.DataFrame(ofs_stats['Hits']),
                         pd.DataFrame(ofs_stats['False alarms'])
                         ['False alarms'],
                         pd.DataFrame(ofs_stats['Misses'])
                         ['Misses']],axis=1)
        df2 = pd.concat([pd.DataFrame(ofs_stats['RMSE, all']),
                         pd.DataFrame(ofs_stats['RMSE, ice'])
                         ['RMSE, ice']],axis=1)

        if not is_df_nans(df1) and not is_df_nans(df2):
            # Bar plots!
            try:
                make_bar_plots(df1, df2, prop, cast, logger)
                make_overview_map(df1, df2, prop, cast, logger)
                make_daily_error_gif(df1, prop, cast, logger)
                make_summary_table(df_ofs, prop, cast, logger)
                export_kpi_json(df_ofs, prop, cast, logger)
                logger.info('Finished O&M ice plots! Good bye.')
            except Exception as ex:
                logger.error('Exception caught in make_scorecard_plot, map, or GIF processing: %s', ex)
        else:
            logger.warning('Ice stats dictionary is full of NaNs. No scorecard '
                           'plots will be made.')

    logger.info('Program complete.')

# Execution:
if __name__ == '__main__':

    # Parse (optional and required) command line arguments
    parser = argparse.ArgumentParser(
        prog='python make_OM_view_ice.py',
        usage='%(prog)s',
        description='make generalized plots of ice skill for quick O&M checks',
    )
    parser.add_argument(
        '-p',
        '--Path',
        required=True,
        help='Home path',
    )
    parser.add_argument(
        '-ws',
        '--Whichcasts',
        required=True,
        help="whichcast: 'Nowcast','Forecast_A','Forecast_B', all",
    )
    parser.add_argument(
        '-c',
        '--config',
        help='Path to configuration file (default: conf/ofs_dps.conf)')

    args = parser.parse_args()

    prop1 = model_properties.ModelProperties()
    prop1.path = args.Path
    prop1.whichcasts = args.Whichcasts.lower()
    prop1.config_file = args.config

    # Exclude forecast_a
    if 'forecast_a' in prop1.whichcasts:
        print('Forecast_a is not available! Use nowcast and/or forecast_b.')
        sys.exit(-1)

    # Go!
    make_OM_view_ice(prop1, None)
