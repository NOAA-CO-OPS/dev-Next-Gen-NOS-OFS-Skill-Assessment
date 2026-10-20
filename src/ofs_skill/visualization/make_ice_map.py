"""
Created on Mon Jan 13 08:31:05 2025
@author: PL

Main map-making module for do_iceskill.py.
Takes in stacked map data (2D arrays) and loops through each array
to generate Matplotlib subplots and export JSON maps.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap

import ofs_skill.visualization.processing_2d as processing_2d


def make_ice_map(prop, lon_o, lat_o, x_o, y_o, mapdata, time, mapstuff, logger):
    """
    Main map-making function that generates figures with subplotted maps and JSON spatial data.

    Args:
        prop (object): Properties object containing path configurations and run settings.
        lon_o (np.ndarray): 2D array of observation longitudes.
        lat_o (np.ndarray): 2D array of observation latitudes.
        x_o (np.ndarray): X-coordinates for map projection.
        y_o (np.ndarray): Y-coordinates for map projection.
        mapdata (np.ndarray): 3D array (stack) of 2D map layers to plot.
        time (datetime): Current time of the map evaluation.
        mapstuff (str): Keyword specifying the map type ('rmse means', 'clim', 'daily', etc.).
        logger (logging.Logger): Logger instance for recording execution steps.
    """
    # Flags for output types
    makestatic = getattr(prop, 'makestatic', False)
    makejson = getattr(prop, 'makejson', True)
    brdr = 0.2 # Map border padding

    # Parse temporal bounds for dynamic titling
    start_date = prop.start_date_full.split('T')[0]
    end_date = prop.end_date_full.split('T')[0]
    datestr = str(time).split()[0]
    dayselapsed = (datetime.strptime(prop.end_date_full, '%Y-%m-%dT%H:%M:%SZ') -
                   datetime.strptime(prop.start_date_full, '%Y-%m-%dT%H:%M:%SZ')).days + 1

    # Standardize string prefixes for file naming
    ofs_up = prop.ofs.upper()
    prefix = f'{prop.ofs}_{prop.whichcast}'
    json_prefix = f'{prefix}_{start_date}--{end_date}_map_'
    daily_prefix = time.strftime('%m-%d-%Y-%HZ-') + prefix

    # Configuration Dictionary: Centralizes titles, color limits, colorbars,
    # and file names based on the requested 'mapstuff' type to keep logic DRY.
    configs = {
        'rmse means': {
            'titles': [f'{ofs_up} mean observed ice coverage, {start_date} - {datestr}',
                       f'{ofs_up} mean modeled ice coverage, {start_date} - {datestr}',
                       f'{ofs_up} RMSE, {start_date} - {datestr}'],
            'minmax': [[0, 100], [0, 100], [0, 100]],
            'cbars': ['Ice coverage (%)', 'Ice coverage (%)', 'RMSE (%)'],
            'cmaps': ['terrain', 'terrain', 'plasma'],
            'map_name': f'{prefix}_map_conc_.png',
            'json_name': json_prefix,
            'json_types': ['mean_iceconc_obs', 'mean_iceconc_mod', 'rmse']
        },
        'clim': {
            'titles': [f'{ofs_up} mean climatology ice coverage, {start_date} - {datestr}'],
            'minmax': [[0, 100]],
            'cbars': ['Ice coverage (%)'],
            'cmaps': ['Greens'],
            'map_name': f'{prefix}_map_clim_.png',
            'json_name': json_prefix,
            'json_types': ['mean_iceconc_clim']
        },
        'daily': {
            'titles': [f'{ofs_up} observed ice coverage, {datestr}',
                       f'{ofs_up} modeled ice coverage, {datestr}',
                       f'{ofs_up} Modeled - observed ice coverage, {datestr}'],
            'minmax': [[0, 100], [0, 100], [-100, 100]],
            'cbars': ['Ice coverage (%)', 'Ice coverage (%)', 'Error (%)'],
            'cmaps': ['cool', 'cool', 'seismic'],
            'map_name': f'{daily_prefix}_daily_map.png',
            'json_name': daily_prefix,
            'json_types': ['_daily_obs', '_daily_mod', '_daily_diff']
        },
        'extents': {
            'titles': [f'{ofs_up} total observed ice days, \n{start_date} - {datestr} (total days elapsed = {dayselapsed})',
                       f'{ofs_up} total modeled ice days, \n{start_date} - {datestr} (total days elapsed = {dayselapsed})'],
            'minmax': [[0, dayselapsed], [0, dayselapsed]],
            'cbars': ['Ice days', 'Ice days'],
            'cmaps': ['viridis', 'viridis'],
            'map_name': f'{prefix}_map_extent_.png',
            'json_name': json_prefix,
            'json_types': ['obs_ice_days', 'model_ice_days']
        },
        'diff': {
            'titles': [f'{ofs_up} mean error, {start_date} - {datestr}',
                       f'{ofs_up} maximum error, {start_date} - {datestr}',
                       f'{ofs_up} minimum error, {start_date} - {datestr}'],
            'minmax': [[-100, 100], [-100, 100], [-100, 100]],
            'cbars': ['Error (%)', 'Error (%)', 'Error (%)'],
            'cmaps': ['Spectral', 'Spectral', 'Spectral'],
            'map_name': f'{prefix}_map_error_.png',
            'json_name': json_prefix,
            'json_types': ['mean_error', 'max_error', 'min_error']
        },
        'csi': {
            'titles': [f'{ofs_up} Critical Success Index hits, \n{start_date} - {datestr} (total days elapsed = {dayselapsed})',
                       f'{ofs_up} Critical Success Index false alarms, \n{start_date} - {datestr} (total days elapsed = {dayselapsed})',
                       f'{ofs_up} Critical Success Index misses, \n{start_date} - {datestr} (total days elapsed = {dayselapsed})'],
            'minmax': [[0, 100], [0, 100], [0, 100]],
            'cbars': ['% of ice days', '% of ice days', '% of ice days'],
            'cmaps': ['brg', 'brg', 'brg'],
            'map_name': f'{prefix}_map_csi_.png',
            'json_name': json_prefix,
            'json_types': ['csi_hit', 'csi_falarm', 'csi_miss']
        }
    }

    logger.info(f'Making maps for {mapstuff}')
    # Validate map type
    if mapstuff not in configs:
        logger.info(f'Incorrect map type ({mapstuff}) specified! Exiting.')
        sys.exit(-1)

    # Load matched configuration
    cfg = configs[mapstuff]
    minmax_vals = np.array(cfg['minmax'])

    # Initialize subplots for static map generation
    if makestatic:
        fig, axs = plt.subplots(len(mapdata), 1)
        if len(mapdata) == 1:
            axs = [axs] # Ensure iterable for single subplots
        fig.set_figheight(10)
        fig.set_figwidth(8)

    # Loop through each layer in the provided map stack
    for j in range(len(mapdata)):
        logger.info(f'Looping through #{j} layer of map stack')

        # Build Matplotlib / Basemap Visuals
        if makestatic:
            ax = axs[j]
            ax.set_title(cfg['titles'][j], fontsize=14)

            # Setup Basemap projection limits based on coordinate inputs
            bmap = Basemap(projection='merc', resolution='i', area_thresh=1.0, ax=ax,
                           llcrnrlon=lon_o.min()-brdr, llcrnrlat=lat_o.min()-brdr,
                           urcrnrlon=lon_o.max()+brdr, urcrnrlat=lat_o.max()+brdr)

            bmap.drawmapboundary()
            bmap.drawcoastlines()
            bmap.drawparallels(np.arange(0., 90., 1.), labels=[1, 0, 0, 0], fontsize=10, linewidth=0.25)
            bmap.drawmeridians(np.arange(180., 360., 2.), labels=[0, 0, 0, 1], fontsize=10, linewidth=0.25)
            bmap.fillcontinents(color='grey', lake_color='aqua')

            cmap_type = cfg['cmaps'][j]
            # Handle discrete boundaries for extent maps using viridis
            if cmap_type == 'viridis':
                norm = mpl.colors.BoundaryNorm(np.arange(0, dayselapsed + 5, 5), mpl.cm.viridis.N)
                i_m = bmap.pcolor(x_o, y_o, mapdata[j, :, :], cmap=mpl.cm.viridis, norm=norm, shading='auto')
            else:
                i_m = bmap.pcolor(x_o, y_o, mapdata[j, :, :], cmap=cmap_type,
                                  vmin=minmax_vals[j, 0], vmax=minmax_vals[j, 1], shading='auto')

            # Apply colorbar and legends
            cbar = bmap.colorbar(i_m)
            cbar.set_label(cfg['cbars'][j], fontsize=13)

            if cmap_type != 'terrain':
                plt.legend(handles=[mpatches.Patch(color='aqua', label='Open water')], loc='lower right')

            fig.tight_layout()
            logger.info(f'Made static map subplot #{j}')

        # Write matching JSON representations for web/interactive components
        if makejson:
            logger.info(f'Starting JSON from map stack layer #{j}')
            out_file = os.path.join(prop.visuals_json_maps_ice_path, f"{cfg['json_name']}{cfg['json_types'][j]}.json")
            processing_2d.write_2d_arrays_to_json(lat_o, lon_o, mapdata[j, :, :], out_file)
            logger.info(f'Finished JSON from map stack layer #{j}')

    # Save final plot payload
    if makestatic:
        logger.info('Saving static map')
        fig.savefig(os.path.join(prop.visuals_maps_ice_path, cfg['map_name']), format='png', dpi=250, bbox_inches='tight')
        plt.close('all')

    logger.info(f'Finished with {mapstuff} maps! Back to main...')
