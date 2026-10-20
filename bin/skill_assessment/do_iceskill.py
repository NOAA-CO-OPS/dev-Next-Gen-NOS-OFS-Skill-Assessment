'''
Introduction and notes

Documentation for do_iceskill.py

Directory Location:  /bin/skill_assessment/

Technical Contact(s): Name:  PL

Abstract:

During a run, for each day, the ice skill assessment:

1) Downloads ice concentration maps from the Great Lakes Surface Environmental
    Analysis (GLSEA) for the time period of interest, and clips it to an OFS area;
2) Fetches available nowcast and/or forecast GLOFS guidance of ice
    concentration for the same time period and OFS area;
3) Produces 1D time series of GLSEA and modeled ice concentration with skill
    statistics at specified locations within the OFS;
4) Interpolates the model output to the regular GLSEA grid, so they are
    directly comparable;
5) Produces basin-wide skill statistics and 2D skill statistics maps.

Language:  Python 3.11

Estimated Execution Time: depends on date range; typically <20 minutes

Scripts/Programs Called:
1) get_icecover_observations.py -- retrieves GLSEA netcdfs
2) get_icecover_model.py -- retrieves model netcdfs, and concatenates them
3) find_ofs_ice_stations.py -- gets inventory of observation stations in an
    OFS, then finds model nodes & GLSEA cells that correspond to them, and
    finally extracts time series of model & GLSEA ice concentration.
4) create_1dplot_ice.py -- main plotting function for ice concentration time
    series and statistics time series.
5) make_ice_map.py -- main map-making function. Makes static .png maps and json
    maps.


Input arguments:
        "-o", "--ofs", required=True, help="""Choose from the list on the
        ofs_extents/ folder, you can also create your own shapefile,
        add it at the ofs_extents/ folder and call it here""", )

        "-p", "--path", required=True,
        help="Inventory File path where ofs_extents/ folder is located", )

        "-s", "--StartDate_full", required=True,
        help="Start Date_full YYYY-MM-DDThh:mm:ssZ e.g.'2023-01-01T12:34:00Z'")

        "-e", "--EndDate_full", required=True,
        help="End Date_full YYYY-MM-DDThh:mm:ssZ e.g. '2023-01-01T12:34:00Z'")

        "-ws", "--Whichcasts", required=True,
        help="whichcasts: 'Nowcast', 'Forecast_A', 'Forecast_B'", )

        "-da", "--DailyAverage", required=False,
        help="Use a daily average model output instead of single hour;
        True/False", )

        "-c", "--config", required=False,
        help="Path to configuration file (default: conf/ofs_dps.conf)", )

Output: See the README for all output types, locations, and filenames:
    https://github.com/NOAA-CO-OPS/" \
        "NOS_Shared_Cyberinfrastructure_and_Skill_Assessment/blob/main/" \
            "README.md#4-great-lakes-ice-skill-assessment"

Author Name:  PL       Creation Date:  07/2024


'''
from __future__ import annotations

import argparse
import csv
import os
import sys
import warnings
from datetime import date, datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.interpolate as interp
from bin.model_processing import get_icecover_model
from bin.obs_retrieval import get_icecover_observations
from bin.visualization import create_1dplot_ice
from mpl_toolkits.basemap import Basemap

from ofs_skill.model_processing import (
    model_properties,
    model_source,
)
from ofs_skill.obs_retrieval import find_ofs_ice_stations, utils
from ofs_skill.skill_assessment import nos_metrics
from ofs_skill.visualization import make_ice_boxplots, make_ice_map


def make_2d_mask(array_to_mask, conditional_array, threshold):
    """
    Applies a threshold mask to a 2D array.

    Args:
        array_to_mask (np.ndarray): The primary array to be masked.
        conditional_array (np.ndarray, optional): A secondary array to check the threshold against. If None, checks the primary array.
        threshold (float): The threshold value below which elements are set to NaN.

    Returns:
        np.ndarray: The masked array with values below the threshold replaced by np.nan.
    """
    masked = array_to_mask.copy()
    if conditional_array is not None:
        masked[conditional_array < threshold] = np.nan
    else:
        masked[masked < threshold] = np.nan
    return masked


def iceonoff(time_all_dt, meanicecover, logger):
    """
    Finds and returns ice onset and thaw dates for an ice concentration time series.

    Args:
        time_all_dt (list): List of datetime.date objects corresponding to the time series.
        meanicecover (np.ndarray or list): 1D array of mean ice concentration values.
        logger (logging.Logger): Logger instance for outputting status messages.

    Returns:
        tuple: (iceon, iceoff) where both are datetime.date objects or None if not found.
    """
    counter = 0
    iceon = None
    iceoff = None
    # Find ice onset date
    for i in range(len(time_all_dt)):
        if meanicecover[i] >= 10:
            counter += 1
        else:
            counter = 0
        if counter == 5:
            iceon = time_all_dt[i]
            logger.info('Ice onset date found!')
            break
    if iceon is None:
        logger.info('Ice onset date not found!')

    # Find ice thaw date
    try:
        logger.info('Trying ice thaw enumerate...')
        idx = len(meanicecover) - next(
            i for i, val in enumerate(reversed(meanicecover), 1) if val >= 10
        )
        logger.info('Completed ice thaw enumerate!')
        if (
            len([idx]) > 0
            and (len(meanicecover)-1)-idx >= 5
            and sum(np.isnan(meanicecover[idx:])) <= 2
        ):
            iceoff = time_all_dt[idx+5]
            logger.info('Ice thaw date found!')
        else:
            logger.info('Ice thaw date not found!')
    except StopIteration:
        logger.warning('StopIteration exception: ice thaw date not found!')

    logger.info('Completed ice onset/thaw date-finding, return to main...')
    return iceon, iceoff


def ice_climatology(prop, time_all_dt, ice_clim):
    """
    Handles loading and parsing 1D and 2D Great Lakes ice cover climatology.

    Args:
        prop (ModelProperties): Object containing model configuration and paths.
        time_all_dt (list): List of datetime.date objects for the target time period.
        ice_clim (np.ndarray): 3D array of climatology data.

    Returns:
        tuple: (icecover_hist, icecover_hist_2d) containing 1D and 2D climatology arrays matched to the time series.
    """
    filename = utils.resolve_asset_path(prop.path, 'conf', 'gl_1d_clim.csv')
    df = pd.read_csv(filename, header=0)
    df['DateTime'] = pd.to_datetime(
        df[['Year', 'Month', 'Day']],
    )

    # Select dates
    dateindex = []
    for i in range(0, len(time_all_dt)):
        tempindex = df.index[(
            df['DateTime'].dt.month == time_all_dt[i].month
        ) & (
            df['DateTime'].dt.day == time_all_dt[i].day
        )]
        if len(tempindex) > 0:
            dateindex.append(tempindex[0])

    if not dateindex:
        return np.array([]), []

    dfsubset = df.iloc[dateindex]
    icecover_hist = dfsubset[prop.ofs.removesuffix('2')].to_numpy()

    # Now do 2D (generated internally)
    num_days = ice_clim.shape[0]
    # Use a leap year (e.g., 2020) if the 2D climatology array has 366 days, otherwise a standard year (2021)
    base_year = 2020 if num_days == 366 else 2021

    # Generate the date range and format as 'M-D' (e.g., '1-1', '12-31') to match the datesstr logic
    date_range = pd.date_range(start=f'{base_year}-01-01', periods=num_days)
    uniq = [f'{d.month}-{d.day}' for d in date_range]

    # Get climatology days that correspond to time_all_dt
    slices = []
    for i in range(0, len(time_all_dt)):
        datesstr = f'{time_all_dt[i].month}-{time_all_dt[i].day}'
        for j in range(0, len(uniq)):
            if datesstr == uniq[j]:
                slices.append(np.array(ice_clim[j, :, :]))

    icecover_hist_2d = np.stack(slices) if len(time_all_dt) == len(slices) else []
    return icecover_hist, icecover_hist_2d


def pair_ice(
    time_m, icecover_m, time_o, icecover_o, prop, logger,
    ice_clim,
):
    """
    Pairs the observed and modeled ice concentration time series, ensuring temporal alignment.

    Args:
        time_m (np.ndarray): Modeled time array.
        icecover_m (np.ndarray): Modeled ice concentration array.
        time_o (np.ndarray): Observed time array.
        icecover_o (np.ndarray): Observed ice concentration array.
        prop (ModelProperties): Object containing model configuration.
        logger (logging.Logger): Logger instance.
        ice_clim (np.ndarray): 3D array of climatology data.

    Returns:
        tuple: Aligned paired arrays and time lists:
               (icecover_o_pair, icecover_m_pair, time_all, time_all_dt, icecover_hist, icecover_hist_2d)
    """
    time_all = []
    time_all_dt = []
    icecover_o_pair = []
    icecover_m_pair = []
    for j in range(0, len(time_o)):
        my_obs_date = pd.to_datetime(time_o[j])

        if prop.ice_dt == 'daily':
            time_all.append(my_obs_date)
            time_all_dt.append(
                datetime.strptime(str(my_obs_date), '%Y-%m-%d %H:%M:%S').date())
            icecover_o_pair.append(np.array(icecover_o[j][:][:]))
            try:
                icecover_m_pair.append(np.array(icecover_m[j][:]))
            except IndexError:
                logger.error('Model and GLSEA ice arrays are different sizes!')
        if prop.ice_dt == 'hourly':
            for i in range(0, len(time_m)):
                my_mod_date = pd.to_datetime(time_m[i])
                if (
                    my_mod_date.day == my_obs_date.day
                    and my_mod_date.year == my_obs_date.year
                    and my_mod_date.month == my_obs_date.month
                ):
                    time_all.append(my_mod_date)
                    time_all_dt.append(
                        datetime.strptime(str(my_mod_date),
                            '%Y-%m-%d %H:%M:%S').date())
                    icecover_o_pair.append(np.array(icecover_o[j][:][:]))
                    icecover_m_pair.append(np.array(icecover_m[i][:]))

    icecover_o_pair = np.stack(icecover_o_pair)
    icecover_m_pair = np.stack(icecover_m_pair)

    # Load climatology (now active for both daily and hourly)
    icecover_hist, icecover_hist_2d = ice_climatology(prop, time_all_dt, ice_clim)

    return icecover_o_pair, icecover_m_pair, time_all, \
        time_all_dt, icecover_hist, icecover_hist_2d


def do_iceskill(prop, logger):
    """
    Main ice skill assessment pipeline function.

    Retrieves observations and model data, spatially and temporally pairs them,
    calculates 1D and 2D skill metrics, generates statistical maps, and exports results to CSV.

    Args:
        prop (ModelProperties): Object containing model configuration, paths, and date ranges.
        logger (logging.Logger): Logger instance.
    """

    _conf = getattr(prop, 'config_file', None)
    if logger is None:
        logger = utils.init_root_logger(
            prop.path, utils.Utils(_conf).get_config_file())

    logger.info('--- Starting ice skill assessment, put on a coat ---')

    dir_params = utils.Utils(_conf).read_config_section('directories', logger)

    # Do forecast_a start and end date reshuffle
    if 'forecast_a' in prop.whichcasts:
        if prop.forecast_hr is None:
            error_message = (
                'prop.forecast_hr is required if prop.whichcast is '
                'forecast_a. Abort!'
            )
            logger.error(error_message)
            sys.exit(-1)
        elif prop.forecast_hr is not None:
            try:
                int(prop.forecast_hr[:-2])
            except ValueError:
                error_message = (
                    f'Please check Forecast Hr format - '
                    f'{prop.forecast_hr}. Abort!'
                )
                logger.error(error_message)
                sys.exit(-1)
            if prop.forecast_hr[-2:] != 'hr':
                error_message = (
                    f'Please check Forecast Hr format - '
                    f'{prop.forecast_hr}. Abort!'
                )
                logger.error(error_message)
                sys.exit(-1)

    # Set filetype to fields (this is not currently used but might be later)
    prop.ofsfiletype = 'fields'

    # Start Date and End Date validation
    try:
        prop.start_date_full_before = prop.start_date_full
        prop.end_date_full_before = prop.end_date_full
        start_date_obj = datetime.strptime(prop.start_date_full, '%Y-%m-%dT%H:%M:%SZ')
        end_date_obj = datetime.strptime(prop.end_date_full, '%Y-%m-%dT%H:%M:%SZ')
    except ValueError:
        error_message = (
            f'Please check Start Date - '
            f'{prop.start_date_full}, End Date - '
            f'{prop.end_date_full}. Abort!'
        )
        logger.error(error_message)
        sys.exit(-1)

    if start_date_obj > end_date_obj:
        error_message = (
            f'End Date {prop.end_date_full} '
            f'is before Start Date {prop.start_date_full}. Abort!'
        )
        logger.error(error_message)
        sys.exit(-1)

    # Make sure home path is set
    if prop.path is None:
        prop.path = dir_params['home']

    # Make sure start and end months are in ice season
    monthlist = ['11', '12', '01', '02', '03', '04', '05']
    if (
        prop.start_date_full.split('-')[1] not in monthlist
        or
        prop.end_date_full.split('-')[1] not in monthlist
    ):
        logger.error(
            'Start and/or end months are not in ice season months %s',
            monthlist,
        )
        sys.exit(-1)

    # Check OFS -- if not Great Lakes, then quit
    ofscheck = ['leofs', 'loofs', 'lmhofs', 'lsofs', 'loofs2']
    if prop.ofs not in ofscheck:
        logger.error(
            "Ice skill can't be run for %s. Input a Great Lakes OFS.",
            prop.ofs,
        )
        sys.exit(-1)

    # prop.path validation
    prop.ofs_extents_path = utils.resolve_asset_path(
        prop.path, dir_params['ofs_extents_dir'],
    )
    if not os.path.exists(prop.ofs_extents_path):
        error_message = (
            f'ofs_extents/ folder is not found. '
            f'Please check prop.path - {prop.path}. Abort!'
        )
        logger.error(error_message)
        sys.exit(-1)

    # prop.ofs validation
    shape_file = f'{prop.ofs_extents_path}/{prop.ofs}.shp'
    if not os.path.isfile(shape_file):
        error_message = (
            f'Shapefile {prop.ofs} is not found at '
            f'the folder {prop.ofs_extents_path}. Abort!'
        )
        logger.error(error_message)
        sys.exit(-1)

    # Daily model average argument verification
    if prop.dailyavg is None:
        prop.dailyavg = False

    # Enforce mutual exclusivity for Target Horizon and Daily Average
    if getattr(prop, 'target_horizon', None) is not None and prop.dailyavg:
        logger.error(
            'The Target Horizon (-th) and Daily Average (-da) arguments '
            'cannot be used together. Please choose one or the other. Au revoir!'
        )
        sys.exit(-1)

    # Directory tree set-up
    # stats csv files
    prop.data_skill_stats_path = os.path.join(
        prop.path, dir_params['data_dir'], dir_params['skill_dir'],
        dir_params['stats_dir'], dir_params['stats_ice_dir'],
    )
    os.makedirs(prop.data_skill_stats_path, exist_ok=True)
    # 1D paired csv files
    prop.data_skill_ice1dpair_path = os.path.join(
        prop.path, dir_params['data_dir'], dir_params['skill_dir'],
        dir_params['1d_ice_pair_dir'],
    )
    os.makedirs(prop.data_skill_ice1dpair_path, exist_ok=True)
    # visuals -- static maps
    prop.visuals_maps_ice_path = os.path.join(
        prop.path, dir_params['data_dir'], dir_params['visual_dir'],
        dir_params['visual_ice_dir'], dir_params['visual_ice_static_maps'],
    )
    os.makedirs(prop.visuals_maps_ice_path, exist_ok=True)
    # visuals -- JSON maps
    prop.visuals_json_maps_ice_path = os.path.join(
        prop.path, dir_params['data_dir'], dir_params['visual_dir'],
        dir_params['visual_ice_dir'], dir_params['visual_ice_json_maps'],
    )
    os.makedirs(prop.visuals_json_maps_ice_path, exist_ok=True)
    # visuals -- 1D time series
    prop.visuals_1d_ice_path = os.path.join(
        prop.path, dir_params['data_dir'], dir_params['visual_dir'],
        dir_params['visual_ice_dir'], dir_params['visual_ice_time_series'],
    )
    os.makedirs(prop.visuals_1d_ice_path, exist_ok=True)
    # visuals -- stats
    prop.visuals_stats_ice_path = os.path.join(
        prop.path, dir_params['data_dir'], dir_params['visual_dir'],
        dir_params['visual_ice_dir'], dir_params['visual_ice_stats'],
    )
    os.makedirs(prop.visuals_stats_ice_path, exist_ok=True)
    # GLSEA analysis
    prop.data_observations_2d_satellite_path = os.path.join(
        prop.path,
        dir_params['data_dir'],
        dir_params['observations_dir'],
        dir_params['2d_satellite_ice_dir'],
    )
    os.makedirs(prop.data_observations_2d_satellite_path, exist_ok=True)
    # concated model save
    prop.data_model_ice_path = os.path.join(
        prop.path, dir_params['data_dir'], dir_params['model_dir'],
        dir_params['model_icesave_dir'],
    )
    os.makedirs(prop.data_model_ice_path, exist_ok=True)
    # Example (local) ice data
    prop.model_path = os.path.join(
        dir_params['model_historical_dir'], prop.ofs, dir_params['netcdf_dir'],
    )

    # Parse whichcasts argument
    prop.whichcasts = prop.whichcasts.replace('[', '').replace(']', '').split(',')

    #########################################################

    # Read variables from configuration
    ice_params = utils.Utils(_conf).read_config_section('ice_skill_settings', logger) or {}

    shouldimakemaps = str(ice_params.get('shouldimakemaps', 'True')).lower() == 'true'
    shouldimakeplots = str(ice_params.get('shouldimakeplots', 'True')).lower() == 'true'
    dailyplotdays = int(ice_params.get('dailyplotdays', 10))
    threshold_exte = float(ice_params.get('threshold_exte', 10))
    stathresh = float(ice_params.get('stathresh', 1))

    # Store output rendering flags in prop to pass to make_ice_map
    prop.makestatic = str(ice_params.get('makestatic', 'False')).lower() == 'true'
    prop.makejson = str(ice_params.get('makejson', 'True')).lower() == 'true'

    # Hardcoded Constants:
    brdr = 0.2                          # margin for plotting
    seasonrun = 'yes'                   # Run for the current ice season?

    # CHECK ice_dt and daily average settings.
    if prop.ice_dt != 'daily' and prop.dailyavg is True:
        prop.dailyavg = False

    # Expand whichcasts if multiple target horizons are provided
    expanded_casts = []
    target_horizons = getattr(prop, 'target_horizon', None)

    for cast in prop.whichcasts:
        if 'forecast' in cast.lower() and isinstance(target_horizons, list):
            for hr in target_horizons:
                expanded_casts.append(f'{cast.lower()}_{hr:03d}h')
        else:
            expanded_casts.append(cast.lower())

    prop.whichcasts = expanded_casts

    ########################################################
    # Loop through whichcasts --> GO!
    for cast in prop.whichcasts:
        prop.whichcast = cast.lower()
        # Adjust start date if using forecast and doing a current season run.
        if prop.whichcast == 'forecast_b' and seasonrun == 'yes':
            prop.oldstartdate = prop.start_date_full
            if (
                (end_date_obj - start_date_obj).days > 60
            ):
                logger.info('Adjusting start date for forecast_b...')
                start_date_forecast = date.today() - timedelta(days=60)
                prop.start_date_full = datetime.strftime(
                    start_date_forecast,
                    '%Y-%m-%dT%H:%M:%SZ',
                )
                logger.info(
                    'Forecast_b start date changed to %s from %s',
                    prop.start_date_full, prop.oldstartdate,
                )
                # Make sure start and end months are in ice season
                if (
                    prop.start_date_full.split('-')[1] not in monthlist
                    or
                    prop.end_date_full.split('-')[1] not in monthlist
                ):
                    logger.error(
                        'Start and/or end months not in ice season %s',
                        monthlist,
                    )
                    sys.exit(-1)

    # -------------------------------------------------------------------------
        # Download, concatenate, and mask/clip satellite and 2D climatology
        obsice, ice_clim = get_icecover_observations.get_icecover_observations(
            prop, logger,
        )
        logger.info('Grabbed ice cover observations')
        # Concatenate existing model output
        icecover_m, lon_m, lat_m, time_m, h_m = get_icecover_model.\
        get_icecover_model(prop, logger)
        logger.info('Grabbed ice cover model output')
    # -------------------------------------------------------------------------

        # -- Read lat, lon and ice cover from GLSEA netCDF file (observations)
        lon_o = np.asarray(obsice.variables['lon'][:])
        lat_o = np.asarray(obsice.variables['lat'][:])
        icecover_o = np.asarray(obsice.variables['ice_concentration'][:, :, :])
        time_o = np.asarray(obsice.variables['time'][:])

        # Tile lat & lon into arrays
        lat_o, lon_o = np.meshgrid(lat_o, lon_o, indexing='ij')
        logger.info('GLSEA parsing complete')

        # First check time arrays for size compatibility
        set_o = set(time_o)
        set_m = set(time_m)
        difference_o_minus_m = set_o - set_m
        difference_m_minus_o = set_m - set_o

        if len(difference_m_minus_o) > 0 or len(difference_o_minus_m) > 0:
            logger.error('Model & GLSEA dates are mismatched! Please '
                         'delete the "data" directory and try again!')

        # Now, pair model output to observations, get master time arrays
        icecover_o, icecover_m, time_all, time_all_dt, icecover_hist, \
            icecover_hist_2d = pair_ice(
                time_m,
                icecover_m,
                time_o,
                icecover_o,
                prop,
                logger,
                ice_clim)
        logger.info('Done with data pairing')

        # Get OFS station inventory, find model & obs lat & lon nearest to
        # stations, extract 1D time series, and save to file for each whichcast
        inventory = find_ofs_ice_stations.find_ofs_ice_stations(
            prop, lon_m, lat_m, lon_o, lat_o, time_all_dt, time_all, icecover_o,
            icecover_m, icecover_hist_2d, logger,
        )
        logger.info('Completed inventory and .int files')

    # -------------------------------------------------------------------------
        # ---> Let's do some statistics now, ok? ok.

        # -- 1D obs & model statistics through time
        ice_1d_stats = {k: [] for k in [
            'mod_meanicecover', 'mod_stdmic', 'mod_extent', 'obs_meanicecover',
            'obs_stdmic', 'obs_extent', 'extent_error', 'r_all', 'rmse_either',
            'rmse_all', 'skill_score', 'csi_all', 'csi_misses', 'csi_falsealarms'
        ]}

        # -- 2D statistics mapped through time
        ice_2d_stats = {k: [] for k in [
            'obsmoddiff_all', 'obs_all', 'mod_all', 'obs_extent_map_all',
            'mod_extent_map_all', 'overlap_map_all', 'falarm_map_all',
            'miss_map_all', 'total_extent'
        ]}

        dayrange = ((time_all_dt[-1] - time_all_dt[0]).days)+1

        # Pre-calculate common mapping objects
        ice_map = Basemap(
            projection='merc',
            resolution='i', area_thresh=1.0,
            llcrnrlon=lon_o.min()-brdr,
            llcrnrlat=lat_o.min()-brdr,
            urcrnrlon=lon_o.max()+brdr,
            urcrnrlat=lat_o.max()+brdr,
        )
        xo, yo = ice_map(lon_o, lat_o)
        xm, ym = ice_map(lon_m, lat_m)

        for i in range(0, len(time_all)):
            logger.info(
                '%s percent complete: %s',
                prop.whichcast,
                np.round((i/dayrange)*100, decimals=0),
            )

            # Extract and interpolate ice arrays
            icecover_o_mask = np.array(icecover_o[i][:][:])
            icecover_m_interp = interp.griddata(
                (xm, ym), np.array(icecover_m[i, :]*100), (xo, yo),
                method='nearest',
            )
            # Apply land mask to interpolated model grid
            icecover_m_mask = icecover_m_interp - (icecover_o_mask*0)

            # Statistics Pre-processing
            logger.info('Stats pre-processing -- applying masks...')

            # Create a shared mask for open water and missing data
            icecover_add = icecover_o_mask + icecover_m_mask
            valid_mask = ~np.isnan(icecover_add)

            # Apply low ice threshold for continuous calculations
            icecover_o_mask2 = np.where((icecover_add >= stathresh) & valid_mask, icecover_o_mask, np.nan)
            icecover_m_mask2 = np.where((icecover_add >= stathresh) & valid_mask, icecover_m_mask, np.nan)

            # Calculate stats
            logger.info('Calculating stats!')

            # Mean ice cover & standard deviation
            ice_1d_stats['mod_meanicecover'].append(np.nanmean(icecover_m_mask))
            ice_1d_stats['mod_stdmic'].append(np.nanstd(icecover_m_mask))
            ice_1d_stats['obs_meanicecover'].append(np.nanmean(icecover_o_mask))
            ice_1d_stats['obs_stdmic'].append(np.nanstd(icecover_o_mask))

            # Percent ice cover (extent)
            mod_valid_count = (icecover_m_mask >= 0).sum()
            obs_valid_count = (icecover_o_mask >= 0).sum()

            mod_extent_pct = ((icecover_m_mask >= threshold_exte).sum() / mod_valid_count) * 100 if mod_valid_count > 0 else 0
            obs_extent_pct = ((icecover_o_mask >= threshold_exte).sum() / obs_valid_count) * 100 if obs_valid_count > 0 else 0

            ice_1d_stats['mod_extent'].append(mod_extent_pct)
            ice_1d_stats['obs_extent'].append(obs_extent_pct)
            ice_1d_stats['r_all'].append(np.nan)

            # RMSE calculations
            if np.nansum(~np.isnan(icecover_m_mask)) >= 2 and np.nansum(~np.isnan(icecover_o_mask)) >= 2:
                rmse_all = nos_metrics.rmse(icecover_m_mask, icecover_o_mask)
                ice_1d_stats['rmse_all'].append(rmse_all)

                # Skill Score
                mse2_fC = np.nanmean((icecover_m_mask - icecover_hist_2d[i, :, :])**2)
                skillscore = 1 - ((rmse_all**2) / mse2_fC) if mse2_fC > 0 else np.nan
                ice_1d_stats['skill_score'].append(skillscore)
            else:
                ice_1d_stats['rmse_all'].append(np.nan)
                ice_1d_stats['skill_score'].append(np.nan)

            if np.nansum(~np.isnan(icecover_m_mask2)) >= 2 and np.nansum(~np.isnan(icecover_o_mask2)) >= 2:
                ice_1d_stats['rmse_either'].append(nos_metrics.rmse(icecover_m_mask2, icecover_o_mask2))
            else:
                ice_1d_stats['rmse_either'].append(np.nan)

            # Daily ice extent & CSI mapping
            obs_extent_map = np.where(icecover_o_mask >= threshold_exte, 1, 0).astype(float)
            mod_extent_map = np.where(icecover_m_mask >= threshold_exte, 1, 0).astype(float)

            # Keep NaNs where data is invalid
            obs_extent_map[~valid_mask] = np.nan
            mod_extent_map[~valid_mask] = np.nan

            ice_2d_stats['obs_extent_map_all'].append(obs_extent_map)
            ice_2d_stats['mod_extent_map_all'].append(mod_extent_map)

            total_extent = np.where((mod_extent_map + obs_extent_map) > 0, 1, 0).astype(float)
            total_extent[~valid_mask] = np.nan
            ice_2d_stats['total_extent'].append(total_extent)

            overlap_map = np.where((mod_extent_map + obs_extent_map) == 2, 1, 0).astype(float)
            overlap_map[~valid_mask] = np.nan
            ice_2d_stats['overlap_map_all'].append(overlap_map)

            csi_map = mod_extent_map - obs_extent_map
            falarm_map = np.where(csi_map == 1, 1, 0).astype(float)
            falarm_map[~valid_mask] = np.nan
            ice_2d_stats['falarm_map_all'].append(falarm_map)

            miss_map = np.where(csi_map == -1, 1, 0).astype(float)
            miss_map[~valid_mask] = np.nan
            ice_2d_stats['miss_map_all'].append(miss_map)

            # 1D CSI metrics
            hits = np.nansum(overlap_map)
            falarms = np.nansum(falarm_map)
            misses = np.nansum(miss_map)
            total_events = hits + falarms + misses

            if total_events > 0:
                ice_1d_stats['csi_all'].append(hits / total_events)
                ice_1d_stats['csi_misses'].append(misses / total_events)
                ice_1d_stats['csi_falsealarms'].append(falarms / total_events)
                ice_1d_stats['extent_error'].append((falarms + misses) / total_events)
            else:
                ice_1d_stats['csi_all'].append(np.nan)
                ice_1d_stats['csi_misses'].append(np.nan)
                ice_1d_stats['csi_falsealarms'].append(np.nan)
                ice_1d_stats['extent_error'].append(np.nan)

            # 2D Diff mapping
            ice_2d_stats['obsmoddiff_all'].append(icecover_m_mask2 - icecover_o_mask2)
            ice_2d_stats['obs_all'].append(icecover_o_mask)
            ice_2d_stats['mod_all'].append(icecover_m_mask)

            # Output daily map
            if shouldimakemaps and (
                (prop.ice_dt == 'hourly' and time_all[i].hour == 12) or
                (prop.ice_dt == 'daily' and (time_all_dt[-1] - time_all_dt[i]).days <= dailyplotdays)
            ):
                mapdata = np.stack((icecover_o_mask2, icecover_m_mask2, ice_2d_stats['obsmoddiff_all'][i]))
                make_ice_map.make_ice_map(prop, lon_o, lat_o, xo, yo, mapdata, time_all[i], 'daily', logger)

            # Final Step: Aggregate 2D Stats
            if i == len(time_all)-1 and dayrange >= 1:
                logger.info('Calculating cumulative 2D maps...')

                # Stack arrays for fast vectorized processing
                obsmoddiff_stack = np.stack(ice_2d_stats['obsmoddiff_all'])
                obs_stack = np.stack(ice_2d_stats['obs_all'])
                mod_stack = np.stack(ice_2d_stats['mod_all'])

                # Generate global masks to hide open water across the entire time period
                ice_mask = np.nanmax(obs_stack + mod_stack, axis=0) >= stathresh
                openwater_mask = np.where(ice_mask, 1, np.nan)
                obs_mask = np.nanmax(obs_stack, axis=0) >= stathresh
                obs_mask = np.where(obs_mask, 1, np.nan)
                mod_mask = np.nanmax(mod_stack, axis=0) >= stathresh
                mod_mask = np.where(mod_mask, 1, np.nan)

                ext_mask = np.nanmax(obs_stack + mod_stack, axis=0) >= threshold_exte
                openwater_ext_mask = np.where(ext_mask, 1, np.nan)

                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', category=RuntimeWarning)

                    # Mean/Min/Max differences
                    ice_2d_stats['obsmoddiff_allmean'] = np.nanmean(obsmoddiff_stack, axis=0) * openwater_mask
                    ice_2d_stats['obsmoddiff_allmax'] = np.nanmax(obsmoddiff_stack, axis=0) * openwater_mask
                    ice_2d_stats['obsmoddiff_allmin'] = np.nanmin(obsmoddiff_stack, axis=0) * openwater_mask

                    ice_2d_stats['obs_allmean'] = np.nanmean(obs_stack, axis=0) * obs_mask
                    ice_2d_stats['mod_allmean'] = np.nanmean(mod_stack, axis=0) * mod_mask
                    # Calculate mean climatology 2D grid
                    if icecover_hist_2d is not None and len(icecover_hist_2d) > 0:
                        ice_2d_stats['clim_allmean'] = np.nanmean(icecover_hist_2d, axis=0)
                    else:
                        ice_2d_stats['clim_allmean'] = np.full(obs_stack[0].shape, np.nan)
                    # RMSE
                    ice_2d_stats['rmse_2d'] = np.sqrt(np.nanmean((obsmoddiff_stack**2), axis=0)) * openwater_mask

                    # Total Ice Days
                    obs_extent_stack = np.stack(ice_2d_stats['obs_extent_map_all'])
                    mod_extent_stack = np.stack(ice_2d_stats['mod_extent_map_all'])

                    ice_2d_stats['obs_icedays_all'] = np.nansum(obs_extent_stack, axis=0) * openwater_ext_mask
                    ice_2d_stats['mod_icedays_all'] = np.nansum(mod_extent_stack, axis=0) * openwater_ext_mask

                    # CSI Map Metrics
                    csi_norm = np.nansum(np.stack(ice_2d_stats['total_extent']), axis=0)

                    # Prevent division by zero warnings
                    csi_norm_safe = np.where(csi_norm == 0, np.nan, csi_norm)

                    ice_2d_stats['hit_map_allsum'] = (np.nansum(np.stack(ice_2d_stats['overlap_map_all']), axis=0) / csi_norm_safe) * 100 * openwater_ext_mask
                    ice_2d_stats['miss_map_allsum'] = (np.nansum(np.stack(ice_2d_stats['miss_map_all']), axis=0) / csi_norm_safe) * 100 * openwater_ext_mask
                    ice_2d_stats['falarm_map_allsum'] = (np.nansum(np.stack(ice_2d_stats['falarm_map_all']), axis=0) / csi_norm_safe) * 100 * openwater_ext_mask

                # Find ice-on and ice-off dates, if doing a season-long run
                if time_all_dt[0].month in [11, 12]:
                    logger.info('Starting ice onset/thaw date-finding routine...')
                    obs_iceon, obs_iceoff = iceonoff(time_all_dt, ice_1d_stats['obs_meanicecover'], logger)
                    mod_iceon, mod_iceoff = iceonoff(time_all_dt, ice_1d_stats['mod_meanicecover'], logger)
                    clim_iceon, clim_iceoff = iceonoff(time_all_dt, icecover_hist, logger)

                    iceondiff = (mod_iceon - obs_iceon).days if (mod_iceon and obs_iceon) else None
                    iceoffdiff = (mod_iceoff - obs_iceoff).days if (mod_iceoff and obs_iceoff) else None

                    logger.info('Writing ice onset/thaw table...')
                    iceonoffall = [
                        [' ', 'Ice onset', 'Ice thaw'],
                        ['Observed', str(obs_iceon), str(obs_iceoff)],
                        ['Modeled', str(mod_iceon), str(mod_iceoff)],
                        ['Climatology', str(clim_iceon), str(clim_iceoff)],
                        ['Model-obs difference (days)', str(iceondiff), str(iceoffdiff)],
                    ]

                    title = os.path.join(prop.data_skill_stats_path, f'skill_{prop.ofs}_iceonoff_{prop.whichcast}.csv')
                    with open(title, 'w', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow(iceonoffall[0])
                        for row in iceonoffall[1:]:
                            writer.writerow(row)
                    logger.info('Ice on/off table saved!')

                # Make 2D stats maps
                if shouldimakemaps:
                    logger.info('Starting all stats maps...')
                    # Generate the Climatology Map
                    make_ice_map.make_ice_map(
                        prop, lon_o, lat_o, xo, yo,
                        np.stack([ice_2d_stats['clim_allmean']]),
                        time_all[i], 'clim', logger
                    )
                    make_ice_map.make_ice_map(
                        prop, lon_o, lat_o, xo, yo,
                        np.stack((ice_2d_stats['obs_allmean'], ice_2d_stats['mod_allmean'], ice_2d_stats['rmse_2d'])),
                        time_all[i], 'rmse means', logger
                    )
                    make_ice_map.make_ice_map(
                        prop, lon_o, lat_o, xo, yo,
                        np.stack((ice_2d_stats['obs_icedays_all'], ice_2d_stats['mod_icedays_all'])),
                        time_all[i], 'extents', logger
                    )
                    make_ice_map.make_ice_map(
                        prop, lon_o, lat_o, xo, yo,
                        np.stack((ice_2d_stats['obsmoddiff_allmean'], ice_2d_stats['obsmoddiff_allmax'], ice_2d_stats['obsmoddiff_allmin'])),
                        time_all[i], 'diff', logger
                    )
                    make_ice_map.make_ice_map(
                        prop, lon_o, lat_o, xo, yo,
                        np.stack((ice_2d_stats['hit_map_allsum'], ice_2d_stats['falarm_map_allsum'], ice_2d_stats['miss_map_allsum'])),
                        time_all[i], 'csi', logger
                    )
                    logger.info('All stats maps complete!')

                if shouldimakeplots:
                    logger.info('Starting histograms...')
                    make_ice_boxplots.make_ice_boxplots(obs_stack, mod_stack, time_all_dt, prop, logger)
                    plt.close('all')
                    logger.info('Box plots complete!')

            elif i == len(time_all)-1 and dayrange < 1:
                logger.info('Day range is < 1, so no maps or cumulative stats!')

        # Before plotting, make pandas dataframe with stats time series
        logger.info('Compiling time series stats for table output...')
        pd.DataFrame({
            'time_all_dt': time_all_dt,
            'obs_meanicecover': ice_1d_stats['obs_meanicecover'],
            'mod_meanicecover': ice_1d_stats['mod_meanicecover'],
            'obs_stdmic': ice_1d_stats['obs_stdmic'],
            'mod_stdmic': ice_1d_stats['mod_stdmic'],
            'icecover_hist': icecover_hist,
            'SS': ice_1d_stats['skill_score'],
            'rmse_all': ice_1d_stats['rmse_all'],
            'rmse_either': ice_1d_stats['rmse_either'],
            'obs_extent': ice_1d_stats['obs_extent'],
            'mod_extent': ice_1d_stats['mod_extent'],
            'r_all': ice_1d_stats['r_all'],
            'csi_all': ice_1d_stats['csi_all'],
            'csi_falsealarms': ice_1d_stats['csi_falsealarms'],
            'csi_misses': ice_1d_stats['csi_misses'],
            'extent_error': ice_1d_stats['extent_error'],
        }).to_csv(
            os.path.join(prop.data_skill_stats_path, f'skill_{prop.ofs}_icestatseries_{prop.whichcast}.csv')
        )

        logger.info('Time series of OFS-wide skill stats is created successfully.')

        # Switch date back to user-inputted start date if doing a season run
        if seasonrun == 'yes' and prop.whichcast == 'forecast_b':
            prop.start_date_full = prop.oldstartdate

    if shouldimakeplots:
        if dayrange >= 2:
            create_1dplot_ice.create_1dplot_icestats(prop, time_all, logger)
            create_1dplot_ice.create_1dplot_ice(prop, inventory, time_all, logger)
        else:
            logger.info('Need >=5 days for stats & 1D time series plots! No 1D plots made.')

    logger.info('Program complete! Go get coffee and bagel')


# Execution:
def _run_pipeline(run_args):
    """
    Execute the ice skill pipeline with the given command-line arguments.

    Args:
        run_args (argparse.Namespace): Parsed command-line arguments.
    """
    prop1 = model_properties.ModelProperties()
    ofs = getattr(run_args, 'OFS', None) or getattr(run_args, 'ofs', None)
    prop1.ofs = ofs.lower()
    prop1.path = getattr(run_args, 'Path', None) or getattr(run_args, 'path', None)
    prop1.config_file = getattr(run_args, 'config', None)
    prop1.start_date_full = run_args.StartDate_full
    prop1.end_date_full = run_args.EndDate_full
    whichcasts = getattr(run_args, 'Whichcasts', None)
    prop1.whichcasts = ','.join(whichcasts) if isinstance(
        whichcasts, list) else whichcasts
    prop1.target_horizon = getattr(run_args, 'Target_Horizon', None)
    prop1.ice_dt = str(getattr(run_args, 'TimeStep', 'daily'))
    prop1.dailyavg = getattr(run_args, 'DailyAverage', False)
    # Map the combined neighborhood flag and search radius
    avg_radius = getattr(run_args, 'Neighborhood_Average', None)
    if avg_radius is not None and avg_radius > 0:
        prop1.Neighborhood_Average = True
        prop1.Search_Radius = avg_radius
    else:
        prop1.Neighborhood_Average = False
        prop1.Search_Radius = 0
    prop1.model_source = model_source.model_source(prop1.ofs)

    do_iceskill(prop1, None)


def main(argv=None):
    """
    Entry point for the do-iceskill console script.

    Parses command line arguments and routes execution to the main pipeline
    or a GUI if no OFS is specified.

    Args:
        argv (list, optional): Command line arguments. Defaults to sys.argv.
    """
    parser = argparse.ArgumentParser(prog='do-iceskill')
    parser.add_argument(
        '-o',
        '--ofs',
        required=False,
        default=None,
        help="""Choose from the list in the
        ofs_extents/ folder, you can also create your own shapefile,
        add it at the ofs_extents/ folder and call it here""", )
    parser.add_argument(
        '-p',
        '--path',
        required=False,
        help='Inventory File path where ofs_extents folder is located', )
    parser.add_argument(
        '-s',
        '--StartDate_full',
        required=False,
        help="Start Date_full YYYY-MM-DDThh:mm:ssZ e.g.'2023-01-01T12:34:00Z'",
    )
    parser.add_argument(
        '-e',
        '--EndDate_full',
        required=False,
        help="End Date_full YYYY-MM-DDThh:mm:ssZ e.g. '2023-01-01T12:34:00Z'",
    )
    parser.add_argument(
        '-ws',
        '--Whichcasts',
        required=False,
        help="whichcasts: 'nowcast', 'forecast_b', 'hindcast'", )
    parser.add_argument(
        '-da',
        '--DailyAverage',
        action='store_true',
        help='Use a daily average model output instead of single hour; True '
        'or False (False is default)', )
    parser.add_argument(
        '-ts',
        '--TimeStep',
        required=False,
        default='daily',
        help='Set assessment time step: hourly or daily (daily is default)', )
    parser.add_argument(
        '-c', '--config',
        help='Path to configuration file (default: conf/ofs_dps.conf)')
    parser.add_argument(
        '-avg',
        '--Neighborhood_Average',
        type=float,
        default=None,
        metavar='RADIUS_METERS',
        help='Enable neighborhood averaging and set the search radius in meters. If not provided, neighborhood averaging is off.', )
    parser.add_argument(
        '-th',
        '--Target_Horizon',
        type=int,
        nargs='+',  # Accepts multiple inputs: -th 24 48 120
        required=False,
        default=None,
        help='Set specific forecast lead times in hours (e.g., 24 48 120; min = 1, max = 120) to extract.', )
    args = parser.parse_args(argv)

    if args.ofs is None:
        from ofs_skill.visualization import create_gui_ice
        create_gui_ice.create_gui_ice(runner=_run_pipeline)
    else:
        _run_pipeline(args)


if __name__ == '__main__':
    main()
