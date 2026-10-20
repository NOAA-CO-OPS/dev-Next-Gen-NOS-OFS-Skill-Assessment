"""
Created on Tue Jul 30 10:01:36 2024
@author: PL

Spatially matches observation stations used in 1D skill assessment to ice model grid cells.
Extracts paired observation and model time series, writes the pairings to files,
and generates summary metric tables.
"""
from __future__ import annotations

import os
import sys
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from numpy import isnan

from ofs_skill.obs_retrieval import utils
from ofs_skill.obs_retrieval.ofs_inventory_stations import ofs_inventory_stations
from ofs_skill.skill_assessment import nos_metrics


def haversine_distance_meters(lon1, lat1, lon2, lat2):
    """
    Calculates the great-circle distance between two points (or arrays of points)
    on a sphere given their longitudes and latitudes.

    Args:
        lon1, lat1: Longitude/latitude of the first point in decimal degrees.
        lon2, lat2: Longitude/latitude of the second point in decimal degrees.

    Returns:
        float or np.ndarray: Distance between coordinates in meters.
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    # Haversine formula core
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 2 * np.arcsin(np.sqrt(a)) * 6371000

def find_ofs_ice_stations(prop, lon_m, lat_m, lon_o, lat_o, time_all_dt, time_all,
                          icecover_o, icecover_m, icecover_hist_2d, logger):
    """
    Spatially matches observation stations to model nodes, extracts time series data,
    calculates summary metrics (RMSE, Bias, Pearson R), and writes output to file.

    Args:
        prop (object): Properties object with configuration details and run paths.
        lon_m, lat_m (np.ndarray): Model longitude and latitude arrays.
        lon_o, lat_o (np.ndarray): Observation longitude and latitude arrays.
        time_all_dt (list): List of evaluated datetime objects.
        time_all (list): List of evaluated time strings.
        icecover_o, icecover_m (np.ndarray): Observation and modeled ice cover matrices.
        icecover_hist_2d (np.ndarray): Historical climatology matrix for the region.
        logger (logging.Logger): Logger for execution tracing.

    Returns:
        pd.DataFrame: Final ice station inventory dataframe.
    """

    # Pad start and end dates by 3 days to guarantee full temporal coverage during slicing
    start_date = datetime.strptime(prop.start_date_full[:10], '%Y-%m-%d')
    end_date = datetime.strptime(prop.end_date_full[:10], '%Y-%m-%d')
    start_str = (start_date - timedelta(days=3)).strftime('%Y%m%d')
    end_str = (end_date + timedelta(days=3)).strftime('%Y%m%d')

    # Load configuration and establish control output directories
    _conf = getattr(prop, 'config_file', None)
    dir_params = utils.Utils(_conf).read_config_section('directories', logger)
    control_path = os.path.join(prop.path, dir_params['control_files_dir'])
    os.makedirs(control_path, exist_ok=True)
    inv_file = os.path.join(control_path, f'inventory_all_{prop.ofs}.csv')

    # Load existing station inventory or generate a new one if missing
    try:
        inventory = pd.read_csv(inv_file)
        logger.info(f'Inventory file found at {inv_file}')
    except FileNotFoundError:
        logger.info('Inventory file not found. Creating new inventory file...')
        try:
            ofs_inventory_stations(prop.ofs, start_str, end_str, prop.path,
                                   ['co-ops', 'ndbc', 'usgs', 'chs'], logger, config_file=_conf)
            inventory = pd.read_csv(inv_file)
        except Exception as ex:
            logger.error(f'Error creating inventory: {ex}')
            raise

    # Organize spatial coordinate structures for minimum distance searching
    stationlonlat = inventory[['X', 'Y']].values
    modellonlat = np.column_stack((lon_m, lat_m))
    obslonlat = np.column_stack((lon_o.flatten(), lat_o.flatten()))

    mod_rowcol, mod_xy, lon_index, lat_index = [], [], [], []

    # Loop through target stations to find closest model and observation nodes
    for target in stationlonlat:
        mod_dist = np.linalg.norm(modellonlat - target, axis=1)
        obs_dist = np.linalg.norm(obslonlat - target, axis=1)

        mod_min, obs_min = np.min(mod_dist), np.min(obs_dist)

        # Enforce strict 0.01 degree distance threshold for direct matches
        if mod_min <= 0.01 and obs_min <= 0.01:
            m_idx = np.argmin(mod_dist)
            mod_rowcol.append(m_idx)
            mod_xy.append(modellonlat[m_idx])

            o_idx = np.argmin(obs_dist)
            obs_target = obslonlat[o_idx]

            lon_idx_val = list(np.where(lon_o[0, :] == obs_target[0])[0])
            lat_idx_val = list(np.where(lat_o[:, 0] == obs_target[1])[0])

            # Fallback logic: If target observation node has only NaN data, find next closest valid node
            # within a relaxed 0.05 degree search radius
            while sum(~isnan(icecover_o[:, lat_idx_val, lon_idx_val])) == 0:
                obs_dist[o_idx] = 100000 # Invalidate current empty node
                if np.min(obs_dist) <= 0.05:
                    o_idx = np.argmin(obs_dist)
                    obs_target = obslonlat[o_idx]
                    lon_idx_val = list(np.where(lon_o[0, :] == obs_target[0])[0])
                    lat_idx_val = list(np.where(lat_o[:, 0] == obs_target[1])[0])
                else:
                    lon_idx_val, lat_idx_val = [-999], [-999]
                    break

            lon_index.append(lon_idx_val)
            lat_index.append(lat_idx_val)
        else:
            mod_rowcol.append(-999)
            mod_xy.append([-999, -999])
            lon_index.append([-999])
            lat_index.append([-999])

    # Finalize index stacks
    obs_rowcol = np.column_stack((np.concatenate(lat_index), np.concatenate(lon_index)))
    mod_rowcol = np.array(mod_rowcol)

    # Initialize optional neighborhood average parameters (extracting a cluster of points)
    search_radius = getattr(prop, 'Search_Radius', 500) if getattr(prop, 'Neighborhood_Average', False) else 0

    # Output storage dictionary for metrics
    results = {key: [] for key in ['r', 'rmse', 'bias', 'bias_std', 'mod_mean', 'obs_mean',
                                   'inv_id', 'mod_node', 'src', 'x', 'y', 'xm', 'ym', 'name']}

    # Extract time series and compute skill assessment metrics for matched stations
    for i, row in inventory.iterrows():
        # Skip failed matches
        if obs_rowcol[i, 0] == -999 or obs_rowcol[i, 1] == -999 or mod_rowcol[i] == -999:
            continue

        results['inv_id'].append(row['ID'])
        results['src'].append(row['Source'])
        results['mod_node'].append(mod_rowcol[i])
        results['x'].append(row['X'])
        results['y'].append(row['Y'])
        results['xm'].append(mod_xy[i][0])
        results['ym'].append(mod_xy[i][1])
        results['name'].append(row['Name'])

        # Isolate individual series
        obs_series = icecover_o[:, obs_rowcol[i, 0], obs_rowcol[i, 1]]
        mod_series = icecover_m[:, mod_rowcol[i]] * 100
        clim_series = icecover_hist_2d[:, obs_rowcol[i, 0], obs_rowcol[i, 1]]
        mod_std = np.zeros_like(mod_series)
        obs_std = np.zeros_like(obs_series)

        # Apply spatial neighborhood averaging if triggered
        if search_radius > 0:
            m_dist = haversine_distance_meters(modellonlat[mod_rowcol[i], 0], modellonlat[mod_rowcol[i], 1], modellonlat[:, 0], modellonlat[:, 1])
            mod_idx = np.where(m_dist <= search_radius)[0]
            if len(mod_idx) > 0:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', category=RuntimeWarning)
                    ens_mod = icecover_m[:, mod_idx] * 100
                    mod_series, mod_std = np.nanmean(ens_mod, axis=1), np.nanstd(ens_mod, axis=1)

            o_dist = haversine_distance_meters(lon_o[0, obs_rowcol[i, 1]], lat_o[obs_rowcol[i, 0], 0], obslonlat[:, 0], obslonlat[:, 1])
            obs_idx = np.where(o_dist <= search_radius)[0]
            if len(obs_idx) > 0:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', category=RuntimeWarning)
                    ens_obs = icecover_o.reshape(icecover_o.shape[0], -1)[:, obs_idx]
                    obs_series, obs_std = np.nanmean(ens_obs, axis=1), np.nanstd(ens_obs, axis=1)

        # Strip unpairable NaNs prior to correlation math
        valid = ~isnan(obs_series) & ~isnan(mod_series)
        o_val, m_val = obs_series[valid], mod_series[valid]

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            # Calculate Pearson correlation (R) only if a reliable length exists
            r_val = nos_metrics.pearson_r(m_val, o_val) if len(o_val) > 5 else np.nan

        # Collect summary statistics
        results['r'].append(np.round(r_val, 2))
        results['rmse'].append(np.round(nos_metrics.rmse(mod_series, obs_series), 2))
        results['bias'].append(np.round(nos_metrics.mean_bias(mod_series - obs_series), 2))
        results['bias_std'].append(np.round(nos_metrics.standard_deviation(mod_series - obs_series), 2))
        results['mod_mean'].append(np.round(np.nanmean(mod_series), 2))
        results['obs_mean'].append(np.round(np.nanmean(obs_series), 2))

        # Write explicit localized timeseries for each station to an intermediate file
        out_csv = os.path.join(prop.data_skill_ice1dpair_path, f"{prop.ofs}_iceconc_{row['ID']}_{mod_rowcol[i]}_{prop.whichcast}_pair.int")
        pd.DataFrame({'DateTime': time_all_dt, 'OBS': obs_series, 'OFS': mod_series,
                      'OBS_STDEV': obs_std, 'OFS_STDEV': mod_std, 'CLIM': clim_series}).to_csv(out_csv)
        logger.info(f"Paired data for {prop.ofs} {prop.whichcast} station {row['ID']} created.")

    # Generate overarching summary tables
    if len(results['inv_id']) > 0:
        pd.DataFrame({
            'ID': results['inv_id'], 'NODE': results['mod_node'], 'mean_obs_ice_conc': results['obs_mean'],
            'mean_model_ice_conc': results['mod_mean'], 'bias': results['bias'], 'bias_standard_dev': results['bias_std'],
            'rmse': results['rmse'], 'r': results['r'], 'start_date': time_all_dt[0], 'end_date': time_all_dt[-1]
        }).to_csv(os.path.join(prop.data_skill_stats_path, f'skill_{prop.ofs}_iceconc_{prop.whichcast}.csv'))

        iceinv = pd.DataFrame({
            'ID': results['inv_id'], 'NODE': results['mod_node'], 'Name': results['name'], 'Source': results['src'],
            'Ice_Source': 'NOAA CoastWatch/National Ice Center', 'X': results['x'], 'Y': results['y'],
            'X_mod': results['xm'], 'Y_mod': results['ym']
        })
        iceinv.to_csv(os.path.join(control_path, f'inventory_all_ice_{prop.ofs}.csv'))
        logger.info(f'Summary tables for {prop.ofs} created successfully.')
        return iceinv
    else:
        logger.error(f'Summary tables failed: columns have 0 length for {prop.ofs}.')
        sys.exit(-1)
