"""
Created on Tue Jul 30 10:01:36 2024

@author: PL

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
    Calculate the great circle distance between two points
    or arrays of points on the earth (specified in decimal degrees)
    """
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371000 # Radius of earth in meters
    return c * r

def find_ofs_ice_stations(
    prop, lon_m, lat_m, lon_o, lat_o, time_all_dt, time_all, icecover_o,
    icecover_m, icecover_hist_2d, logger,
):
    '''
    This function spatially matches observation stations used in the main 1D
    skill assessment to ice model grid cells, and extracts paired obs and model
    time series at that location. The paired series is then written to file,
    along with a stats table for each station location and an inventory file
    listing observation stations used in the analysis.
    '''

    start_date_full = prop.start_date_full
    end_date_full = prop.end_date_full
    path = prop.path
    ofs = prop.ofs

    # This is adding +- 3 days to make sure when the data is sliced
    # it has data from beginning to end
    start_date_full = start_date_full.replace('-', '')
    end_date_full = end_date_full.replace('-', '')
    start_date_full = start_date_full.replace('Z', '')
    end_date_full = end_date_full.replace('Z', '')
    start_date = start_date_full.split('T')[0]
    end_date = end_date_full.split('T')[0]
    start_date_full = start_date_full.replace('T', '-')
    end_date_full = end_date_full.replace('T', '-')

    start_dt = datetime.strptime(start_date, '%Y%m%d') - timedelta(days=3)
    end_dt = datetime.strptime(end_date, '%Y%m%d') + timedelta(days=3)

    start_date = start_dt.strftime('%Y%m%d')
    end_date = end_dt.strftime('%Y%m%d')

    _conf = getattr(prop, 'config_file', None)
    dir_params = utils.Utils(_conf).read_config_section('directories', logger)

    control_files_path = os.path.join(
        path, dir_params['control_files_dir'],
    )
    os.makedirs(control_files_path, exist_ok=True)

    # This part of the script will load the inventory file, if the
    # inventory file is not found it will then create a new one by running the
    # ofs_inventory function.
    try:
        inventory = pd.read_csv(
            r'' + f'{control_files_path}/inventory_all_{ofs}.csv',
        )
        logger.info(
            'Inventory file (inventory_all_%s.csv) found in "%s/". If you '
            'instead want to create a new Inventory file, please change '
            'the name/delete the current inventory_all_%s.csv', ofs,
            control_files_path, ofs,
        )
    except FileNotFoundError:
        try:
            logger.info(
                'Inventory file not found. Creating Inventory file!. '
                'This might take a couple of minutes',
            )
            ofs_inventory_stations(
                ofs, start_date, end_date, path, ['co-ops', 'ndbc', 'usgs', 'chs'],
                logger,
                config_file=_conf,
            )
            inventory = pd.read_csv(
                r'' + f'{control_files_path}/inventory_all_{ofs}.csv',
            )
            logger.info('Inventory file created successfully')
        except Exception as ex:
            logger.error(
                'Errors happened when creating inventory files -- %s.',
                str(ex), )
            raise Exception(
                'Errors happened when creating inventory files',
            ) from ex
    ###
    # Done with station inventory, now find nearest model node.
    # First organize model, observed, and station lat & lon and reshape.
    # We are using raw model output, not interpolated model output!
    ###
    stationlonlat = np.array(inventory[['X', 'Y']])
    modellonlat = np.array([lon_m, lat_m])
    modellonlat = modellonlat.T
    lon_o_flat = lon_o.flatten()
    lat_o_flat = lat_o.flatten()
    obslonlat = np.array([lon_o_flat, lat_o_flat])
    obslonlat = obslonlat.T
    ###
    # Start looping through station coords and finding the minimum distance
    # between station (target) --> model and station (target) --> observed.
    ###
    mod_rowcol = []
    lon_index = []
    lat_index = []
    mod_xy = []
    # xm = []
    # ym = []
    for i in range(len(stationlonlat)):
        target = stationlonlat[i, :]
        moddistances = np.linalg.norm(
            modellonlat-target,
            axis=1,
        )
        obsdistances = np.linalg.norm(
            obslonlat-target,
            axis=1,
        )
        if np.min(moddistances) <= 0.01 and np.min(obsdistances) <= 0.01:
            mod_rowcol.append(np.argmin(moddistances))
            mod_xy.append(modellonlat[np.argmin(moddistances), :])
            obs_mindist = np.argmin(obsdistances)
            obs_targetlatlon = obslonlat[obs_mindist, :]
            lontemp = lon_o[0, :]
            lattemp = lat_o[:, 0]
            lon_index_value = (
                list(np.where(lontemp == obs_targetlatlon[0])[0]))
            lat_index_value = (
                list(np.where(lattemp == obs_targetlatlon[1])[0]))
            # mod_rowcol.append(modellonlat[min_index,:])

            if sum(~isnan(icecover_o[:, lat_index_value, lon_index_value])) > 0:
                ###
                # If values in series (it's not all NaNs), use it
                ###
                lon_index.append(lon_index_value)
                lat_index.append(lat_index_value)

            elif sum(~isnan(icecover_o[:,
                                       lat_index_value,
                                       lon_index_value])) == 0:
                ###
                # If all NaNs, look for next closest observation location--but
                # still within 0.01 threshold distance
                ###
                while sum(
                    ~isnan
                    (icecover_o[:, lat_index_value, lon_index_value]),
                ) == 0:

                    # Some arbitrarily big number
                    obsdistances[obs_mindist] = 100000
                    if np.min(obsdistances) <= 0.05:
                        obs_mindist = np.argmin(obsdistances)
                        obs_targetlatlon = obslonlat[obs_mindist, :]
                        lontemp = lon_o[0, :]
                        lattemp = lat_o[:, 0]
                        lon_index_value = (
                            list(np.where(lontemp == obs_targetlatlon[0])[0]))
                        lat_index_value = (
                            list(np.where(lattemp == obs_targetlatlon[1])[0]))
                        if sum(~isnan(icecover_o[:,
                                                 lat_index_value,
                                                 lon_index_value])) > 0:
                            lon_index.append(lon_index_value)
                            lat_index.append(lat_index_value)
                            break
                    else:
                        # mod_rowcol.append(-999)
                        lon_index.append([-999])
                        lat_index.append([-999])
                        break
        else:
            mod_xy.append(-999)
            mod_rowcol.append(-999)
            lon_index.append([-999])
            lat_index.append([-999])

    ###
    # Reorganize lat/lon indices output from for-loop
    ###
    lon_index_np = np.array(lon_index)
    lat_index_np = np.array(lat_index)
    lon_index_np = np.reshape((lon_index_np), len(lon_index_np))
    lat_index_np = np.reshape((lat_index_np), len(lat_index_np))
    obs_rowcol = np.array([lat_index_np, lon_index_np])
    obs_rowcol = obs_rowcol.T
    mod_rowcol = np.stack(mod_rowcol)

    ###
    # Do pseudo-ensemble/neighborhood methods by getting area of model output
    # around GLSEA analysis grid cell.
    ###
    # Use a flag from 'prop' to activate, defaulting to False.
    # search_radius should now represent a distance in meters (e.g., 500)
    use_neighborhood = getattr(prop, 'Neighborhood_Average', False)
    search_radius = getattr(prop, 'Search_Radius', 500) if use_neighborhood else 0

    ###
    # Stats and plotting below
    ###
    r_stations = []
    rmse_stations = []
    bias_stations = []
    bias_stdev_stations = []
    mod_mean_stations = []
    obs_mean_stations = []
    inventory_all = []
    mod_rowcol_all = []
    source_all = []
    xo_all = []
    yo_all = []
    xm_all = []
    ym_all = []
    name_all = []
    for i in range(len(inventory['Name'])):
        if obs_rowcol[i, 0] != -999 and obs_rowcol[i, 1] != -999 and \
            mod_rowcol[i] != -999:
            inventory_all.append(inventory['ID'].at[i])
            source_all.append(inventory['Source'].at[i])
            mod_rowcol_all.append(mod_rowcol[i])
            xo_all.append(inventory['X'].at[i])
            yo_all.append(inventory['Y'].at[i])
            xm_all.append(mod_xy[i][0])
            ym_all.append(mod_xy[i][1])
            name_all.append(inventory['Name'].at[i])

            # Extract cell-to-cell time series
            obs_series = np.array(
                icecover_o[:, obs_rowcol[i, 0], obs_rowcol[i, 1]])
            mod_series = np.array(icecover_m[:, mod_rowcol[i]]*100)
            clim_series = np.array(
                icecover_hist_2d[:, obs_rowcol[i, 0], obs_rowcol[i, 1]])
            # Initialize standard deviation arrays
            mod_std = np.array(mod_series*0)
            obs_std = np.array(obs_series*0)
            ###
            # Apply search radius using Origin Lat/Lon in Meters
            ###
            if search_radius > 0:
                # --- Model Neighborhood by Haversine Distance ---
                mod_lon_origin = modellonlat[mod_rowcol[i], 0]
                mod_lat_origin = modellonlat[mod_rowcol[i], 1]

                # Calculate distance from origin to all model nodes in meters
                moddistances = haversine_distance_meters(
                    mod_lon_origin, mod_lat_origin,
                    modellonlat[:, 0], modellonlat[:, 1]
                )

                # Find all model nodes within the search_radius (in meters)
                modindex = np.where(moddistances <= search_radius)[0]

                if len(modindex) > 0:
                    ens_mod_series = np.array(icecover_m[:, modindex] * 100)
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', category=RuntimeWarning)
                        mod_series = np.nanmean(ens_mod_series, axis=1)
                        mod_std = np.nanstd(ens_mod_series, axis=1)

                # --- Observation Neighborhood by Haversine Distance ---
                obs_lat_origin = lat_o[obs_rowcol[i, 0], 0]
                obs_lon_origin = lon_o[0, obs_rowcol[i, 1]]

                obsdistances = haversine_distance_meters(
                    obs_lon_origin, obs_lat_origin,
                    obslonlat[:, 0], obslonlat[:, 1]
                )

                # Find all observation grid points within the search_radius (in meters)
                obsindex_flat = np.where(obsdistances <= search_radius)[0]

                if len(obsindex_flat) > 0:
                    icecover_o_flat = icecover_o.reshape(icecover_o.shape[0], -1)
                    ens_obs_series = icecover_o_flat[:, obsindex_flat]

                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', category=RuntimeWarning)
                        obs_series = np.nanmean(ens_obs_series, axis=1)
                        obs_std = np.nanstd(ens_obs_series, axis=1)

            # Calculate stats
            badnans = ~isnan(obs_series) * ~isnan(mod_series)
            obs_series_nan = obs_series[badnans]
            mod_series_nan = mod_series[badnans]
            # R
            if len(obs_series_nan) > 5:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    r_value = nos_metrics.pearson_r(mod_series_nan, obs_series_nan)
                r_stations.append(np.round(r_value, decimals=2))
            else:
                r_value = np.nan
                r_stations.append(r_value)
            # RMSE
            rmse_value = np.round(
                nos_metrics.rmse(mod_series, obs_series), 2,
            )
            rmse_stations.append(rmse_value)
            # Mean bias
            bias_value = np.round(nos_metrics.mean_bias(mod_series - obs_series), 2)
            bias_stations.append(bias_value)
            # bias st dev
            bias_stdev = np.round(nos_metrics.standard_deviation(mod_series - obs_series), 2)
            bias_stdev_stations.append(bias_stdev)
            # Model & obs means
            mod_mean = np.round(np.nanmean(mod_series), 2)
            obs_mean = np.round(np.nanmean(obs_series), 2)
            mod_mean_stations.append(mod_mean)
            obs_mean_stations.append(obs_mean)

            # Write model and obs 1D time series to file
            if (
                len(mod_rowcol) != 0
                and len(inventory['ID']) != 0
                and len(obs_mean_stations) != 0
                and len(mod_mean_stations) != 0
                and len(rmse_stations) != 0
                and len(r_stations) != 0
                and len(bias_stations) != 0
                and len(bias_stdev_stations) != 0
            ):
                pd.DataFrame(
                    {
                        'DateTime': time_all_dt,
                        'OBS': obs_series,
                        'OFS': mod_series,
                        'OBS_STDEV': obs_std,
                        'OFS_STDEV': mod_std,
                        'CLIM': clim_series,
                    },
                ).to_csv(
                    r'' + f'{prop.data_skill_ice1dpair_path}/'
                          f'{prop.ofs}_'
                          f'iceconc_'
                          f"{inventory.at[i,'ID']}_"
                          f'{mod_rowcol[i]}_'
                          f'{prop.whichcast}_pair.int',
                )

                logger.info(
                    'Paired data for %s %s station %s ice concentration '
                    'is created successfully.',
                    prop.ofs,
                    prop.whichcast,
                    f"{inventory.at[i,'ID']}",
                )
            else:
                logger.error(
                    'Paired data for %s %s ice concentration '
                    'is NOT created successfully. Row length = 0!',
                    prop.ofs,
                    prop.whichcast,
                )
                sys.exit(-1)

    # Write stats & ice inventory tables -- do some checks first
    if (
        len(mod_rowcol) != 0
        and len(inventory['ID']) != 0
        and len(obs_mean_stations) != 0
        and len(mod_mean_stations) != 0
        and len(rmse_stations) != 0
        and len(r_stations) != 0
        and len(bias_stations) != 0
        and len(bias_stdev_stations) != 0
    ):
        pd.DataFrame(
            {
                'ID': inventory_all,
                'NODE': mod_rowcol_all,
                'mean_obs_ice_conc': obs_mean_stations,
                'mean_model_ice_conc': mod_mean_stations,
                'bias': bias_stations,
                'bias_standard_dev': bias_stdev_stations,
                'rmse': rmse_stations,
                'r': r_stations,
                'start_date': time_all_dt[0],
                'end_date': time_all_dt[-1],
            },
        ).to_csv(
            r'' + f'{prop.data_skill_stats_path}/'
                  f'skill_{prop.ofs}_'
            f'iceconc_{prop.whichcast}.csv',
        )
        logger.info(
            'Summary skill table for prop.ofs %s ice concentration '
            'is created successfully.',
            prop.ofs,
        )

        # Ice inventory
        iceinv = pd.DataFrame(
            {
                'ID': inventory_all,
                'NODE': mod_rowcol_all,
                'Name': name_all,
                'Source': source_all,
                'Ice_Source': 'NOAA CoastWatch/National Ice Center',
                'X': xo_all,
                'Y': yo_all,
                'X_mod': xm_all,
                'Y_mod': ym_all,
            },
        )
        iceinv.to_csv(
            r'' + f'{control_files_path}/'
                  f'inventory_all_ice_{ofs}.csv',
        )
        logger.info(
            'Ice inventory table for prop.ofs %s '
            'is created successfully.',
            prop.ofs,
        )
    else:
        logger.error(
            'Summary skill & inventory tables for %s ice '
            'is NOT created successfully -- columns have length of zero!',
            prop.ofs,
        )

    return iceinv
