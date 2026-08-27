"""
This is the entry point for the final model 1d extraction function,
it opens the path and looks for the model ctl file,if model ctl file is
found, then the script uses it for extracting the model timeseries if
model ctl file is not found, all the predefined function for finding the
nearest node and depth are applied and a new model ctl file is created along
with the time series
"""

import argparse

from ofs_skill.model_processing import model_properties
from ofs_skill.model_processing.get_node_ofs import get_node_ofs
from ofs_skill.model_processing.model_source import get_model_source


def _run_pipeline(run_args):
    """Execute model time-series extraction with the given arguments."""
    prop1 = model_properties.ModelProperties()
    prop1.ofs = run_args.OFS.lower()
    prop1.path = run_args.Path
    prop1.config_file = getattr(run_args, 'config', None)
    prop1.start_date_full = run_args.StartDate_full
    prop1.end_date_full = run_args.EndDate_full

    whichcasts = getattr(run_args, 'Whichcasts', None) or getattr(
        run_args, 'Whichcast', 'nowcast')
    if isinstance(whichcasts, list):
        prop1.whichcast = ','.join(whichcasts)
    else:
        prop1.whichcast = whichcasts

    prop1.datum = getattr(run_args, 'Datum', 'MLLW').upper()
    prop1.model_source = get_model_source(run_args.OFS)
    prop1.ofsfiletype = getattr(run_args, 'OFS_Filetype', None) or getattr(
        run_args, 'FileType', 'stations')
    prop1.horizonskill = getattr(run_args, 'Horizon_Skill', False)

    forecast_hr = getattr(run_args, 'Forecast_Hr', -999)
    prop1.forecast_hr = str(forecast_hr) if forecast_hr != -999 else '00hr'

    var_sel = getattr(run_args, 'Var_Selection', None)
    if var_sel is None:
        prop1.var_list = 'water_level,water_temperature,salinity,currents'
    elif isinstance(var_sel, list):
        prop1.var_list = ','.join(var_sel)
    else:
        prop1.var_list = var_sel

    prop1.user_input_location = getattr(run_args, 'UserInput', False) or getattr(
        run_args, 'User_Input', False)

    # Station owner filter, used when build-ctl-only mode auto-fetches
    # observations. Mirrors get-station-observations: comma-separated list
    # of providers, default all.
    owners = getattr(run_args, 'Station_Owner', None)
    if owners is None:
        prop1.stationowner = 'co-ops,ndbc,usgs,chs'
    elif isinstance(owners, list):
        prop1.stationowner = ','.join(owners)
    else:
        prop1.stationowner = owners.lower()

    # Build-ctl-only mode (issue #189): write the model control file(s) and
    # a station-distance report/map, then stop before extracting time
    # series. Lets users inspect and hand-edit obs-model matches first.
    prop1.build_ctl_only = getattr(run_args, 'Build_Ctl_Only', False)
    prop1.build_ctl_map = not getattr(run_args, 'No_Map', False)

    if 'l' in prop1.ofs[0] and prop1.datum == 'MLLW':
        prop1.datum = 'IGLD85'

    # In build-ctl-only mode the model matcher needs an observation control
    # file to match against (unless custom XY coordinates are supplied).
    # Fetch observations first if the obs ctl is missing so the whole
    # "build and inspect" step works from a single command.
    if prop1.build_ctl_only and not prop1.user_input_location:
        _ensure_obs_ctl(prop1)

    get_node_ofs(prop1, None)


def _ensure_obs_ctl(prop1):
    """Fetch observation station data if the obs control files are absent.

    Build-ctl-only mode matches model nodes against the observation
    station control files. When those are missing (e.g. a fresh working
    directory), retrieve them first so the build step is self-contained.
    """
    import logging
    import os

    from ofs_skill.obs_retrieval import utils
    from ofs_skill.obs_retrieval.write_obs_ctlfile import write_obs_ctlfile
    from ofs_skill.utils import cache_manifest  # Add this import

    logger = logging.getLogger('root')
    dir_params = utils.Utils(
        getattr(prop1, 'config_file', None)).read_config_section(
        'directories', logger)
    control_files_path = os.path.join(
        prop1.path, dir_params['control_files_dir'])

    if not getattr(prop1, 'stationowner', ''):
        prop1.stationowner = 'co-ops,ndbc,usgs,chs'

    var_list = prop1.var_list.split(',') if isinstance(prop1.var_list, str) else prop1.var_list

    # Convert ISO dates (YYYY-MM-DDThh:mm:ssZ) to expected YYYYMMDD format
    start_date_fmt = prop1.start_date_full[:10].replace('-', '') if prop1.start_date_full else None
    end_date_fmt = prop1.end_date_full[:10].replace('-', '') if prop1.end_date_full else None

    # 1. Validate the inventory cache signature
    # (ensure_fresh will automatically delete the inventory if parameters changed)
    inventory_sig = cache_manifest.inventory_signature(
        prop1.ofs, start_date_fmt, end_date_fmt, prop1.stationowner, currents_bins_csv=None
    )
    inventory_path = os.path.join(control_files_path, f'inventory_all_{prop1.ofs}.csv')

    cache_manifest.ensure_fresh(
        inventory_path, inventory_sig, control_files_path, 'inventory', logger
    )

    # 2. Check if the inventory is present AND all requested variables have their .ctl files
    var_name_map = {
        'water_level': 'wl',
        'water_temperature': 'temp',
        'salinity': 'salt',
        'currents': 'cu',
    }

    missing_data = False
    if not os.path.isfile(inventory_path):
        missing_data = True
    else:
        for var in var_list:
            short_name = var_name_map.get(var, var)
            ctl_path = os.path.join(control_files_path, f'{prop1.ofs}_{short_name}_station.ctl')
            if not os.path.isfile(ctl_path):
                missing_data = True
                break

    # If the inventory is fresh and all requested files exist, skip the API fetch!
    if not missing_data:
        return

    # Otherwise, execute the fetch routine
    write_obs_ctlfile(
        start_date=start_date_fmt,
        end_date=end_date_fmt,
        datum=prop1.datum,
        path=prop1.path,
        ofs=prop1.ofs,
        stationowner=prop1.stationowner,
        var_list=var_list,
        logger=logger,
        config_file=prop1.config_file
    )


def main(argv=None):
    """Entry point for the get-node-ofs console script."""
    parser = argparse.ArgumentParser(
        prog='get-node-ofs',
        usage='%(prog)s',
        description='Create model control files & time series',
    )
    parser.add_argument(
        '-o', '--OFS',
        required=False,
        default=None,
        help="""Choose from the list on the ofs_extents/folder,
        you can also create your own shapefile, add it at the
        ofs_extents/folder and call it here""", )
    parser.add_argument(
        '-p', '--Path',
        required=False,
        help='Path to your working directory', )
    parser.add_argument(
        '-s', '--StartDate_full',
        required=False,
        help='Assessment start date: YYYY-MM-DDThh:mm:ssZ '
        "e.g. '2023-01-01T12:34:00Z'")
    parser.add_argument(
        '-e', '--EndDate_full',
        required=False,
        help='Assessment end date: YYYY-MM-DDThh:mm:ssZ '
        "e.g. '2023-01-01T12:34:00Z'")
    parser.add_argument(
        '-d', '--Datum',
        required=False,
        default='MLLW',
        help="datum options: 'MHW', 'MHHW' \
        'MLW', 'MLLW', 'NAVD88', 'XGEOID20B', 'IGLD85', 'LWD'")
    parser.add_argument(
        '-ws', '--Whichcast',
        required=False,
        default='nowcast',
        help="whichcasts: 'nowcast', 'forecast_b', 'forecast_a'", )
    parser.add_argument(
        '-t', '--FileType',
        required=False,
        default='stations',
        help="OFS model output file type to use: 'fields' or 'stations'", )
    parser.add_argument(
        '-f',
        '--Forecast_Hr',
        required=False,
        default='00hr',
        help='Specify model cycle to assess. Used with forecast_a mode only: '
        "'02hr', '06hr', '12hr', ... ", )
    parser.add_argument(
        '-hs',
        '--Horizon_Skill',
        action='store_true',
        help='Use all available forecast horizons between the '
        'start and end dates? True or False (boolean)')
    parser.add_argument(
        '-vs',
        '--Var_Selection',
        required=False,
        default='water_level,water_temperature,salinity,currents',
        help='Which variables do you want to skill assess? Options are: '
            'water_level, water_temperature, salinity, and currents. Choose '
            'any combination. Default (no argument) is all variables.')
    parser.add_argument(
        '-ui',
        '--User_Input',
        action='store_true',
        help='Input custom coordinates for model time series extraction? '
        'True or False (boolean)')
    parser.add_argument(
        '-b',
        '--Build_Ctl_Only',
        action='store_true',
        help='Build/verify the model control file(s) and write an '
        'obs-model station distance report (CSV + interactive map), then '
        'stop before extracting time series. Lets you inspect and '
        'hand-edit obs-model station matches before a full run.')
    parser.add_argument(
        '--No_Map',
        action='store_true',
        help='In build-ctl-only mode (-b), skip the interactive '
        'station-pair map and only write the CSV distance report.')
    parser.add_argument(
        '-so', '--Station_Owner',
        required=False,
        default=None,
        help='Station owner(s) to retrieve when build-ctl-only mode (-b) '
        "auto-fetches observations. Comma-separated, from: 'co-ops', "
        "'ndbc', 'usgs', 'chs'. Default is all providers.")
    parser.add_argument(
        '-c', '--config',
        help='Path to configuration file (default: conf/ofs_dps.conf)')

    args = parser.parse_args(argv)

    if args.OFS is None:
        from ofs_skill.visualization import create_gui_model
        create_gui_model.create_gui_model(runner=_run_pipeline)
    else:
        _run_pipeline(args)


if __name__ == '__main__':
    main()
