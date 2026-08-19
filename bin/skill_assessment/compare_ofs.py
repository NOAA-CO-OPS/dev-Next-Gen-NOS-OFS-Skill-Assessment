"""
Created on Mon Jul  6 13:58:20 2026

@author: PWL
"""

import argparse
import logging.config
import os
import shutil
import sys

from ofs_skill.model_processing import model_properties
from ofs_skill.obs_retrieval import utils


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
    prop.filecheck = True
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

    # Import the CLI scripts including plot_model_timeseries, which houses the
    # comparison plotting routines
    from create_1dplot import create_1dplot
    from get_shapefile_intersection import get_shapefile_intersection
    from plot_model_timeseries import (
        generate_comparisons,
        generate_stat_comparisons,
    )

    # loggin'
    logger = setup_logger(args.home_path, args.config)

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
