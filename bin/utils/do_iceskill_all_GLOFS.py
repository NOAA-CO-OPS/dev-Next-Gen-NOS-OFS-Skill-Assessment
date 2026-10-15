import argparse

# Import the main pipeline functions
from bin.skill_assessment.do_iceskill import do_iceskill
from bin.skill_assessment.make_OM_view_ice import make_OM_view_ice
from ofs_skill.model_processing import model_properties, model_source
from ofs_skill.obs_retrieval import utils


def run_all_lakes(argv=None):
    parser = argparse.ArgumentParser(description='Run ice skill assessment for all Great Lakes and generate OM view.')
    parser.add_argument('-p', '--path', required=True, help='Inventory File path')
    parser.add_argument('-s', '--StartDate_full', required=True, help='Start Date (YYYY-MM-DDThh:mm:ssZ)')
    parser.add_argument('-e', '--EndDate_full', required=True, help='End Date (YYYY-MM-DDThh:mm:ssZ)')
    parser.add_argument('-ws', '--Whichcasts', required=True, help='nowcast, forecast_b, or all')
    parser.add_argument('-ts', '--TimeStep', required=False, default='daily', help='hourly or daily')
    parser.add_argument('-da', '--DailyAvg', action='store_true', help='Use daily average data')
    parser.add_argument('-avg', '--Neighborhood_Average', type=float, default=None, metavar='RADIUS_METERS', help='Enable neighborhood averaging and set the search radius in meters. If not provided, neighborhood averaging is off.')
    parser.add_argument('-c', '--config', help='Path to configuration file')
    args = parser.parse_args(argv)

    lakes = ['leofs', 'lmhofs', 'loofs', 'lsofs']
    logger = utils.init_root_logger(args.path)

    # 1. Loop through all Great Lakes OFS
    for lake in lakes:
        logger.info(f'=== Starting Ice Skill Assessment for {lake.upper()} ===')
        prop = model_properties.ModelProperties()
        prop.ofs = lake
        prop.path = args.path
        prop.config_file = args.config
        prop.start_date_full = args.StartDate_full
        prop.end_date_full = args.EndDate_full
        prop.whichcasts = args.Whichcasts
        prop.ice_dt = args.TimeStep
        prop.dailyavg = args.DailyAvg
        prop.Neighborhood_Average = args.Neighborhood_Average is not None
        prop.Search_Radius = args.Neighborhood_Average if prop.Neighborhood_Average else 0
        prop.model_source = model_source.model_source(lake)

        try:
            do_iceskill(prop, logger)
        except Exception as e:
            logger.error(f'Failed processing {lake.upper()}: {e}')

    # 2. Run the aggregated OM View
    logger.info('=== Starting Aggregated O&M View Generation ===')
    om_prop = model_properties.ModelProperties()
    om_prop.path = args.path
    om_prop.whichcasts = args.Whichcasts
    om_prop.config_file = args.config

    make_OM_view_ice(om_prop, logger)

if __name__ == '__main__':
    run_all_lakes()
