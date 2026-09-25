"""
Create inventory of eMOLT (Environmental Monitors on Lobster Traps) stations.
"""

import io
import urllib.error
import urllib.request
from logging import Logger

import pandas as pd


def inventory_emolt_station(
    lat1: float,
    lat2: float,
    lon1: float,
    lon2: float,
    start_date: str,
    end_date: str,
    logger: Logger,
    config_file=None,
) -> pd.DataFrame | None:

    lat1, lat2, lon1, lon2 = float(lat1), float(lat2), float(lon1), float(lon2)

    start_dt = f'{start_date[:4]}-{start_date[4:6]}-{start_date[6:]}T00%3A00%3A00Z'
    end_dt = f'{end_date[:4]}-{end_date[4:6]}-{end_date[6:]}T23%3A59%3A59Z'

    dataset_id = 'eMOLT_RT_QAQC'
    base_url = 'https://erddap.emolt.net/erddap/tabledap'

    query_url = (
        f'{base_url}/{dataset_id}.csvp'
        f'?tow_id,longitude,latitude'
        f'&longitude>={lon1}&longitude<={lon2}'
        f'&latitude>={lat1}&latitude<={lat2}'
        f'&time>={start_dt}&time<={end_dt}'
        f'&distinct()'
    )

    logger.info('Calling eMOLT ERDDAP service...')
    logger.info(f'eMOLT Query URL: {query_url}')

    try:
        req = urllib.request.Request(
            query_url,
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        with urllib.request.urlopen(req) as response:
            data_bytes = response.read()

        df = pd.read_csv(io.BytesIO(data_bytes))

    except urllib.error.HTTPError as ex:
        if ex.code == 404:
            logger.info('No eMOLT stations found within bounding box and time range (404)')
            return None
        logger.error('eMOLT data download failed at %s -- HTTP Error: %s', query_url, str(ex))
        return None
    except Exception as ex:
        logger.error('eMOLT data download failed at %s -- %s', query_url, str(ex))
        return None

    if df.empty:
        logger.info('No eMOLT stations found within bounding box and time range')
        return None

    # ERDDAP .csvp appends units to headers (e.g., 'longitude (degrees_east)').
    # Force rename them to match the exact order we requested in the URL.
    df.columns = ['tow_id', 'longitude', 'latitude']

    # FORCE coordinates to be numeric (decimals). If any garbage text is returned, it becomes NaN.
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')

    # Drop duplicates and any rows where coordinates failed to convert
    df = df.drop_duplicates(subset=['tow_id']).dropna(subset=['tow_id', 'longitude', 'latitude'])

    inventory_emolt = pd.DataFrame({
        'ID': df['tow_id'].astype(str),
        'X': df['longitude'],
        'Y': df['latitude'],
        'Source': 'eMOLT',
        'Name': 'eMOLT_Tow_' + df['tow_id'].astype(str), # Adding prefix bypasses the filter
        'has_wl': False,
        'has_temp': True,
        'has_salt': False,
        'has_cu': False,
    })

    logger.info('inventory_emolt_station.py ran successfully - found %d stations', len(inventory_emolt))
    return inventory_emolt
