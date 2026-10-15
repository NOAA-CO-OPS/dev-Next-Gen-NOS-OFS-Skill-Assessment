import os
import urllib.request
from datetime import datetime, timedelta
from urllib.error import HTTPError

import xarray as xr


def generate_date_range(start_date, end_date):
    """Generate a list of datetime objects between start and end dates."""
    dates = []
    current_date = start_date
    while current_date <= end_date:
        dates.append(current_date)
        current_date += timedelta(days=1)
    return dates

def download_glsea_data(dates, download_dir, url_root):
    """
    Downloads GLSEA netcdfs using a temporary file to guarantee no partial downloads.
    """
    os.makedirs(download_dir, exist_ok=True)
    downloaded_files = []

    for mydate in dates:
        filename = f"{mydate.strftime('%Y')}_{mydate.strftime('%j')}_glsea_ice.nc"
        url = (
            f'{url_root}'
            f"{mydate.strftime('%Y')}/"
            f"{mydate.strftime('%m')}/"
            f'{filename}'
        )

        save_path = os.path.join(download_dir, filename)
        tmp_path = save_path + '.tmp'

        if not os.path.isfile(save_path):
            try:
                print(f'Downloading: {url}')
                # 1. Download to a temporary file
                urllib.request.urlretrieve(url, tmp_path)
                # 2. Rename only if the download was successful
                os.rename(tmp_path, save_path)
                downloaded_files.append(save_path)
            except HTTPError as e:
                print(f"File not found or HTTP error for {mydate.strftime('%Y-%m-%d')}: {e}")
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception as e:
                print(f'Failed to download {url}: {e}')
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
        else:
            downloaded_files.append(save_path)

    return downloaded_files

def create_climatology_and_save(file_list, output_npy_path):
    """
    Concatenates files, calculates the daily average across all years,
    and saves the numpy array.
    """
    print('Opening multiple datasets...')
    # Concatenation approach utilized in get_icecover_observations.py
    ds = xr.open_mfdataset(
        file_list,
        combine='nested',
        concat_dim='time',
        lock=False
    )

    print('Calculating daily climatology (averaging all years by day of year)...')
    # Group by day of year (1-366) and take the mean over the time dimension
    climatology = ds.groupby('time.dayofyear').mean('time')

    # Assuming the variable name in the netcdf is 'ice_cover' or 'z'
    # (Adjust 'ice_cover' below to match the exact variable name in the GLSEA netcdf)
    var_name = [var for var in climatology.data_vars][0]
    print(f'Saving climatology to {output_npy_path}...')
    # Save directly to compressed netCDF
    encoding = {var_name: {'zlib': True, 'complevel': 4}} # complevel 1-9
    climatology.to_netcdf(output_npy_path.replace('.npy', '.nc'), encoding=encoding)
    print('Done!')

if __name__ == '__main__':
    # Parameters
    START_DATE = datetime(1995, 1, 1)
    END_DATE = datetime.now() - timedelta(days=1)

    URL_ROOT = 'https://apps.glerl.noaa.gov/thredds/fileServer/glsea_ice_nc/'
    DOWNLOAD_DIR = './glsea_downloads'
    OUTPUT_FILE = 'gl_2d_clim.npy'

    dates_to_fetch = generate_date_range(START_DATE, END_DATE)

    # 2. Download missing data (will now correctly skip existing, valid files)
    files = download_glsea_data(dates_to_fetch, DOWNLOAD_DIR, URL_ROOT)

    # 3. List valid files and process
    valid_files = [os.path.join(DOWNLOAD_DIR, f) for f in os.listdir(DOWNLOAD_DIR) if f.endswith('.nc')]

    if valid_files:
        create_climatology_and_save(valid_files, OUTPUT_FILE)
    else:
        print('No valid files found to process.')
