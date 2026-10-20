import os
import urllib.request
from datetime import datetime, timedelta
from urllib.error import HTTPError

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def generate_date_range(start_date, end_date, skip_months=(6, 7, 8, 9, 10)):
    """
    Generate a list of datetime objects between start and end dates,
    excluding summer/early fall months.
    """
    dates = []
    current_date = start_date
    while current_date <= end_date:
        if current_date.month not in skip_months:
            dates.append(current_date)
        current_date += timedelta(days=1)
    return dates

def remove_partial_downloads(directory_path):
    """
    Validates downloaded NetCDF files by attempting to read their headers.
    Removes HTML error pages or incomplete files.
    """
    print('Validating downloaded NetCDF files...')
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path) and filename.endswith('.nc'):
            if os.path.getsize(file_path) < 100 * 1024:
                print(f'Removing invalid file (<100KB): {file_path}')
                try:
                    os.remove(file_path)
                except OSError:
                    pass
                continue

            try:
                with xr.open_dataset(file_path, engine='netcdf4'):
                    pass
            except Exception as e:
                print(f'Removing corrupt file: {file_path} - Error: {e}')
                try:
                    os.remove(file_path)
                except OSError:
                    pass

def download_glsea_data(dates, download_dir, url_root, url_root_backup):
    """
    Downloads GLSEA NetCDF files, falling back to ERDDAP if THREDDS fails.
    """
    os.makedirs(download_dir, exist_ok=True)
    downloaded_files = []

    for mydate in dates:
        # GLSEA file naming convention[cite: 1]
        filename = f"{mydate.strftime('%Y')}_{mydate.strftime('%j')}_glsea_ice.nc"
        date_path = f"{mydate.strftime('%Y')}/{mydate.strftime('%m')}/{filename}"

        url_thredds = f'{url_root}{date_path}'
        url_erddap = f'{url_root_backup}{date_path}'

        save_path = os.path.join(download_dir, filename)

        if not os.path.isfile(save_path):
            try:
                print(f'Downloading from THREDDS: {url_thredds}')
                urllib.request.urlretrieve(url_thredds, save_path)
                downloaded_files.append(save_path)
            except (HTTPError, urllib.error.URLError) as e:
                print(f"THREDDS failed for {mydate.strftime('%Y-%m-%d')}: {e}. Trying ERDDAP...")
                try:
                    urllib.request.urlretrieve(url_erddap, save_path)
                    print(f'Successfully downloaded from ERDDAP: {url_erddap}')
                    downloaded_files.append(save_path)
                except Exception as e_backup:
                    print(f"Failed to download from both sources for {mydate.strftime('%Y-%m-%d')}: {e_backup}")
        else:
            downloaded_files.append(save_path)

    return downloaded_files

def create_climatology_and_save(file_list, output_nc_path):
    """
    Computes daily climatology sequentially, preserving the GLSEA land mask
    (land = NaN, water = 0-100), and saves a zlib-compressed NetCDF file.
    """
    print(f'Processing {len(file_list)} NetCDF files sequentially...')

    sum_by_doy = {}
    count_by_doy = {}
    template_ds = None
    var_name = None
    water_mask = None

    for idx, fpath in enumerate(file_list):
        if idx % 200 == 0 or idx == len(file_list) - 1:
            print(f'Processing file {idx + 1}/{len(file_list)}...')

        try:
            with xr.open_dataset(fpath, engine='netcdf4') as ds:
                if var_name is None:
                    var_name = [v for v in ds.data_vars if 'ice' in v.lower() or v == 'z'][0]
                    template_ds = ds.isel(time=0).drop_vars(
                        [v for v in ds.data_vars if v != var_name]
                    )
                    # Land is represented as NaN in GLSEA[cite: 3]
                    first_slice = ds[var_name].isel(time=0).values
                    water_mask = ~np.isnan(first_slice)

                doy = int(ds.time.dt.dayofyear.values[0])
                data_slice = ds[var_name].isel(time=0).values

                if doy not in sum_by_doy:
                    sum_by_doy[doy] = np.zeros(water_mask.shape, dtype=np.float64)
                    count_by_doy[doy] = 0

                sum_by_doy[doy] += np.nan_to_num(data_slice, nan=0.0)
                count_by_doy[doy] += 1
        except Exception as e:
            print(f'Skipping unreadable file {fpath}: {e}')

    print('Computing daily mean across all years and applying land mask...')
    shape_2d = water_mask.shape
    clim_array = np.full((366, *shape_2d), np.nan, dtype=np.float32)

    for doy in range(1, 367):
        if doy in sum_by_doy and count_by_doy[doy] > 0:
            mean_day = sum_by_doy[doy] / count_by_doy[doy]
            clim_array[doy - 1, water_mask] = mean_day[water_mask]
        else:
            # Summer days: water cells set to 0.0% ice; land cells remain NaN
            clim_array[doy - 1, water_mask] = 0.0

    # Build xarray dataset
    doy_coords = np.arange(1, 367)
    clim_ds = xr.Dataset(
        {var_name: (['dayofyear', *template_ds[var_name].dims], clim_array)},
        coords={'dayofyear': doy_coords, **template_ds.coords}
    )

    # Set up DEFLATE (zlib) compression for the ice concentration variable
    encoding = {
        var_name: {
            'zlib': True,
            'complevel': 4,
            'dtype': 'float32',
        }
    }

    print(f'Saving compressed climatology dataset to {output_nc_path}...')
    clim_ds.to_netcdf(output_nc_path, encoding=encoding, engine='netcdf4')

    return clim_ds, var_name

def plot_daily_timeseries(clim_ds, var_name, output_plot_path='ice_climatology_timeseries.png'):
    """
    Plots the spatially averaged daily ice concentration climatology over water cells
    (ignoring NaN land points using nanmean)[cite: 3].
    """
    print('Generating land-masked time-series plot...')

    clim_array = clim_ds[var_name].values
    daily_water_mean = np.nanmean(clim_array, axis=(1, 2))

    if 'dayofyear' in clim_ds.coords:
        days_of_year = clim_ds['dayofyear'].values
    else:
        days_of_year = np.arange(1, len(daily_water_mean) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(days_of_year, daily_water_mean, color='#1f77b4', linewidth=2, label='Mean Ice Concentration')

    plt.title('Great Lakes Daily Ice Concentration Climatology (Water Area Average)')
    plt.xlabel('Day of Year (1–366)')
    plt.ylabel('Ice Concentration (%)')
    plt.xlim(1, 366)
    plt.ylim(0, max(np.nanmax(daily_water_mean) * 1.1, 10))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    print(f'Saving plot to {output_plot_path}...')
    plt.savefig(output_plot_path, dpi=300)
    plt.close()
    print('Plot saved successfully!')

if __name__ == '__main__':
    START_DATE = datetime(1995, 1, 1)
    END_DATE = datetime.now() - timedelta(days=1)

    # Endpoints defined in ofs_dps.conf[cite: 2]
    URL_ROOT = 'https://apps.glerl.noaa.gov/thredds/fileServer/glsea_ice_nc/'
    URL_ROOT_BACKUP = 'https://apps.glerl.noaa.gov/erddap/files/GL_Ice_Concentration_GCS/'
    DOWNLOAD_DIR = './glsea_downloads'
    OUTPUT_NC_FILE = 'gl_2d_clim.nc'

    should_process = True

    # Check if a valid 366-day file already exists
    if os.path.isfile(OUTPUT_NC_FILE):
        with xr.open_dataset(OUTPUT_NC_FILE) as existing_ds:
            vname = [v for v in existing_ds.data_vars if 'ice' in v.lower() or v == 'z'][0]
            if existing_ds[vname].shape[0] == 366:
                print(f"Found existing valid 366-day climatology file: '{OUTPUT_NC_FILE}'. Skipping download and processing...")
                plot_daily_timeseries(existing_ds, vname)
                should_process = False
            else:
                print(f"Existing '{OUTPUT_NC_FILE}' contains {existing_ds[vname].shape[0]} days instead of 366. Recomputing...")
                should_process = True

    if should_process:
        # 1. Generate date list skipping summer months (June–October)
        dates_to_fetch = generate_date_range(START_DATE, END_DATE)

        # 2. Download missing files
        download_glsea_data(dates_to_fetch, DOWNLOAD_DIR, URL_ROOT, URL_ROOT_BACKUP)

        # 3. Clean up invalid/HTML downloads
        remove_partial_downloads(DOWNLOAD_DIR)

        # 4. List valid files
        valid_files = [
            os.path.join(DOWNLOAD_DIR, f)
            for f in sorted(os.listdir(DOWNLOAD_DIR))
            if f.endswith('.nc')
        ]

        if valid_files:
            # 5. Build full 366-day climatology array and save to compressed NetCDF
            clim_ds, variable_name = create_climatology_and_save(valid_files, OUTPUT_NC_FILE)

            # 6. Plot average time series
            plot_daily_timeseries(clim_ds, variable_name)
        else:
            print('No valid NetCDF files found to process.')
