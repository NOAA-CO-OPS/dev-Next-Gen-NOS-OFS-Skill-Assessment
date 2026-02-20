from datetime import datetime, timedelta
from os import PathLike
from pathlib import Path
from typing import Collection
import argparse
import os
import re
import fiona
import fiona.crs
import numpy as np
import rasterio
from rasterio.enums import Resampling
import scipy.interpolate
import xarray as xr
import geopandas as gpd
import shapely
import shapely.geometry
import shapely.wkt
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
import rasterio.features
import fiona.crs
import ast
from shapely.geometry import Polygon, Point
import pandas as pd
import sys


NRT_DELAY = timedelta(hours=1)


def polygons_intersect_2d(poly1, poly2):
    p1 = Polygon(poly1)
    p2 = Polygon(poly2)
    return p1.intersects(p2)


def ensure_clockwise(poly):
    ring = Polygon(poly)
    if not ring.exterior.is_ccw:
        return poly  # already clockwise
    return poly[::-1]


def spherical_point_in_poly(p, poly_xyz, tol=1e-4):  # Loosened tolerance
    angle_sum = 0
    for i in range(len(poly_xyz)):
        a = poly_xyz[i]
        b = poly_xyz[(i + 1) % len(poly_xyz)]
        va = normalize(a - p)
        vb = normalize(b - p)
        cross = np.cross(va, vb)
        sin_theta = np.linalg.norm(cross)
        cos_theta = np.dot(va, vb)
        angle = np.arctan2(sin_theta, cos_theta)
        orientation = np.sign(np.dot(p, cross))
        angle_sum += orientation * angle
    return abs(abs(angle_sum) - 2 * np.pi) < tol


def is_point_on_arc(p, a, b, tol=1e-10):
    angle_ab = np.arccos(np.clip(np.dot(a, b), -1, 1))
    angle_ap = np.arccos(np.clip(np.dot(a, p), -1, 1))
    angle_pb = np.arccos(np.clip(np.dot(p, b), -1, 1))
    return abs((angle_ap + angle_pb) - angle_ab) < tol


def normalize(v, eps=1e-12):
    norm = np.linalg.norm(v)
    return v if norm < eps else v / norm


def segments_intersect_gc(a1, a2, b1, b2, tol=1e-10):
    n1 = normalize(np.cross(a1, a2))
    n2 = normalize(np.cross(b1, b2))
    cross = np.cross(n1, n2)
    if np.linalg.norm(cross) < tol:
        return False  # ÃƒÂ¢Ã…â€œÃ¢â‚¬Â¦ HANDLE NEAR-PARALLEL CASES
    intersect_pts = [normalize(cross), normalize(-cross)]
    for p in intersect_pts:
        if is_point_on_arc(p, a1, a2) and is_point_on_arc(p, b1, b2):
            return True
    return False


def latlon_to_xyz(lat_deg, lon_deg):
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    return np.array([x, y, z])


def get_geospatial_bounds(nc_file):
    try:
        # Open file if a string path is passed
        if isinstance(nc_file, str):
            nc_file = xr.open_dataset(nc_file)

        # 1. Check for WKT-style geospatial_bounds
        if 'geospatial_bounds' in nc_file.attrs:
            return nc_file.attrs['geospatial_bounds']

        # 2. Check for 'bbox' as attribute (not via .ncattrs())
        elif 'bbox' in nc_file.attrs:
            bbox = nc_file.attrs['bbox']  # Should be a list or array: [lon_min, lat_min, lon_max, lat_max]
            lon_min, lat_min, lon_max, lat_max = bbox

        # 3. Fallback to individual lat/lon bounds in attrs
        else:
            lat_min = nc_file.attrs.get('geospatial_lat_min')
            lat_max = nc_file.attrs.get('geospatial_lat_max')
            lon_min = nc_file.attrs.get('geospatial_lon_min')
            lon_max = nc_file.attrs.get('geospatial_lon_max')

            # If any of the above are missing, calculate from coordinate data
            if None in [lat_min, lat_max, lon_min, lon_max]:
                if 'lat' in nc_file.coords and 'lon' in nc_file.coords:
                    lat_vals = nc_file['lat'].values
                    lon_vals = nc_file['lon'].values

                    lat_min = float(lat_vals.min())
                    lat_max = float(lat_vals.max())
                    lon_min = float(lon_vals.min())
                    lon_max = float(lon_vals.max())
                else:
                    raise ValueError("No 'lat' and 'lon' coordinates found in dataset")

        # Return WKT polygon
        wkt_poly = (
            f"POLYGON (({lon_min} {lat_min}, {lon_min} {lat_max}, "
            f"{lon_max} {lat_max}, {lon_max} {lat_min}, {lon_min} {lat_min}))"
        )
        return wkt_poly

    except Exception as e:
        print(f"Error reading bounds: {e}")
        return None


def polygons_intersect_spherical(poly1_latlon, poly2_latlon):
    poly1 = [latlon_to_xyz(lat, lon) for lon, lat in poly1_latlon]
    poly2 = [latlon_to_xyz(lat, lon) for lon, lat in poly2_latlon]

    for i in range(len(poly1)):
        a1 = poly1[i]
        a2 = poly1[(i + 1) % len(poly1)]
        for j in range(len(poly2)):
            b1 = poly2[j]
            b2 = poly2[(j + 1) % len(poly2)]
            if segments_intersect_gc(a1, a2, b1, b2):
                return True

    if any(spherical_point_in_poly(p, poly2) for p in poly1) or \
       any(spherical_point_in_poly(p, poly1) for p in poly2):
        return True

    return False


def parse_wkt_polygon(wkt_str):
    pattern = r'POLYGON\s*\(\(\s*(.+?)\s*\)\)'
    match = re.search(pattern, wkt_str)
    if not match:
        raise ValueError("Invalid WKT POLYGON format.")
    coord_pairs = match.group(1).split(',')
    polygon = []
    for pair in coord_pairs:
        lon, lat = map(float, pair.strip().split())
        polygon.append((lon, lat))

        # ÃƒÂ¢Ã…â€œÃ¢â‚¬Â¦ ENSURE POLYGON IS CLOSED
    if polygon[0] != polygon[-1]:
        polygon.append(polygon[0])

    return polygon


def export_ascii(data_da, outfile):
    data = data_da.values.copy()
    data = np.where(np.isnan(data), -9999, data)

    lats = data_da['lat'].values
    lons = data_da['lon'].values

    if lats[0] < lats[-1]:
        data = np.flipud(data)
        lats = lats[::-1]

    if lons[0] > lons[-1]:
        data = data[:, ::-1]
        lons = lons[::-1]

    nrows = len(lats)
    ncols = len(lons)

    cellsize_x = np.mean(np.diff(lons))
    cellsize_y = np.mean(np.diff(lats))

    xllcorner = lons[0]  # leftmost longitude
    yllcorner = lats[-1]  # bottommost latitude


    with open(outfile, "w") as f:
        f.write(f"ncols        {ncols}\n")
        f.write(f"nrows        {nrows}\n")
        f.write(f"xllcorner    {xllcorner}\n")
        f.write(f"yllcorner    {yllcorner}\n")
        f.write(f"cellsize     {cellsize_x}\n")
        f.write("NODATA_value -9999\n")
        np.savetxt(f, data, fmt="%.4f")


    prj_path = outfile.with_suffix(".prj")

    wgs84_wkt = 'GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]]'

    with open(prj_path, "w") as f:
        f.write(wgs84_wkt)



def clip_by_ofs(ds, gdf):
    lon = ds['lon'].values
    lat = ds['lat'].values
    lon2d, lat2d = np.meshgrid(lon, lat)
    
    points = gpd.GeoDataFrame(
        geometry=[Point(x, y) for x, y in zip(lon2d.flatten(), lat2d.flatten())],
        crs="EPSG:4326"
    )
    
    points_in = gpd.sjoin(points, gdf, predicate="within", how="inner")
    
    mask = np.zeros(lon2d.size, dtype=bool)
    mask[points_in.index] = True
    mask = mask.reshape(lon2d.shape)
    
    mask_da = xr.DataArray(mask, coords=[ds.lat, ds.lon], dims=["lat", "lon"])
    
    return mask_da


def wrap_lon(lon):
    return ((lon + 180) % 360) - 180


def check_for_overlap(
    date_obj, 
    data_dir, 
    gdf, 
    ofs, 
    mode, 
    start_time=None, 
    end_time=None
):
    """
    usegc: US East Coast and Gulf of America
    uswc: US West Coast
    glna: Great Lakes North America
    ushiL US Hawaii
    akns: Alaska North Slope
    gak: Gulf of Alaska
    prvi: Puerto Rico/Virgin Islands
    """

    hf_datasets = ["usegc", "uswc", "glna", "ushi", "akns", "gak", "prvi"]

    lon_min, lat_min, lon_max, lat_max = gdf.total_bounds
    study_area = [
        (lon_min, lat_min),
        (lon_min, lat_max),
        (lon_max, lat_max),
        (lon_max, lat_min),
        (lon_min, lat_min),
    ]
    study_area = ensure_clockwise(study_area)

    matching_files = {}

    for hfd in hf_datasets:
        try:
            url = f"https://dods.ndbc.noaa.gov/thredds/dodsC/hfradar_{hfd}_2km"

            ds = xr.open_dataset(url)
            geospatial_bounds = get_geospatial_bounds(ds)

            if geospatial_bounds:
                file_polygon = parse_wkt_polygon(geospatial_bounds)
                file_polygon = [(wrap_lon(lon), lat) for lon, lat in file_polygon]
                file_polygon = ensure_clockwise(file_polygon)

                if polygons_intersect_2d(study_area, file_polygon):
                    ds = xr.open_dataset(url)
                    matching_files[url] = ds

        except Exception as e:
            print(e)
        
    process_files(matching_files, date_obj, data_dir, gdf, ofs, mode, start_time, end_time)


def process_files(
    matching_files, 
    date_obj, 
    data_dir,
    gdf, 
    ofs, 
    mode, 
    start_time=None, 
    end_time=None
):

    if mode == "daily":
        today = datetime.utcnow().date()

        if date_obj.date() == today:
            et = datetime.utcnow() - NRT_DELAY
            st = et - timedelta(days=1) - NRT_DELAY 

        else:
            st = date_obj.date()
            et = date_obj.date() + timedelta(days=1)

    elif mode == "hourly":
        if start_time is not None and end_time is not None:
            et = end_time
            st = start_time


    st = st.replace(minute=0, second=0, microsecond=0)
    et = et.replace(minute=0, second=0, microsecond=0)

    for url, ds in matching_files.items():
        dtp = ds.sel(time=slice(st, et))

        if dtp.time.size == 0:
            continue

        dtp_u = dtp["u"]
        dtp_v = dtp["v"]

        mask_dtp = clip_by_ofs(dtp, gdf)

        u_data = dtp_u.where(mask_dtp)
        v_data = dtp_v.where(mask_dtp)

        u_data = u_data.assign_coords(lat=dtp['lat'], lon=dtp['lon'])
        v_data = v_data.assign_coords(lat=dtp['lat'], lon=dtp['lon'])

        if '_FillValue' in dtp_u.attrs:
            u_data = u_data.where(u_data != dtp_u._FillValue)
        if '_FillValue' in dtp_v.attrs:
            v_data = v_data.where(v_data != dtp_v._FillValue)

        if mode == "daily":
            u_avg = u_data.mean(dim="time")
            v_avg = v_data.mean(dim="time")

            mag = np.sqrt(u_avg**2 + v_avg**2)
            dir_rad = np.arctan2(u_avg, v_avg)
            direction = (np.degrees(dir_rad) + 360) % 360

            mag_outfile = data_dir / f"mag_{date_obj.strftime('%Y%m%d')}.asc"
            dir_outfile = data_dir / f"dir_{date_obj.strftime('%Y%m%d')}.asc"

            export_ascii(mag, mag_outfile)
            export_ascii(direction, dir_outfile)

        elif mode == "hourly":
            for t in range(len(u_data.time)):
                u_hour = u_data.isel(time=t)
                v_hour = v_data.isel(time=t)

                mag = np.sqrt(u_hour**2 + v_hour**2)
                dir_rad = np.arctan2(u_hour, v_hour)
                direction = (np.degrees(dir_rad) + 360) % 360

                timestamp = pd.to_datetime(u_hour.time.values).strftime("%Y%m%d_%H%M")
                mag_outfile = data_dir / f"mag_{timestamp}.asc"
                dir_outfile = data_dir / f"dir_{timestamp}.asc"

                export_ascii(mag, mag_outfile)
                export_ascii(direction, dir_outfile)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("my_date", help="Date for data collection")
    parser.add_argument("catalogue", help="File directory to write the files to")
    parser.add_argument("my_box", help="Bounding box (shapefile)")
    #parser.add_argument("var_name_list", nargs='*', help="Variables of interest (e.g., sst)")
    parser.add_argument("ofs", help="OFS of interest")

    parser.add_argument("--mode", choices=["daily", "hourly"], default="daily")
    parser.add_argument("--start", help="Start time YYYYMMDDHH (UTC)")
    parser.add_argument("--end", help="End time YYYYMMDDHH (UTC)")

    args = parser.parse_args()

    date_obj = datetime.strptime(args.my_date, "%Y%m%d")
    
    data_dir = Path(args.catalogue)

    mode = args.mode

    if args.start:
        start_time = args.start

    if args.end:
        end_time = args.end

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if os.path.exists(Path(args.my_box)):
        gdf = gpd.read_file(Path(args.my_box))
        bounds = gdf.total_bounds
        lon_min, lat_min, lon_max, lat_max = bounds

        if mode == "daily":
            check_for_overlap(date_obj, data_dir, gdf, args.ofs, mode)

        elif mode == "hourly":
            check_for_overlap(date_obj, data_dir, gdf, args.ofs, mode, start_time, end_time)


