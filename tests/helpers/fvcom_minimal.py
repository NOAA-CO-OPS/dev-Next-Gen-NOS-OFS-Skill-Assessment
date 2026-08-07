"""Shared FVCOM synthetic NetCDF helpers for ctl-writer tests."""

from __future__ import annotations

import numpy as np
import xarray as xr


def build_fvcom_minimal_dataset(tmp_path):
    """Build two small FVCOM stations files + combine via xr.open_mfdataset.

    With ``data_vars='minimal'`` and a single time-concat dim, static
    vars (``lon``, ``lat``, ``h``, ``siglay``) come back at their native
    shape (no time replication). ``zeta`` is the only time-varying var
    here.

    Returns
    -------
    (xr.Dataset, np.ndarray, np.ndarray)
        The combined dataset and the original 1-D lon / lat arrays the
        caller can use to verify the written ctl file.
    """
    n_station = 6
    n_siglay = 3
    n_time = 4

    lon_1d = np.linspace(-71.0, -68.0, n_station, dtype=np.float64)
    lat_1d = np.linspace(41.0, 44.0, n_station, dtype=np.float64)
    h_1d = np.linspace(5.0, 30.0, n_station, dtype=np.float64)
    siglay = np.tile(
        np.linspace(-1.0, 0.0, n_siglay, dtype=np.float64)[:, None],
        (1, n_station),
    )

    def _make_file(path, t_offset):
        ds = xr.Dataset(
            data_vars={
                'lon': (('station',), lon_1d),
                'lat': (('station',), lat_1d),
                'h': (('station',), h_1d),
                'siglay': (('siglay', 'station'), siglay),
                'zeta': (
                    ('time', 'station'),
                    np.zeros((n_time, n_station), dtype=np.float64),
                ),
            },
            coords={
                'time': (
                    np.datetime64('2026-02-16T00')
                    + (t_offset + np.arange(n_time)) * np.timedelta64(1, 'h')
                ),
            },
        )
        ds.to_netcdf(path)
        ds.close()

    f1 = tmp_path / 'fvcom_stations_a.nc'
    f2 = tmp_path / 'fvcom_stations_b.nc'
    _make_file(f1, 0)
    _make_file(f2, n_time)

    combined = xr.open_mfdataset(
        [str(f1), str(f2)],
        data_vars='minimal',
        combine='nested',
        concat_dim='time',
    )

    # Pre-condition: static coords MUST remain 1-D under 'minimal'.
    assert combined['lon'].dims == ('station',), combined['lon'].dims
    assert combined['lat'].dims == ('station',), combined['lat'].dims

    return combined, lon_1d, lat_1d


def write_minimal_config(tmp_path):
    """Write a minimal INI config the writer needs for read_config_section.

    Delegates to the shared superset helper so the test suite has a single
    definition of the minimal ``ofs_dps.conf``.
    """
    from tests.helpers.api_mocks import write_minimal_ofs_config

    return write_minimal_ofs_config(tmp_path)


def write_obs_station_ctl(control_dir, ofs, name_var, stations):
    """Write a minimal obs station.ctl file the writer will read."""
    # Lazy import: keep helper module loadable without ofs_skill on path.
    from ofs_skill.utils.file_headers import OBS_CTL_HEADER

    path = control_dir / f'{ofs}_{name_var}_station.ctl'
    lines = []
    for sid, lat, lon, depth in stations:
        lines.append(f'{sid} {sid}_COOPS "Station {sid}"')
        lines.append(f'  {lat} {lon} {depth} 0.0 MLLW')
    path.write_text(OBS_CTL_HEADER + '\n'.join(lines) + '\n')
    return path
