"""Shared STOFS-2D-Global (ADCIRC) synthetic fields helpers for ctl-writer tests.

The dataset mirrors docs/adcirc_fields_file_notes.md: ``x``/``y`` are
``(node,)`` float64 in -180..180, ``zeta`` is ``(time, node)`` float32 at
hourly steps with dry values stored as exactly -99999.0 (no ``_FillValue``
attribute, no NaN), and ``depth`` is ``(node,)``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

ADCIRC_FILL = -99999.0

# Node layout (lon, lat) and role.
ADCIRC_NODES = [
    (-70.00, 40.00),   # 0 wet at every step
    (-65.00, 35.00),   # 1 wet at every step
    (-60.00, 30.00),   # 2 dry (fill) at every step
    (-55.00, 25.00),   # 3 dry (fill) at step 1 only
    (-50.00, 20.00),   # 4 wet at every step, isolated
]
ALWAYS_DRY_NODE = 2
ONE_STEP_DRY_NODE = 3
ONE_STEP_DRY_STEP = 1
N_TIME = 6
T0 = np.datetime64('2026-10-20T00:00')


def build_adcirc_minimal_dataset(tmp_path: Path) -> xr.Dataset:
    """Write a small STOFS-2D-Global-like fields file and open it.

    Args:
        tmp_path: Pytest temporary directory for the NetCDF file.

    Returns:
        The opened dataset (lazy, chunked by single time step like intake).
    """
    n_node = len(ADCIRC_NODES)
    x = np.array([p[0] for p in ADCIRC_NODES], dtype=np.float64)
    y = np.array([p[1] for p in ADCIRC_NODES], dtype=np.float64)
    depth = np.linspace(5.0, 50.0, n_node, dtype=np.float64)
    # Distinct values per (step, node) so extraction can be checked exactly.
    zeta = (0.1 * np.arange(N_TIME)[:, None]
            + 0.01 * np.arange(n_node)[None, :]).astype(np.float32)
    zeta[:, ALWAYS_DRY_NODE] = ADCIRC_FILL
    zeta[ONE_STEP_DRY_STEP, ONE_STEP_DRY_NODE] = ADCIRC_FILL

    ds = xr.Dataset(
        data_vars={
            'x': (('node',), x),
            'y': (('node',), y),
            'depth': (('node',), depth),
            'zeta': (('time', 'node'), zeta),
        },
        coords={'time': T0 + np.arange(N_TIME) * np.timedelta64(1, 'h')},
    )
    path = tmp_path / 'stofs_2d_glo.t00z.fields.cwl.nc'
    # No _FillValue on zeta (or on the coords), as in the real file.
    ds.to_netcdf(path, encoding={name: {'_FillValue': None}
                                 for name in ('x', 'y', 'depth', 'zeta')})
    ds.close()

    opened = xr.open_dataset(path, chunks={'time': 1})
    assert '_FillValue' not in opened['zeta'].attrs
    assert '_FillValue' not in opened['zeta'].encoding
    assert opened['zeta'].dtype == np.float32
    assert opened['x'].dims == ('node',) and opened['x'].dtype == np.float64
    return opened


def write_adcirc_config(tmp_path: Path, max_dist_km: float = 4.0) -> Path:
    """Minimal ofs_dps.conf plus the [settings] key the ADCIRC path reads."""
    from tests.helpers.api_mocks import write_minimal_ofs_config

    cfg_path = write_minimal_ofs_config(tmp_path)
    with open(cfg_path, 'a', encoding='utf-8') as f:
        f.write('\n[settings]\n'
                f'station_match_max_dist_km={max_dist_km}\n')
    return cfg_path
