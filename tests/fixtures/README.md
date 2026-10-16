# Offline fixtures for CI

This directory holds **deterministic** sample data for unit and integration
tests. PR CI must never require live NODD, USGS, CO-OPS, or NDBC access.

Primary pipeline samples are cut from real production runs (CBOFS water
level, station 8637689), trimmed to a short overlapping window.

## Layout

```
tests/fixtures/
├── README.md
├── coops_mdapi_snapshots.json         # CO-OPS ADCP MDAPI snapshots (existing)
├── coops_adcp_audit_snapshot.csv      # ADCP audit regression ground truth
└── pipeline/                          # 1D skill-assessment stage fixtures
    ├── inventory_all_cbofs.csv
    ├── cbofs_wl_station.ctl
    ├── cbofs_wl_model_station.ctl
    ├── 8637689_cbofs_wl_station.obs
    ├── 8637689_cbofs_wl_45_nowcast_stations_model.prd
    ├── cbofs_wl_8637689_45_nowcast_stations_pair.int
    ├── cbofs_wl_8637689_45_nowcast_stations_pair.golden_rows.int
    ├── skill_cbofs_water_level_nowcast_stations.csv
    ├── cb0201_b01_stofs_3d_atl_cu_station.obs   # ADCP currents extra
    └── error_ranges.csv                         # copy of conf/error_ranges.csv
```

Helpers: `tests/helpers/api_mocks.py` (USGS / CO-OPS / NDBC mocks,
`load_julian_disk_series`, minimal `ofs_dps.conf` writer).

## Notes

- On-disk obs/model are space-delimited julian `.obs` / `.prd` files — **not**
  `DateTime,OBS` CSV. That CSV shape exists only in in-memory dataframes.
- Model CTL node `45` for station `8637689` is embedded in `.prd` / `.pair`
  filenames; keep those consistent if you edit.
- `skill_*.csv` column `obs_water_depth` is a known wrong-field bug upstream;
  do not assert it as ground truth.
- `coops_mdapi_snapshots.json` already lives at `tests/fixtures/`; do not
  duplicate it under `pipeline/`.

## `*_pair.golden_rows.int`

Byte-for-byte ground truth for the `.int` writer, used by
`tests/pairing_performance_regression_test.py`. It holds only the **data
rows** — no header. The real header line ends in a trailing space, which
the `trailing-whitespace` pre-commit hook would strip out of any committed
file, producing a fixture that could never match real output; the test
prepends the header itself instead.

Regenerate (only if the paired-series *format* deliberately changes) with:

```python
import logging

from ofs_skill.skill_assessment.format_paired_one_d import paired_scalar
from ofs_skill.skill_assessment.get_skill import _format_int_rows
from tests.helpers.api_mocks import PIPELINE_FIXTURES, load_julian_disk_series

logger = logging.getLogger('regen')

rows, _ = paired_scalar(
    load_julian_disk_series(
        PIPELINE_FIXTURES / '8637689_cbofs_wl_station.obs'),
    load_julian_disk_series(
        PIPELINE_FIXTURES / '8637689_cbofs_wl_45_nowcast_stations_model.prd'),
    '20260327-18:00:00', '20260327-22:42:00', logger, lookback_hours=0)
(PIPELINE_FIXTURES
 / 'cbofs_wl_8637689_45_nowcast_stations_pair.golden_rows.int'
 ).write_text(_format_int_rows(rows), encoding='utf-8')
```

The test also asserts the file's SHA-256, so update that literal too.

## Refresh procedure

1. Prefer mentor/production-trimmed samples under `pipeline/`.
2. Keep windows short (≤48 points at native cadence) so CI stays fast.
3. Never commit `conf/api_keys.conf` or real tokens. Scheduled live tests use
   GitHub Actions secret `API_USGS_PAT` only.
4. Model NetCDF for extract tests is still built in `tmp_path` (synthetic),
   not checked into git.
