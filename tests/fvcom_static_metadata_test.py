"""
Tests for FVCOM static-metadata handling at multi-file open (issue #311).

A multi-month FVCOM stations window can span files whose static
variables disagree without any station having moved:

- ``x`` / ``y`` are the station coordinates in the model's Cartesian
  projection. A projection change shifts every value by ~1e6 m, but
  nothing on the FVCOM path reads them.
- ``lon`` can flip between the -180..180 and 0..360 conventions, so the
  same station reads -66.98 in one file and 293.02 in the next.
- ``h`` can change when a station's bathymetry is corrected. That one is
  real, and is deliberately reconciled rather than hidden.

Under the default strict compare any of these aborts the combine with
``MergeError: conflicting values for variable 'x'``, and the validator's
majority rule discards the minority files. These tests pin that the
window combines with every time step retained, and that what differed is
reported rather than silently overridden.
"""

import logging

import numpy as np
import pytest
import xarray as xr

from ofs_skill.model_processing.intake_scisa import (
    FVCOM_DROP_VARIABLES,
    get_station_dim,
    station_dims_compatible,
)
from ofs_skill.model_processing.model_file_validation import (
    _differing_static_vars,
    validate_model_files,
)

LOG = logging.getLogger('fvcom_static_metadata_test')

N_TIME = 8
N_STATION = 6
N_SIGLAY = 4

# Western Atlantic longitudes, the convention FVCOM archives normally use.
BASE_LON = np.array([-66.98, -67.21, -68.20, -70.24, -70.56, -70.74])
BASE_LAT = np.array([44.65, 44.53, 44.10, 42.62, 42.35, 42.25])

# The reconciled open, as intake_model configures it for FVCOM stations.
RECONCILED = {
    'combine': 'nested',
    'concat_dim': 'time',
    'data_vars': 'minimal',
    'compat': 'override',
    'coords': 'minimal',
}
STRICT = {'combine': 'nested', 'concat_dim': 'time', 'data_vars': 'minimal'}


def _fvcom_stations_ds(t0='2026-05-14T00', lon=None, x_offset=0.0,
                       h_station1=1.79):
    """One FVCOM stations file.

    ``lon`` overrides the longitude convention, ``x_offset`` simulates a
    projection change, and ``h_station1`` a bathymetry correction.
    """
    lon = BASE_LON if lon is None else lon
    n_siglev = N_SIGLAY + 1
    h = np.linspace(5.0, 30.0, N_STATION)
    h[1] = h_station1
    return xr.Dataset(
        data_vars={
            'x': (('station',),
                  np.linspace(1.0e5, 2.0e5, N_STATION) + x_offset),
            'y': (('station',),
                  np.linspace(4.0e6, 4.1e6, N_STATION) + x_offset),
            'lon': (('station',), lon),
            'lat': (('station',), BASE_LAT),
            'h': (('station',), h),
            'siglay': (('siglay', 'station'),
                       np.zeros((N_SIGLAY, N_STATION))),
            'siglev': (('siglev', 'station'),
                       np.zeros((n_siglev, N_STATION))),
            'zeta': (('time', 'station'),
                     np.random.default_rng(0).normal(
                         0, 1, (N_TIME, N_STATION))),
        },
        coords={'time': np.datetime64(t0)
                + np.arange(N_TIME) * np.timedelta64(6, 'm')},
    )


def _write(tmp_path, name, ds):
    path = tmp_path / name
    ds.to_netcdf(path, format='NETCDF3_CLASSIC', unlimited_dims=['time'])
    return str(path)


def _six_month_window(tmp_path):
    """Three files standing in for the three differences seen in a real
    six-month archive: projection change, longitude convention flip, and
    a bathymetry correction at one station."""
    return [
        _write(tmp_path, 'a.nc', _fvcom_stations_ds('2026-05-14T00')),
        _write(tmp_path, 'b.nc',
               _fvcom_stations_ds('2026-05-14T01', x_offset=1.97e6)),
        _write(tmp_path, 'c.nc',
               _fvcom_stations_ds('2026-05-14T02', lon=BASE_LON + 360.0,
                                  h_station1=3.65)),
    ]


# --------------------------------------------------------------------
# The combine
# --------------------------------------------------------------------

@pytest.mark.integration
def test_strict_combine_fails_on_unused_projection_variable(tmp_path):
    """Establishes the bug: the batch dies on 'x', which is never read."""
    files = _six_month_window(tmp_path)
    with pytest.raises(xr.MergeError, match="'x'"):
        xr.open_mfdataset(files, **STRICT)


@pytest.mark.integration
def test_reconciled_combine_keeps_every_time_step(tmp_path):
    files = _six_month_window(tmp_path)
    combined = xr.open_mfdataset(
        files, drop_variables=list(FVCOM_DROP_VARIABLES), **RECONCILED)

    # Nothing dropped: all three files' time steps are present.
    assert combined.sizes['time'] == 3 * N_TIME
    # The unused projection variables never reach the combine.
    assert not set(FVCOM_DROP_VARIABLES) & set(combined.variables)
    # Static values come from the first file, in its own convention.
    np.testing.assert_array_equal(combined['lon'].values, BASE_LON)
    np.testing.assert_array_equal(combined['lat'].values, BASE_LAT)
    assert float(combined['h'].values[1]) == pytest.approx(1.79)


@pytest.mark.integration
def test_dropping_projection_vars_is_not_enough_on_its_own(tmp_path):
    """Why the fix is not just a drop_variables entry.

    Wrapping a 0..360 longitude back does not reproduce the original
    bits, so a wrapped file still fails a strict compare.
    """
    files = _six_month_window(tmp_path)
    with pytest.raises(xr.MergeError):
        xr.open_mfdataset(
            files, drop_variables=list(FVCOM_DROP_VARIABLES), **STRICT)

    wrapped = ((BASE_LON + 360.0 + 180.0) % 360.0) - 180.0
    assert np.allclose(wrapped, BASE_LON)
    assert wrapped.tobytes() != BASE_LON.tobytes()


@pytest.mark.integration
def test_reconciled_combine_preserves_data_values(tmp_path):
    """Reconciling static metadata must not touch the data itself."""
    sources = [_fvcom_stations_ds('2026-05-14T00'),
               _fvcom_stations_ds('2026-05-14T01', x_offset=1.97e6)]
    files = [_write(tmp_path, 'a.nc', sources[0]),
             _write(tmp_path, 'b.nc', sources[1])]
    combined = xr.open_mfdataset(
        files, drop_variables=list(FVCOM_DROP_VARIABLES), **RECONCILED)
    expected = np.concatenate([s['zeta'].values for s in sources])
    np.testing.assert_allclose(combined['zeta'].values, expected)


# --------------------------------------------------------------------
# Validator: report, do not drop
# --------------------------------------------------------------------

def _validate(files, **kwargs):
    valid, dropped, _ = validate_model_files(
        files, 'netcdf4', 'time', 'stations', LOG, **kwargs)
    return len(valid), len(dropped)


def test_reconcile_mode_keeps_every_file(tmp_path):
    files = _six_month_window(tmp_path)
    assert _validate(files, ignore_vars=FVCOM_DROP_VARIABLES,
                     reconcile_statics=True) == (3, 0)


def test_without_reconcile_mode_the_minority_is_still_dropped(tmp_path):
    """Non-FVCOM sources keep the existing majority behaviour."""
    files = [
        _write(tmp_path, 'a.nc', _fvcom_stations_ds('2026-05-14T00')),
        _write(tmp_path, 'b.nc', _fvcom_stations_ds('2026-05-14T01')),
        _write(tmp_path, 'c.nc',
               _fvcom_stations_ds('2026-05-14T02', h_station1=3.65)),
    ]
    assert _validate(files) == (2, 1)


def test_ignored_variables_are_not_a_configuration_difference(tmp_path):
    """A projection change alone must not register as a difference."""
    files = [
        _write(tmp_path, 'a.nc', _fvcom_stations_ds('2026-05-14T00')),
        _write(tmp_path, 'b.nc',
               _fvcom_stations_ds('2026-05-14T01', x_offset=1.97e6)),
    ]
    # Fingerprinting everything makes the odd file out a minority.
    assert _validate(files) == (1, 1)
    # Ignoring what the combine never sees keeps both.
    assert _validate(files, ignore_vars=FVCOM_DROP_VARIABLES) == (2, 0)


def test_reconcile_mode_reports_which_variables_differ(tmp_path, caplog):
    files = _six_month_window(tmp_path)
    with caplog.at_level(logging.WARNING):
        _validate(files, ignore_vars=FVCOM_DROP_VARIABLES,
                  reconcile_statics=True)
    warning = '\n'.join(r.getMessage() for r in caplog.records)
    assert 'Static station metadata is not identical' in warning
    # The variables that actually matter are named...
    assert 'h' in warning and 'lon' in warning
    # ...and the ignored ones are not.
    assert 'x' not in warning.split('differing in: ')[1].split('.')[0]


def test_differing_static_vars_identifies_only_changed_names():
    per_path = {
        'a': {'lat': 'aa', 'lon': 'bb', 'h': 'cc'},
        'b': {'lat': 'aa', 'lon': 'bb', 'h': 'dd'},
        'c': {'lat': 'aa', 'lon': 'ee', 'h': 'cc'},
    }
    assert _differing_static_vars(per_path) == ['h', 'lon']


def test_differing_static_vars_counts_missing_as_changed():
    per_path = {'a': {'lat': 'aa', 'h': 'cc'}, 'b': {'lat': 'aa'}}
    assert _differing_static_vars(per_path) == ['h']


# --------------------------------------------------------------------
# Station-count compatibility (issue #311)
# --------------------------------------------------------------------

@pytest.mark.parametrize(
    'dims, compatible, ref',
    [
        ([279, 279, 279], True, None),            # all equal
        ([275, 279, 279], False, 0),              # count increases
        ([279, 279, 275], False, 2),              # count DECREASES
        ([279, 275, 279], False, 1),              # dips and recovers
        ([279, 275, 271], False, 2),              # decreases twice
        ([279, 279], True, None),                 # the minimum batch
    ],
)
def test_station_dims_compatible(dims, compatible, ref):
    """A change in station count must be caught in either direction.

    The previous test was ``np.nanmax(np.diff(dims)) != 0``, which for
    [279, 279, 275] gives diffs [0, -4] and a maximum of 0 -- reporting
    a shrinking batch as compatible.
    """
    assert station_dims_compatible(dims) == (compatible, ref)


def test_station_dims_compatible_rejects_the_old_max_diff_logic():
    """Pin the exact case the old expression missed."""
    dims = [279, 279, 275]
    assert np.nanmax(np.diff(dims)) == 0      # what the old check saw
    assert station_dims_compatible(dims)[0] is False


@pytest.mark.integration
def test_get_station_dim_detects_a_shrinking_batch(tmp_path):
    """End to end: a batch whose station count drops must not be routed
    to the direct combine, which would fail on the station dimension."""
    files = [
        _write(tmp_path, 'a.nc', _fvcom_stations_ds('2026-05-14T00')),
        _write(tmp_path, 'b.nc', _fvcom_stations_ds('2026-05-14T01')),
        _write(tmp_path, 'c.nc',
               _fvcom_stations_ds('2026-05-14T02').isel(
                   station=slice(0, N_STATION - 2))),
    ]
    compat, ref = get_station_dim(
        'netcdf4', files, list(FVCOM_DROP_VARIABLES), LOG)
    assert compat is False
    assert ref == 2, 'reference should be the file with the fewest stations'


@pytest.mark.integration
def test_get_station_dim_accepts_a_uniform_batch(tmp_path):
    files = [
        _write(tmp_path, 'a.nc', _fvcom_stations_ds('2026-05-14T00')),
        _write(tmp_path, 'b.nc', _fvcom_stations_ds('2026-05-14T01')),
    ]
    compat, _ = get_station_dim(
        'netcdf4', files, list(FVCOM_DROP_VARIABLES), LOG)
    assert compat is True
