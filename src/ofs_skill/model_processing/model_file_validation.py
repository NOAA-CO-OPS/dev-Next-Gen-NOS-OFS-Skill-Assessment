"""
Model File Validation

Pre-open validation of model netCDF files so that incomplete, corrupt,
or structurally inconsistent files are dropped with a warning instead
of crashing the multi-file open or the extraction that follows
(issue #194: multi-month runs failing on files left behind by a
forecast interruption).

Each local file is opened individually and checked for:

1. **Readability** — the file opens under the engine chosen for the
   batch. Zero-byte files, garbage, and truncated HDF5 files fail here.
2. **A usable time axis** — the concat dimension exists and has at
   least one step. Header-only files written during an interrupted
   forecast fail here.
3. **A strictly increasing time axis** — the full time coordinate is
   loaded and must be strictly increasing. This is the truncation
   detector for NetCDF-3: libnetCDF silently zero-fills reads past
   EOF (no error!), so a file truncated by an interrupted forecast
   opens cleanly and returns zeros for the missing records — including
   time values of 0, which later produce duplicate/non-monotonic
   timestamps that crash resampling (and quietly poison the data
   before that). Since time is a record variable, any truncation deep
   enough to lose data corrupts the time tail, so this check catches
   it. Truncated HDF5 files fail at open instead (check 1).
4. **Data variables present** — a file with coordinates but no data
   variables is dropped.

After the per-file checks, dimension sizes are compared across the
surviving files: a file whose non-time dimensions disagree with the
majority of the batch (e.g. a different number of sigma layers or mesh
nodes) is dropped, since the multi-file combine would otherwise fail.
The ``station`` dimension of stations files is exempt — differing
station counts are legitimate (grid updates) and are handled downstream
by ``remove_extra_stations``, which slices files to the common set.

Remote URLs are passed through unvalidated: per-file remote opens are
expensive, and NODD-served files are the canonical copies. The
h5netcdf open-time fallback in ``intake_model`` still protects that
path.
"""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from logging import Logger
from typing import Any, Optional

import numpy as np
import xarray as xr

_REMOTE_PREFIXES = ('http://', 'https://', 's3://')


def _is_remote(path: Any) -> bool:
    if not isinstance(path, str):
        return False
    # Unwrap fsspec protocol chaining (simplecache::https://...)
    tail = path.split('::')[-1] if '::' in path else path
    return tail.startswith(_REMOTE_PREFIXES)


def _check_file(path: str, engine: str,
                time_name: Optional[str]) -> tuple[Optional[str], dict]:
    """Validate one local file.

    Returns ``(reason, dims)`` — ``reason`` is None when the file is
    valid, otherwise a short description of why it was rejected;
    ``dims`` is the file's dimension-size mapping (empty on failure).
    """
    try:
        ds = xr.open_dataset(path, engine=engine, decode_times=False)
    except Exception as ex:  # noqa: BLE001 - any open failure = bad file
        return (f'unreadable ({type(ex).__name__}: '
                f'{str(ex)[:120]})'), {}
    try:
        dims = dict(ds.sizes)
        if time_name:
            if time_name not in dims:
                return (f"missing '{time_name}' dimension "
                        '(incomplete file?)'), {}
            if dims[time_name] < 1:
                return f"empty '{time_name}' dimension", {}
        if len(ds.data_vars) == 0:
            return 'no data variables (header-only file?)', {}
        if time_name:
            try:
                tvals = np.asarray(ds[time_name].values, dtype='float64')
            except Exception as ex:  # noqa: BLE001
                return (f'time coordinate unreadable — truncated file? '
                        f'({type(ex).__name__}: {str(ex)[:120]})'), {}
            # Model time axes are strictly increasing within a file.
            # libnetCDF silently zero-fills NetCDF-3 reads past EOF, so
            # a truncated file reads back with a zeroed time tail —
            # detectable here, invisible to the open itself.
            if tvals.size > 1:
                diffs = np.diff(tvals)
                if not np.all(diffs > 0):
                    bad = int(np.argmin(diffs > 0)) + 1
                    return ('time axis not strictly increasing at step '
                            f'{bad} of {tvals.size} — zero-filled or '
                            'truncated file?'), {}
        return None, dims
    finally:
        ds.close()


def _majority_dims(per_file_dims: list[dict], exempt: set) -> dict:
    """Most common size for every non-exempt dimension in the batch."""
    counts: dict[str, Counter] = {}
    for dims in per_file_dims:
        for name, size in dims.items():
            if name in exempt:
                continue
            counts.setdefault(name, Counter())[size] += 1
    return {name: counter.most_common(1)[0][0]
            for name, counter in counts.items()}


def validate_model_files(
    file_list: list,
    engine: str,
    time_name: Optional[str],
    ofsfiletype: str,
    logger: Logger,
) -> tuple[list, list[tuple[str, str]]]:
    """
    Drop unreadable, incomplete, or structurally inconsistent files.

    Parameters
    ----------
    file_list : list of str
        Model file paths and/or remote URLs about to be opened together.
    engine : str
        xarray engine chosen for the batch (see netcdf_engine.py).
    time_name : str or None
        Name of the concat/time dimension ('time', 'ocean_time', ...).
    ofsfiletype : str
        'stations' or 'fields'. For stations files the station
        dimension is exempt from the consistency check (handled by
        remove_extra_stations downstream).
    logger : Logger
        Logger instance; one WARNING per dropped file plus a summary.

    Returns
    -------
    tuple of (list, list)
        ``(valid_files, dropped)`` where ``dropped`` holds
        ``(path, reason)`` pairs. Remote URLs are always passed
        through as valid.

    Raises
    ------
    SystemExit
        If the batch contained files but none survived validation.
    """
    local = [f for f in file_list if isinstance(f, str)
             and not _is_remote(f)]
    if not local:
        return list(file_list), []

    # netcdf4/scipy hold process-global C state, so per-file probing is
    # serialized for those engines; h5netcdf probes in parallel.
    if engine == 'h5netcdf' and len(local) > 1:
        with ThreadPoolExecutor(max_workers=min(8, len(local))) as pool:
            checks = list(pool.map(
                lambda p: _check_file(p, engine, time_name), local))
    else:
        checks = [_check_file(p, engine, time_name) for p in local]

    dropped: list[tuple[str, str]] = []
    surviving: dict[str, dict] = {}
    for path, (reason, dims) in zip(local, checks):
        if reason is None:
            surviving[path] = dims
        else:
            dropped.append((path, reason))

    # Batch-level consistency: drop files whose non-time dims disagree
    # with the batch majority (a minority file would crash the combine).
    exempt = {time_name} if time_name else set()
    if ofsfiletype == 'stations':
        exempt.add('station')
    if len(surviving) > 1:
        majority = _majority_dims(list(surviving.values()), exempt)
        for path, dims in list(surviving.items()):
            bad = {name: (size, majority[name])
                   for name, size in dims.items()
                   if name not in exempt and size != majority[name]}
            if bad:
                detail = ', '.join(
                    f'{name}={size} (batch majority {want})'
                    for name, (size, want) in sorted(bad.items()))
                dropped.append(
                    (path, f'dimension mismatch with batch: {detail}'))
                del surviving[path]

    if not dropped:
        return list(file_list), []

    for path, reason in dropped:
        logger.warning('Skipping model file %s: %s', path, reason)
    logger.warning(
        'Dropped %d of %d model file(s) that failed validation; '
        'continuing with the remaining %d. Skill results will not '
        'include data from the dropped files.',
        len(dropped), len(file_list), len(file_list) - len(dropped))

    dropped_set = {path for path, _ in dropped}
    valid = [f for f in file_list
             if not (isinstance(f, str) and f in dropped_set)]
    if not valid:
        logger.error(
            'All %d model files failed validation — cannot continue. '
            'See the warnings above for per-file reasons.',
            len(file_list))
        raise SystemExit
    return valid, dropped
