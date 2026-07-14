"""
Helpers to check whether cached time-series artifacts cover the run window.

The 1D pipeline caches its intermediate artifacts (``*_station.obs``,
``*_model.prd``, ``*_pair.int``) under filenames that do not encode the
assessment start/end dates. A persistent working directory can therefore
serve files left over from an earlier run window. Downstream, the plotting
step crops every series to the current window
(``combine_obs_across_casts``), so a stale artifact renders as a one-point
plot when the two windows are adjacent (daily operational runs share a
boundary timestamp) or as a blank plot when they are disjoint.

These helpers let the existence checks in ``get_skill`` and
``create_1dplot`` detect stale artifacts and trigger regeneration instead.
"""

from datetime import datetime

# An artifact is considered stale when its time range overlaps less than
# this fraction of the requested run window. Kept deliberately loose so
# genuine data gaps (late model cycles, short gauge outages) do not force
# pointless regeneration on every run.
MIN_OVERLAP_FRACTION = 0.5


def parse_run_window(prop, logger=None):
    """Parse ``prop.start_date_full``/``prop.end_date_full`` to datetimes.

    Both date formats used across the pipeline are accepted. Returns a
    ``(start, end)`` tuple, or ``None`` if the dates cannot be parsed (in
    which case callers should skip staleness checks).
    """
    start_raw = getattr(prop, 'start_date_full', None)
    end_raw = getattr(prop, 'end_date_full', None)
    for fmt in ('%Y-%m-%dT%H:%M:%SZ', '%Y%m%d-%H:%M:%S'):
        try:
            return (datetime.strptime(start_raw, fmt),
                    datetime.strptime(end_raw, fmt))
        except (TypeError, ValueError):
            continue
    if logger is not None:
        logger.warning(
            'Could not parse run window from start=%s end=%s; '
            'skipping cached-file staleness checks.',
            getattr(prop, 'start_date_full', None),
            getattr(prop, 'end_date_full', None))
    return None


def _row_datetime(line):
    """Timestamp of one whitespace-separated data row, or None.

    ``.obs``, ``.prd``, and ``*_pair.int`` files all carry
    ``year month day hour minute`` in columns 1-5 (0-based) after the
    julian-date column. Header rows and malformed lines return None.
    """
    fields = line.split()
    if len(fields) < 6:
        return None
    try:
        return datetime(int(float(fields[1])), int(float(fields[2])),
                        int(float(fields[3])), int(float(fields[4])),
                        int(float(fields[5])))
    except ValueError:
        return None


def read_first_last_timestamps(path):
    """Return (first, last) data-row timestamps of a cached artifact.

    Returns ``(None, None)`` when the file cannot be read or contains no
    parseable data rows.
    """
    first = None
    last = None
    try:
        with open(path, encoding='utf-8') as file_handle:
            for line in file_handle:
                stamp = _row_datetime(line)
                if stamp is None:
                    continue
                if first is None:
                    first = stamp
                last = stamp
    except OSError:
        return None, None
    return first, last


def covers_run_window(path, start_dt, end_dt,
                      min_fraction=MIN_OVERLAP_FRACTION, logger=None):
    """True if the file's time range overlaps enough of the run window.

    Fails open (returns True) when the file cannot be parsed, so
    unreadable files keep flowing through the pre-existing error handling
    instead of being regenerated on every run.
    """
    first, last = read_first_last_timestamps(path)
    if first is None or last is None:
        if logger is not None:
            logger.warning(
                'Could not read timestamps from %s; '
                'skipping staleness check for this file.', path)
        return True
    window_seconds = (end_dt - start_dt).total_seconds()
    if window_seconds <= 0:
        return True
    overlap_seconds = (min(end_dt, last)
                       - max(start_dt, first)).total_seconds()
    return overlap_seconds / window_seconds >= min_fraction
