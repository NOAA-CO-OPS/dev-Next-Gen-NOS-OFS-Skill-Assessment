"""
Observation Time Series Formatting

Format observation data from pandas DataFrames into standardized
text format for skill assessment.
"""

import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


def remove_sigma_outliers(
    timeseries: pd.DataFrame,
    column: str = 'OBS',
    n_sigma: float = 5.0,
    min_points: int = 10,
) -> pd.DataFrame:
    """
    Remove observations that fall outside a +/- n-sigma band about the mean.

    This reproduces the outlier-rejection performed by the legacy NOS Fortran
    skill-assessment code (``sorc/refwl.f``), which computes the mean and
    standard deviation over the entire water level series and discards any
    observation outside ``mean +/- 5 * std``. As in the Fortran code, the
    filter is only applied when at least ``min_points`` valid observations are
    present; otherwise the series is returned unchanged.

    Parameters
    ----------
    timeseries : pd.DataFrame
        DataFrame containing the observation column to filter.
    column : str, optional
        Name of the column holding observation values. Default is 'OBS'.
    n_sigma : float, optional
        Number of standard deviations that defines the acceptance band.
        Default is 5.0 to match ``refwl.f``.
    min_points : int, optional
        Minimum number of valid (non-NaN) observations required before the
        filter is applied. Default is 10 to match ``refwl.f``.

    Returns
    -------
    pd.DataFrame
        The input DataFrame with rows outside the +/- n-sigma band removed.
        The original DataFrame is not modified.

    Notes
    -----
    - NaN observations are ignored when computing the mean and standard
      deviation and are preserved in the returned frame (downstream code is
      responsible for NaN handling).
    - The sample standard deviation (ddof=1) is used, matching the
      ``SD = SQRT(SD/(NTMP-1))`` computation in ``refwl.f``.
    """
    if timeseries is None or timeseries.empty or column not in timeseries.columns:
        return timeseries

    values = pd.to_numeric(timeseries[column], errors='coerce')
    valid = values.dropna()

    # Match refwl.f: only filter when there are at least min_points obs.
    if len(valid) < min_points:
        return timeseries

    mean = valid.mean()
    std = valid.std(ddof=1)

    # If std is zero or not finite, there is nothing to reject.
    if not np.isfinite(std) or std == 0.0:
        return timeseries

    lower = mean - n_sigma * std
    upper = mean + n_sigma * std

    # Keep NaN rows (they are not outliers) and rows within the band.
    within_band = values.isna() | ((values >= lower) & (values <= upper))
    return timeseries.loc[within_band].copy()


def format_scalar(
    timeseries: pd.DataFrame,
    start_date_full: str,
    end_date_full: str,
    lookback_hours: int = 24,
) -> list[str]:
    """
    Format scalar observation data (water level, temperature, salinity).

    Converts pandas DataFrame to fixed-width formatted strings suitable
    for model-observation pairing.

    Parameters
    ----------
    timeseries : pd.DataFrame
        DataFrame with 'DateTime' and 'OBS' columns
    start_date_full : str
        Start date in format 'YYYYMMDD-HH:MM:SS'
    end_date_full : str
        End date in format 'YYYYMMDD-HH:MM:SS'
    lookback_hours : int, optional
        Number of hours before start_date to include in the output.
        Default is 24 hours. When used in nowcast+forecast_a mode,
        the caller can pass 24 to ensure overlap between casts.
        Set to 0 to disable the lookback.

    Returns
    -------
    List[str]
        List of formatted strings, one per observation

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'DateTime': pd.date_range('2025-01-01', periods=3, freq='H'),
    ...     'OBS': [1.23, 1.45, 1.67]
    ... })
    >>> formatted = format_scalar(df, '20250101-00:00:00', '20250101-02:00:00')
    >>> print(formatted[0])
    2460676.50000000 2025  1  1  0  0    1.2300

    Notes
    -----
    Output format (fixed-width columns):
        julian_date year month day hour minute value

    - julian_date: Julian date (float, 13.8 format)
    - year: 4-digit year
    - month, day, hour, minute: 2-digit integers
    - value: Observation value (9.4 format)

    Missing data (values < -999 or > 999) are converted to NaN.
    """
    # Parse date range
    start_dt_full = datetime.strptime(start_date_full, '%Y%m%d-%H:%M:%S')
    end_dt_full = datetime.strptime(end_date_full, '%Y%m%d-%H:%M:%S')

    # Filter to date range (lookback ensures overlap between consecutive casts)
    mask = (
        (timeseries['DateTime'] >= start_dt_full - timedelta(hours=lookback_hours)) &
        (timeseries['DateTime'] <= end_dt_full)
    )
    timeseries = timeseries.loc[mask].copy()

    # Calculate Julian date
    julian = pd.array(timeseries['DateTime']).to_julian_date()
    julian = julian.round(4)

    # Extract date components
    year = pd.to_datetime(timeseries['DateTime']).dt.strftime('%Y').to_numpy()
    month = pd.to_datetime(timeseries['DateTime']).dt.strftime('%m').to_numpy()
    day = pd.to_datetime(timeseries['DateTime']).dt.strftime('%d').to_numpy()
    hour = pd.to_datetime(timeseries['DateTime']).dt.strftime('%H').to_numpy()
    minute = pd.to_datetime(timeseries['DateTime']).dt.strftime('%M').to_numpy()

    # Filter out missing data values (< -999 or > 999)
    timeseries.loc[timeseries['OBS'] < -999, 'OBS'] = np.nan
    timeseries.loc[timeseries['OBS'] > 999, 'OBS'] = np.nan

    obs = timeseries['OBS'].to_numpy()

    # Format as fixed-width strings
    formatted_series = []
    for i in range(len(obs)):
        formatted_series.append(
            f'{float(julian[i]):13.8f} {int(year[i]):4d} {int(month[i]):2d} {int(day[i]):2d} {int(hour[i]):2d} {int(minute[i]):2d} {float(obs[i]):9.4f}'
        )

    return formatted_series


def format_vector(
    timeseries: pd.DataFrame,
    start_date_full: str,
    end_date_full: str,
    lookback_hours: int = 24,
) -> list[str]:
    """
    Format vector observation data (currents).

    Converts current speed and direction to u/v components and formats
    as fixed-width strings.

    Parameters
    ----------
    timeseries : pd.DataFrame
        DataFrame with 'DateTime', 'OBS' (speed), and 'DIR' (direction) columns
    start_date_full : str
        Start date in format 'YYYYMMDD-HH:MM:SS'
    end_date_full : str
        End date in format 'YYYYMMDD-HH:MM:SS'
    lookback_hours : int, optional
        Number of hours before start_date to include in the output.
        Default is 24 hours. When used in nowcast+forecast_a mode,
        the caller can pass 24 to ensure overlap between casts.
        Set to 0 to disable the lookback.

    Returns
    -------
    List[str]
        List of formatted strings, one per observation

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'DateTime': pd.date_range('2025-01-01', periods=2, freq='H'),
    ...     'OBS': [0.5, 0.6],  # m/s
    ...     'DIR': [90, 180]     # degrees clockwise from North
    ... })
    >>> formatted = format_vector(df, '20250101-00:00:00', '20250101-01:00:00')

    Notes
    -----
    Output format:
        julian_date year month day hour minute speed direction u v

    Direction convention:
    - Clockwise from North (meteorological convention)
    - u = speed * sin(direction)
    - v = speed * cos(direction)

    Missing data (values < -999 or > 999) are converted to NaN.
    """
    # Parse date range
    start_dt_full = datetime.strptime(start_date_full, '%Y%m%d-%H:%M:%S')
    end_dt_full = datetime.strptime(end_date_full, '%Y%m%d-%H:%M:%S')

    # Filter to date range (lookback ensures overlap between consecutive casts)
    mask = (
        (timeseries['DateTime'] >= start_dt_full - timedelta(hours=lookback_hours)) &
        (timeseries['DateTime'] <= end_dt_full)
    )
    timeseries = timeseries.loc[mask].copy()

    # Calculate Julian date
    julian = pd.array(timeseries['DateTime']).to_julian_date()
    julian = julian.round(4)

    # Extract date components
    year = pd.to_datetime(timeseries['DateTime']).dt.strftime('%Y').to_numpy()
    month = pd.to_datetime(timeseries['DateTime']).dt.strftime('%m').to_numpy()
    day = pd.to_datetime(timeseries['DateTime']).dt.strftime('%d').to_numpy()
    hour = pd.to_datetime(timeseries['DateTime']).dt.strftime('%H').to_numpy()
    minute = pd.to_datetime(timeseries['DateTime']).dt.strftime('%M').to_numpy()

    # Filter out missing data values
    timeseries.loc[timeseries['OBS'] < -999, 'OBS'] = np.nan
    timeseries.loc[timeseries['OBS'] > 999, 'OBS'] = np.nan
    timeseries.loc[timeseries['DIR'] < -999, 'DIR'] = np.nan
    timeseries.loc[timeseries['DIR'] > 999, 'DIR'] = np.nan

    obs = timeseries['OBS'].to_numpy()  # Speed
    ang = timeseries['DIR'].to_numpy()  # Direction

    # Convert to u,v components
    # Direction is clockwise from North, so:
    # u = speed * sin(direction)
    # v = speed * cos(direction)
    u, v = [], []
    for i in range(len(ang)):
        u.append(float(obs[i]) * math.sin(math.radians(float(ang[i]))))
        v.append(float(obs[i]) * math.cos(math.radians(float(ang[i]))))

    # Format as fixed-width strings
    formatted_series = []
    for i in range(len(obs)):
        formatted_series.append(
            f'{float(julian[i]):13.8f} {int(year[i]):4d} {int(month[i]):2d} {int(day[i]):2d} {int(hour[i]):2d} {int(minute[i]):2d} {float(obs[i]):9.4f} {float(ang[i]):9.4f} {float(u[i]):9.4f} {float(v[i]):9.4f}'
        )

    return formatted_series


# Legacy function names for backward compatibility
def scalar(timeseries: pd.DataFrame, start_date_full: str, end_date_full: str) -> list[str]:
    """Legacy function name - use format_scalar() instead."""
    return format_scalar(timeseries, start_date_full, end_date_full)


def vector(timeseries: pd.DataFrame, start_date_full: str, end_date_full: str) -> list[str]:
    """Legacy function name - use format_vector() instead."""
    return format_vector(timeseries, start_date_full, end_date_full)
