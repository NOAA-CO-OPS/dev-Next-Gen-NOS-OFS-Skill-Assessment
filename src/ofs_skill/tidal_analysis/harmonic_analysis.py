"""
Core harmonic analysis module wrapping UTide to match NOS Fortran conventions.

Replaces the legacy ``harm29d.f``, ``harm15.f``, and ``lsqha.f`` programs.
UTide's iteratively-reweighted least-squares solver provides a single,
unified interface that automatically handles short, standard, and long
records.

References
----------
- Codiga, D.L. (2011). Unified Tidal Analysis and Prediction Using the
  UTide Matlab Functions.  Technical Report 2011-01, URI-GSO.
- Zhang et al. (2006). NOAA Technical Report NOS CS 24.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from utide import solve

from .constituents import NOS_37_CONSTITUENTS

logger = logging.getLogger(__name__)


def harmonic_analysis(
    time: pd.DatetimeIndex,
    values: np.ndarray,
    latitude: float,
    constit: list[str] | None = None,
    method: str = 'auto',
    min_duration_days: float = 15.0,
    logger: logging.Logger | None = None,
) -> dict:
    """
    Perform harmonic analysis on a water level or current time series.

    This is the Python replacement for ``harm29d.f`` (29-day Fourier HA),
    ``harm15.f`` (15-day Fourier HA), and ``lsqha.f`` (least-squares HA).
    Under the hood it calls :func:`utide.solve` with NOS-convention defaults.

    Parameters
    ----------
    time : pd.DatetimeIndex
        Timestamps of observations (UTC).
    values : np.ndarray
        Observed values (water level in metres, or current speed).
    latitude : float
        Station latitude in decimal degrees (needed for nodal corrections).
    constit : list of str, optional
        Constituent names to resolve.  If ``None`` (default), the NOS
        standard 37 constituents are requested; UTide will automatically
        drop any that cannot be separated given the record length.
    method : str, optional
        ``"auto"`` (default) — UTide selects internally.  Provided for
        forward-compatibility if explicit method selection is added later.
    min_duration_days : float, optional
        Minimum record length required (default 15.0 days).
    logger : logging.Logger, optional
        Logger instance for diagnostic messages.

    Returns
    -------
    dict
        ``"coef"`` : UTide coefficient structure (Bunch) —
            contains ``name``, ``A``, ``g``, ``A_ci``, ``g_ci``, ``mean``.
        ``"constituents"`` : :class:`pandas.DataFrame` —
            columns ``Name``, ``Amplitude``, ``Phase``, ``SNR``.
        ``"mean"`` : float — mean water level H0.
        ``"method_used"`` : str — description of effective method class.

    Raises
    ------
    ValueError
        If the record is shorter than *min_duration_days* or if *values*
        contains no finite data.
    """
    _log = logger or logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if len(time) != len(values):
        raise ValueError(
            f"time ({len(time)}) and values ({len(values)}) must have the "
            f"same length."
        )

    finite_mask = np.isfinite(values)
    if not np.any(finite_mask):
        raise ValueError('values contains no finite data.')

    duration_days = (time[-1] - time[0]).total_seconds() / 86400.0
    if duration_days < min_duration_days:
        raise ValueError(
            f"Record length {duration_days:.1f} days is less than the "
            f"minimum {min_duration_days} days required for harmonic analysis."
        )

    # ------------------------------------------------------------------
    # Constituent list
    # ------------------------------------------------------------------
    if constit is None:
        constit = list(NOS_37_CONSTITUENTS)

    method_label = _classify_method(duration_days)
    _log.info(
        'Running harmonic analysis: %.1f-day record, %d constituents '
        "requested, method class '%s'.",
        duration_days, len(constit), method_label,
    )

    # ------------------------------------------------------------------
    # Call UTide solver
    # ------------------------------------------------------------------
    coef = solve(
        t=time,
        u=values,
        lat=latitude,
        constit=constit,
        method='ols',           # ordinary least squares (closest to legacy)
        conf_int='linear',      # linear confidence intervals
        Rayleigh_min=0.9,       # constituent separation criterion
    )

    # ------------------------------------------------------------------
    # Package results
    # ------------------------------------------------------------------
    # SNR proxy: amplitude / half-width of 95 % confidence interval
    snr = coef.A / np.where(coef.A_ci > 0, coef.A_ci, np.inf)

    results_df = pd.DataFrame({
        'Name': coef.name,
        'Amplitude': coef.A,
        'Phase': coef.g,
        'SNR': snr,
    })

    mean_level = float(coef.mean) if hasattr(coef, 'mean') else np.nan
    _log.info(
        'Harmonic analysis complete. Mean=%.4f, %d constituents resolved.',
        mean_level, len(results_df),
    )

    return {
        'coef': coef,
        'constituents': results_df,
        'mean': mean_level,
        'method_used': method_label,
    }


def _classify_method(duration_days: float) -> str:
    """Return a human-readable label for the effective HA method class."""
    if duration_days < 20:
        return 'short_record'
    elif duration_days < 180:
        return 'standard'
    else:
        return 'long_record_lsq'
