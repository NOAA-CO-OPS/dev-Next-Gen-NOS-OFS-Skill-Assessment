"""
STOFS-3D fields-mode station -> wet-node preprocessing.

This module implements a station-type-aware nearest-node selector used
ONLY for STOFS-3D-Atl / STOFS-3D-Pac ``fields`` output when extracting
1D time series for temperature, salinity, and currents. It is invoked as
an optional preprocessing hook from ``write_ofs_ctlfile`` and, when it
succeeds, overrides the geometric nearest node produced by the default
``indexing.index_nearest_node`` path with a node selected using the
model element connectivity and nodal bathymetry.

Rationale
---------
STOFS-3D fields files carry an unstructured SCHISM mesh. Picking the
single geometrically-closest node can land on a node whose column is
dry / too shallow for the station's sensor depth, which produces bad
skill statistics. Instead we mirror the approach validated in a
standalone STOFS-vs-OFS study:

CO-OPS temperature / salinity
    1. Resolve the sensor depth in the model datum (xgeoid20b) from the
       CO-OPS ``sensors`` + ``datums`` metadata, and the station's record
       low water level (``min``) also in the model datum, via the same
       ``vdatum_resilient.convert`` path the water-level code uses.
    2. Build a KDTree over element centroids (from
       ``SCHISM_hgrid_face_nodes`` + node x/y).
    3. Walk the nearest elements; accept the first element whose three
       (or four) corner nodes are ALL deeper than the record low
       (``depth > -min_wl``, SCHISM ``depth`` positive-down). The chosen
       node is the corner of that element closest to the station.

NDBC
    Far offshore; simply keep the geometric nearest wet node (the
    default indexing result), no element test.

USGS
    Not implemented yet (documented stub); will also require the
    bathymetry-based wet-node search once the exact rule is provided.

The module is intentionally self-contained and additive: if anything
fails (missing connectivity var, metadata error, datum-conversion
failure) it returns ``None`` for that station so the caller falls back
to the default geometric nearest node. It never raises out to the
caller for a single-station problem.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# OFS this preprocessing applies to.
STOFS3D_FIELDS_OFS = ('stofs_3d_atl', 'stofs_3d_pac')

# Variables (name_var) this preprocessing applies to.
# All variables use the full bathymetry-aware wet-node search (sensor
# depth + record-low water level from CO-OPS metadata). For 'wl' (water
# level, a 2-D surface variable) the resolved sensor depth is NOT
# back-patched into the obs depth because no vertical-layer search is
# needed downstream.
SUPPORTED_NAME_VARS = ('temp', 'salt', 'cu', 'wl')

# Number of nearest element centroids to probe per station before giving
# up on the "all corner nodes deep enough" test. The reference
# implementation used k=50; raised to 100 so shallow/edge stations have a
# better chance of finding a qualifying always-wet element before falling
# back to the default geometric node.
_KDTREE_K = 100

# SCHISM native datum for the STOFS-3D fields depth field. ``depth`` in
# out2d is referenced to xgeoid20b (positive-down).
_STOFS3D_NATIVE_DATUM = 'xgeoid20b'


def _face_node_arrays(model: Any, logger_: logging.Logger):
    """Return (raw_face_nodes, node_x, node_y, depth) numpy arrays.

    ``SCHISM_hgrid_face_nodes`` is a (face, maxnodes) connectivity array
    with 1-based node indices and ``NaN`` (or <= 0) padding for triangles
    inside a quad-capable array. ``depth`` is positive-down nodal
    bathymetry. Returns ``None`` if any required variable is absent.
    """
    required = (
        'SCHISM_hgrid_face_nodes',
        'SCHISM_hgrid_node_x',
        'SCHISM_hgrid_node_y',
        'depth',
    )
    for name in required:
        if name not in model.variables:
            logger_.warning(
                'STOFS-3D wet-node preprocessing: model is missing %r; '
                'falling back to default nearest-node selection.', name)
            return None
    try:
        # These grid variables are time-independent, but the combined
        # multi-file STOFS-3D dataset may carry a broadcast ``time`` dim
        # on them. Slice off time BEFORE materializing so we do not
        # allocate (ntime x nface x 4) — a multi-GiB blow-up. ``.values``
        # on the full dask-backed connectivity is what previously tried to
        # allocate 12 GiB.
        def _first_time_slice(var):
            if 'time' in var.dims:
                return var.isel(time=0)
            return var

        raw_nodes = np.asarray(
            _first_time_slice(model['SCHISM_hgrid_face_nodes']).values,
            dtype=float)
        node_x = np.asarray(
            _first_time_slice(model['SCHISM_hgrid_node_x']).values,
            dtype=float)
        node_y = np.asarray(
            _first_time_slice(model['SCHISM_hgrid_node_y']).values,
            dtype=float)
        depth = np.asarray(
            _first_time_slice(model['depth']).values, dtype=float)
    except (KeyError, ValueError, TypeError) as ex:
        logger_.warning(
            'STOFS-3D wet-node preprocessing: could not materialize grid '
            'arrays (%s); falling back to default selection.', ex)
        return None
    return raw_nodes, node_x, node_y, depth


def _build_centroid_tree(raw_nodes, node_x, node_y):
    """Build a KDTree over element centroids from face connectivity.

    Handles mixed triangle/quad meshes: padding (NaN or <= 0) is ignored
    and ``nanmean`` divides by the true corner count per element.

    Returns ``(tree, centroids_xy)`` or raises ImportError if scipy is
    unavailable (caller handles the fallback).
    """
    from scipy.spatial import KDTree  # local import: optional heavy dep

    mask = ~np.isnan(raw_nodes) & (raw_nodes > 0)
    node_indices = raw_nodes - 1  # SCHISM is 1-based
    lookup_idx = np.zeros(node_indices.shape, dtype=int)
    lookup_idx[mask] = node_indices[mask].astype(int)

    face_x = np.where(mask, node_x[lookup_idx], np.nan)
    face_y = np.where(mask, node_y[lookup_idx], np.nan)

    centroids_x = np.nanmean(face_x, axis=1)
    centroids_y = np.nanmean(face_y, axis=1)
    centroids_xy = np.column_stack((centroids_x, centroids_y))
    tree = KDTree(centroids_xy)
    return tree, centroids_xy


def _select_wet_node_by_element(
    obs_lon: float,
    obs_lat: float,
    threshold_depth: float,
    tree: Any,
    raw_nodes: np.ndarray,
    node_x: np.ndarray,
    node_y: np.ndarray,
    depth: np.ndarray,
    k: int = _KDTREE_K,
) -> int | None:
    """Return the 0-based node index for the closest qualifying element.

    Walks the ``k`` nearest element centroids to (obs_lon, obs_lat).
    Accepts the first element whose every real corner node has
    ``depth > threshold_depth`` (SCHISM depth is positive-down, so this
    means "the whole element is deeper than the threshold"). Within the
    accepted element, returns the corner node closest to the station.

    ``threshold_depth`` is expected to be ``-min_wl`` in the reference
    convention (record-low water level, positive-down in model datum),
    i.e. every node must be deeper than the lowest expected water column.
    Returns ``None`` when no element within the ``k`` nearest qualifies.
    """
    n_faces = raw_nodes.shape[0]
    k_eff = int(min(k, n_faces))
    _, idxs = tree.query([obs_lon, obs_lat], k=k_eff)
    idxs = np.atleast_1d(idxs)

    for face_idx in idxs:
        face_row = raw_nodes[int(face_idx)]
        valid = ~np.isnan(face_row) & (face_row > 0)
        valid_nodes = face_row[valid].astype(int) - 1  # 0-based
        if valid_nodes.size == 0:
            continue
        node_depths = depth[valid_nodes]
        if np.all(node_depths > threshold_depth):
            d2 = ((node_x[valid_nodes] - obs_lon) ** 2
                  + (node_y[valid_nodes] - obs_lat) ** 2)
            return int(valid_nodes[int(np.argmin(d2))])
    return None


def _resolve_coops_depths(
    prop: Any,
    station_id: str,
    obs_lat: float,
    obs_lon: float,
    logger_: logging.Logger,
) -> tuple[float, float] | None:
    """Resolve (sensor_depth_model, min_wl_model) for a CO-OPS station.

    Both values are returned in the model native datum (xgeoid20b),
    positive-down for use against SCHISM ``depth``.

    * ``sensor_depth_model``: the temperature/salinity sensor elevation
      converted to xgeoid20b and sign-flipped to positive-down. Used as
      the vertical interpolation target.
    * ``min_wl_model``: the station's record-low observed water level
      (``min`` from the datums endpoint) converted to xgeoid20b. The
      element wet test requires all corner-node bathymetry to be deeper
      than ``-min_wl_model``.

    Returns ``None`` when the metadata or datum conversion is
    unavailable so the caller can fall back.
    """
    # Local imports keep this module import-light and avoid a heavy
    # obs_retrieval import chain at model_processing import time.
    from ofs_skill.obs_retrieval import utils, vdatum_resilient
    from ofs_skill.obs_retrieval.retrieve_t_and_c_station import (
        get_station_datums,
        get_station_sensors,
    )

    _conf = getattr(prop, 'config_file', None)
    url_params = utils.Utils(_conf).read_config_section('urls', logger_)
    mdapi_url = url_params['co_ops_mdapi_base_url']

    variable = getattr(prop, '_sdp_variable', 'water_temperature')

    sensors = get_station_sensors(station_id, mdapi_url, logger_)
    datums = get_station_datums(station_id, mdapi_url, logger_)
    if sensors is None or datums is None:
        logger_.warning(
            'CO-OPS station %s: sensor/datum metadata unavailable; '
            'cannot resolve model-datum sensor depth.', station_id)
        return None

    sensor = _match_sensor(sensors, variable)
    if sensor is None:
        logger_.warning(
            'CO-OPS station %s: no matching %s sensor with an elevation; '
            'cannot resolve sensor depth.', station_id, variable)
        return None

    sensor_elev = sensor['elevation']
    sensor_refdatum = str(sensor['refdatum'])

    # Convert the sensor elevation (in its reference datum) to the model
    # datum. CO-OPS station datums are relative to STND; the datums
    # endpoint gives datum values in meters above STND. Convert the
    # sensor elevation to xgeoid20b via vdatum where possible, otherwise
    # bridge through NAVD88 offsets present in the datums payload.
    sensor_model = _elev_to_model_datum(
        sensor_elev, sensor_refdatum, datums, obs_lat, obs_lon,
        station_id, logger_, vdatum_resilient)
    if sensor_model is None:
        return None

    min_wl_stnd = datums.get('min')
    min_wl_model: float = sensor_model
    if min_wl_stnd is None:
        logger_.warning(
            'CO-OPS station %s: no record-low (min) water level in datums '
            'payload; using sensor depth as the wet-test threshold.',
            station_id)
    else:
        _min_wl_model = _elev_to_model_datum(
            float(min_wl_stnd), 'STND', datums, obs_lat, obs_lon,
            station_id, logger_, vdatum_resilient)
        if _min_wl_model is not None:
            min_wl_model = _min_wl_model

    # Positive-down for comparison against SCHISM depth. Elevations in a
    # geopotential datum are positive-up, so depth = -elevation.
    sensor_depth_model = -sensor_model
    min_wl_down = -min_wl_model
    return sensor_depth_model, min_wl_down


def _match_sensor(sensors: list, variable: str) -> dict | None:
    """Pick the sensor dict matching the variable with a real elevation."""
    name_key = {
        'water_temperature': 'water temperature',
        'salinity': 'conductivity',
    }.get(variable, 'water temperature')
    for s in sensors:
        try:
            name = str(s.get('name', '')).lower()
            elev = s.get('elevation')
        except AttributeError:
            continue
        if elev is None:
            continue
        if name_key in name:
            return {'elevation': float(elev),
                    'refdatum': s.get('refdatum', 'MLLW')}
    # Fallback: first sensor with an elevation.
    for s in sensors:
        elev = s.get('elevation')
        if elev is not None:
            return {'elevation': float(elev),
                    'refdatum': s.get('refdatum', 'MLLW')}
    return None


# Maps CO-OPS reference-datum strings to the datum names understood by
# ``coastalmodeling_vdatum``.  CO-OPS tidal datums that vdatum can build a
# direct pipeline to xgeoid20b for are listed here; anything not present
# (STND, MTL, DTL, GT, MN, ...) is handled by the NAVD88-offset bridge.
_COOPS_TO_VDATUM = {
    'MLLW': 'mllw',
    'MLW': 'mlw',
    'MSL': 'lmsl',
    'LMSL': 'lmsl',
    'MHW': 'mhw',
    'MHHW': 'mhhw',
    'NAVD88': 'navd88',
    'NAVD': 'navd88',
    'IGLD85': 'igld85',
    'IGLD': 'igld85',
    'LWD': 'lwd',
}


def _elev_to_model_datum(
    elevation: float,
    refdatum: str,
    datums: dict,
    lat: float,
    lon: float,
    station_id: str,
    logger_: logging.Logger,
    vdatum_resilient: Any,
) -> float | None:
    """Convert an elevation from ``refdatum`` to xgeoid20b (positive-up).

    Strategy:
    1. If ``refdatum`` is a tidal/vertical datum vdatum can build a
       pipeline for, convert it directly to xgeoid20b in one step.
    2. Otherwise (STND, MTL, GT, ... or a failed/non-finite direct
       conversion) fall back to bridging through NAVD88 using the CO-OPS
       datums payload offsets, then convert NAVD88 -> xgeoid20b.
    """
    ref_up = str(refdatum).upper()

    # --- 1. Direct conversion when vdatum supports the source datum. ---
    vd_from = _COOPS_TO_VDATUM.get(ref_up)
    if vd_from is not None:
        try:
            _, _, z = vdatum_resilient.convert(
                vd_from, _STOFS3D_NATIVE_DATUM, lat, lon, elevation,
                epoch=None, station_id=station_id, logger=logger_)
            if np.isfinite(z):
                return float(z)
            logger_.warning(
                'CO-OPS station %s: direct %s->xgeoid20b conversion '
                'returned a non-finite value; trying NAVD88 bridge.',
                station_id, vd_from)
        except Exception as ex:  # noqa: BLE001 - fall back to the bridge
            logger_.warning(
                'CO-OPS station %s: direct %s->xgeoid20b conversion failed '
                '(%s); trying NAVD88 bridge.', station_id, vd_from, ex)

    # --- 2. NAVD88-offset bridge fallback. ---
    datum_list = datums.get('datums') or []
    datum_vals = {str(d['name']).upper(): float(d['value'])
                  for d in datum_list
                  if d.get('value') is not None}
    # NAVD88 may live at the top level rather than inside 'datums'.
    navd88 = datum_vals.get('NAVD88')
    if navd88 is None and 'NAVD88' in datums:
        try:
            navd88 = float(datums['NAVD88'])
        except (TypeError, ValueError):
            navd88 = None
    if navd88 is None:
        logger_.warning(
            'CO-OPS station %s: no NAVD88 datum value available to bridge '
            '%s -> xgeoid20b.', station_id, refdatum)
        return None

    if ref_up == 'STND':
        ref_value = 0.0
    elif ref_up in datum_vals:
        ref_value = datum_vals[ref_up]
    elif ref_up == 'NAVD88':
        ref_value = navd88
    else:
        logger_.warning(
            'CO-OPS station %s: reference datum %r not found in datums '
            'payload; cannot convert to model datum.', station_id, refdatum)
        return None

    # Elevation is measured above ``refdatum``; express it above NAVD88.
    # datum values are heights above STND, so:
    #   height_above_STND = elevation + ref_value
    #   height_above_NAVD88 = height_above_STND - navd88
    elev_navd88 = (elevation + ref_value) - navd88

    # NAVD88 -> xgeoid20b conversion. vdatum returns z' referenced to the
    # target datum for an input z referenced to the source.
    try:
        _, _, z = vdatum_resilient.convert(
            'navd88', _STOFS3D_NATIVE_DATUM, lat, lon, elev_navd88,
            epoch=None, station_id=station_id, logger=logger_)
    except Exception as ex:  # noqa: BLE001 - fall back cleanly on any failure
        logger_.warning(
            'CO-OPS station %s: NAVD88->xgeoid20b conversion failed (%s); '
            'cannot resolve model-datum elevation.', station_id, ex)
        return None
    if not np.isfinite(z):
        logger_.warning(
            'CO-OPS station %s: NAVD88->xgeoid20b conversion returned a '
            'non-finite value.', station_id)
        return None
    return float(z)


def preprocess_stofs3d_nodes(
    prop: Any,
    extract: Any,
    default_nearest_node: list,
    model: Any,
    name_var: str,
    logger_: logging.Logger | None = None,
) -> list | None:
    """Override geometric nearest nodes with bathymetry-aware wet nodes.

    Called from ``write_ofs_ctlfile`` for STOFS-3D fields temp/salt/cu/wl.
    Returns a new ``list_of_nearest_node`` (same length/order as
    ``default_nearest_node``) or ``None`` to signal "use the default".

    Per-station behaviour by source (parsed from the obs ctl file):
    - ``CO-OPS`` : element wet test using the station's sensor depth and
      record-low water level resolved from CO-OPS metadata (same logic
      for all variables). For temp/salt/cu the resolved sensor depth is
      also back-patched into the obs depth so ``index_nearest_depth``
      picks the right vertical layer. For ``wl`` (2-D surface variable)
      the back-patch is skipped — no vertical-layer search is needed.
    - ``NDBC``   : keep the default geometric nearest node.
    - ``USGS``   : not yet implemented — keep default and log once.

    Any per-station failure falls back to the default node for that
    station without aborting the run.
    """
    log = logger_ or logger
    if prop.ofs not in STOFS3D_FIELDS_OFS:
        return None
    if getattr(prop, 'ofsfiletype', None) != 'fields':
        return None
    if name_var not in SUPPORTED_NAME_VARS:
        return None

    info_rows = extract[0]
    coord_rows = extract[-1]
    n = len(default_nearest_node)
    if n == 0:
        return None

    arrays = _face_node_arrays(model, log)
    if arrays is None:
        return None
    raw_nodes, node_x, node_y, depth = arrays

    try:
        tree, _ = _build_centroid_tree(raw_nodes, node_x, node_y)
    except ImportError:
        log.warning(
            'scipy KDTree unavailable; STOFS-3D wet-node preprocessing '
            'disabled, using default nearest-node selection.')
        return None
    except Exception as ex:  # noqa: BLE001 - never abort the run here
        log.warning(
            'STOFS-3D centroid tree build failed (%s); using default '
            'nearest-node selection.', ex)
        return None

    # STOFS-3D-Pac node_x is in [0, 360]; obs lon is in [-180, 180].
    # Normalize the tree/query space consistently: node coords are used
    # as-is and obs lon is shifted to match.
    pac_shift = prop.ofs == 'stofs_3d_pac'

    result = list(default_nearest_node)
    n_overridden = 0
    n_coops = 0
    usgs_warned = False

    for i in range(n):
        try:
            source = info_rows[i][3]
        except (IndexError, TypeError):
            source = ''
        source = str(source).upper()

        try:
            obs_lat = float(coord_rows[i][0])
            obs_lon = float(coord_rows[i][1])
        except (IndexError, TypeError, ValueError):
            continue

        if pac_shift and obs_lon < 0:
            obs_lon_q = obs_lon + 360
        else:
            obs_lon_q = obs_lon

        if 'CO-OPS' in source or 'COOPS' in source:
            station_id = info_rows[i][0]
            depths = _resolve_coops_depths(
                prop, str(station_id), obs_lat, obs_lon, log)
            if depths is None:
                continue  # keep default node for this station
            sensor_depth_model, min_wl_down = depths
            n_coops += 1
            node_idx = _select_wet_node_by_element(
                obs_lon_q, obs_lat, min_wl_down, tree,
                raw_nodes, node_x, node_y, depth)
            if node_idx is None:
                log.warning(
                    'CO-OPS station %s: no wet element found deeper than '
                    'record-low water (%.2f m); keeping default node.',
                    station_id, min_wl_down)
                continue
            result[i] = node_idx
            n_overridden += 1
            if name_var != 'wl':
                # Back-patch obs depth to the resolved sensor depth so the
                # vertical-layer search downstream targets the sensor level.
                # Water level is a 2-D surface variable — no depth needed.
                try:
                    coord_rows[i][3] = f'{sensor_depth_model:.2f}'
                except (IndexError, TypeError):
                    pass
            log.info(
                'CO-OPS station %s: wet node %d selected; '
                'record-low threshold %.2f m%s.',
                station_id, node_idx, min_wl_down,
                '' if name_var == 'wl'
                else f', sensor depth {sensor_depth_model:.2f} m (model datum)'
            )

        elif 'NDBC' in source:
            # Offshore: keep the default geometric nearest node.
            continue

        elif 'USGS' in source:
            if not usgs_warned:
                log.info(
                    'USGS STOFS-3D wet-node rule not yet implemented; '
                    'using default nearest-node selection for USGS '
                    'stations.')
                usgs_warned = True
            continue
        # Unknown source: leave default.

    if n_overridden == 0:
        log.info(
            'STOFS-3D wet-node preprocessing made no overrides for %s '
            '(%d CO-OPS stations processed); using default selection.',
            name_var, n_coops)
        return None

    log.info(
        'STOFS-3D wet-node preprocessing overrode %d of %d %s nodes.',
        n_overridden, n, name_var)
    return result
