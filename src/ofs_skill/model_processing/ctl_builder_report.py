"""
Control-File Builder Reporting

Helpers that summarise a freshly built set of model control files so a
user can evaluate obs-model station matches *before* committing to a full
extraction/skill run (see issue #189).

Two artefacts are produced per OFS after ``write_ofs_ctlfile`` has written
the ``{ofs}_{var}_model_station.ctl`` / ``{ofs}_{var}_model.ctl`` files:

1. A CSV distance report (``{ofs}_ctl_station_pairs.csv``) listing every
   observation station. Matched stations carry the observation
   coordinates, the matched model node/point coordinates, and the
   great-circle distance between them, with a ``beyond_threshold`` column
   flagging pairs beyond the configured ``station_match_max_dist_km``
   cutoff. Unmatched stations carry ``matched=no`` and a ``reason``
   explaining why no model location was paired (e.g. "nearest model
   location 6.2 km away (> 4.0 km cutoff)").

2. An interactive plotly map (``{ofs}_ctl_station_pairs.html``) that plots
   each observation station and its matched model location, joined by a
   line, so mismatches jump out visually.

Both are best-effort: any variable whose ctl files are missing/blank is
simply skipped, and a failure to build the map never aborts the build.
"""

import csv
import os
from logging import Logger
from typing import Any

from ofs_skill.model_processing.station_distance import calculate_station_distance
from ofs_skill.obs_retrieval.station_ctl_file_extract import station_ctl_file_extract

# Variable -> ctl-file short name. Matches name_convent() in get_node_ofs.
_VAR_NAMES = {
    'water_level': 'wl',
    'water_temperature': 'temp',
    'salinity': 'salt',
    'currents': 'cu',
}

_CSV_FIELDS = [
    'variable',
    'station_id',
    'station_name',
    'obs_lat',
    'obs_lon',
    'matched',
    'model_node_index',
    'model_lat',
    'model_lon',
    'distance_km',
    'beyond_threshold',
    'reason',
]


def _model_ctl_path(prop: Any, name_var: str) -> str:
    """Return the model ctl path for this run's filetype."""
    if getattr(prop, 'ofsfiletype', 'stations') == 'fields':
        suffix = 'model'
    else:
        suffix = 'model_station'
    return os.path.join(
        prop.control_files_path, f'{prop.ofs}_{name_var}_{suffix}.ctl'
    )


def _load_model_ctl_rows(ctl_path: str) -> dict[str, dict[str, Any]]:
    """Parse a model ctl into ``{station_id: {node, lat, lon}}``.

    Model ctl format (space-delimited)::

        <node/station index> <layer> <lat> <lon> <station_id> <shift/depth>

    Missing files, blank files, and malformed rows yield an empty dict /
    are skipped so a partial ctl never aborts the report.
    """
    rows: dict[str, dict[str, Any]] = {}
    if not os.path.isfile(ctl_path) or os.path.getsize(ctl_path) == 0:
        return rows
    try:
        with open(ctl_path, encoding='utf-8') as fh:
            for raw in fh:
                parts = raw.split()
                # Need at least node, layer, lat, lon, id, trailing col.
                if len(parts) < 6:
                    continue
                try:
                    node = int(parts[0])
                    lat = float(parts[2])
                    lon = float(parts[3])
                except (TypeError, ValueError):
                    # Header row or malformed line -- skip.
                    continue
                station_id = parts[-2]
                rows[station_id] = {
                    'node': node, 'lat': lat, 'lon': lon,
                }
    except OSError:
        return rows
    return rows


def _load_obs_coords(
    ctl_path: str,
) -> dict[str, dict[str, Any]]:
    """Parse an obs station ctl into ``{station_id: {lat, lon, name}}``.

    Uses the shared ``station_ctl_file_extract`` so the parsing (including
    header handling and virtual-bin currents IDs) stays consistent with
    the rest of the package. Returns an empty dict when the file is
    missing/blank/unparseable.
    """
    coords: dict[str, dict[str, Any]] = {}
    extract = station_ctl_file_extract(ctl_path)
    if not extract:
        return coords
    station_info, coord_info = extract
    for info, coord in zip(station_info, coord_info):
        try:
            station_id = info[0]
            name = info[2] if len(info) > 2 else station_id
            lat = float(coord[0])
            lon = float(coord[1])
        except (IndexError, TypeError, ValueError):
            continue
        coords[station_id] = {'lat': lat, 'lon': lon, 'name': name}
    return coords


def _ledger_reasons(ledger: Any) -> dict[str, str]:
    """Return ``{station_id: reason}`` of drops recorded by the matcher.

    ``index_nearest_station`` records a ``node_match`` (or
    ``node_match_collision``) drop on the ledger for every obs station that
    failed the distance cutoff, with a fully-formed reason string (e.g.
    "nearest model location 6.2 km away (> 4.0 km cutoff)"). The ledger is
    shared across all requested variables in build mode, so the same obs
    station ID maps to the same reason regardless of variable. When
    multiple drops exist for one ID the last (most specific) reason wins.
    """
    reasons: dict[str, str] = {}
    if ledger is None:
        return reasons
    try:
        for rec in getattr(ledger, 'drops', []):
            reasons[str(rec.station_id)] = str(rec.reason)
    except (AttributeError, TypeError):
        return {}
    return reasons


def build_station_pair_records(
    prop: Any, logger: Logger, max_dist_km: float,
    ledger: Any = None,
) -> list[dict[str, Any]]:
    """Assemble obs/model station-pair records for all variables.

    For each variable in ``prop.var_list``, reads the obs station ctl and
    the model ctl written by ``write_ofs_ctlfile`` and joins them on
    station ID, computing the great-circle distance for each matched pair.

    Every observation station in the obs ctl produces a record:

    * ``matched='yes'`` rows carry the model node index, coordinates, and
      distance.
    * ``matched='no'`` rows carry a ``reason`` explaining why no model
      location was paired (pulled from the station-drop ``ledger`` when the
      matcher recorded one, e.g. "nearest model location 6.2 km away
      (> 4.0 km cutoff)"; a generic fallback is used otherwise).

    Returns a flat list of dict records (one per obs station per variable).
    Variables whose obs ctl is missing/blank contribute no records.
    """
    var_list = prop.var_list
    if isinstance(var_list, str):
        var_list = [v.strip() for v in var_list.split(',') if v.strip()]

    drop_reasons = _ledger_reasons(ledger)

    records: list[dict[str, Any]] = []
    for variable in var_list:
        name_var = _VAR_NAMES.get(variable)
        if name_var is None:
            continue

        obs_path = os.path.join(
            prop.control_files_path,
            f'{prop.ofs}_{name_var}_station.ctl',
        )
        obs_coords = _load_obs_coords(obs_path)
        model_rows = _load_model_ctl_rows(_model_ctl_path(prop, name_var))

        if not obs_coords:
            logger.info(
                'No obs stations to report for %s (%s); obs ctl '
                'missing or blank.', variable, name_var,
            )
            continue

        # Iterate over the OBS stations so unmatched ones are included. Any
        # model-ctl station not present in the obs ctl (unusual, but
        # possible) is appended afterwards so nothing is silently dropped.
        seen_ids: set[str] = set()
        for station_id, obs in obs_coords.items():
            seen_ids.add(station_id)
            model = model_rows.get(station_id)
            obs_lat = obs['lat']
            obs_lon = obs['lon']
            name = obs['name']

            if model is not None:
                dist = calculate_station_distance(
                    obs_lat, obs_lon, model['lat'], model['lon'],
                )
                beyond = dist > max_dist_km
                if beyond:
                    logger.warning(
                        'Station %s (%s): matched model location is %.2f '
                        'km away, beyond the %.1f km cutoff.',
                        station_id, variable, dist, max_dist_km,
                    )
                records.append({
                    'variable': variable,
                    'station_id': station_id,
                    'station_name': name,
                    'obs_lat': obs_lat,
                    'obs_lon': obs_lon,
                    'matched': 'yes',
                    'model_node_index': model['node'],
                    'model_lat': model['lat'],
                    'model_lon': model['lon'],
                    'distance_km': round(dist, 3),
                    'beyond_threshold': 'yes' if beyond else 'no',
                    'reason': '',
                })
            else:
                # No model match. Prefer the matcher's recorded reason;
                # fall back to a generic explanation.
                reason = drop_reasons.get(
                    station_id,
                    f'no model location within the {max_dist_km:.1f} km '
                    f'cutoff',
                )
                logger.info(
                    'Station %s (%s) unmatched: %s',
                    station_id, variable, reason,
                )
                records.append({
                    'variable': variable,
                    'station_id': station_id,
                    'station_name': name,
                    'obs_lat': obs_lat,
                    'obs_lon': obs_lon,
                    'matched': 'no',
                    'model_node_index': '',
                    'model_lat': '',
                    'model_lon': '',
                    'distance_km': '',
                    'beyond_threshold': '',
                    'reason': reason,
                })

        # Model-ctl entries with no corresponding obs station (rare).
        for station_id, model in model_rows.items():
            if station_id in seen_ids:
                continue
            records.append({
                'variable': variable,
                'station_id': station_id,
                'station_name': '',
                'obs_lat': '',
                'obs_lon': '',
                'matched': 'yes',
                'model_node_index': model['node'],
                'model_lat': model['lat'],
                'model_lon': model['lon'],
                'distance_km': '',
                'beyond_threshold': '',
                'reason': 'model ctl station not found in obs ctl',
            })

    return records


def write_distance_report(
    prop: Any, records: list[dict[str, Any]], logger: Logger,
) -> str | None:
    """Write the station-pair distance CSV. Returns the path, or None.

    The report is written to ``prop.control_files_path`` next to the ctl
    files it summarises so it travels with them.
    """
    if not records:
        logger.info('No matched station pairs to write to the CSV report.')
        return None

    out_path = os.path.join(
        prop.control_files_path, f'{prop.ofs}_ctl_station_pairs.csv',
    )
    try:
        with open(out_path, 'w', encoding='utf-8', newline='') as fh:
            writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDS)
            writer.writeheader()
            writer.writerows(records)
    except OSError as ex:
        logger.error('Could not write ctl distance report: %s', ex)
        return None

    logger.info(
        'Wrote control-file distance report (%d pairs): %s',
        len(records), out_path,
    )
    return out_path


def make_station_pair_map(
    prop: Any, records: list[dict[str, Any]], logger: Logger,
) -> str | None:
    """Build an interactive plotly map of obs/model station pairs.

    Each matched observation station and its model location are plotted
    and joined by a line so the user can eyeball match quality across the
    OFS. Pairs beyond the distance threshold are drawn with a red
    connector. Unmatched observation stations (with obs coordinates but no
    model match) are plotted as red X markers so they are easy to spot.
    Returns the saved HTML path, or None on failure/no data.
    """
    # Records with obs coordinates split into matched (have model coords)
    # and unmatched (no model match).
    with_obs = [
        r for r in records
        if r['obs_lat'] != '' and r['obs_lon'] != ''
    ]
    matched = [r for r in with_obs if r['matched'] == 'yes'
               and r['model_lat'] != '' and r['model_lon'] != '']
    unmatched = [r for r in with_obs if r['matched'] != 'yes']

    if not with_obs:
        logger.info('No mappable stations; skipping the pair map.')
        return None

    try:
        import plotly.graph_objects as go  # noqa: PLC0415
    except ImportError:
        logger.warning(
            'plotly is not available; skipping the station-pair map.')
        return None

    out_dir = getattr(prop, 'plotly_maps', None) or prop.control_files_path
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(
        out_dir, f'{prop.ofs}_ctl_station_pairs.html',
    )

    try:
        fig = go.Figure()

        # Connector lines (one trace per pair keeps the hover clean and
        # lets us colour beyond-threshold pairs red).
        legend_ok_shown = False
        legend_bad_shown = False
        for rec in matched:
            beyond = rec['beyond_threshold'] == 'yes'
            colour = 'red' if beyond else 'gray'
            show_legend = (
                (beyond and not legend_bad_shown)
                or (not beyond and not legend_ok_shown)
            )
            if beyond:
                legend_bad_shown = legend_bad_shown or show_legend
            else:
                legend_ok_shown = legend_ok_shown or show_legend
            fig.add_trace(go.Scattermap(
                mode='lines',
                lon=[rec['obs_lon'], rec['model_lon']],
                lat=[rec['obs_lat'], rec['model_lat']],
                line=dict(color=colour, width=1),
                hoverinfo='skip',
                showlegend=show_legend,
                legendgroup='beyond' if beyond else 'within',
                name=(
                    'Beyond threshold' if beyond else 'Within threshold'
                ),
            ))

        # Matched observation station markers.
        if matched:
            fig.add_trace(go.Scattermap(
                mode='markers',
                lon=[r['obs_lon'] for r in matched],
                lat=[r['obs_lat'] for r in matched],
                marker=dict(size=9, color='#1f77b4'),
                name='Observation station',
                text=[
                    f"{r['station_id']} ({r['variable']})<br>"
                    f"{r['station_name']}<br>"
                    f"dist: {r['distance_km']} km"
                    for r in matched
                ],
                hoverinfo='text',
            ))

            # Matched model location markers.
            fig.add_trace(go.Scattermap(
                mode='markers',
                lon=[r['model_lon'] for r in matched],
                lat=[r['model_lat'] for r in matched],
                marker=dict(size=7, color='#ff7f0e'),
                name='Matched model node',
                text=[
                    f"node {r['model_node_index']} ({r['variable']})<br>"
                    f"dist: {r['distance_km']} km"
                    for r in matched
                ],
                hoverinfo='text',
            ))

        # Unmatched observation station markers -- red so they stand out.
        if unmatched:
            fig.add_trace(go.Scattermap(
                mode='markers',
                lon=[r['obs_lon'] for r in unmatched],
                lat=[r['obs_lat'] for r in unmatched],
                marker=dict(size=10, color='red'),
                name='Unmatched observation',
                text=[
                    f"{r['station_id']} ({r['variable']})<br>"
                    f"{r['station_name']}<br>"
                    f"UNMATCHED: {r['reason']}"
                    for r in unmatched
                ],
                hoverinfo='text',
            ))
        # Collect all valid latitudes and longitudes
        all_lats = [r['obs_lat'] for r in with_obs] + [r['model_lat'] for r in matched]
        all_lons = [r['obs_lon'] for r in with_obs] + [r['model_lon'] for r in matched]

        # Calculate exact center
        center_lat = (min(all_lats) + max(all_lats)) / 2.0
        center_lon = (min(all_lons) + max(all_lons)) / 2.0

        # Calculate coordinate range
        lat_diff = max(all_lats) - min(all_lats)
        lon_diff = max(all_lons) - min(all_lons)

        if lat_diff == 0 and lon_diff == 0:
            # Fallback for a single station
            auto_zoom = 8.0
        else:
            import math
            # Prevent division by zero if all stations share a single axis
            lat_diff = max(lat_diff, 0.001)
            lon_diff = max(lon_diff, 0.001)

            # Map tiles use a base-2 logarithmic scale.
            # 180 degrees (lat) and 360 degrees (lon) represent the full globe.
            zoom_lat = math.log2(180 / lat_diff)
            zoom_lon = math.log2(360 / lon_diff)

            # Take the smaller zoom to fit both dimensions, minus 1.5 for edge padding
            auto_zoom = min(zoom_lat, zoom_lon) - 1.5

        fig.update_layout(
            map_style='carto-positron',
            map=dict(
                center=dict(lat=center_lat, lon=center_lon),
                zoom=auto_zoom
            ),
            height=700,
            width=1000,
            title=dict(
                text=(
                    f'{prop.ofs.upper()} observation-model station pairs'
                ),
                x=0.5,
            ),
            legend=dict(x=0.01, y=0.99),
            margin=dict(l=0, r=0, t=50, b=0),
        )

        import plotly  # noqa: PLC0415
        plotly.offline.plot(
            fig, filename=out_path, auto_open=False,
            config={'scrollZoom': True},
        )
    except Exception as ex:  # pylint: disable=broad-exception-caught
        logger.error('Could not build the station-pair map: %s', ex)
        return None

    logger.info('Wrote station-pair map: %s', out_path)
    return out_path


def report_ctl_matches(
    prop: Any, logger: Logger, max_dist_km: float,
    make_map: bool = True, ledger: Any = None,
) -> list[dict[str, Any]]:
    """Top-level entry: build records, write the CSV, and (opt.) the map.

    ``ledger`` is an optional :class:`StationLedger` carrying the
    per-station drop reasons recorded by the matcher, used to explain
    unmatched stations in the report.

    Returns the list of station-pair records so callers/tests can inspect
    them. Never raises on reporting failures -- the ctl files are already
    written and are the primary deliverable.
    """
    records = build_station_pair_records(
        prop, logger, max_dist_km, ledger=ledger)
    write_distance_report(prop, records, logger)
    if make_map:
        make_station_pair_map(prop, records, logger)
    return records
