"""
OFS Model Properties

This module defines the ModelProperties class which holds configuration
and path information for OFS model operations.
"""

import copy
from typing import Any

# Attributes excluded from deepcopy. ``_cached_model`` holds the loaded
# xarray model dataset (a dask graph referencing hundreds of backing
# files); deep-copying it is expensive and no consumer needs a private
# copy — every access goes through ``getattr(..., None)`` and treats a
# missing attribute as a cache miss. Without this guard, the per-station
# ``copy.deepcopy(prop)`` in the plotting fan-out duplicated the whole
# dataset twice per station.
_DEEPCOPY_SKIP_ATTRS = frozenset({'_cached_model', '_cached_model_key'})


class ModelProperties:
    """
    Properties and configuration for OFS Model operations.

    This class stores all configuration needed for model data processing,
    including OFS identifier, file paths, time ranges, and data paths.

    Attributes
    ----------
    ofs : str
        OFS identifier (e.g., 'cbofs', 'ngofs2')
    whichcast : str
        Forecast type (e.g., 'nowcast', 'forecast_a')
    whichcasts : str
        Comma-separated list of forecast types
    forecast_hr : str
        Forecast hour
    path : Path or str
        Root working directory path
    datum : str
        Vertical datum (e.g., 'MLLW', 'NAVD88')
    datum_list : str
        List of datums to process
    start_date_full : str
        Start date in ISO format (YYYY-MM-DDTHH:MM:SSZ)
    end_date_full : str
        End date in ISO format
    startdate : str
        Start date (shortened format)
    enddate : str
        End date (shortened format)
    ofsfiletype : str
        File type ('stations' or 'fields')
    control_files_path : str
        Path to control files
    model_path : str
        Path to model data
    ofs_extents_path : str
        Path to OFS extent shapefiles
    data_model_1d_node_path : str
        Path for 1D model node data
    data_model_2d_json_path : str
        Path for 2D model JSON data
    data_observations_1d_station_path : str
        Path for 1D observation station data
    data_observations_2d_station_path : str
        Path for 2D observation station data
    data_observations_2d_json_path : str
        Path for 2D observation JSON data
    data_skill_1d_pair_path : str
        Path for 1D skill paired data
    data_skill_1d_table_path : str
        Path for 1D skill tables
    data_skill_stats_path : str
        Path for skill statistics
    data_skill_2d_json_path : str
        Path for 2D skill JSON data
    visuals_1d_station_path : str
        Path for 1D station visualizations
    visuals_2d_station_path : str
        Path for 2D station visualizations
    data_skill_ice1dpair_path : str
        Path for ice 1D paired data
    visuals_maps_ice_path : str
        Path for ice map visualizations
    visuals_1d_ice_path : str
        Path for 1D ice visualizations
    visuals_stats_ice_path : str
        Path for ice statistics visualizations
    data_observations_2d_satellite_path : str
        Path for satellite observation data
    data_model_ice_path : str
        Path for model ice data
    model_source : str
        Model source type (e.g., 'fvcom', 'roms', 'schism')

    Examples
    --------
    >>> prop = ModelProperties()
    >>> prop.ofs = "cbofs"
    >>> prop.datum = "MLLW"
    >>> prop.path = Path("./")
    """

    def __init__(self):
        """Initialize ModelProperties with default values."""
        # Many of these attributes are reassigned downstream to bool/None
        # values (e.g. user_input_location is a bool once set from the CLI,
        # forecast_hr is Optional[str]). Typed as Any so dynamic attribute
        # sets from argparse don't trigger mypy assignment errors.
        self.ofs: Any = ''
        self.whichcast: Any = ''
        self.whichcasts: Any = ''
        self.forecast_hr: Any = ''
        self.path: Any = ''
        self.datum: Any = ''
        self.datum_list: Any = ''
        self.start_date_full: Any = ''
        self.end_date_full: Any = ''
        self.startdate: Any = ''
        self.enddate: Any = ''
        self.ofsfiletype: Any = ''
        self.stationowner: Any = ''
        self.user_input_location: Any = ''
        self.horizonskill: Any = ''
        self.var_list: Any = ''
        self.filecheck: Any = ''
        # Extension attrs set dynamically by various CLI entrypoints.
        self.currents_bins_csv: Any = None
        self.filepath: Any = ''
        # Build-ctl-only mode (issue #189): when True, get_node_ofs writes
        # the model control file(s) and a station-distance report, then
        # stops before extracting time series. build_ctl_map toggles the
        # accompanying interactive station-pair map.
        self.build_ctl_only: Any = False
        self.build_ctl_map: Any = True

        # Path attributes
        self.control_files_path: str = ''
        self.model_path: str = ''
        self.ofs_extents_path: str = ''
        self.data_model_1d_node_path: str = ''
        self.data_model_2d_json_path: str = ''
        self.data_observations_1d_station_path: str = ''
        self.data_observations_2d_station_path: str = ''
        self.data_observations_2d_json_path: str = ''
        self.data_skill_1d_pair_path: str = ''
        self.data_skill_1d_table_path: str = ''
        self.data_skill_stats_path: str = ''
        self.data_skill_2d_json_path: str = ''
        self.visuals_1d_station_path: str = ''
        self.visuals_2d_station_path: str = ''

        # Ice-specific paths & variables
        self.ice_dt: str = ''
        self.dailyavg: str = ''
        self.data_skill_ice1dpair_path: str = ''
        self.visuals_maps_ice_path: str = ''
        self.visuals_1d_ice_path: str = ''
        self.visuals_stats_ice_path: str = ''
        self.data_observations_2d_satellite_path: str = ''
        self.data_model_ice_path: str = ''

        self.model_source: str = ''
        self.config_file = None

    def __repr__(self) -> str:
        """String representation of ModelProperties."""
        return f"ModelProperties(ofs='{self.ofs}', datum='{self.datum}')"

    def __deepcopy__(self, memo):
        """Deep-copy all attributes except the cached model dataset.

        Copies come back without ``_cached_model`` / ``_cached_model_key``;
        consumers read those via ``getattr(..., None)`` and treat their
        absence as a cache miss, so behavior is unchanged apart from the
        copy no longer dragging a full dask graph along.
        """
        cls = self.__class__
        clone = cls.__new__(cls)
        memo[id(self)] = clone
        for key, value in self.__dict__.items():
            if key in _DEEPCOPY_SKIP_ATTRS:
                continue
            setattr(clone, key, copy.deepcopy(value, memo))
        return clone
