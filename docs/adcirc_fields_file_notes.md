# STOFS-2D-Global (ADCIRC) fields file: observed facts

These notes record what one real STOFS-2D-Global fields file contains. They are input for implementing the ADCIRC branch of `index_nearest_node` in `src/ofs_skill/model_processing/indexing.py`, which still raises `NotImplementedError` ([indexing.py L520–521](https://github.com/NOAA-CO-OPS/dev-Next-Gen-NOS-OFS-Skill-Assessment/blob/b3744b815ce7e86e86e68452b5f4becc918d5e0b/src/ofs_skill/model_processing/indexing.py#L520-L521)).

- Inspected 2026-10-21, against repo commit `b3744b815ce7e86e86e68452b5f4becc918d5e0b`.
- No source code was changed.

## File inspected

| Item | Value |
|---|---|
| File | `stofs_2d_glo.t00z.fields.cwl.nc`, cycle 2026-10-20 00z. The name matches what [list_of_files.py L489–492](https://github.com/NOAA-CO-OPS/dev-Next-Gen-NOS-OFS-Skill-Assessment/blob/b3744b815ce7e86e86e68452b5f4becc918d5e0b/src/ofs_skill/model_processing/list_of_files.py#L489-L492) builds. |
| Source | <https://noaa-gestofs-pds.s3.amazonaws.com/stofs_2d_glo.20261020/stofs_2d_glo.t00z.fields.cwl.nc> |
| Local copy | `~/.ofs_cache/s3/stofs_2d_glo.20261020/stofs_2d_glo.t00z.fields.cwl.nc`. This is the default cache root from [utils.py `get_s3_cache_dir`](https://github.com/NOAA-CO-OPS/dev-Next-Gen-NOS-OFS-Skill-Assessment/blob/b3744b815ce7e86e86e68452b5f4becc918d5e0b/src/ofs_skill/obs_retrieval/utils.py#L364-L389), using the layout from [model_file_validation.py `cached_path_for_url`](https://github.com/NOAA-CO-OPS/dev-Next-Gen-NOS-OFS-Skill-Assessment/blob/b3744b815ce7e86e86e68452b5f4becc918d5e0b/src/ofs_skill/model_processing/model_file_validation.py#L83-L111). |
| Size | 10,117,858,606 bytes (9.42 GiB). The downloaded size matches the S3 listing. |
| Format | netCDF-4 / HDF5: the magic bytes are `\x89HDF\r\n\x1a\n`, and `ncdump -k` reports `netCDF-4`. No variable is compressed (`zlib=False` everywhere). |
| Engine the repo picks | `h5netcdf`, chosen by [netcdf_engine.py `resolve_engine`](https://github.com/NOAA-CO-OPS/dev-Next-Gen-NOS-OFS-Skill-Assessment/blob/b3744b815ce7e86e86e68452b5f4becc918d5e0b/src/ofs_skill/model_processing/netcdf_engine.py#L201-L243) when sniffing the local file. This agrees with [netcdf_engine_test.py L186–192](https://github.com/NOAA-CO-OPS/dev-Next-Gen-NOS-OFS-Skill-Assessment/blob/b3744b815ce7e86e86e68452b5f4becc918d5e0b/tests/netcdf_engine_test.py#L186-L192). |
| Model / run | ADCIRC, `runid = "STOFS 2D GLOBAL v5.6.5"`. `rundes = "2026102000 :-6 hr nowcast and +180 hr forecast"`. See the raw header below. |

## Answers to the seven questions

| # | Question | Finding |
|---|---|---|
| 1 | Variables; do `x`, `y`, `zeta`, `time` exist; node dimension | All four exist. The node dimension is **`node`**. `x(node)` and `y(node)` are float64. `zeta(time, node)` is float32. `time(time)` is float64, in seconds. There are **no `lon`/`lat` variables**. The full table is below. |
| 2 | Range of `x` and `y` | `x` runs from −179.9999897 to 179.9999522, so the convention is **−180..180**. No value is above 180, 10,305,231 are below 0, and none is exactly ±180. `y` runs from −85.2305 to 89.8863. Neither has NaNs. There are 5,958 nodes with \|x\| > 179.9 (3,095 east of 179.9 and 2,863 west of −179.9), so the mesh is continuous across the antimeridian. |
| 3 | `zeta` fill attributes; nodes that are always missing | `zeta` has **no `_FillValue` and no `missing_value` attribute**. Its only attributes are `units` and `long_name`. The file has a **global** attribute `_FillValue = -99999.`. Dry or land values are stored as **−99999.0**, not NaN. **0 nodes are all-NaN.** No NaN appears anywhere in `zeta`. **3,598,895 nodes (28.15 %) are −99999.0 at all 186 steps.** 4,034,088 nodes are −99999.0 at least once. Each step has 3,735,849–3,866,931 such values. The valid (non-fill) range is −6.742 to 8.628 m. No value equals the netCDF default fill (9.969e36). |
| 4 | Node count | **12,785,004** (`node` dimension). |
| 5 | Time step and count | **186 steps**, uniformly **3600 s** apart. The first step is 2026-10-19T19:00Z, which is cycle − 5 h; the last is 2026-10-27T12:00Z, which is cycle + 180 h. That makes 6 nowcast steps followed by 180 forecast steps. The **1 step per hour assumption** in [intake_scisa.py L951–954](https://github.com/NOAA-CO-OPS/dev-Next-Gen-NOS-OFS-Skill-Assessment/blob/b3744b815ce7e86e86e68452b5f4becc918d5e0b/src/ofs_skill/model_processing/intake_scisa.py#L951-L954) **holds**. So does the run length `n_t_total = 6 + 180 = 186` ([L957–960](https://github.com/NOAA-CO-OPS/dev-Next-Gen-NOS-OFS-Skill-Assessment/blob/b3744b815ce7e86e86e68452b5f4becc918d5e0b/src/ofs_skill/model_processing/intake_scisa.py#L957-L960); the 4 cycles/day and 180 h forecast come from [get_fcst_cycle.py L197–220](https://github.com/NOAA-CO-OPS/dev-Next-Gen-NOS-OFS-Skill-Assessment/blob/b3744b815ce7e86e86e68452b5f4becc918d5e0b/src/ofs_skill/model_processing/get_fcst_cycle.py#L197-L220)). The `time:units` string has a trailing comment (`"seconds since 2026-05-26 12:00:00        ! NCDASE - BASE_DAT"`), but xarray still decodes it to `datetime64[ns]` correctly. |
| 6 | Format and size | Same as the table above: netCDF-4/HDF5, uncompressed, 10,117,858,606 bytes. |
| 7 | Drop list against actual variables | The list at [intake_scisa.py L284–287](https://github.com/NOAA-CO-OPS/dev-Next-Gen-NOS-OFS-Skill-Assessment/blob/b3744b815ce7e86e86e68452b5f4becc918d5e0b/src/ofs_skill/model_processing/intake_scisa.py#L284-L287) is `nvel, element, adcirc_mesh, nvell, max_nvell, ibtype, nbvv`. Five of these are variables in the file and get dropped: `element`, `adcirc_mesh`, `nvell`, `ibtype`, `nbvv`. `nvel` and `max_nvell` are **dimension names, not variables**, so dropping them has no effect; xarray accepts the names without error. **Kept but not used by the pipeline:** `depth(node)` is float64 and 102,280,032 bytes. `barsp`, `barht`, `ibconn` and `barsb` are all `(nvel=12421)` and tiny. The largest dropped variable is `element` at 298,504,032 bytes. `zeta` is 9,512,042,976 bytes and must be kept. |

## Variables

| Variable | Dimensions | dtype | Shape | Storage | Dropped by intake? |
|---|---|---|---|---|---|
| `time` | (time) | float64 | (186,) | chunked [512] | no |
| `x` | (node) | float64 | (12785004,) | contiguous | no (becomes a coordinate) |
| `y` | (node) | float64 | (12785004,) | contiguous | no (becomes a coordinate) |
| `element` | (nfaces, nvertex) | int32 | (24875336, 3) | contiguous | **yes** |
| `adcirc_mesh` | () | int32 | () | contiguous | **yes** |
| `nvell` | (nbou) | int32 | (262,) | contiguous | **yes** |
| `ibtype` | (nbou) | int32 | (262,) | contiguous | **yes** |
| `nbvv` | (nvel) | int32 | (12421,) | contiguous | **yes** |
| `barsp` | (nvel) | float64 | (12421,) | contiguous | no |
| `barht` | (nvel) | float64 | (12421,) | contiguous | no |
| `ibconn` | (nvel) | int32 | (12421,) | contiguous | no |
| `barsb` | (nvel) | float64 | (12421,) | contiguous | no |
| `depth` | (node) | float64 | (12785004,) | contiguous | no |
| `zeta` | (time, node) | float32 | (186, 12785004) | chunked [1, 3196251] | no |

Dimensions: `time` = 186 (UNLIMITED), `node` = 12,785,004, `nele` = 24,875,336 (no variable uses it), `nfaces` = 24,875,336, `nvertex` = 3, `nbou` = 262, `nvel` = 12,421, `max_nvell` = 1,772.

`depth` ranges from −648.2 to 10,657.3 m, and 4,098,765 nodes are negative (above the geoid). Of the 3,598,895 nodes that are always −99999.0, 3,596,912 have `depth` ≤ 0 and 1,983 have `depth` > 0.

## Repo intake path on this file

I called [`intake_model`](https://github.com/NOAA-CO-OPS/dev-Next-Gen-NOS-OFS-Skill-Assessment/blob/b3744b815ce7e86e86e68452b5f4becc918d5e0b/src/ofs_skill/model_processing/intake_scisa.py#L180-L576) with `[local_path]` and `model_source='adcirc'`, `ofs='stofs_2d_glo'`, `ofsfiletype='fields'`. It applies [`fix_adcirc_dataset`](https://github.com/NOAA-CO-OPS/dev-Next-Gen-NOS-OFS-Skill-Assessment/blob/b3744b815ce7e86e86e68452b5f4becc918d5e0b/src/ofs_skill/model_processing/intake_scisa.py#L874-L1020) itself at [L540–541](https://github.com/NOAA-CO-OPS/dev-Next-Gen-NOS-OFS-Skill-Assessment/blob/b3744b815ce7e86e86e68452b5f4becc918d5e0b/src/ofs_skill/model_processing/intake_scisa.py#L540-L541). I ran it once for each `whichcast`, then called [`indexing._static_coord_1d`](https://github.com/NOAA-CO-OPS/dev-Next-Gen-NOS-OFS-Skill-Assessment/blob/b3744b815ce7e86e86e68452b5f4becc918d5e0b/src/ofs_skill/model_processing/indexing.py#L79-L121) on the result.

| whichcast | `time` length after intake | Unique times | First → last | `x` / `y` dims | `_static_coord_1d('x')` / `('y')` |
|---|---|---|---|---|---|
| nowcast | 12 | 6 | 2026-10-19T19:00 → 2026-10-20T00:00 | `('node',)` / `('node',)` | 1-D `ndarray`, shape (12785004,), float64 |
| forecast_b | 12 | 6 | 2026-10-20T01:00 → 2026-10-20T06:00 | `('node',)` / `('node',)` | same |
| forecast_a | 360 | 180 | 2026-10-20T01:00 → 2026-10-27T12:00 | `('node',)` / `('node',)` | same |

- `x` and `y` come out as **`(node,)` coordinates**, not data variables. `depth` carries `coordinates = "time y x"`, and xarray promotes both to coordinates from that.
- `_static_coord_1d` returns 1-D arrays for both, because `node` is not in `_TIME_CONCAT_DIM_CANDIDATES = ('time', 'ocean_time')`.
- The data variables after intake are `zeta(time, node)`, `depth(node)`, and the four `(nvel)` variables. A scalar `filename` coordinate is added.
- Reading permanently dry nodes through xarray/h5netcdf (the same engine intake uses) returns **−99999.0**, not NaN. For one such node, `zeta.isnull().all()` over time is **False**.
- `ofs_skill/__init__.py` and `model_processing/__init__.py` import the full pipeline. To load only `intake_scisa` and `indexing`, the three parent packages were registered as stub modules whose `__path__` points at the real source directories, so the real submodules ran unchanged.
  - The environment was a scratch venv with Python 3.12, xarray 2026.9.0, intake 2.0.9, intake-xarray 2.0.0, dask 2026.8.0, h5netcdf 1.8.1, netCDF4 1.7.5 and numpy 2.5.4.
  - There is no `conf/ofs_dps.conf`, so the run used `conf/ofs_dps.conf.example` as its config fallback.

## Contradictions with current assumptions

1. **Dry nodes are −99999.0, not NaN.**
   - Nothing in the intake path masks them, because `zeta` has no `_FillValue` or `missing_value` attribute. Only the global attribute carries `-99999.`.
   - `grep` for `99999` (excluding `999999`) in `src/` and `bin/` finds no matches.
   - These existing checks therefore will not detect dry ADCIRC nodes:
     - the all-NaN tests in the STOFS-3D branch of `index_nearest_node` ([indexing.py L504–505](https://github.com/NOAA-CO-OPS/dev-Next-Gen-NOS-OFS-Skill-Assessment/blob/b3744b815ce7e86e86e68452b5f4becc918d5e0b/src/ofs_skill/model_processing/indexing.py#L504-L505));
     - the `np.isnan` check in the writer ([write_ofs_ctlfile.py L746–750](https://github.com/NOAA-CO-OPS/dev-Next-Gen-NOS-OFS-Skill-Assessment/blob/b3744b815ce7e86e86e68452b5f4becc918d5e0b/src/ofs_skill/model_processing/write_ofs_ctlfile.py#L746-L750)).
   - 28 % of nodes are always −99999.0, and another 435,193 are −99999.0 at some steps.
2. **The coordinates are `x`/`y`, not `lon`/`lat`.**
   - The [`index_nearest_node` docstring (L149)](https://github.com/NOAA-CO-OPS/dev-Next-Gen-NOS-OFS-Skill-Assessment/blob/b3744b815ce7e86e86e68452b5f4becc918d5e0b/src/ofs_skill/model_processing/indexing.py#L149) says ADCIRC uses `'lon', 'lat'`.
   - So does the static-coordinate preload list ([get_node_ofs.py L376](https://github.com/NOAA-CO-OPS/dev-Next-Gen-NOS-OFS-Skill-Assessment/blob/b3744b815ce7e86e86e68452b5f4becc918d5e0b/src/ofs_skill/model_processing/get_node_ofs.py#L376)). That function skips names that are missing ([L381–382](https://github.com/NOAA-CO-OPS/dev-Next-Gen-NOS-OFS-Skill-Assessment/blob/b3744b815ce7e86e86e68452b5f4becc918d5e0b/src/ofs_skill/model_processing/get_node_ofs.py#L381-L382)), so with this file it would preload nothing for ADCIRC.
   - The ADCIRC writer branch already uses `x`/`y` ([write_ofs_ctlfile.py L959–981](https://github.com/NOAA-CO-OPS/dev-Next-Gen-NOS-OFS-Skill-Assessment/blob/b3744b815ce7e86e86e68452b5f4becc918d5e0b/src/ofs_skill/model_processing/write_ofs_ctlfile.py#L959-L981)).
3. **Fields files are about 10 GB, not 100–500 MB.**
   - The comment at [intake_scisa.py L403–408](https://github.com/NOAA-CO-OPS/dev-Next-Gen-NOS-OFS-Skill-Assessment/blob/b3744b815ce7e86e86e68452b5f4becc918d5e0b/src/ofs_skill/model_processing/intake_scisa.py#L403-L408) gives 100–500 MB, and remote fields files are not cached.
   - This `cwl` file is 10.1 GB. The same-cycle `fields.cwl.noanomaly.nc` and `fields.htp.nc` are 19.6 GB each, and `fields.cwl.vel.nc` is 38.7 GB (from the S3 listing of `stofs_2d_glo.20261020/`).
4. **The guard message is outdated.** [get_node_ofs.py L1629–1635](https://github.com/NOAA-CO-OPS/dev-Next-Gen-NOS-OFS-Skill-Assessment/blob/b3744b815ce7e86e86e68452b5f4becc918d5e0b/src/ofs_skill/model_processing/get_node_ofs.py#L1629-L1635) says "STOFS-2D-Global is only available as station output files". Fields files are published on NODD; one is the file inspected here.
5. **Each timestamp appears twice in a single-file fields run.**
   - [intake_scisa.py L316–317](https://github.com/NOAA-CO-OPS/dev-Next-Gen-NOS-OFS-Skill-Assessment/blob/b3744b815ce7e86e86e68452b5f4becc918d5e0b/src/ofs_skill/model_processing/intake_scisa.py#L316-L317) duplicates a one-file list. Duplicate times are dropped only for stations ([L557–560](https://github.com/NOAA-CO-OPS/dev-Next-Gen-NOS-OFS-Skill-Assessment/blob/b3744b815ce7e86e86e68452b5f4becc918d5e0b/src/ofs_skill/model_processing/intake_scisa.py#L557-L560)).
   - So with one fields file every timestamp appears twice: 12 steps for nowcast and forecast_b, 360 for forecast_a.
   - `fix_adcirc_dataset` logged "Time coordinate is not strictly increasing" and "non-uniform spacing (max = 3600 s, min = −18000 s)" for nowcast and forecast_b ([L1005–1013](https://github.com/NOAA-CO-OPS/dev-Next-Gen-NOS-OFS-Skill-Assessment/blob/b3744b815ce7e86e86e68452b5f4becc918d5e0b/src/ofs_skill/model_processing/intake_scisa.py#L1005-L1013)).
6. **The nowcast window starts at cycle − 5 h.**
   - The file has no step at cycle − 6 h. The nowcast slice is cycle − 5 h through cycle, inclusive (6 steps).
   - The `fix_adcirc_dataset` docstring example ([L919](https://github.com/NOAA-CO-OPS/dev-Next-Gen-NOS-OFS-Skill-Assessment/blob/b3744b815ce7e86e86e68452b5f4becc918d5e0b/src/ofs_skill/model_processing/intake_scisa.py#L919)) says "06:00 UTC -> 12:00 UTC" for a 12z run. The positional slicing itself matches the file.

These assumptions held:

- `x`, `y`, `zeta` and `time` exist.
- The coordinates are 1-D `(node,)`, and `_static_coord_1d` returns 1-D arrays.
- Longitude runs −180..180.
- The step is hourly, with 186 steps per run.
- The file is HDF5, and the repo picks the `h5netcdf` engine.
- The S3 path resolves for real STOFS runs. The project lead reported on 2026-10-21 that every STOFS run (ADCIRC and SCHISM) uses `netcdf_dir=` (empty) in its `.conf` file. With an empty value, the `stofs_2d_glo` bucket prefix is empty ([list_of_files.py L81–91](https://github.com/NOAA-CO-OPS/dev-Next-Gen-NOS-OFS-Skill-Assessment/blob/b3744b815ce7e86e86e68452b5f4becc918d5e0b/src/ofs_skill/model_processing/list_of_files.py#L81-L91)). The URL is then the bucket-root path, which returned HTTP 200 for this file.
  - Only the shipped `conf/ofs_dps.conf.example` sets `netcdf_dir=netcdf` ([L53](https://github.com/NOAA-CO-OPS/dev-Next-Gen-NOS-OFS-Skill-Assessment/blob/b3744b815ce7e86e86e68452b5f4becc918d5e0b/conf/ofs_dps.conf.example#L53)). With that value the URL is `https://noaa-gestofs-pds.s3.amazonaws.com/netcdf/stofs_2d_glo.20261020/stofs_2d_glo.t00z.fields.cwl.nc`, which returned HTTP 404.

## Raw `ncdump -h`

Run on the local copy with `ncdump` from netCDF-C, as shipped in the `DC` conda environment.

```text
netcdf stofs_2d_glo.t00z.fields.cwl {
dimensions:
	time = UNLIMITED ; // (186 currently)
	node = 12785004 ;
	nele = 24875336 ;
	nfaces = 24875336 ;
	nvertex = 3 ;
	nbou = 262 ;
	nvel = 12421 ;
	max_nvell = 1772 ;
variables:
	double time(time) ;
		time:long_name = "model time" ;
		time:standard_name = "time" ;
		time:units = "seconds since 2026-05-26 12:00:00        ! NCDASE - BASE_DAT" ;
		time:base_date = "2026-05-26 12:00:00        ! NCDASE - BASE_DATE" ;
	double x(node) ;
		x:long_name = "longitude" ;
		x:standard_name = "longitude" ;
		x:units = "degrees_east" ;
		x:positive = "east" ;
	double y(node) ;
		y:long_name = "latitude" ;
		y:standard_name = "latitude" ;
		y:units = "degrees_north" ;
		y:positive = "north" ;
	int element(nfaces, nvertex) ;
		element:long_name = "element" ;
		element:cf_role = "face_node_connectivity" ;
		element:start_index = 1 ;
		element:units = "nondimensional" ;
	int adcirc_mesh ;
		adcirc_mesh:long_name = "mesh_topology" ;
		adcirc_mesh:cf_role = "mesh_topology" ;
		adcirc_mesh:topology_dimension = 2 ;
		adcirc_mesh:node_coordinates = "x y" ;
		adcirc_mesh:face_node_connectivity = "element" ;
	int nvell(nbou) ;
		nvell:long_name = "number of nodes in each normal flow specified boundary segment" ;
		nvell:units = "nondimensional" ;
	int ibtype(nbou) ;
		ibtype:long_name = "type of normal flow (discharge) boundary" ;
		ibtype:units = "nondimensional" ;
	int nbvv(nvel) ;
		nbvv:long_name = "node numbers on normal flow boundary segment" ;
		nbvv:units = "nondimensional" ;
	double barsp(nvel) ;
		barsp:long_name = "external/internal barrier boundary weir coefficient of supercritical flow" ;
		barsp:units = "nondimensional" ;
	double barht(nvel) ;
		barht:long_name = "external/internal barrier boundary height above geoid" ;
		barht:units = "m" ;
	int ibconn(nvel) ;
		ibconn:long_name = "internal barrier nodal connections" ;
		ibconn:units = "nondimensional" ;
	double barsb(nvel) ;
		barsb:long_name = "internal barrier boundary weir coefficient of subcritical flow" ;
		barsb:units = "nondimensional" ;
	double depth(node) ;
		depth:long_name = "distance  below geoid" ;
		depth:standard_name = "depth below geoid" ;
		depth:coordinates = "time y x" ;
		depth:location = "node" ;
		depth:mesh = "adcirc_mesh" ;
		depth:units = "m" ;
	float zeta(time, node) ;
		zeta:units = "meters" ;
		zeta:long_name = "Corrected Water Level (original zeta + bias_correction)" ;

// global attributes:
		:_FillValue = -99999. ;
		:model = "ADCIRC" ;
		:version = "unknown" ;
		:git_hash = "unknown" ;
		:grid_type = "Triangular" ;
		:description = "2026102000 :-6 hr nowcast and +180 hr forecast ! 32 CHARACTER ALPHANUMERIC RUN D" ;
		:agrid = "OceanMesh2D" ;
		:rundes = "2026102000 :-6 hr nowcast and +180 hr forecast ! 32 CHARACTER ALPHANUMERIC RUN D" ;
		:runid = "STOFS 2D GLOBAL v5.6.5     ! 24 CHARACTER ALPHANUMERIC RUN IDENTIFICATION" ;
		:title = "STOFS_2D_GLOBAL.V3.1.0     ! NCPROJ - PROJECT TITLE" ;
		:institution = "NOS/OCS/CSDL/CMMB          ! NCINST - PROJECT INSTITUTION" ;
		:source = "Dogwood/Cactus             ! NCSOUR - PROJECT SOURCE" ;
		:history = "PRODUCTION                 ! NCHIST - PROJECT HISTORY" ;
		:references = "http://www.adcirc.org      ! NCREF  - PROJECT REFERENCES" ;
		:comments = "STOFS_2D_GLOBAL.V3.1.0     ! NCCOM  - PROJECT COMMENTS" ;
		:host = "NOS/OCS/CSDL/CMMB          ! NCHOST - PROJECT HOST" ;
		:convention = "CF-1.0                     ! NCCONV - CONVENTIONS" ;
		:Conventions = "UGRID-0.9.0" ;
		:contact = "Yuji.Funaoshi@noaa.gov     ! NCCONT - CONTACT INFORMATION" ;
		:creation_date = "2026-10-20  3:46:50  00:00" ;
		:modification_date = "2026-10-20  3:46:50  00:00" ;
		:fort.15 = "==== Input File Parameters (below) ====" ;
		:dt = 6. ;
		:ihot = 368 ;
		:ics = 22 ;
		:nolibf = 1 ;
		:nolifa = 2 ;
		:nolica = 1 ;
		:nolicat = 1 ;
		:nwp = 7 ;
		:ncor = 1 ;
		:ntip = 2 ;
		:nws = 14 ;
		:nramp = 1 ;
		:tau0 = 0.053333 ;
		:statim = 0. ;
		:reftim = 0. ;
		:rnday = 146.5 ;
		:dramp = 6.75 ;
		:a00 = 0.8 ;
		:b00 = 0.2 ;
		:c00 = 0. ;
		:h0 = 0.1 ;
		:slam0 = 0. ;
		:sfea0 = 45. ;
		:cf = 0.0005 ;
		:eslm = -0.2 ;
		:cori = 0. ;
		:ntif = 8 ;
		:nbfr = 0 ;
}
```
