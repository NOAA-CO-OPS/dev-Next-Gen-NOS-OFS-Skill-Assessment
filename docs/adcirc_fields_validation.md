# STOFS-2D-Global (ADCIRC) fields mode: cluster validation

This page records the cluster validation of ADCIRC fields-mode support
(`ofs='stofs_2d_glo'`, `-t fields`) on branch
`feature/adcirc-fields-nearest-node`. Each fields run is compared with the
matching stations run, which reads the STOFS-2D-Global points files.

- The runs used branch `validation/adcirc-fields-nearest-node`, commit
  `8fc79ca22708b823bd1c02fa5fb7cef5a6913f43`. That commit is the feature
  branch at `239769e` plus the run folders under `dan_runs/`.
- The analysis read only the run logs, control files, `.prd`/`.obs` files and
  skill CSVs. No STOFS model file was opened.

## Summary

All four runs completed (exit 0).

- **Nowcast, cluster vs laptop:** the cluster nowcast results are identical to
  the earlier laptop run over the same window. That covers the matched nodes,
  the distances, and all 240 fields and 215 stations model series.
- **Matching:** fields mode matched all 240 observation stations within 4 km,
  and stations mode matched 215. The always-wet rule moved 29 stations to a
  farther node, and none was left unmatched.
- **Series:** every fields series is hourly, monotonic and NaN-free, and lines
  up with the stations series at zero lag (214 of 215 stations).
- **Main open issue:** fields and stations output disagree by a near-constant
  offset at 35 stations (nowcast) and 28 (forecast_a). At most flagged stations
  the stations product is closer to the observations. The cause is not known.

## Runs

| Run | Folder | Window or cycle | Type | `-vs` |
|---|---|---|---|---|
| (a) | `stofs2d-fields-nowcast-2day` | nowcast, 2026-10-17T00:00:00Z to 2026-10-19T00:00:00Z | fields | default (all four variables) |
| (b) | `stofs2d-stations-nowcast-2day` | same as (a) | stations | `water_level` |
| (c1) | `stofs2d-fields-forecasta-1cycle` | forecast_a, cycle 2026-10-14 00z, 180 h to 2026-10-21T12:00:00Z | fields | `water_level` |
| (c2) | `stofs2d-stations-forecasta-1cycle` | same as (c1) | stations | `water_level` |

All four used `-o stofs_2d_glo -d MSL -so co-ops`. The forecast_a runs pass
`-s 2026-10-14T00:00:00Z -f 00z`. `get_fcst_dates` then sets the end to
cycle + 180 h (`src/ofs_skill/model_processing/get_fcst_cycle.py:227-368`).

Commands, as printed at the top of each `.out`:

```text
(a)  python ./bin/visualization/create_1dplot.py -c ./dan_runs/stofs2d-fields-nowcast-2day/stofs2d-fields-nowcast-2day.conf -p ./dan_runs/stofs2d-fields-nowcast-2day -o stofs_2d_glo -s 2026-10-17T00:00:00Z -e 2026-10-19T00:00:00Z -d MSL -ws nowcast -t fields -so co-ops
(b)  python ./bin/visualization/create_1dplot.py -c ./dan_runs/stofs2d-stations-nowcast-2day/stofs2d-stations-nowcast-2day.conf -p ./dan_runs/stofs2d-stations-nowcast-2day -o stofs_2d_glo -s 2026-10-17T00:00:00Z -e 2026-10-19T00:00:00Z -d MSL -ws nowcast -t stations -vs water_level -so co-ops
(c1) python ./bin/visualization/create_1dplot.py -c ./dan_runs/stofs2d-fields-forecasta-1cycle/stofs2d-fields-forecasta-1cycle.conf -p ./dan_runs/stofs2d-fields-forecasta-1cycle -o stofs_2d_glo -s 2026-10-14T00:00:00Z -e 2026-10-21T12:00:00Z -f 00z -d MSL -ws forecast_a -t fields -vs water_level -so co-ops
(c2) python ./bin/visualization/create_1dplot.py -c ./dan_runs/stofs2d-stations-forecasta-1cycle/stofs2d-stations-forecasta-1cycle.conf -p ./dan_runs/stofs2d-stations-forecasta-1cycle -o stofs_2d_glo -s 2026-10-14T00:00:00Z -e 2026-10-21T12:00:00Z -f 00z -d MSL -ws forecast_a -t stations -vs water_level -so co-ops
```

Model files came from NODD through the S3 fallback, with `netcdf_dir=` empty.
Fields files are streamed and not cached. Points files are cached under
`~/.ofs_cache/s3`.

### Edits made on the cluster

The cluster's `git diff` shows two kinds of change against `8fc79ca`, the
same in all four run folders:

- **Job scripts:** `#SBATCH --cpus-per-task=64` became `#SBATCH --cpus-per-task=4`.
- **Run confs, `[parallelization]` section:** `parallel_workflow=False` became
  `True`, and `parallel_plotting=False` became `True`.

These edits are now on the validation branch as commit `7d91629` ("Match
validation run folders to what ran on the cluster"). The committed files are
byte-identical to the ones that ran.

### How the runs were executed

- **Concurrency:** runs (b), (c1) and (c2) were submitted together and all
  started at 2026-10-23 21:04:29 UTC.
- **Run (a) was a second run in the same folder.** Its `.out` covers 20:27:58
  to 20:55:22 UTC. That run reused the model control file, `.prd` files and
  observations from an earlier run (about 1.5 h) whose `.out`/`.err` it
  overwrote. Its log has no intake, dry-scan or nearest-node stages.
- **Timing of the earlier run (a):** the reused files' modification times give
  approximate times, assuming the copy back from the cluster preserved them:
  - observation control file at 18:57:51 UTC;
  - model control file at 19:13:22 UTC;
  - `.prd` files at 19:17:03–19:17:06 UTC.

  The validation branch held only commit `8fc79ca` until 2026-10-26, so the
  earlier run used the same code.

## 1. Completion

| Run | Exit status (`/usr/bin/time`) | `Commit:` in `.out` | Last line | Last traceback |
|---|---|---|---|---|
| (a) | 0 | `8fc79ca2…` | `Finished create_1dplot!` | `ValueError: Unsupported vertical datum conversion 'mlw'->'msl'` (skill step, handled per station) |
| (b) | 0 | `8fc79ca2…` | `Finished create_1dplot!` | `UnboundLocalError: cannot access local variable 'scalar_data'` (batch precompute, handled; see below) |
| (c1) | 0 | `8fc79ca2…` | `Finished create_1dplot!` | `ValueError: Unsupported vertical datum conversion 'mllw'->'msl'` (skill step, handled per station) |
| (c2) | 0 | `8fc79ca2…` | `Finished create_1dplot!` | `UnboundLocalError: cannot access local variable 'scalar_data'` (batch precompute, handled) |

Every traceback was caught and logged, and the run carried on. Both kinds come
from code this branch does not change.

- **Datum `ValueError`** (15 tracebacks in (a), 22 in (c1)).
  - **Path:** `get_skill.py:552` → `metrics_paired_one_d.py:192` →
    `get_station_tidal_data.py:34` → `vdatum_resilient.py:258`.
  - **Cause:** tidal predictions came back in a datum other than MSL, and
    converting to `'msl'` is not supported.
  - **Effect:** that station drops out of the fields skill table. The laptop
    run hit the same error.
- **`UnboundLocalError`** (one per stations run).
  - **Path:** `_precompute_scalar_data` in
    `src/ofs_skill/model_processing/get_node_ofs.py:1010-1053` has branches for
    `fvcom`, `roms` and `schism` only. For `adcirc`, `scalar_data` is never set.
  - **Effect:** the stations run falls back to per-station extraction
    (`Batch precomputation failed, falling back to per-station extraction`).
    The laptop stations run logged the same warning.
  - **Not fixed here:** stations mode is outside this PR.

## a. Variable handling (run (a))

| Check | Result |
|---|---|
| `restrict_adcirc_variables` warnings | Each appears exactly once: `ADCIRC (stofs_2d_glo) has no water temperature or salinity output; removing ['water_temperature', 'salinity'] from the variable list.` and `ADCIRC (stofs_2d_glo) currents are not yet supported in this pipeline; removing currents from the variable list.` |
| Observations | Only `control_files/stofs_2d_glo_wl_station.ctl` and 240 `*_wl_station.obs` files. No temperature, salinity or currents control or observation files. |
| Outputs | Only water-level `.prd`, `.int` and skill CSVs (`*_wl_*`, `*water_level*`), plus `skill_stofs_2d_glo_all_stations.csv` and `stofs_2d_glo_nowcast_filename_key.csv`. |
| "GOMOFS extent" warning | Absent: 0 case-insensitive matches for `gomofs` in all four `.out` files. |

Runs (b), (c1) and (c2) passed `-vs water_level`, so they log no restriction
warning, as expected.

`data/visual/` is empty in all four returned folders because the plots were
not copied back from the cluster. Every log records `Completed plot for
station …`.

## b. Wall time, peak memory and stages

From the `/usr/bin/time -v` report at the end of each `.err`:

| Run | Wall time | Peak RSS | CPU | User + system CPU |
|---|---|---|---|---|
| (a), second run, reused artifacts | 0:27:24 | 0.35 GiB (361,916 kB) | 4% | 70 s |
| (b) | 3:56:30 | 1.76 GiB (1,843,780 kB) | 1% | 238 s |
| (c1) | 4:20:58 | 1.63 GiB (1,712,776 kB) | 2% | 401 s |
| (c2) | 3:03:40 | 0.76 GiB (800,196 kB) | 1% | 177 s |

Stages from log timestamps (UTC, 2026-10-23/24):

| Stage | (c1) fields forecast_a | (b) stations nowcast | (c2) stations forecast_a |
|---|---|---|---|
| Inventory | 21:05:03–21:05:07 (4 s) | 21:05:05–21:05:08 (3 s) | 21:05:05–21:05:11 (6 s) |
| Observation retrieval | 21:05:07–21:33:17 (28 min) | 21:05:08–21:35:17 (30 min) | 21:05:11–21:34:53 (30 min) |
| Intake | 21:33:17–21:33:24 (7 s, 1 remote fields file) | 21:35:17–21:35:30 (13 s, 16 points files downloaded) | 21:34:53–21:34:55 (2 s, 1 downloaded, 1 reused) |
| Dry scan | 21:33:35–21:40:00 (logged 386.4 s for 180 unique steps) | n/a | n/a |
| Nearest-node (KD-tree, cutoff) | 21:40:00–21:40:04 (4 s) | n/a (name match) | n/a |
| Model ctl written | 21:40:12 | 21:35:33 | 21:34:56 |
| Extraction | 21:40:12–21:46:39 (6.5 min) | 21:35:33–21:36:13 (40 s) | 21:34:56–21:35:20 (24 s) |
| Skill (pairing, tidal predictions, metrics) | 21:46:39–01:05:45 (3 h 19 min) | 21:36:13–23:11:34 (1 h 35 min) | 21:35:20–22:18:42 (43 min) |
| Plots | 01:05:45–01:25:26 (20 min) | 23:11:34–01:00:57 (1 h 49 min) | 22:18:42–00:08:08 (1 h 49 min) |

- **Second run (a):** skill took 20:28:29–20:49:40 (21 min) and plots
  20:49:40–20:55:20 (6 min).
- **Earlier run (a):** from the file times above, the model control file
  (intake, dry scan, nearest-node) was written about 15.5 min after the
  observation control file. Extraction took about 3.7 min more.
- **What took the time:** CO-OPS returned HTTP 403 throughout the three
  concurrent runs. The logs record 17,880–24,120 failed tidal-prediction
  requests per run, plus 365–375 water-level observation retries per run.
  - **Affected stages:** skill and plots, which fetch tidal predictions, took
    most of the wall time in all three.
  - **Model stages:** short in comparison. The fields dry scan of 180 steps
    took 6.4 min, against about 11 min for 96 steps on the laptop.

## c. Per-station matching

- **Dry rule:** a nearer node counts as excluded as dry when the matched node
  differs from the nearest node of any kind and is farther away. The nearest
  node of any kind and the exact distances come from the laptop analysis of
  the STOFS-2D-Global mesh, which is the same for every cycle.
- **Inexact distances:** for 4 forecast_a stations the matched node is one the
  laptop analysis never measured, so the distance comes from the 3-decimal node
  coordinates in the model control file (about ±0.1 km).
- **Full tables:** Appendices C (nowcast) and E (forecast_a).

| | Nowcast (a) vs (b) | forecast_a (c1) vs (c2) |
|---|---|---|
| Observation stations (same set in both runs) | 240 | 240 |
| Matched in fields mode (≤ 4 km) | 240 | 240 |
| Matched in stations mode | 215 | 215 |
| Matched in both | 215 | 215 |
| Fields only | 25 | the same 25 |
| Stations only | 0 | 0 |
| Distance, median / max | 0.068 / 1.763 km | 0.068 / 1.763 km |
| Matched to a farther node because a nearer one is dry | 29 (moved 0.004–1.738 km farther) | 29 (moved 0.004–1.738 km farther) |
| Unmatched because of dryness | 0 | 0 |

- **Fields-only stations:** the 25 stations matched only in fields mode are the
  ones the stations runs log as `No matching model station found for obs station
  …`. They have no entry in the STOFS-2D-Global points file station list.
- **Nodes that differ between nowcast and forecast_a:** six stations. The dry
  scan covers different time steps in each (96 nowcast steps, 180 forecast_a
  steps).

| ID | Nowcast node (km) | forecast_a node (km) | Nearest node of any kind (km) | Dry-moved: nowcast / forecast_a |
|---|---|---|---|---|
| 8573364 | 6382746 (0.105) | 6390507 (≈0.17) | 6382748 (0.046) | yes / yes |
| 8636580 | 11057120 (0.118) | 11081856 (≈0.18) | 11057120 (0.118) | no / yes |
| 8727520 | 8896819 (0.100) | 8870576 (0.025) | 8870576 (0.025) | yes / no |
| 8772985 | 9089747 (0.671) | 9064268 (≈0.49) | 8959604 (0.027) | yes / yes |
| 8773259 | 7085279 (0.099) | 7069751 (0.090) | 7069751 (0.090) | yes / no |
| 9435380 | 1488391 (0.019) | 1477928 (≈0.08) | 1488391 (0.019) | no / yes |

## d. Fields vs stations model series

These statistics cover stations matched in both modes. Bias = fields −
stations, over the common hourly times in the assessment window. A station is
FLAGged when |bias| > 0.05 m or correlation < 0.95. Per-station values and
the two runs' skill tables side by side are in Appendices D and F.

| | Nowcast | forecast_a |
|---|---|---|
| Stations compared | 215 | 215 |
| Common hourly times | 49 (214 stations), 45 (8571421) | 180 (211), 166–176 (4) |
| RMSE, median / max | 0.015 / 0.845 m | 0.012 / 0.905 m |
| Bias, median | −0.004 m | −0.004 m |
| Correlation, median / min | 0.9998 / 0.739 | 0.99999 / 0.782 |
| FLAG | 35 | 28 |
| Flagged by correlation < 0.95 | 3 (8637689, 8771972, 8775792) | 1 (8771972) |

- **Mostly constant offsets:** most flags are near-constant offsets with high
  correlation. The largest:

  | Station | Nowcast bias | forecast_a bias |
  |---|---|---|
  | Valdez 9454240 | +0.842 m | +0.905 m |
  | Juneau 9452210 | +0.369 m | +0.442 m |
  | Yorktown 8637689 | −0.348 m | −0.310 m |
  | Nikiski 9455760 | −0.343 m | not among the largest |
  | Elfin Cove 9452634 | not among the largest | +0.338 m |

  Across the 215 stations, the per-station biases of the two windows correlate
  at 0.983.
- **Which product matches the observations:** for the flagged stations, the
  skill tables show the **stations** product closer to the observations in 29
  of 32 nowcast cases and 20 of 24 forecast_a cases. Example: Valdez nowcast
  skill bias is +0.84 m for fields and 0.00 m for stations.
- **Cause not known.** The fields `zeta` is labelled "Corrected Water Level
  (original zeta + bias_correction)" and the points `zeta` "water surface
  elevation above geoid" (`docs/adcirc_fields_file_notes.md`). The pipeline
  applies no datum shift to either for MSL (`Datum offset … : 0` in every
  extraction line).
- **Dry rule not the cause:** only 2 nowcast flags (8447930, 8775792) and no
  forecast_a flag are among the stations moved by the dry rule.
- **Station 8771972 (lag 3 h):** its best lag is +3 h in both windows
  (correlation 0.74 nowcast, 0.78 forecast_a). It is the only station not best
  aligned at lag 0.

## e. Fields series checks

| | Nowcast | forecast_a |
|---|---|---|
| Fields `.prd` files | 240 | 240 |
| Steps per series | 96 | 180 |
| Span | 2026-10-16 19:00 to 2026-10-20 18:00 | 2026-10-14 01:00 to 2026-10-21 12:00 |
| Hourly, strictly increasing, no duplicate times | all 240 | all 240 |
| Best lag against stations series | 0 h for 214 of 215 (8771972: +3 h) | 0 h for 214 of 215 (8771972: +3 h) |

## f. NaN steps

Zero in every fields series (240 nowcast, 240 forecast_a), over the whole file
and within the window. This is expected: the always-wet rule excludes any node
that is dry at any step.

## g. forecast_a (cycle 2026-10-14 00z)

- **Completion:** both runs completed, with exit 0 and commit `8fc79ca2…`
  (section 1).
- **Matching:** 240 matched in fields mode and 215 in stations mode, with the
  same 25 fields-only stations as nowcast. 29 stations were moved by the dry
  rule and none left unmatched (section c).
- **Fields vs stations:** median RMSE 0.012 m, median correlation 0.99999,
  28 flagged (section d).
- **Missing observations:** run (c1) has 238 observation files. 8720219 and
  8722670 are missing, during the CO-OPS 403 period. Their fields `.prd`
  files exist.
- **Skill-table coverage:**

  | Run | Rows in skill table |
  |---|---|
  | (c1) fields | 216 of 240 matched |
  | (c2) stations | 215 of 215 matched |

  The fields stations without a row are listed in the analysis output. Causes
  in the log: the datum `ValueError` above, and the two missing observation
  files.

## h. Comparison with the laptop run (nowcast)

The laptop run used the same window, `-so co-ops` and code. It is kept outside
the repo as `adcirc_fields_validation_laptop_2026-10-22.md`.

| Compared | Result |
|---|---|
| Observation control file | identical: 240 stations, same coordinates |
| Fields model control file | identical: 240 rows, same node index and node coordinates |
| Stations model control file | identical: 215 rows |
| Distances (km) | identical (max difference 0.0) |
| Fields `.prd` series (240) | identical: same timestamps and values |
| Stations `.prd` series (215) | identical |
| Observation `.obs` files (240) | identical apart from line endings (laptop CRLF, cluster LF) |
| Fields skill table | 225 rows on the cluster, 208 on the laptop. Every metric is identical for the 200 stations in both. Cluster only: 8775241, 8775283, 8775296, 8775792, 8777812, 9410170, 9410230, 9410678, 9410840, 9418767, 9419750, 9431647, 9432780, 9435380, 9437540, 9439040, 9439099, 9454050, 9455090, 9455760, 9459881, 9461380, 9497645, 9751364, 9755371. Laptop only: 8779280, 8779748, 9412110, 9457292, 9751401, 9751639, 9752235, 9752695. |
| Stations skill table | 215 rows on the cluster, 213 on the laptop. Every metric is identical for the 213 in both. Cluster only: 9419750, 9459450. |

The only differences are which stations made it into the skill tables. That
depends on which tidal-prediction requests succeeded during the CO-OPS 403
periods, and on the datum `ValueError` (section 1).

## Control-file reuse across runs (code reading, not run)

In the default configuration, a model control file from one run is not reused
by a later run over a different window, including with `--Continue_Run`.

- **Window is part of the signature:** each model control file is stamped with
  the run's start and end dates
  (`src/ofs_skill/utils/cache_manifest.py:155-156`).
- **Ctl writer:** it checks that stamp
  (`src/ofs_skill/model_processing/write_ofs_ctlfile.py:611-613`). If the
  window differs, the file is deleted (`cache_manifest.py:400-408`) and rebuilt
  (`write_ofs_ctlfile.py:615-623`, then `indexing.index_nearest_node` at
  `:669`). The rebuild reruns the dry scan
  (`src/ofs_skill/model_processing/indexing.py:249`).
- **create_1dplot:** its own gate does the same check
  (`bin/visualization/create_1dplot.py:362-367`).
- **`--Continue_Run`:** it relaxes the window check only for observations and
  paired/`.prd` files (`get_station_observations.py:1088-1089`,
  `get_skill.py:874`, `get_skill.py:1121`). Model control files are still
  rebuilt. It is disabled for forecast_a (`create_1dplot.py:1161-1172`).
- **Exception, `keep_stale_artifacts=True`:** a control file from another
  window is kept (`cache_manifest.py:385`, `:392-398`, `:335-336`), and its
  ADCIRC nodes are **not** re-checked for dryness. The default is `False`
  (`conf/ofs_dps.conf.example:315`).
- **Partial safeguard:** in fields runs, any −99999 value at a reused node is
  still set to NaN at extraction (`get_node_ofs.py:1422`, `:1482`).

Run (a)'s second run is the same-window case: the signature matched, so the
control file and `.prd` files were reused as designed.

## Other observations

- **CO-OPS HTTP 403s affected every run.** Tidal-prediction requests failed
  17,880–24,120 times per run in (b), (c1) and (c2), and 2,373 times in the
  second run (a). These failures decide which stations reach the skill tables
  and plots, and most of each run's wall time went to these stages.
- **CPU use was 1–4%.** The runs wait on the network, so
  `--cpus-per-task=4` loses nothing against 64.

---

Appendices C–F follow. In the per-station tables:

- **km source:** "exact" distances come from the mesh coordinates measured in
  the laptop analysis. "ctl" distances use the 3-decimal node coordinates in
  the model control file.
- **stations index:** the station's index in the points file.
- **Skill columns:** the `skill_stofs_2d_glo_water_level_<cast>_<fields|stations>.csv`
  tables. A blank means the station is not in that table.

## Appendix C. Per-station matching, nowcast, 2026-10-17T00Z to 2026-10-19T00Z

| ID | Name | fields | stations | fields node | km | km source | nearest node of any kind, km | nearer node dry | stations index | reason if one mode only |
|---|---|---|---|---|---|---|---|---|---|---|
| 1611400 | Nawiliwili | yes | yes | 2115047 | 0.070 | exact | 0.070 |  | 201 |  |
| 1612340 | Honolulu | yes | yes | 2196805 | 0.069 | exact | 0.069 |  | 202 |  |
| 1612401 | Pearl Harbor | yes | no | 1694349 | 0.065 | exact | 0.065 |  |  | stations run: "No matching model station found" (not in points file station list) |
| 1612480 | Mokuoloe | yes | yes | 2067685 | 0.037 | exact | 0.037 |  | 203 |  |
| 1615680 | Kahului, Kahului Harbor | yes | yes | 2444047 | 0.064 | exact | 0.064 |  | 204 |  |
| 1617433 | Kawaihae | yes | yes | 2266382 | 0.094 | exact | 0.063 | yes | 205 |  |
| 1617760 | Hilo, Hilo Bay, Kuhio Bay | yes | yes | 2567990 | 0.095 | exact | 0.054 | yes | 206 |  |
| 1619910 | Sand Island, Midway Islands | yes | yes | 1681601 | 0.783 | exact | 0.783 |  | 207 |  |
| 1630000 | Apra Harbor, Guam | yes | yes | 1051396 | 0.071 | exact | 0.071 |  | 224 |  |
| 1631428 | Pago Bay, Guam | yes | yes | 1209204 | 0.071 | exact | 0.071 |  | 225 |  |
| 1770000 | Pago Pago, American Samoa | yes | yes | 3753584 | 0.369 | exact | 0.161 | yes | 226 |  |
| 1820000 | Kwajalein, Marshall Islands | yes | yes | 2162012 | 0.077 | exact | 0.077 |  | 227 |  |
| 1890000 | Wake Island, Pacific Ocean | yes | yes | 1706413 | 0.152 | exact | 0.152 |  | 228 |  |
| 2695535 | Bermuda Biological Station | yes | no | 12471348 | 0.409 | exact | 0.409 |  |  | stations run: "No matching model station found" (not in points file station list) |
| 2695540 | Bermuda, St. Georges Island | yes | no | 12471349 | 0.801 | exact | 0.801 |  |  | stations run: "No matching model station found" (not in points file station list) |
| 8410140 | Eastport | yes | yes | 10423350 | 0.108 | exact | 0.108 |  | 0 |  |
| 8411060 | Cutler Farris Wharf | yes | yes | 11319611 | 0.150 | exact | 0.055 | yes | 1 |  |
| 8413320 | Bar Harbor | yes | yes | 11391369 | 0.049 | exact | 0.049 |  | 2 |  |
| 8418150 | Portland | yes | yes | 10952780 | 0.056 | exact | 0.056 |  | 3 |  |
| 8419870 | Seavey Island | yes | yes | 11050155 | 0.037 | exact | 0.037 |  | 585 |  |
| 8443970 | Boston | yes | yes | 10662974 | 0.094 | exact | 0.094 |  | 6 |  |
| 8447386 | Fall River | yes | yes | 11151953 | 0.169 | exact | 0.052 | yes | 7 |  |
| 8447435 | Chatham | yes | yes | 12376886 | 0.050 | exact | 0.050 |  | 573 |  |
| 8447636 | New Bedford Harbor | yes | no | 12240173 | 0.042 | exact | 0.042 |  |  | stations run: "No matching model station found" (not in points file station list) |
| 8447930 | Woods Hole | yes | yes | 12227775 | 0.092 | exact | 0.045 | yes | 8 |  |
| 8449130 | Nantucket Island | yes | yes | 12517134 | 0.152 | exact | 0.065 | yes | 9 |  |
| 8452660 | Newport | yes | yes | 12385477 | 0.075 | exact | 0.075 |  | 10 |  |
| 8452944 | Conimicut Light | yes | yes | 11465433 | 0.077 | exact | 0.077 |  | 11 |  |
| 8454000 | Providence | yes | yes | 10403365 | 0.045 | exact | 0.045 |  | 12 |  |
| 8461490 | New London | yes | yes | 12083637 | 0.063 | exact | 0.063 |  | 14 |  |
| 8465705 | New Haven | yes | yes | 12473999 | 0.083 | exact | 0.083 |  | 15 |  |
| 8467150 | Bridgeport | yes | yes | 12551763 | 0.048 | exact | 0.048 |  | 16 |  |
| 8510560 | Montauk | yes | yes | 12613478 | 0.101 | exact | 0.101 |  | 17 |  |
| 8516945 | Kings Point | yes | yes | 12717748 | 0.052 | exact | 0.052 |  | 18 |  |
| 8518750 | The Battery | yes | yes | 12723926 | 0.105 | exact | 0.105 |  | 19 |  |
| 8518962 | Turkey Point Hudson River NERRS | yes | no | 12740023 | 0.181 | exact | 0.181 |  |  | stations run: "No matching model station found" (not in points file station list) |
| 8518979 | Coxsackie, Hudson River | yes | no | 12743897 | 0.073 | exact | 0.073 |  |  | stations run: "No matching model station found" (not in points file station list) |
| 8531680 | Sandy Hook | yes | yes | 12696408 | 0.062 | exact | 0.062 |  | 21 |  |
| 8534720 | Atlantic City | yes | yes | 12585344 | 0.154 | exact | 0.154 |  | 22 |  |
| 8536110 | Cape May | yes | yes | 12286984 | 0.034 | exact | 0.034 |  | 23 |  |
| 8537121 | Ship John Shoal | yes | yes | 11004835 | 0.170 | exact | 0.170 |  | 24 |  |
| 8539094 | Burlington, Delaware River | yes | yes | 5519961 | 0.079 | exact | 0.079 |  | 26 |  |
| 8540433 | Marcus Hook | yes | yes | 7290577 | 0.107 | exact | 0.107 |  | 27 |  |
| 8545240 | Philadelphia | yes | yes | 6206641 | 0.009 | exact | 0.009 |  | 28 |  |
| 8546252 | Bridesburg | yes | yes | 5979109 | 0.068 | exact | 0.068 |  | 634 |  |
| 8548989 | Newbold | yes | yes | 5326150 | 0.054 | exact | 0.054 |  | 29 |  |
| 8551762 | Delaware City | yes | yes | 8785824 | 0.094 | exact | 0.094 |  | 30 |  |
| 8555889 | Brandywine Shoal Light | yes | yes | 12327360 | 0.242 | exact | 0.242 |  | 635 |  |
| 8557380 | Lewes | yes | yes | 12371093 | 0.073 | exact | 0.073 |  | 32 |  |
| 8570283 | Ocean City Inlet | yes | yes | 12436577 | 0.062 | exact | 0.062 |  | 33 |  |
| 8571421 | Bishops Head | yes | yes | 9577268 | 0.236 | exact | 0.110 | yes | 34 |  |
| 8571892 | Cambridge | yes | yes | 7141231 | 0.124 | exact | 0.124 |  | 35 |  |
| 8573364 | Tolchester Beach | yes | yes | 6382746 | 0.105 | exact | 0.046 | yes | 36 |  |
| 8573927 | Chesapeake City | yes | yes | 7222275 | 0.051 | exact | 0.051 |  | 37 |  |
| 8574680 | Baltimore | yes | yes | 6113506 | 0.050 | exact | 0.050 |  | 38 |  |
| 8575512 | Annapolis | yes | yes | 6985745 | 0.041 | exact | 0.041 |  | 39 |  |
| 8577330 | Solomons Island | yes | yes | 9149498 | 0.092 | exact | 0.092 |  | 40 |  |
| 8594900 | Washington | yes | yes | 5193966 | 0.022 | exact | 0.022 |  | 41 |  |
| 8631044 | Wachapreague | yes | no | 11973757 | 0.159 | exact | 0.046 | yes |  | stations run: "No matching model station found" (not in points file station list) |
| 8632200 | Kiptopeke | yes | yes | 11975615 | 0.071 | exact | 0.071 |  | 42 |  |
| 8635027 | Dahlgren | yes | yes | 7274222 | 0.038 | exact | 0.038 |  | 676 |  |
| 8635750 | Lewisetta | yes | yes | 9928546 | 0.040 | exact | 0.040 |  | 44 |  |
| 8636580 | Windmill Point | yes | yes | 11057120 | 0.118 | exact | 0.118 |  | 45 |  |
| 8637689 | Yorktown USCG Training Center | yes | yes | 11228916 | 0.041 | exact | 0.041 |  | 46 |  |
| 8638610 | Sewells Point | yes | yes | 11056750 | 0.043 | exact | 0.043 |  | 47 |  |
| 8638901 | CBBT, Chesapeake Channel | yes | yes | 12244211 | 0.409 | exact | 0.409 |  | 648 |  |
| 8639348 | Money Point | yes | yes | 10383577 | 0.027 | exact | 0.027 |  | 49 |  |
| 8651370 | Duck | yes | yes | 12436645 | 0.201 | exact | 0.201 |  | 50 |  |
| 8652587 | Oregon Inlet Marina | yes | yes | 12372996 | 0.206 | exact | 0.030 | yes | 51 |  |
| 8654467 | USCG Station Hatteras | yes | yes | 12418067 | 0.034 | exact | 0.034 |  | 52 |  |
| 8656483 | Beaufort, Duke Marine Lab | yes | yes | 12071012 | 0.073 | exact | 0.073 |  | 53 |  |
| 8658120 | Wilmington | yes | yes | 11447083 | 0.153 | exact | 0.153 |  | 54 |  |
| 8658163 | Wrightsville Beach | yes | yes | 12321343 | 0.092 | exact | 0.092 |  | 55 |  |
| 8661070 | Springmaid Pier | yes | yes | 12247145 | 0.161 | exact | 0.161 |  | 56 |  |
| 8665530 | Charleston | yes | yes | 11379559 | 0.047 | exact | 0.047 |  | 58 |  |
| 8670870 | Fort Pulaski | yes | yes | 11426157 | 0.043 | exact | 0.043 |  | 59 |  |
| 8679598 | Kings Bay MSF Pier | yes | yes | 11613575 | 0.084 | exact | 0.084 |  | 829 |  |
| 8720030 | Fernandina Beach | yes | yes | 11615072 | 0.062 | exact | 0.062 |  | 60 |  |
| 8720218 | Mayport (Bar Pilots Dock) | yes | yes | 11614891 | 0.061 | exact | 0.061 |  | 61 |  |
| 8720219 | Dames Point | yes | yes | 10471022 | 0.058 | exact | 0.058 |  | 696 |  |
| 8720226 | Southbank Riverwalk, St Johns River | yes | yes | 10077188 | 0.079 | exact | 0.079 |  | 698 |  |
| 8720357 | I-295 Buckman Bridge | yes | yes | 10022375 | 0.194 | exact | 0.194 |  | 62 |  |
| 8721604 | Trident Pier, Port Canaveral | yes | yes | 12002497 | 0.051 | exact | 0.051 |  | 63 |  |
| 8722670 | Lake Worth Pier, Atlantic Ocean | yes | yes | 11942331 | 0.067 | exact | 0.067 |  | 64 |  |
| 8722956 | South Port Everglades | yes | yes | 11596855 | 0.201 | exact | 0.062 | yes | 825 |  |
| 8723214 | Virginia Key | yes | yes | 10992109 | 0.035 | exact | 0.035 |  | 65 |  |
| 8723970 | Vaca Key, Florida Bay | yes | yes | 10435399 | 0.066 | exact | 0.066 |  | 66 |  |
| 8724580 | Key West | yes | yes | 10372778 | 0.162 | exact | 0.020 | yes | 67 |  |
| 8725114 | Naples Bay, North | yes | yes | 9658768 | 0.029 | exact | 0.029 |  | 68 |  |
| 8725520 | Fort Myers | yes | yes | 8087023 | 0.054 | exact | 0.054 |  | 69 |  |
| 8726384 | Port Manatee | yes | yes | 8365161 | 0.068 | exact | 0.068 |  | 70 |  |
| 8726520 | St. Petersburg | yes | yes | 8261748 | 0.023 | exact | 0.023 |  | 71 |  |
| 8726607 | Old Port Tampa | yes | yes | 7587030 | 0.082 | exact | 0.082 |  | 72 |  |
| 8726674 | East Bay | yes | no | 6915498 | 0.036 | exact | 0.036 |  |  | stations run: "No matching model station found" (not in points file station list) |
| 8726724 | Clearwater Beach | yes | yes | 9508045 | 0.125 | exact | 0.125 |  | 74 |  |
| 8727520 | Cedar Key | yes | yes | 8896819 | 0.100 | exact | 0.025 | yes | 75 |  |
| 8728690 | Apalachicola | yes | yes | 8659818 | 0.027 | exact | 0.027 |  | 76 |  |
| 8729108 | Panama City | yes | yes | 8821913 | 0.044 | exact | 0.044 |  | 77 |  |
| 8729210 | Panama City Beach | yes | yes | 9468801 | 0.160 | exact | 0.160 |  | 78 |  |
| 8729840 | Pensacola | yes | yes | 8447512 | 0.198 | exact | 0.076 | yes | 699 |  |
| 8735180 | Dauphin Island | yes | yes | 9253171 | 0.036 | exact | 0.036 |  | 79 |  |
| 8735391 | Dog River Bridge | yes | yes | 7529581 | 0.056 | exact | 0.056 |  | 80 |  |
| 8735523 | East Fowl River Bridge | yes | no | 8114751 | 0.079 | exact | 0.079 |  |  | stations run: "No matching model station found" (not in points file station list) |
| 8736897 | Coast Guard Sector Mobile | yes | yes | 7454290 | 0.026 | exact | 0.026 |  | 700 |  |
| 8737048 | Mobile State Docks | yes | yes | 7346582 | 0.059 | exact | 0.059 |  | 81 |  |
| 8737138 | Chickasaw Creek | yes | yes | 6915980 | 0.083 | exact | 0.083 |  | 701 |  |
| 8738043 | West Fowl River Bridge | yes | no | 8396177 | 1.763 | exact | 0.025 | yes |  | stations run: "No matching model station found" (not in points file station list) |
| 8739803 | Bayou La Batre Bridge | yes | no | 8018428 | 0.034 | exact | 0.034 |  |  | stations run: "No matching model station found" (not in points file station list) |
| 8741533 | Pascagoula NOAA Lab | yes | yes | 8042473 | 0.024 | exact | 0.024 |  | 703 |  |
| 8747437 | Bay Waveland Yacht Club | yes | yes | 7712785 | 0.078 | exact | 0.078 |  | 82 |  |
| 8760721 | Pilottown | yes | yes | 8378663 | 0.047 | exact | 0.047 |  | 83 |  |
| 8760922 | Pilots Station East, S.W. Pass | yes | yes | 9736063 | 0.028 | exact | 0.028 |  | 84 |  |
| 8761305 | Shell Beach | yes | yes | 7295178 | 0.105 | exact | 0.105 |  | 85 |  |
| 8761724 | Grand Isle | yes | yes | 9279697 | 0.086 | exact | 0.073 | yes | 704 |  |
| 8761927 | New Canal Station | yes | yes | 6345735 | 0.050 | exact | 0.050 |  | 86 |  |
| 8762075 | Port Fourchon, Belle Pass | yes | yes | 9279768 | 0.043 | exact | 0.043 |  | 87 |  |
| 8762482 | West Bank 1, Bayou Gauche | yes | no | 5993026 | 0.089 | exact | 0.089 |  |  | stations run: "No matching model station found" (not in points file station list) |
| 8764044 | Berwick, Atchafalaya River | yes | yes | 6492963 | 0.066 | exact | 0.066 |  | 88 |  |
| 8764227 | LAWMA, Amerada Pass | yes | yes | 8068752 | 0.052 | exact | 0.052 |  | 89 |  |
| 8764314 | Eugene Island, North of, Atchafalaya Bay | yes | yes | 8850040 | 0.069 | exact | 0.069 |  | 705 |  |
| 8766072 | Freshwater Canal Locks | yes | yes | 8824404 | 0.077 | exact | 0.077 |  | 706 |  |
| 8767816 | Lake Charles | yes | yes | 6699783 | 0.369 | exact | 0.158 | yes | 707 |  |
| 8767961 | Bulk Terminal | yes | yes | 6710731 | 0.039 | exact | 0.039 |  | 708 |  |
| 8768094 | Calcasieu Pass | yes | yes | 8531520 | 0.056 | exact | 0.056 |  | 709 |  |
| 8770475 | Port Arthur | yes | yes | 7515568 | 0.140 | exact | 0.034 | yes | 90 |  |
| 8770520 | Rainbow Bridge | yes | yes | 7053385 | 0.091 | exact | 0.075 | yes | 91 |  |
| 8770613 | Morgans Point, Barbours Cut | yes | yes | 6828284 | 0.049 | exact | 0.049 |  | 93 |  |
| 8770777 | Manchester | yes | yes | 6127860 | 0.048 | exact | 0.048 |  | 94 |  |
| 8770808 | High Island | yes | yes | 8400336 | 0.079 | exact | 0.079 |  | 710 |  |
| 8770822 | Texas Point, Sabine Pass | yes | yes | 8479957 | 0.040 | exact | 0.040 |  | 95 |  |
| 8770971 | Rollover Pass | yes | yes | 8585940 | 0.035 | exact | 0.035 |  | 96 |  |
| 8771013 | Eagle Point, Galveston Bay | yes | yes | 7316816 | 0.138 | exact | 0.138 |  | 97 |  |
| 8771341 | Galveston Bay Entrance, North Jetty | yes | yes | 8399960 | 0.064 | exact | 0.064 |  | 98 |  |
| 8771450 | Galveston Pier 21 | yes | yes | 8612551 | 0.034 | exact | 0.034 |  | 99 |  |
| 8771486 | Galveston Railroad Bridge | yes | yes | 8094674 | 0.177 | exact | 0.177 |  | 712 |  |
| 8771972 | San Luis Pass | yes | yes | 8929875 | 0.068 | exact | 0.068 |  | 714 |  |
| 8772471 | Freeport Harbor | yes | no | 8825253 | 0.072 | exact | 0.072 |  |  | stations run: "No matching model station found" (not in points file station list) |
| 8772985 | Sargent | yes | no | 9089747 | 0.671 | exact | 0.027 | yes |  | stations run: "No matching model station found" (not in points file station list) |
| 8773037 | Seadrift | yes | yes | 7460198 | 0.068 | exact | 0.068 |  | 101 |  |
| 8773146 | Matagorda City | yes | no | 8482899 | 0.636 | exact | 0.016 | yes |  | stations run: "No matching model station found" (not in points file station list) |
| 8773259 | Port Lavaca | yes | yes | 7085279 | 0.099 | exact | 0.090 | yes | 102 |  |
| 8773701 | Port O'Connor | yes | yes | 8587433 | 0.061 | exact | 0.061 |  | 103 |  |
| 8773767 | Matagorda Bay Entrance Channel | yes | yes | 8931373 | 0.051 | exact | 0.051 |  | 715 |  |
| 8774230 | Aransas Wildlife Refuge | yes | no | 8507119 | 0.066 | exact | 0.066 |  |  | stations run: "No matching model station found" (not in points file station list) |
| 8774513 | Copano Bay | yes | yes | 8049205 | 0.126 | exact | 0.126 |  | 104 |  |
| 8774770 | Rockport | yes | yes | 8298457 | 0.053 | exact | 0.053 |  | 105 |  |
| 8775132 | La Quinta Channel North | yes | no | 7615419 | 0.058 | exact | 0.058 |  |  | stations run: "No matching model station found" (not in points file station list) |
| 8775222 | Viola Turning Basin | yes | no | 6921937 | 0.100 | exact | 0.100 |  |  | stations run: "No matching model station found" (not in points file station list) |
| 8775237 | Port Aransas | yes | yes | 8879488 | 0.103 | exact | 0.103 |  | 106 |  |
| 8775241 | Aransas, Aransas Pass | yes | yes | 9089180 | 0.045 | exact | 0.045 |  | 716 |  |
| 8775283 | Enbridge, Ingleside | yes | yes | 8350134 | 0.084 | exact | 0.084 |  | 107 |  |
| 8775296 | USS Lexington, Corpus Christi Bay | yes | yes | 7697478 | 0.023 | exact | 0.023 |  | 108 |  |
| 8775792 | Packery Channel | yes | yes | 8721901 | 0.159 | exact | 0.024 | yes | 109 |  |
| 8776139 | S. Bird Island | yes | no | 8853604 | 0.028 | exact | 0.028 |  |  | stations run: "No matching model station found" (not in points file station list) |
| 8776604 | Baffin Bay | yes | no | 8587975 | 0.058 | exact | 0.058 |  |  | stations run: "No matching model station found" (not in points file station list) |
| 8777812 | Rincon Del San Jose | yes | no | 8171593 | 0.079 | exact | 0.079 |  |  | stations run: "No matching model station found" (not in points file station list) |
| 8778490 | Port Mansfield | yes | no | 8097634 | 0.059 | exact | 0.059 |  |  | stations run: "No matching model station found" (not in points file station list) |
| 8779280 | Realitos Peninsula | yes | yes | 8482560 | 0.041 | exact | 0.041 |  | 720 |  |
| 8779748 | South Padre Island CG Station | yes | yes | 8904560 | 0.040 | exact | 0.040 |  | 111 |  |
| 8779749 | SPI Brazos Santiago | yes | yes | 8956867 | 0.091 | exact | 0.091 |  | 718 |  |
| 8779770 | Port Isabel | yes | yes | 8427435 | 0.077 | exact | 0.077 |  | 112 |  |
| 9410170 | San Diego | yes | yes | 1628109 | 0.053 | exact | 0.053 |  | 128 |  |
| 9410230 | La Jolla | yes | yes | 2431095 | 0.014 | exact | 0.014 |  | 129 |  |
| 9410660 | Los Angeles | yes | yes | 1786436 | 0.022 | exact | 0.022 |  | 131 |  |
| 9410678 | Long Beach Fire Boat Pier | yes | no | 1726662 | 0.010 | exact | 0.010 |  |  | stations run: "No matching model station found" (not in points file station list) |
| 9410840 | Santa Monica | yes | yes | 1984114 | 0.057 | exact | 0.057 |  | 134 |  |
| 9411340 | Santa Barbara | yes | yes | 2173699 | 0.045 | exact | 0.045 |  | 135 |  |
| 9412110 | Port San Luis | yes | yes | 2527827 | 0.030 | exact | 0.030 |  | 138 |  |
| 9413450 | Monterey | yes | yes | 2224669 | 0.051 | exact | 0.051 |  | 140 |  |
| 9414290 | San Francisco | yes | yes | 1972248 | 0.026 | exact | 0.026 |  | 141 |  |
| 9414523 | Redwood City | yes | yes | 810037 | 0.045 | exact | 0.045 |  | 143 |  |
| 9414750 | Alameda | yes | yes | 1315255 | 0.009 | exact | 0.009 |  | 145 |  |
| 9414863 | Richmond | yes | yes | 1449449 | 0.044 | exact | 0.044 |  | 149 |  |
| 9415020 | Point Reyes | yes | yes | 2190380 | 0.055 | exact | 0.055 |  | 151 |  |
| 9415102 | Martinez-Amorco Pier | yes | yes | 570596 | 0.072 | exact | 0.072 |  | 153 |  |
| 9415144 | Port Chicago | yes | yes | 483149 | 0.046 | exact | 0.046 |  | 155 |  |
| 9416131 | Port of West Sacramento, Washington Lake | yes | no | 6863 | 0.057 | exact | 0.057 |  |  | stations run: "No matching model station found" (not in points file station list) |
| 9416841 | Arena Cove | yes | yes | 2398488 | 0.042 | exact | 0.042 |  | 158 |  |
| 9418767 | North Spit | yes | yes | 1956527 | 0.037 | exact | 0.037 |  | 159 |  |
| 9419750 | Crescent City | yes | yes | 1944978 | 0.024 | exact | 0.024 |  | 160 |  |
| 9431647 | Port Orford | yes | yes | 2030661 | 0.028 | exact | 0.028 |  | 162 |  |
| 9432780 | Charleston | yes | yes | 1717637 | 0.049 | exact | 0.049 |  | 163 |  |
| 9435380 | South Beach | yes | yes | 1488391 | 0.019 | exact | 0.019 |  | 164 |  |
| 9437540 | Garibaldi | yes | yes | 1509484 | 0.045 | exact | 0.045 |  | 165 |  |
| 9439040 | Astoria | yes | yes | 1127067 | 0.035 | exact | 0.035 |  | 167 |  |
| 9439099 | Wauna | yes | yes | 406737 | 0.044 | exact | 0.044 |  | 743 |  |
| 9439201 | St Helens | yes | yes | 31625 | 0.028 | exact | 0.028 |  | 733 |  |
| 9440083 | Vancouver | yes | yes | 7837 | 0.053 | exact | 0.053 |  | 732 |  |
| 9440357 | TEMCO Kalama Terminal | yes | yes | 65258 | 0.084 | exact | 0.084 |  | 830 |  |
| 9440422 | Longview | yes | yes | 117904 | 0.063 | exact | 0.063 |  | 738 |  |
| 9440569 | Skamokawa | yes | yes | 560329 | 0.104 | exact | 0.104 |  | 168 |  |
| 9440581 | Cape Disappointment | yes | yes | 1530071 | 0.050 | exact | 0.050 |  | 746 |  |
| 9440910 | Toke Point | yes | yes | 1457458 | 0.023 | exact | 0.023 |  | 169 |  |
| 9441102 | Westport | yes | yes | 1457451 | 0.009 | exact | 0.009 |  | 170 |  |
| 9442396 | La Push, Quillayute River | yes | yes | 1680343 | 0.034 | exact | 0.034 |  | 171 |  |
| 9443090 | Neah Bay | yes | yes | 1375277 | 0.057 | exact | 0.057 |  | 172 |  |
| 9444090 | Port Angeles | yes | yes | 763962 | 0.073 | exact | 0.073 |  | 173 |  |
| 9444900 | Port Townsend | yes | yes | 567418 | 0.106 | exact | 0.106 |  | 174 |  |
| 9445958 | Bremerton | yes | yes | 163828 | 0.122 | exact | 0.122 |  | 828 |  |
| 9446484 | Tacoma | yes | yes | 137207 | 0.535 | exact | 0.535 |  | 175 |  |
| 9447130 | Seattle | yes | yes | 273791 | 0.169 | exact | 0.169 |  | 176 |  |
| 9449424 | Cherry Point | yes | yes | 355228 | 0.037 | exact | 0.037 |  | 178 |  |
| 9449880 | Friday Harbor | yes | yes | 462265 | 0.043 | exact | 0.043 |  | 179 |  |
| 9450460 | Ketchikan | yes | yes | 703846 | 1.714 | exact | 1.714 |  | 180 |  |
| 9451054 | Port Alexander | yes | yes | 940709 | 0.203 | exact | 0.203 |  | 181 |  |
| 9451600 | Sitka | yes | yes | 896895 | 0.269 | exact | 0.269 |  | 182 |  |
| 9452210 | Juneau | yes | yes | 435831 | 0.638 | exact | 0.638 |  | 183 |  |
| 9452400 | Skagway, Taiya Inlet | yes | yes | 381632 | 0.239 | exact | 0.239 |  | 184 |  |
| 9452634 | Elfin Cove | yes | yes | 835083 | 0.746 | exact | 0.746 |  | 185 |  |
| 9453220 | Yakutat, Yakutat Bay | yes | yes | 843707 | 1.412 | exact | 1.412 |  | 186 |  |
| 9454050 | Cordova | yes | yes | 706462 | 0.480 | exact | 0.480 |  | 187 |  |
| 9454240 | Valdez | yes | yes | 533836 | 0.565 | exact | 0.565 |  | 188 |  |
| 9455090 | Seward | yes | yes | 490465 | 0.114 | exact | 0.114 |  | 189 |  |
| 9455500 | Seldovia | yes | yes | 472298 | 0.206 | exact | 0.206 |  | 190 |  |
| 9455760 | Nikiski | yes | yes | 343098 | 0.050 | exact | 0.050 |  | 191 |  |
| 9455920 | Anchorage | yes | yes | 169759 | 0.209 | exact | 0.209 |  | 192 |  |
| 9457292 | Kodiak Island | yes | yes | 584477 | 0.225 | exact | 0.225 |  | 193 |  |
| 9457804 | Alitak | yes | yes | 730823 | 0.045 | exact | 0.045 |  | 194 |  |
| 9459450 | Sand Point | yes | yes | 924305 | 0.068 | exact | 0.068 |  | 195 |  |
| 9459881 | King Cove | yes | yes | 909350 | 0.335 | exact | 0.335 |  | 196 |  |
| 9461380 | Adak Island | yes | yes | 3238200 | 0.112 | exact | 0.112 |  | 197 |  |
| 9461710 | Atka | yes | yes | 3080315 | 0.084 | exact | 0.084 |  | 198 |  |
| 9462450 | Nikolski | yes | yes | 3011359 | 0.621 | exact | 0.621 |  | 199 |  |
| 9462620 | Unalaska | yes | yes | 2843281 | 0.288 | exact | 0.288 |  | 200 |  |
| 9463502 | Port Moller | yes | yes | 1875324 | 0.129 | exact | 0.129 |  | 774 |  |
| 9464212 | Village Cove, St Paul Island | yes | yes | 3502729 | 0.658 | exact | 0.658 |  | 808 |  |
| 9468333 | Unalakleet | yes | yes | 3739659 | 0.522 | exact | 0.522 |  | 775 |  |
| 9468756 | Nome, Norton Sound | yes | yes | 4306014 | 0.125 | exact | 0.125 |  | 795 |  |
| 9491094 | Red Dog Dock | yes | yes | 4727311 | 0.082 | exact | 0.082 |  | 791 |  |
| 9497645 | Prudhoe Bay | yes | yes | 5778216 | 0.343 | exact | 0.343 |  | 761 |  |
| 9751364 | Christiansted Harbor, St Croix | yes | yes | 10974058 | 0.109 | exact | 0.018 | yes | 113 |  |
| 9751381 | Lameshur Bay, St John | yes | yes | 10682958 | 0.051 | exact | 0.051 |  | 114 |  |
| 9751401 | Limetree Bay | yes | yes | 10590193 | 0.018 | exact | 0.018 |  | 115 |  |
| 9751639 | Charlotte Amalie | yes | yes | 10682159 | 0.018 | exact | 0.018 |  | 116 |  |
| 9752235 | Culebra | yes | no | 11045628 | 0.065 | exact | 0.033 | yes |  | stations run: "No matching model station found" (not in points file station list) |
| 9752695 | Esperanza, Vieques Island | yes | yes | 10336446 | 0.058 | exact | 0.054 | yes | 118 |  |
| 9755371 | San Juan, La Puntilla, San Juan Bay | yes | yes | 11552885 | 0.042 | exact | 0.042 |  | 121 |  |
| 9759110 | Magueyes Island | yes | yes | 10480042 | 0.038 | exact | 0.038 |  | 124 |  |
| 9759394 | Mayaguez | yes | yes | 10994220 | 0.071 | exact | 0.026 | yes | 125 |  |
| 9759938 | Mona Island | yes | yes | 11142113 | 0.170 | exact | 0.042 | yes | 126 |  |

## Appendix D. Fields vs stations series, and skill tables, nowcast, 2026-10-17T00Z to 2026-10-19T00Z

| ID | n | series RMSE (m) | series bias (m) | series corr | lag (h) | skill RMSE fields / stations | skill r fields / stations | skill bias fields / stations | central freq fields / stations | FLAG |
|---|---|---|---|---|---|---|---|---|---|---|
| 1611400 | 49 | 0.020 | +0.012 | 0.9935 | 0 | 0.02 / 0.01 | 0.99 / 1 | 0.01 / 0 | 100 / 100 |  |
| 1612340 | 49 | 0.006 | -0.004 | 0.9995 | 0 | 0.01 / 0.01 | 1 / 1 | -0 / 0 | 100 / 100 |  |
| 1612480 | 49 | 0.003 | -0.001 | 0.9998 | 0 | 0.02 / 0.02 | 0.99 / 0.99 | -0 / -0 | 100 / 100 |  |
| 1615680 | 49 | 0.015 | +0.015 | 0.9998 | 0 | 0.02 / 0.03 | 1 / 0.99 | 0.02 / 0 | 100 / 100 |  |
| 1617433 | 49 | 0.006 | +0.006 | 1.0000 | 0 | 0.01 / 0.01 | 1 / 1 | 0.01 / 0 | 100 / 100 |  |
| 1617760 | 49 | 0.003 | -0.000 | 0.9998 | 0 | 0.01 / 0.02 | 1 / 0.99 | -0 / -0 | 100 / 100 |  |
| 1619910 | 49 | 0.030 | -0.030 | 0.9990 | 0 | 0.03 / 0.01 | 1 / 1 | -0.03 / -0 | 100 / 100 |  |
| 1630000 | 49 | 0.064 | +0.060 | 0.9954 | 0 | 0.08 / 0.05 | 0.97 / 0.97 | 0.07 / 0.01 | 96.3 / 95.56 | FLAG |
| 1631428 | 49 | 0.050 | -0.019 | 0.9938 | 0 | 0.05 / 0.05 | 0.99 / 0.99 | -0.01 / 0 | 100 / 97.59 |  |
| 1770000 | 49 | 0.001 | -0.001 | 1.0000 | 0 | 0.02 / 0.02 | 1 / 1 | -0.02 / -0.02 | 100 / 100 |  |
| 1820000 | 49 | 0.211 | +0.210 | 0.9907 | 0 | 0.21 / 0.01 | 0.99 / 1 | 0.21 / 0 | 0 / 100 | FLAG |
| 1890000 | 49 | 0.031 | -0.030 | 0.9993 | 0 | 0.03 / 0.01 | 1 / 1 | -0.03 / -0 | 100 / 100 |  |
| 8410140 | 49 | 0.021 | -0.014 | 1.0000 | 0 | 0.02 / 0.02 | 1 / 1 | -0.01 / 0 | 100 / 100 |  |
| 8411060 | 49 | 0.008 | -0.007 | 1.0000 | 0 | 0.01 / 0.04 | 1 / 1 | -0.01 / -0.01 | 100 / 99.63 |  |
| 8413320 | 49 | 0.022 | -0.021 | 1.0000 | 0 | 0.03 / 0.02 | 1 / 1 | -0.02 / -0 | 100 / 100 |  |
| 8418150 | 49 | 0.064 | -0.060 | 0.9996 | 0 | 0.06 / 0.02 | 1 / 1 | -0.06 / -0 | 100 / 100 | FLAG |
| 8419870 | 49 | 0.002 | +0.002 | 1.0000 | 0 | 0.01 / 0.02 | 1 / 1 | 0 / -0 | 100 / 100 |  |
| 8443970 | 49 | 0.005 | -0.005 | 1.0000 | 0 | 0.01 / 0.03 | 1 / 1 | -0 / -0 | 100 / 100 |  |
| 8447386 | 49 | 0.002 | -0.002 | 1.0000 | 0 | 0.01 / 0.02 | 1 / 1 | -0 / 0 | 100 / 100 |  |
| 8447435 | 49 | 0.000 | +0.000 | 1.0000 | 0 | 0.01 / 0.02 | 1 / 1 | 0 / -0 | 100 / 100 |  |
| 8447930 | 49 | 0.052 | -0.051 | 0.9982 | 0 | 0.05 / 0.02 | 1 / 0.99 | -0.05 / -0 | 100 / 100 | FLAG |
| 8449130 | 49 | 0.025 | -0.025 | 0.9998 | 0 | 0.03 / 0.02 | 1 / 1 | -0.02 / -0 | 100 / 100 |  |
| 8452660 | 49 | 0.000 | -0.000 | 1.0000 | 0 | 0.01 / 0.02 | 1 / 1 | -0 / -0.01 | 100 / 100 |  |
| 8452944 | 49 | 0.016 | -0.014 | 0.9998 | 0 | 0.02 / 0.02 | 1 / 1 | -0.01 / 0 | 100 / 100 |  |
| 8454000 | 49 | 0.003 | -0.000 | 1.0000 | 0 | 0.01 / 0.03 | 1 / 0.99 | 0 / -0 | 100 / 99.81 |  |
| 8461490 | 49 | 0.007 | -0.006 | 0.9999 | 0 | 0.01 / 0.02 | 1 / 1 | -0.01 / -0 | 100 / 100 |  |
| 8465705 | 49 | 0.056 | -0.051 | 0.9990 | 0 | 0.06 / 0.03 | 1 / 1 | -0.06 / -0.01 | 100 / 100 | FLAG |
| 8467150 | 49 | 0.035 | -0.032 | 0.9997 | 0 | 0.04 / 0.02 | 1 / 1 | -0.03 / -0 | 100 / 100 |  |
| 8510560 | 49 | 0.000 | -0.000 | 1.0000 | 0 | 0.05 / 0.05 | 0.95 / 0.96 | -0.01 / -0.01 | 98.15 / 97.78 |  |
| 8516945 | 49 | 0.000 | -0.000 | 1.0000 | 0 | 0.01 / 0.01 | 1 / 1 | 0 / 0 | 100 / 100 |  |
| 8518750 | 49 | 0.037 | -0.036 | 0.9997 | 0 | 0.04 / 0.02 | 1 / 1 | -0.04 / -0 | 100 / 100 |  |
| 8531680 | 49 | 0.055 | -0.049 | 0.9983 | 0 | 0.27 / 0.06 | 0.73 / 0.98 | -0.13 / -0.01 | 90 / 97.84 |  |
| 8534720 | 49 | 0.094 | -0.092 | 0.9986 | 0 | 0.09 / 0.03 | 1 / 0.99 | -0.09 / 0 | 98.15 / 100 | FLAG |
| 8536110 | 49 | 0.000 | +0.000 | 1.0000 | 0 | 0.01 / 0.02 | 1 / 1 | -0 / -0 | 100 / 100 |  |
| 8537121 | 49 | 0.026 | -0.023 | 0.9997 | 0 | 0.02 / 0.01 | 1 / 1 | -0.02 / 0.01 | 100 / 100 |  |
| 8539094 | 49 | 0.004 | -0.002 | 1.0000 | 0 | 0.01 / 0.03 | 1 / 1 | -0 / -0 | 100 / 100 |  |
| 8540433 | 49 | 0.004 | -0.003 | 1.0000 | 0 | 0.01 / 0.02 | 1 / 1 | -0.01 / -0 | 100 / 100 |  |
| 8545240 | 49 | 0.008 | +0.001 | 0.9999 | 0 | 0.01 / 0.02 | 1 / 1 | 0 / 0 | 100 / 100 |  |
| 8546252 | 49 | 0.012 | -0.008 | 0.9999 | 0 | 0.02 / 0.02 | 1 / 1 | -0.01 / -0 | 100 / 100 |  |
| 8548989 | 49 | 0.001 | +0.001 | 1.0000 | 0 | 0.01 / 0.03 | 1 / 1 | -0 / -0 | 100 / 99.44 |  |
| 8551762 | 49 | 0.049 | -0.034 | 0.9974 | 0 | 0.05 / 0.01 | 1 / 1 | -0.04 / 0 | 100 / 100 |  |
| 8555889 | 49 | 0.031 | -0.029 | 0.9997 | 0 | 0.03 / 0.01 | 1 / 1 | -0.03 / 0 | 100 / 100 |  |
| 8557380 | 49 | 0.000 | +0.000 | 1.0000 | 0 | 0.01 / 0.02 | 1 / 1 | 0 / 0 | 100 / 100 |  |
| 8570283 | 49 | 0.021 | -0.013 | 0.9975 | 0 | 0.02 / 0.02 | 0.99 / 0.99 | -0.01 / 0 | 100 / 100 |  |
| 8571421 | 45 | 0.038 | -0.037 | 0.9985 | 0 | 0.08 / 0.01 | 0.98 / 1 | -0.06 / -0.01 | 92.59 / 100 |  |
| 8571892 | 49 | 0.022 | -0.020 | 0.9994 | 0 | 0.02 / 0.01 | 1 / 1 | -0.02 / 0 | 100 / 100 |  |
| 8573364 | 49 | 0.015 | -0.014 | 0.9999 | 0 | 0.02 / 0.01 | 1 / 1 | -0.01 / 0 | 100 / 100 |  |
| 8573927 | 49 | 0.015 | +0.006 | 0.9992 | 0 | 0.02 / 0.01 | 1 / 1 | 0.01 / -0 | 100 / 100 |  |
| 8574680 | 49 | 0.002 | -0.000 | 1.0000 | 0 | 0.01 / 0.01 | 1 / 1 | -0 / -0 | 100 / 100 |  |
| 8575512 | 49 | 0.011 | -0.011 | 0.9999 | 0 | 0.01 / 0.01 | 1 / 1 | -0.01 / 0 | 100 / 100 |  |
| 8577330 | 49 | 0.019 | -0.018 | 0.9997 | 0 | 0.02 / 0.01 | 1 / 1 | -0.02 / 0 | 100 / 100 |  |
| 8594900 | 49 | 0.000 | +0.000 | 1.0000 | 0 | 0.01 / 0.01 | 1 / 1 | 0 / 0 | 100 / 100 |  |
| 8632200 | 49 | 0.033 | -0.028 | 0.9967 | 0 | 0.04 / 0.02 | 1 / 1 | -0.03 / -0 | 100 / 100 |  |
| 8635027 | 49 | 0.027 | -0.027 | 0.9996 | 0 | 0.03 / 0.01 | 1 / 1 | -0.03 / 0 | 100 / 100 |  |
| 8635750 | 49 | 0.033 | -0.032 | 0.9988 | 0 | 0.04 / 0.01 | 1 / 1 | -0.04 / -0 | 100 / 100 |  |
| 8636580 | 49 | 0.025 | -0.025 | 0.9988 | 0 | 0.02 / 0.01 | 1 / 0.99 | -0.02 / 0 | 100 / 100 |  |
| 8637689 | 49 | 0.356 | -0.348 | 0.9156 | 0 | 0.36 / 0.01 | 0.92 / 1 | -0.35 / -0 | 0 / 100 | FLAG |
| 8638610 | 49 | 0.057 | -0.056 | 0.9983 | 0 | 0.06 / 0.01 | 1 / 1 | -0.06 / -0 | 100 / 100 | FLAG |
| 8638901 | 49 | 0.080 | -0.059 | 0.9884 | 0 | 0.14 / 0.15 | 0.85 / 0.83 | -0.02 / 0.04 | 96.3 / 93.7 | FLAG |
| 8639348 | 49 | 0.004 | -0.003 | 1.0000 | 0 | 0.01 / 0.01 | 1 / 1 | -0.01 / -0 | 100 / 100 |  |
| 8651370 | 49 | 0.046 | -0.044 | 0.9993 | 0 | 0.04 / 0.05 | 1 / 0.99 | -0.04 / -0 | 100 / 99.44 |  |
| 8652587 | 49 | 0.008 | -0.008 | 0.9999 | 0 | 0.01 / 0.01 | 1 / 1 | -0.01 / -0 | 100 / 100 |  |
| 8654467 | 49 | 0.072 | -0.067 | 0.9781 | 0 | 0.08 / 0.02 | 0.98 / 0.99 | -0.07 / -0.01 | 100 / 100 | FLAG |
| 8656483 | 49 | 0.039 | -0.037 | 0.9994 | 0 | 0.04 / 0.01 | 1 / 1 | -0.04 / 0 | 100 / 100 |  |
| 8658120 | 49 | 0.010 | -0.005 | 0.9997 | 0 | 0.01 / 0.01 | 1 / 1 | -0 / -0 | 100 / 100 |  |
| 8658163 | 49 | 0.051 | -0.049 | 0.9996 | 0 | 0.08 / 0.04 | 1 / 1 | -0.08 / -0.04 | 100 / 100 |  |
| 8661070 | 49 | 0.000 | +0.000 | 1.0000 | 0 | 0.01 / 0.02 | 1 / 1 | 0 / 0 | 100 / 100 |  |
| 8665530 | 49 | 0.022 | -0.020 | 0.9998 | 0 | 0.03 / 0.01 | 1 / 1 | -0.02 / -0 | 100 / 100 |  |
| 8670870 | 49 | 0.019 | -0.016 | 0.9999 | 0 | 0.02 / 0.03 | 1 / 1 | -0.02 / -0.01 | 100 / 99.81 |  |
| 8679598 | 49 | 0.020 | +0.019 | 0.9999 | 0 | 0.39 / 0.41 | 0.94 / 0.94 | -0.35 / -0.37 | 12.96 / 9.44 |  |
| 8720030 | 49 | 0.026 | -0.022 | 0.9997 | 0 | 0.03 / 0.01 | 1 / 1 | -0.02 / 0 | 100 / 100 |  |
| 8720218 | 49 | 0.016 | -0.015 | 0.9998 | 0 | 0.02 / 0.01 | 1 / 1 | -0.02 / -0 | 100 / 100 |  |
| 8720219 | 49 | 0.005 | -0.004 | 0.9999 | 0 | 0.01 / 0.01 | 1 / 1 | -0 / 0 | 100 / 100 |  |
| 8720226 | 49 | 0.003 | -0.002 | 1.0000 | 0 | 0.01 / 0.01 | 1 / 1 | -0 / 0 | 100 / 100 |  |
| 8720357 | 49 | 0.006 | -0.006 | 1.0000 | 0 | 0.01 / 0.01 | 1 / 1 | 0 / 0.01 | 100 / 100 |  |
| 8721604 | 49 | 0.012 | -0.012 | 1.0000 | 0 | 0.01 / 0.03 | 1 / 0.99 | -0.01 / -0 | 100 / 100 |  |
| 8722670 | 49 | 0.021 | -0.019 | 0.9996 | 0 | 0.03 / 0.01 | 1 / 1 | -0.03 / -0.01 | 100 / 100 |  |
| 8722956 | 49 | 0.027 | -0.026 | 0.9990 | 0 | 0.02 / 0.01 | 1 / 1 | -0.02 / 0 | 100 / 100 |  |
| 8723214 | 49 | 0.010 | -0.009 | 0.9999 | 0 | 0.02 / 0.01 | 1 / 1 | -0.02 / -0.01 | 100 / 100 |  |
| 8723970 | 49 | 0.004 | +0.001 | 0.9996 | 0 | 0.06 / 0.05 | 0.89 / 0.92 | -0.01 / -0.01 | 98.15 / 97.59 |  |
| 8724580 | 49 | 0.033 | -0.028 | 0.9913 | 0 | 0.05 / 0.04 | 0.96 / 0.95 | -0.04 / -0.01 | 98.15 / 97.96 |  |
| 8725114 | 49 | 0.049 | -0.032 | 0.9852 | 0 | 0.05 / 0.01 | 0.98 / 1 | -0.03 / 0 | 100 / 100 |  |
| 8725520 | 49 | 0.002 | +0.001 | 1.0000 | 0 | 0.01 / 0.01 | 1 / 1 | 0 / 0 | 100 / 100 |  |
| 8726384 | 49 | 0.015 | -0.014 | 0.9998 | 0 | 0.02 / 0.01 | 1 / 1 | -0.01 / -0 | 100 / 100 |  |
| 8726520 | 49 | 0.005 | -0.004 | 0.9999 | 0 | 0.01 / 0.01 | 1 / 1 | -0 / 0 | 100 / 100 |  |
| 8726607 | 49 | 0.001 | +0.000 | 1.0000 | 0 | 0.01 / 0.01 | 1 / 1 | -0.01 / -0.01 | 100 / 100 |  |
| 8726724 | 49 | 0.000 | +0.000 | 1.0000 | 0 | 0.01 / 0.02 | 1 / 1 | 0.01 / 0 | 100 / 100 |  |
| 8727520 | 49 | 0.007 | -0.006 | 1.0000 | 0 | 0.01 / 0.01 | 1 / 1 | -0 / 0 | 100 / 100 |  |
| 8728690 | 49 | 0.004 | -0.001 | 0.9999 | 0 | 0.01 / 0.01 | 1 / 1 | -0 / -0 | 100 / 100 |  |
| 8729108 | 49 | 0.005 | -0.003 | 0.9998 | 0 | 0.01 / 0.01 | 1 / 1 | -0 / -0 | 100 / 100 |  |
| 8729210 | 49 | 0.000 | -0.000 | 1.0000 | 0 | 0.01 / 0.01 | 1 / 1 | -0 / -0 | 100 / 100 |  |
| 8729840 | 49 | 0.000 | +0.000 | 1.0000 | 0 | 0.01 / 0.01 | 1 / 1 | 0 / 0 | 100 / 100 |  |
| 8735180 | 49 | 0.017 | -0.015 | 0.9987 | 0 | 0.02 / 0.01 | 1 / 1 | -0.02 / -0 | 100 / 100 |  |
| 8735391 | 49 | 0.002 | +0.001 | 1.0000 | 0 | 0.01 / 0.02 | 1 / 1 | -0.01 / -0.01 | 100 / 100 |  |
| 8736897 | 49 | 0.000 | +0.000 | 1.0000 | 0 | 0.01 / 0.02 | 1 / 1 | -0 / -0 | 100 / 100 |  |
| 8737048 | 49 | 0.000 | +0.000 | 1.0000 | 0 | 0.01 / 0.01 | 1 / 1 | -0 / -0 | 100 / 100 |  |
| 8737138 | 49 | 0.001 | +0.001 | 1.0000 | 0 | 0.01 / 0.01 | 1 / 1 | 0 / 0 | 100 / 100 |  |
| 8741533 | 49 | 0.001 | +0.001 | 0.9999 | 0 | 0.01 / 0.01 | 1 / 1 | 0 / 0 | 100 / 100 |  |
| 8747437 | 49 | 0.005 | +0.001 | 0.9996 | 0 | 0.01 / 0.01 | 1 / 1 | -0.01 / -0.01 | 100 / 100 |  |
| 8760721 | 49 | 0.009 | +0.007 | 0.9997 | 0 | 0.03 / 0.03 | 1 / 1 | 0.03 / 0.02 | 100 / 100 |  |
| 8760922 | 49 | 0.041 | +0.040 | 0.9996 | 0 | 0.53 / 0.57 | 0.97 / 0.97 | -0.53 / -0.57 | 0 / 0 |  |
| 8761305 | 49 | 0.013 | -0.012 | 0.9999 | 0 | 0.02 / 0.01 | 1 / 1 | -0.02 / -0.01 | 100 / 100 |  |
| 8761724 | 49 | 0.008 | -0.005 | 0.9991 | 0 | 0.01 / 0.01 | 1 / 1 | -0.01 / -0 | 100 / 100 |  |
| 8761927 | 49 | 0.013 | +0.011 | 0.9970 | 0 | 0.07 / 0.09 | 0.68 / 0.62 | -0.02 / -0.04 | 88.89 / 87.78 |  |
| 8762075 | 49 | 0.001 | -0.001 | 1.0000 | 0 | 0.01 / 0.01 | 1 / 1 | 0 / 0.01 | 100 / 100 |  |
| 8764044 | 49 | 0.003 | +0.003 | 1.0000 | 0 | 0.02 / 0.02 | 0.99 / 0.99 | 0.01 / 0 | 100 / 100 |  |
| 8764227 | 49 | 0.031 | +0.003 | 0.9913 | 0 | 0.03 / 0.01 | 0.99 / 1 | -0 / -0.01 | 100 / 100 |  |
| 8764314 | 49 | 0.029 | -0.025 | 0.9987 | 0 | 0.06 / 0.04 | 0.98 / 0.98 | -0.03 / -0.01 | 98.15 / 97.96 |  |
| 8766072 | 49 | 0.005 | -0.003 | 0.9999 | 0 | 0.01 / 0.07 | 1 / 0.95 | -0 / -0 | 100 / 94.07 |  |
| 8767816 | 49 | 0.000 | +0.000 | 1.0000 | 0 | 0.01 / 0.01 | 1 / 1 | 0 / -0 | 100 / 100 |  |
| 8767961 | 49 | 0.000 | +0.000 | 1.0000 | 0 | 0.01 / 0.01 | 1 / 1 | -0 / -0 | 100 / 100 |  |
| 8768094 | 49 | 0.005 | -0.004 | 0.9998 | 0 | 0.01 / 0.02 | 1 / 1 | -0 / 0 | 100 / 100 |  |
| 8770475 | 49 | 0.003 | -0.003 | 1.0000 | 0 | 0.01 / 0.03 | 1 / 0.98 | -0 / 0.01 | 100 / 99.63 |  |
| 8770520 | 49 | 0.004 | -0.004 | 1.0000 | 0 | 0.01 / 0.02 | 1 / 0.99 | 0.01 / 0.01 | 100 / 99.81 |  |
| 8770613 | 49 | 0.010 | +0.009 | 0.9999 | 0 | 0.01 / 0.02 | 1 / 1 | 0 / -0.01 | 100 / 100 |  |
| 8770777 | 49 | 0.000 | -0.000 | 1.0000 | 0 | 0.01 / 0.01 | 1 / 1 | 0.01 / 0.01 | 100 / 100 |  |
| 8770808 | 49 | 0.001 | -0.000 | 1.0000 | 0 | 0.01 / 0.02 | 1 / 0.99 | -0 / -0 | 100 / 100 |  |
| 8770822 | 49 | 0.002 | +0.002 | 1.0000 | 0 | 0.26 / 0.27 | 0.95 / 0.95 | -0.25 / -0.26 | 9.26 / 5.56 |  |
| 8770971 | 49 | 0.010 | -0.008 | 0.9998 | 0 | 0.01 / 0.02 | 1 / 1 | -0.01 / -0 | 100 / 100 |  |
| 8771013 | 49 | 0.007 | -0.006 | 0.9999 | 0 | 0.01 / 0.01 | 1 / 1 | -0.01 / -0 | 100 / 100 |  |
| 8771341 | 49 | 0.021 | -0.018 | 0.9975 | 0 | 0.02 / 0.02 | 1 / 1 | -0.02 / -0 | 100 / 100 |  |
| 8771450 | 49 | 0.009 | +0.001 | 0.9983 | 0 | 0.01 / 0.02 | 1 / 1 | -0 / -0 | 100 / 100 |  |
| 8771486 | 49 | 0.009 | -0.002 | 0.9993 | 0 | 2.33 / 1.17 | 0.42 / 0.33 | -1.01 / -0.23 | 69.44 / 92.16 |  |
| 8771972 | 49 | 0.081 | +0.007 | 0.7394 | 3 | 0.11 / 0.01 | 0.64 / 1 | 0.03 / -0 | 85.19 / 100 | FLAG |
| 8773037 | 49 | 0.004 | +0.002 | 0.9993 | 0 | 0.06 / 0.06 | 0.51 / 0.49 | 0.02 / 0.01 | 100 / 100 |  |
| 8773259 | 49 | 0.016 | +0.014 | 0.9996 | 0 | 0.01 / 0.01 | 1 / 1 | 0.01 / -0 | 100 / 100 |  |
| 8773701 | 49 | 0.016 | -0.011 | 0.9957 | 0 | 0.02 / 0.01 | 0.99 / 1 | -0.01 / 0.01 | 100 / 100 |  |
| 8773767 | 49 | 0.012 | -0.011 | 0.9995 | 0 | 0.01 / 0.01 | 1 / 1 | -0.01 / 0 | 100 / 100 |  |
| 8774513 | 49 | 0.010 | +0.001 | 0.9919 | 0 | 0.05 / 0.06 | 0.78 / 0.68 | -0 / -0 | 96.3 / 96.11 |  |
| 8774770 | 49 | 0.000 | -0.000 | 1.0000 | 0 | 0.06 / 0.06 | 0.78 / 0.78 | -0.02 / -0.03 | 96.3 / 94.63 |  |
| 8775237 | 49 | 0.009 | -0.008 | 0.9989 | 0 | 0.01 / 0.03 | 1 / 0.97 | -0.01 / -0 | 100 / 99.63 |  |
| 8775241 | 49 | 0.117 | -0.108 | 0.9636 | 0 | 0.12 / 0.02 | 0.97 / 0.99 | -0.11 / 0 | 85.19 / 100 | FLAG |
| 8775283 | 49 | 0.014 | -0.013 | 0.9968 | 0 | 0.01 / 0.01 | 0.99 / 0.99 | -0.01 / 0 | 100 / 100 |  |
| 8775296 | 49 | 0.008 | -0.007 | 0.9986 | 0 | 0.01 / 0.01 | 0.99 / 0.99 | -0 / 0 | 100 / 100 |  |
| 8775792 | 49 | 0.023 | +0.013 | 0.9328 | 0 | 0.03 / 0.04 | 0.86 / 0.8 | -0.01 / -0.03 | 100 / 100 | FLAG |
| 8779280 | 49 | 0.014 | +0.004 | 0.9947 | 0 |  / 0.08 |  / 0.47 |  / -0.01 |  / 87.04 |  |
| 8779748 | 49 | 0.023 | -0.022 | 0.9986 | 0 |  / 0.02 |  / 1 |  / -0.01 |  / 100 |  |
| 8779749 | 49 | 0.039 | -0.037 | 0.9959 | 0 |  / 0.01 |  / 1 |  / 0 |  / 100 |  |
| 8779770 | 49 | 0.002 | -0.002 | 1.0000 | 0 |  / 0.01 |  / 1 |  / -0 |  / 100 |  |
| 9410170 | 49 | 0.244 | -0.242 | 0.9954 | 0 | 0.25 / 0.02 | 1 / 1 | -0.24 / -0 | 0 / 100 | FLAG |
| 9410230 | 49 | 0.035 | -0.035 | 0.9999 | 0 | 0.04 / 0.01 | 1 / 1 | -0.04 / -0 | 100 / 100 |  |
| 9410660 | 49 | 0.023 | -0.023 | 1.0000 | 0 |  / 0.02 |  / 1 |  / -0 |  / 100 |  |
| 9410840 | 49 | 0.034 | -0.034 | 0.9999 | 0 | 0.03 / 0.02 | 1 / 1 | -0.03 / 0 | 100 / 100 |  |
| 9411340 | 49 | 0.045 | -0.044 | 0.9997 | 0 | 0.04 / 0.02 | 1 / 1 | -0.04 / 0.01 | 100 / 100 |  |
| 9412110 | 49 | 0.254 | -0.251 | 0.9939 | 0 |  / 0.03 |  / 1 |  / -0 |  / 100 | FLAG |
| 9413450 | 49 | 0.062 | -0.061 | 0.9999 | 0 | 0.06 / 0.02 | 1 / 1 | -0.06 / -0 | 100 / 100 | FLAG |
| 9414290 | 49 | 0.022 | -0.021 | 1.0000 | 0 | 0.02 / 0.01 | 1 / 1 | -0.02 / -0 | 100 / 100 |  |
| 9414523 | 49 | 0.000 | -0.000 | 1.0000 | 0 | 0.23 / 0.23 | 0.99 / 0.99 | -0.22 / -0.22 | 22.22 / 22.22 |  |
| 9414750 | 49 | 0.260 | -0.255 | 0.9943 | 0 | 0.25 / 0.01 | 0.99 / 1 | -0.25 / 0 | 0 / 100 | FLAG |
| 9414863 | 49 | 0.084 | -0.082 | 0.9990 | 0 | 0.08 / 0.02 | 1 / 1 | -0.08 / -0 | 100 / 100 | FLAG |
| 9415020 | 49 | 0.047 | -0.047 | 0.9999 | 0 | 0.08 / 0.04 | 0.99 / 1 | -0.06 / -0 | 96.3 / 98.86 |  |
| 9415102 | 49 | 0.010 | -0.010 | 1.0000 | 0 | 0.01 / 0.01 | 1 / 1 | -0.01 / 0 | 100 / 100 |  |
| 9415144 | 49 | 0.002 | +0.001 | 1.0000 | 0 | 0.01 / 0.01 | 1 / 1 | 0 / 0 | 100 / 100 |  |
| 9416841 | 49 | 0.028 | -0.027 | 0.9999 | 0 | 0.03 / 0.01 | 1 / 1 | -0.03 / -0 | 100 / 100 |  |
| 9418767 | 49 | 0.000 | +0.000 | 1.0000 | 0 | 0.01 / 0.02 | 1 / 1 | -0 / -0 | 100 / 100 |  |
| 9419750 | 49 | 0.012 | -0.010 | 0.9999 | 0 | 0.01 / 0.03 | 1 / 1 | -0.01 / 0 | 100 / 100 |  |
| 9431647 | 49 | 0.031 | +0.007 | 0.9989 | 0 | 0.03 / 0.01 | 1 / 1 | 0.01 / 0 | 100 / 100 |  |
| 9432780 | 49 | 0.004 | -0.003 | 1.0000 | 0 | 0.01 / 0.01 | 1 / 1 | -0 / 0 | 100 / 100 |  |
| 9435380 | 49 | 0.006 | -0.005 | 1.0000 | 0 | 0.01 / 0.01 | 1 / 1 | -0 / 0 | 100 / 100 |  |
| 9437540 | 49 | 0.004 | -0.001 | 1.0000 | 0 | 0.01 / 0.01 | 1 / 1 | 0 / 0 | 100 / 100 |  |
| 9439040 | 49 | 0.004 | -0.003 | 1.0000 | 0 | 0.05 / 0.05 | 1 / 1 | -0.02 / -0.01 | 100 / 100 |  |
| 9439099 | 49 | 0.010 | +0.009 | 1.0000 | 0 | 0.01 / 0.01 | 1 / 1 | 0.01 / -0 | 100 / 100 |  |
| 9439201 | 49 | 0.031 | +0.030 | 0.9994 | 0 | 0.03 / 0.01 | 1 / 1 | 0.03 / -0 | 100 / 100 |  |
| 9440083 | 49 | 0.001 | -0.000 | 1.0000 | 0 | 0.01 / 0.01 | 1 / 1 | -0 / -0 | 100 / 100 |  |
| 9440357 | 49 | 0.015 | -0.014 | 0.9999 | 0 | 0.53 / 0.54 | 0.89 / 0.88 | 0.51 / 0.52 | 0 / 0 |  |
| 9440422 | 49 | 0.024 | +0.022 | 0.9997 | 0 | 0.02 / 0.01 | 1 / 1 | 0.02 / -0 | 100 / 100 |  |
| 9440569 | 49 | 0.009 | +0.008 | 1.0000 | 0 | 0.02 / 0.01 | 1 / 1 | 0.01 / 0 | 100 / 100 |  |
| 9440581 | 49 | 0.017 | -0.016 | 0.9999 | 0 | 0.02 / 0.01 | 1 / 1 | -0.02 / -0 | 100 / 100 |  |
| 9440910 | 49 | 0.011 | -0.007 | 0.9999 | 0 | 0.01 / 0.01 | 1 / 1 | -0.01 / 0 | 100 / 100 |  |
| 9441102 | 49 | 0.002 | -0.002 | 1.0000 | 0 | 0.01 / 0.01 | 1 / 1 | -0 / 0 | 100 / 100 |  |
| 9442396 | 49 | 0.014 | -0.013 | 1.0000 | 0 | 0.02 / 0.02 | 1 / 1 | -0.02 / -0 | 100 / 100 |  |
| 9443090 | 49 | 0.004 | +0.001 | 1.0000 | 0 | 0.01 / 0.02 | 1 / 1 | -0 / -0 | 100 / 100 |  |
| 9444090 | 49 | 0.024 | +0.023 | 1.0000 | 0 | 0.03 / 0.01 | 1 / 1 | 0.03 / 0 | 100 / 100 |  |
| 9444900 | 49 | 0.012 | -0.009 | 0.9999 | 0 | 0.02 / 0.01 | 1 / 1 | -0.01 / -0 | 100 / 100 |  |
| 9445958 | 49 | 0.002 | +0.001 | 1.0000 | 0 | 0.22 / 0.22 | 0.98 / 0.98 | -0.14 / -0.14 | 62.96 / 63.89 |  |
| 9446484 | 49 | 0.223 | -0.133 | 0.9834 | 0 | 0.21 / 0.01 | 0.98 / 1 | -0.13 / 0 | 59.26 / 100 | FLAG |
| 9447130 | 49 | 0.207 | -0.129 | 0.9855 | 0 | 0.2 / 0.01 | 0.99 / 1 | -0.12 / -0 | 59.26 / 100 | FLAG |
| 9449424 | 49 | 0.029 | -0.017 | 0.9996 | 0 | 0.03 / 0.01 | 1 / 1 | -0.02 / -0 | 100 / 100 |  |
| 9449880 | 49 | 0.021 | -0.015 | 0.9998 | 0 | 0.02 / 0.02 | 1 / 1 | -0.01 / -0 | 100 / 100 |  |
| 9450460 | 49 | 0.159 | +0.076 | 0.9897 | 0 | 0.16 / 0.01 | 0.99 / 1 | 0.08 / -0 | 53.7 / 100 | FLAG |
| 9451054 | 49 | 0.002 | +0.002 | 1.0000 | 0 | 0.01 / 0.01 | 1 / 1 | 0 / 0 | 100 / 100 |  |
| 9451600 | 49 | 0.098 | -0.093 | 0.9987 | 0 | 0.16 / 0.26 | 1 / 0.99 | 0.15 / 0.25 | 57.41 / 3.7 | FLAG |
| 9452210 | 49 | 0.389 | +0.369 | 0.9916 | 0 | 0.41 / 0.01 | 0.99 / 1 | 0.39 / 0 | 1.85 / 100 | FLAG |
| 9452400 | 49 | 0.024 | +0.023 | 1.0000 | 0 | 0.03 / 0.02 | 1 / 1 | 0.03 / 0 | 100 / 100 |  |
| 9452634 | 49 | 0.251 | +0.247 | 0.9976 | 0 | 0.26 / 0.01 | 1 / 1 | 0.25 / 0 | 0 / 100 | FLAG |
| 9453220 | 49 | 0.055 | +0.007 | 0.9964 | 0 | 0.05 / 0.02 | 1 / 1 | 0.01 / -0 | 100 / 100 |  |
| 9454050 | 49 | 0.012 | +0.007 | 0.9999 | 0 | 0.02 / 0.01 | 1 / 1 | 0.01 / 0 | 100 / 100 |  |
| 9454240 | 49 | 0.845 | +0.842 | 0.9959 | 0 | 0.85 / 0.01 | 1 / 1 | 0.84 / -0 | 0 / 100 | FLAG |
| 9455090 | 49 | 0.056 | +0.055 | 1.0000 | 0 | 0.05 / 0.02 | 1 / 1 | 0.05 / 0 | 100 / 100 | FLAG |
| 9455500 | 49 | 0.129 | -0.045 | 0.9938 | 0 | 0.13 / 0.01 | 0.99 / 1 | -0.05 / -0 | 72.22 / 100 |  |
| 9455760 | 49 | 0.370 | -0.343 | 0.9946 | 0 | 0.38 / 0.01 | 0.99 / 1 | -0.36 / 0 | 11.11 / 100 | FLAG |
| 9455920 | 49 | 0.041 | +0.039 | 1.0000 | 0 |  / 0.69 |  / 1 |  / -0.68 |  / 0 |  |
| 9457292 | 49 | 0.221 | +0.187 | 0.9780 | 0 |  / 0.05 |  / 1 |  / 0.01 |  / 98.42 | FLAG |
| 9457804 | 49 | 0.003 | -0.002 | 1.0000 | 0 | 0.01 / 0.01 | 1 / 1 | 0 / 0 | 100 / 100 |  |
| 9459450 | 49 | 0.146 | -0.139 | 0.9960 | 0 | 0.15 / 0.02 | 1 / 1 | -0.14 / -0 | 59.26 / 100 | FLAG |
| 9459881 | 49 | 0.088 | +0.066 | 0.9925 | 0 | 0.09 / 0.02 | 0.99 / 1 | 0.07 / 0 | 90.74 / 100 | FLAG |
| 9461380 | 49 | 0.151 | +0.146 | 0.9966 | 0 | 0.15 / 0.02 | 1 / 1 | 0.15 / 0 | 40.74 / 100 | FLAG |
| 9461710 | 49 | 0.010 | +0.010 | 1.0000 | 0 | 0.01 / 0.02 | 1 / 1 | 0.01 / -0.01 | 100 / 100 |  |
| 9462450 | 49 | 0.099 | +0.062 | 0.9831 | 0 | 0.09 / 0.04 | 0.98 / 1 | 0.05 / -0.02 | 92.59 / 99.8 | FLAG |
| 9462620 | 49 | 0.270 | +0.251 | 0.9816 | 0 | 0.28 / 0.02 | 0.98 / 1 | 0.26 / 0 | 16.67 / 100 | FLAG |
| 9463502 | 49 | 0.001 | -0.001 | 1.0000 | 0 | 0.05 / 0.06 | 1 / 1 | 0.02 / 0.02 | 94.44 / 93.33 |  |
| 9464212 | 49 | 0.005 | +0.000 | 0.9999 | 0 | 0.01 / 0.02 | 1 / 1 | -0 / -0 | 100 / 100 |  |
| 9468333 | 49 | 0.051 | -0.031 | 0.9976 | 0 | 0.06 / 0.01 | 1 / 1 | -0.04 / -0 | 100 / 100 |  |
| 9468756 | 49 | 0.022 | -0.018 | 0.9995 | 0 | 0.02 / 0.01 | 1 / 1 | -0.01 / 0 | 100 / 100 |  |
| 9491094 | 49 | 0.029 | -0.027 | 0.9992 | 0 | 0.03 / 0.01 | 1 / 1 | -0.03 / -0 | 100 / 100 |  |
| 9497645 | 49 | 0.020 | -0.016 | 0.9998 | 0 | 0.15 / 0.15 | 0.95 / 0.96 | -0.1 / -0.09 | 62.96 / 62.59 |  |
| 9751364 | 49 | 0.006 | -0.006 | 0.9996 | 0 | 0.01 / 0.01 | 1 / 1 | -0.01 / -0 | 100 / 100 |  |
| 9751381 | 49 | 0.011 | -0.011 | 0.9999 | 0 | 0.02 / 0.01 | 0.99 / 0.99 | -0.01 / -0 | 100 / 100 |  |
| 9751401 | 49 | 0.026 | -0.024 | 0.9951 | 0 |  / 0.01 |  / 0.99 |  / -0 |  / 100 |  |
| 9751639 | 49 | 0.053 | -0.051 | 0.9903 | 0 |  / 0.02 |  / 0.98 |  / -0 |  / 100 | FLAG |
| 9752695 | 49 | 0.037 | -0.036 | 0.9935 | 0 |  / 0.05 |  / 0.87 |  / -0 |  / 96.85 |  |
| 9755371 | 49 | 0.009 | -0.008 | 0.9998 | 0 | 0.01 / 0.01 | 1 / 1 | -0.01 / 0 | 100 / 100 |  |
| 9759110 | 49 | 0.097 | +0.051 | 0.9996 | 0 | 0.4 / 0.15 | 0.54 / 0.73 | -0.21 / -0.06 | 73.81 / 84.19 | FLAG |
| 9759394 | 49 | 0.017 | -0.014 | 0.9953 | 0 | 0.02 / 0.01 | 0.99 / 0.99 | -0.01 / -0 | 100 / 100 |  |
| 9759938 | 49 | 0.000 | -0.000 | 1.0000 | 0 | 0.01 / 0.02 | 1 / 0.99 | -0 / -0 | 100 / 100 |  |

## Appendix E. Per-station matching, forecast_a, cycle 2026-10-14 00z

| ID | Name | fields | stations | fields node | km | km source | nearest node of any kind, km | nearer node dry | stations index | reason if one mode only |
|---|---|---|---|---|---|---|---|---|---|---|
| 1611400 | Nawiliwili | yes | yes | 2115047 | 0.070 | exact | 0.070 |  | 201 |  |
| 1612340 | Honolulu | yes | yes | 2196805 | 0.069 | exact | 0.069 |  | 202 |  |
| 1612401 | Pearl Harbor | yes | no | 1694349 | 0.065 | exact | 0.065 |  |  | stations run: "No matching model station found" (not in points file station list) |
| 1612480 | Mokuoloe | yes | yes | 2067685 | 0.037 | exact | 0.037 |  | 203 |  |
| 1615680 | Kahului, Kahului Harbor | yes | yes | 2444047 | 0.064 | exact | 0.064 |  | 204 |  |
| 1617433 | Kawaihae | yes | yes | 2266382 | 0.094 | exact | 0.063 | yes | 205 |  |
| 1617760 | Hilo, Hilo Bay, Kuhio Bay | yes | yes | 2567990 | 0.095 | exact | 0.054 | yes | 206 |  |
| 1619910 | Sand Island, Midway Islands | yes | yes | 1681601 | 0.783 | exact | 0.783 |  | 207 |  |
| 1630000 | Apra Harbor, Guam | yes | yes | 1051396 | 0.071 | exact | 0.071 |  | 224 |  |
| 1631428 | Pago Bay, Guam | yes | yes | 1209204 | 0.071 | exact | 0.071 |  | 225 |  |
| 1770000 | Pago Pago, American Samoa | yes | yes | 3753584 | 0.369 | exact | 0.161 | yes | 226 |  |
| 1820000 | Kwajalein, Marshall Islands | yes | yes | 2162012 | 0.077 | exact | 0.077 |  | 227 |  |
| 1890000 | Wake Island, Pacific Ocean | yes | yes | 1706413 | 0.152 | exact | 0.152 |  | 228 |  |
| 2695535 | Bermuda Biological Station | yes | no | 12471348 | 0.409 | exact | 0.409 |  |  | stations run: "No matching model station found" (not in points file station list) |
| 2695540 | Bermuda, St. Georges Island | yes | no | 12471349 | 0.801 | exact | 0.801 |  |  | stations run: "No matching model station found" (not in points file station list) |
| 8410140 | Eastport | yes | yes | 10423350 | 0.108 | exact | 0.108 |  | 0 |  |
| 8411060 | Cutler Farris Wharf | yes | yes | 11319611 | 0.150 | exact | 0.055 | yes | 1 |  |
| 8413320 | Bar Harbor | yes | yes | 11391369 | 0.049 | exact | 0.049 |  | 2 |  |
| 8418150 | Portland | yes | yes | 10952780 | 0.056 | exact | 0.056 |  | 3 |  |
| 8419870 | Seavey Island | yes | yes | 11050155 | 0.037 | exact | 0.037 |  | 585 |  |
| 8443970 | Boston | yes | yes | 10662974 | 0.094 | exact | 0.094 |  | 6 |  |
| 8447386 | Fall River | yes | yes | 11151953 | 0.169 | exact | 0.052 | yes | 7 |  |
| 8447435 | Chatham | yes | yes | 12376886 | 0.050 | exact | 0.050 |  | 573 |  |
| 8447636 | New Bedford Harbor | yes | no | 12240173 | 0.042 | exact | 0.042 |  |  | stations run: "No matching model station found" (not in points file station list) |
| 8447930 | Woods Hole | yes | yes | 12227775 | 0.092 | exact | 0.045 | yes | 8 |  |
| 8449130 | Nantucket Island | yes | yes | 12517134 | 0.152 | exact | 0.065 | yes | 9 |  |
| 8452660 | Newport | yes | yes | 12385477 | 0.075 | exact | 0.075 |  | 10 |  |
| 8452944 | Conimicut Light | yes | yes | 11465433 | 0.077 | exact | 0.077 |  | 11 |  |
| 8454000 | Providence | yes | yes | 10403365 | 0.045 | exact | 0.045 |  | 12 |  |
| 8461490 | New London | yes | yes | 12083637 | 0.063 | exact | 0.063 |  | 14 |  |
| 8465705 | New Haven | yes | yes | 12473999 | 0.083 | exact | 0.083 |  | 15 |  |
| 8467150 | Bridgeport | yes | yes | 12551763 | 0.048 | exact | 0.048 |  | 16 |  |
| 8510560 | Montauk | yes | yes | 12613478 | 0.101 | exact | 0.101 |  | 17 |  |
| 8516945 | Kings Point | yes | yes | 12717748 | 0.052 | exact | 0.052 |  | 18 |  |
| 8518750 | The Battery | yes | yes | 12723926 | 0.105 | exact | 0.105 |  | 19 |  |
| 8518962 | Turkey Point Hudson River NERRS | yes | no | 12740023 | 0.181 | exact | 0.181 |  |  | stations run: "No matching model station found" (not in points file station list) |
| 8518979 | Coxsackie, Hudson River | yes | no | 12743897 | 0.073 | exact | 0.073 |  |  | stations run: "No matching model station found" (not in points file station list) |
| 8531680 | Sandy Hook | yes | yes | 12696408 | 0.062 | exact | 0.062 |  | 21 |  |
| 8534720 | Atlantic City | yes | yes | 12585344 | 0.154 | exact | 0.154 |  | 22 |  |
| 8536110 | Cape May | yes | yes | 12286984 | 0.034 | exact | 0.034 |  | 23 |  |
| 8537121 | Ship John Shoal | yes | yes | 11004835 | 0.170 | exact | 0.170 |  | 24 |  |
| 8539094 | Burlington, Delaware River | yes | yes | 5519961 | 0.079 | exact | 0.079 |  | 26 |  |
| 8540433 | Marcus Hook | yes | yes | 7290577 | 0.107 | exact | 0.107 |  | 27 |  |
| 8545240 | Philadelphia | yes | yes | 6206641 | 0.009 | exact | 0.009 |  | 28 |  |
| 8546252 | Bridesburg | yes | yes | 5979109 | 0.068 | exact | 0.068 |  | 634 |  |
| 8548989 | Newbold | yes | yes | 5326150 | 0.054 | exact | 0.054 |  | 29 |  |
| 8551762 | Delaware City | yes | yes | 8785824 | 0.094 | exact | 0.094 |  | 30 |  |
| 8555889 | Brandywine Shoal Light | yes | yes | 12327360 | 0.242 | exact | 0.242 |  | 635 |  |
| 8557380 | Lewes | yes | yes | 12371093 | 0.073 | exact | 0.073 |  | 32 |  |
| 8570283 | Ocean City Inlet | yes | yes | 12436577 | 0.062 | exact | 0.062 |  | 33 |  |
| 8571421 | Bishops Head | yes | yes | 9577268 | 0.236 | exact | 0.110 | yes | 34 |  |
| 8571892 | Cambridge | yes | yes | 7141231 | 0.124 | exact | 0.124 |  | 35 |  |
| 8573364 | Tolchester Beach | yes | yes | 6390507 | 0.172 | ctl | 0.046 | yes | 36 |  |
| 8573927 | Chesapeake City | yes | yes | 7222275 | 0.051 | exact | 0.051 |  | 37 |  |
| 8574680 | Baltimore | yes | yes | 6113506 | 0.050 | exact | 0.050 |  | 38 |  |
| 8575512 | Annapolis | yes | yes | 6985745 | 0.041 | exact | 0.041 |  | 39 |  |
| 8577330 | Solomons Island | yes | yes | 9149498 | 0.092 | exact | 0.092 |  | 40 |  |
| 8594900 | Washington | yes | yes | 5193966 | 0.022 | exact | 0.022 |  | 41 |  |
| 8631044 | Wachapreague | yes | no | 11973757 | 0.159 | exact | 0.046 | yes |  | stations run: "No matching model station found" (not in points file station list) |
| 8632200 | Kiptopeke | yes | yes | 11975615 | 0.071 | exact | 0.071 |  | 42 |  |
| 8635027 | Dahlgren | yes | yes | 7274222 | 0.038 | exact | 0.038 |  | 676 |  |
| 8635750 | Lewisetta | yes | yes | 9928546 | 0.040 | exact | 0.040 |  | 44 |  |
| 8636580 | Windmill Point | yes | yes | 11081856 | 0.176 | ctl | 0.118 | yes | 45 |  |
| 8637689 | Yorktown USCG Training Center | yes | yes | 11228916 | 0.041 | exact | 0.041 |  | 46 |  |
| 8638610 | Sewells Point | yes | yes | 11056750 | 0.043 | exact | 0.043 |  | 47 |  |
| 8638901 | CBBT, Chesapeake Channel | yes | yes | 12244211 | 0.409 | exact | 0.409 |  | 648 |  |
| 8639348 | Money Point | yes | yes | 10383577 | 0.027 | exact | 0.027 |  | 49 |  |
| 8651370 | Duck | yes | yes | 12436645 | 0.201 | exact | 0.201 |  | 50 |  |
| 8652587 | Oregon Inlet Marina | yes | yes | 12372996 | 0.206 | exact | 0.030 | yes | 51 |  |
| 8654467 | USCG Station Hatteras | yes | yes | 12418067 | 0.034 | exact | 0.034 |  | 52 |  |
| 8656483 | Beaufort, Duke Marine Lab | yes | yes | 12071012 | 0.073 | exact | 0.073 |  | 53 |  |
| 8658120 | Wilmington | yes | yes | 11447083 | 0.153 | exact | 0.153 |  | 54 |  |
| 8658163 | Wrightsville Beach | yes | yes | 12321343 | 0.092 | exact | 0.092 |  | 55 |  |
| 8661070 | Springmaid Pier | yes | yes | 12247145 | 0.161 | exact | 0.161 |  | 56 |  |
| 8665530 | Charleston | yes | yes | 11379559 | 0.047 | exact | 0.047 |  | 58 |  |
| 8670870 | Fort Pulaski | yes | yes | 11426157 | 0.043 | exact | 0.043 |  | 59 |  |
| 8679598 | Kings Bay MSF Pier | yes | yes | 11613575 | 0.084 | exact | 0.084 |  | 829 |  |
| 8720030 | Fernandina Beach | yes | yes | 11615072 | 0.062 | exact | 0.062 |  | 60 |  |
| 8720218 | Mayport (Bar Pilots Dock) | yes | yes | 11614891 | 0.061 | exact | 0.061 |  | 61 |  |
| 8720219 | Dames Point | yes | yes | 10471022 | 0.058 | exact | 0.058 |  | 696 |  |
| 8720226 | Southbank Riverwalk, St Johns River | yes | yes | 10077188 | 0.079 | exact | 0.079 |  | 698 |  |
| 8720357 | I-295 Buckman Bridge | yes | yes | 10022375 | 0.194 | exact | 0.194 |  | 62 |  |
| 8721604 | Trident Pier, Port Canaveral | yes | yes | 12002497 | 0.051 | exact | 0.051 |  | 63 |  |
| 8722670 | Lake Worth Pier, Atlantic Ocean | yes | yes | 11942331 | 0.067 | exact | 0.067 |  | 64 |  |
| 8722956 | South Port Everglades | yes | yes | 11596855 | 0.201 | exact | 0.062 | yes | 825 |  |
| 8723214 | Virginia Key | yes | yes | 10992109 | 0.035 | exact | 0.035 |  | 65 |  |
| 8723970 | Vaca Key, Florida Bay | yes | yes | 10435399 | 0.066 | exact | 0.066 |  | 66 |  |
| 8724580 | Key West | yes | yes | 10372778 | 0.162 | exact | 0.020 | yes | 67 |  |
| 8725114 | Naples Bay, North | yes | yes | 9658768 | 0.029 | exact | 0.029 |  | 68 |  |
| 8725520 | Fort Myers | yes | yes | 8087023 | 0.054 | exact | 0.054 |  | 69 |  |
| 8726384 | Port Manatee | yes | yes | 8365161 | 0.068 | exact | 0.068 |  | 70 |  |
| 8726520 | St. Petersburg | yes | yes | 8261748 | 0.023 | exact | 0.023 |  | 71 |  |
| 8726607 | Old Port Tampa | yes | yes | 7587030 | 0.082 | exact | 0.082 |  | 72 |  |
| 8726674 | East Bay | yes | no | 6915498 | 0.036 | exact | 0.036 |  |  | stations run: "No matching model station found" (not in points file station list) |
| 8726724 | Clearwater Beach | yes | yes | 9508045 | 0.125 | exact | 0.125 |  | 74 |  |
| 8727520 | Cedar Key | yes | yes | 8870576 | 0.025 | exact | 0.025 |  | 75 |  |
| 8728690 | Apalachicola | yes | yes | 8659818 | 0.027 | exact | 0.027 |  | 76 |  |
| 8729108 | Panama City | yes | yes | 8821913 | 0.044 | exact | 0.044 |  | 77 |  |
| 8729210 | Panama City Beach | yes | yes | 9468801 | 0.160 | exact | 0.160 |  | 78 |  |
| 8729840 | Pensacola | yes | yes | 8447512 | 0.198 | exact | 0.076 | yes | 699 |  |
| 8735180 | Dauphin Island | yes | yes | 9253171 | 0.036 | exact | 0.036 |  | 79 |  |
| 8735391 | Dog River Bridge | yes | yes | 7529581 | 0.056 | exact | 0.056 |  | 80 |  |
| 8735523 | East Fowl River Bridge | yes | no | 8114751 | 0.079 | exact | 0.079 |  |  | stations run: "No matching model station found" (not in points file station list) |
| 8736897 | Coast Guard Sector Mobile | yes | yes | 7454290 | 0.026 | exact | 0.026 |  | 700 |  |
| 8737048 | Mobile State Docks | yes | yes | 7346582 | 0.059 | exact | 0.059 |  | 81 |  |
| 8737138 | Chickasaw Creek | yes | yes | 6915980 | 0.083 | exact | 0.083 |  | 701 |  |
| 8738043 | West Fowl River Bridge | yes | no | 8396177 | 1.763 | exact | 0.025 | yes |  | stations run: "No matching model station found" (not in points file station list) |
| 8739803 | Bayou La Batre Bridge | yes | no | 8018428 | 0.034 | exact | 0.034 |  |  | stations run: "No matching model station found" (not in points file station list) |
| 8741533 | Pascagoula NOAA Lab | yes | yes | 8042473 | 0.024 | exact | 0.024 |  | 703 |  |
| 8747437 | Bay Waveland Yacht Club | yes | yes | 7712785 | 0.078 | exact | 0.078 |  | 82 |  |
| 8760721 | Pilottown | yes | yes | 8378663 | 0.047 | exact | 0.047 |  | 83 |  |
| 8760922 | Pilots Station East, S.W. Pass | yes | yes | 9736063 | 0.028 | exact | 0.028 |  | 84 |  |
| 8761305 | Shell Beach | yes | yes | 7295178 | 0.105 | exact | 0.105 |  | 85 |  |
| 8761724 | Grand Isle | yes | yes | 9279697 | 0.086 | exact | 0.073 | yes | 704 |  |
| 8761927 | New Canal Station | yes | yes | 6345735 | 0.050 | exact | 0.050 |  | 86 |  |
| 8762075 | Port Fourchon, Belle Pass | yes | yes | 9279768 | 0.043 | exact | 0.043 |  | 87 |  |
| 8762482 | West Bank 1, Bayou Gauche | yes | no | 5993026 | 0.089 | exact | 0.089 |  |  | stations run: "No matching model station found" (not in points file station list) |
| 8764044 | Berwick, Atchafalaya River | yes | yes | 6492963 | 0.066 | exact | 0.066 |  | 88 |  |
| 8764227 | LAWMA, Amerada Pass | yes | yes | 8068752 | 0.052 | exact | 0.052 |  | 89 |  |
| 8764314 | Eugene Island, North of, Atchafalaya Bay | yes | yes | 8850040 | 0.069 | exact | 0.069 |  | 705 |  |
| 8766072 | Freshwater Canal Locks | yes | yes | 8824404 | 0.077 | exact | 0.077 |  | 706 |  |
| 8767816 | Lake Charles | yes | yes | 6699783 | 0.369 | exact | 0.158 | yes | 707 |  |
| 8767961 | Bulk Terminal | yes | yes | 6710731 | 0.039 | exact | 0.039 |  | 708 |  |
| 8768094 | Calcasieu Pass | yes | yes | 8531520 | 0.056 | exact | 0.056 |  | 709 |  |
| 8770475 | Port Arthur | yes | yes | 7515568 | 0.140 | exact | 0.034 | yes | 90 |  |
| 8770520 | Rainbow Bridge | yes | yes | 7053385 | 0.091 | exact | 0.075 | yes | 91 |  |
| 8770613 | Morgans Point, Barbours Cut | yes | yes | 6828284 | 0.049 | exact | 0.049 |  | 93 |  |
| 8770777 | Manchester | yes | yes | 6127860 | 0.048 | exact | 0.048 |  | 94 |  |
| 8770808 | High Island | yes | yes | 8400336 | 0.079 | exact | 0.079 |  | 710 |  |
| 8770822 | Texas Point, Sabine Pass | yes | yes | 8479957 | 0.040 | exact | 0.040 |  | 95 |  |
| 8770971 | Rollover Pass | yes | yes | 8585940 | 0.035 | exact | 0.035 |  | 96 |  |
| 8771013 | Eagle Point, Galveston Bay | yes | yes | 7316816 | 0.138 | exact | 0.138 |  | 97 |  |
| 8771341 | Galveston Bay Entrance, North Jetty | yes | yes | 8399960 | 0.064 | exact | 0.064 |  | 98 |  |
| 8771450 | Galveston Pier 21 | yes | yes | 8612551 | 0.034 | exact | 0.034 |  | 99 |  |
| 8771486 | Galveston Railroad Bridge | yes | yes | 8094674 | 0.177 | exact | 0.177 |  | 712 |  |
| 8771972 | San Luis Pass | yes | yes | 8929875 | 0.068 | exact | 0.068 |  | 714 |  |
| 8772471 | Freeport Harbor | yes | no | 8825253 | 0.072 | exact | 0.072 |  |  | stations run: "No matching model station found" (not in points file station list) |
| 8772985 | Sargent | yes | no | 9064268 | 0.486 | ctl | 0.027 | yes |  | stations run: "No matching model station found" (not in points file station list) |
| 8773037 | Seadrift | yes | yes | 7460198 | 0.068 | exact | 0.068 |  | 101 |  |
| 8773146 | Matagorda City | yes | no | 8482899 | 0.636 | exact | 0.016 | yes |  | stations run: "No matching model station found" (not in points file station list) |
| 8773259 | Port Lavaca | yes | yes | 7069751 | 0.090 | exact | 0.090 |  | 102 |  |
| 8773701 | Port O'Connor | yes | yes | 8587433 | 0.061 | exact | 0.061 |  | 103 |  |
| 8773767 | Matagorda Bay Entrance Channel | yes | yes | 8931373 | 0.051 | exact | 0.051 |  | 715 |  |
| 8774230 | Aransas Wildlife Refuge | yes | no | 8507119 | 0.066 | exact | 0.066 |  |  | stations run: "No matching model station found" (not in points file station list) |
| 8774513 | Copano Bay | yes | yes | 8049205 | 0.126 | exact | 0.126 |  | 104 |  |
| 8774770 | Rockport | yes | yes | 8298457 | 0.053 | exact | 0.053 |  | 105 |  |
| 8775132 | La Quinta Channel North | yes | no | 7615419 | 0.058 | exact | 0.058 |  |  | stations run: "No matching model station found" (not in points file station list) |
| 8775222 | Viola Turning Basin | yes | no | 6921937 | 0.100 | exact | 0.100 |  |  | stations run: "No matching model station found" (not in points file station list) |
| 8775237 | Port Aransas | yes | yes | 8879488 | 0.103 | exact | 0.103 |  | 106 |  |
| 8775241 | Aransas, Aransas Pass | yes | yes | 9089180 | 0.045 | exact | 0.045 |  | 716 |  |
| 8775283 | Enbridge, Ingleside | yes | yes | 8350134 | 0.084 | exact | 0.084 |  | 107 |  |
| 8775296 | USS Lexington, Corpus Christi Bay | yes | yes | 7697478 | 0.023 | exact | 0.023 |  | 108 |  |
| 8775792 | Packery Channel | yes | yes | 8721901 | 0.159 | exact | 0.024 | yes | 109 |  |
| 8776139 | S. Bird Island | yes | no | 8853604 | 0.028 | exact | 0.028 |  |  | stations run: "No matching model station found" (not in points file station list) |
| 8776604 | Baffin Bay | yes | no | 8587975 | 0.058 | exact | 0.058 |  |  | stations run: "No matching model station found" (not in points file station list) |
| 8777812 | Rincon Del San Jose | yes | no | 8171593 | 0.079 | exact | 0.079 |  |  | stations run: "No matching model station found" (not in points file station list) |
| 8778490 | Port Mansfield | yes | no | 8097634 | 0.059 | exact | 0.059 |  |  | stations run: "No matching model station found" (not in points file station list) |
| 8779280 | Realitos Peninsula | yes | yes | 8482560 | 0.041 | exact | 0.041 |  | 720 |  |
| 8779748 | South Padre Island CG Station | yes | yes | 8904560 | 0.040 | exact | 0.040 |  | 111 |  |
| 8779749 | SPI Brazos Santiago | yes | yes | 8956867 | 0.091 | exact | 0.091 |  | 718 |  |
| 8779770 | Port Isabel | yes | yes | 8427435 | 0.077 | exact | 0.077 |  | 112 |  |
| 9410170 | San Diego | yes | yes | 1628109 | 0.053 | exact | 0.053 |  | 128 |  |
| 9410230 | La Jolla | yes | yes | 2431095 | 0.014 | exact | 0.014 |  | 129 |  |
| 9410660 | Los Angeles | yes | yes | 1786436 | 0.022 | exact | 0.022 |  | 131 |  |
| 9410678 | Long Beach Fire Boat Pier | yes | no | 1726662 | 0.010 | exact | 0.010 |  |  | stations run: "No matching model station found" (not in points file station list) |
| 9410840 | Santa Monica | yes | yes | 1984114 | 0.057 | exact | 0.057 |  | 134 |  |
| 9411340 | Santa Barbara | yes | yes | 2173699 | 0.045 | exact | 0.045 |  | 135 |  |
| 9412110 | Port San Luis | yes | yes | 2527827 | 0.030 | exact | 0.030 |  | 138 |  |
| 9413450 | Monterey | yes | yes | 2224669 | 0.051 | exact | 0.051 |  | 140 |  |
| 9414290 | San Francisco | yes | yes | 1972248 | 0.026 | exact | 0.026 |  | 141 |  |
| 9414523 | Redwood City | yes | yes | 810037 | 0.045 | exact | 0.045 |  | 143 |  |
| 9414750 | Alameda | yes | yes | 1315255 | 0.009 | exact | 0.009 |  | 145 |  |
| 9414863 | Richmond | yes | yes | 1449449 | 0.044 | exact | 0.044 |  | 149 |  |
| 9415020 | Point Reyes | yes | yes | 2190380 | 0.055 | exact | 0.055 |  | 151 |  |
| 9415102 | Martinez-Amorco Pier | yes | yes | 570596 | 0.072 | exact | 0.072 |  | 153 |  |
| 9415144 | Port Chicago | yes | yes | 483149 | 0.046 | exact | 0.046 |  | 155 |  |
| 9416131 | Port of West Sacramento, Washington Lake | yes | no | 6863 | 0.057 | exact | 0.057 |  |  | stations run: "No matching model station found" (not in points file station list) |
| 9416841 | Arena Cove | yes | yes | 2398488 | 0.042 | exact | 0.042 |  | 158 |  |
| 9418767 | North Spit | yes | yes | 1956527 | 0.037 | exact | 0.037 |  | 159 |  |
| 9419750 | Crescent City | yes | yes | 1944978 | 0.024 | exact | 0.024 |  | 160 |  |
| 9431647 | Port Orford | yes | yes | 2030661 | 0.028 | exact | 0.028 |  | 162 |  |
| 9432780 | Charleston | yes | yes | 1717637 | 0.049 | exact | 0.049 |  | 163 |  |
| 9435380 | South Beach | yes | yes | 1477928 | 0.079 | ctl | 0.019 | yes | 164 |  |
| 9437540 | Garibaldi | yes | yes | 1509484 | 0.045 | exact | 0.045 |  | 165 |  |
| 9439040 | Astoria | yes | yes | 1127067 | 0.035 | exact | 0.035 |  | 167 |  |
| 9439099 | Wauna | yes | yes | 406737 | 0.044 | exact | 0.044 |  | 743 |  |
| 9439201 | St Helens | yes | yes | 31625 | 0.028 | exact | 0.028 |  | 733 |  |
| 9440083 | Vancouver | yes | yes | 7837 | 0.053 | exact | 0.053 |  | 732 |  |
| 9440357 | TEMCO Kalama Terminal | yes | yes | 65258 | 0.084 | exact | 0.084 |  | 830 |  |
| 9440422 | Longview | yes | yes | 117904 | 0.063 | exact | 0.063 |  | 738 |  |
| 9440569 | Skamokawa | yes | yes | 560329 | 0.104 | exact | 0.104 |  | 168 |  |
| 9440581 | Cape Disappointment | yes | yes | 1530071 | 0.050 | exact | 0.050 |  | 746 |  |
| 9440910 | Toke Point | yes | yes | 1457458 | 0.023 | exact | 0.023 |  | 169 |  |
| 9441102 | Westport | yes | yes | 1457451 | 0.009 | exact | 0.009 |  | 170 |  |
| 9442396 | La Push, Quillayute River | yes | yes | 1680343 | 0.034 | exact | 0.034 |  | 171 |  |
| 9443090 | Neah Bay | yes | yes | 1375277 | 0.057 | exact | 0.057 |  | 172 |  |
| 9444090 | Port Angeles | yes | yes | 763962 | 0.073 | exact | 0.073 |  | 173 |  |
| 9444900 | Port Townsend | yes | yes | 567418 | 0.106 | exact | 0.106 |  | 174 |  |
| 9445958 | Bremerton | yes | yes | 163828 | 0.122 | exact | 0.122 |  | 828 |  |
| 9446484 | Tacoma | yes | yes | 137207 | 0.535 | exact | 0.535 |  | 175 |  |
| 9447130 | Seattle | yes | yes | 273791 | 0.169 | exact | 0.169 |  | 176 |  |
| 9449424 | Cherry Point | yes | yes | 355228 | 0.037 | exact | 0.037 |  | 178 |  |
| 9449880 | Friday Harbor | yes | yes | 462265 | 0.043 | exact | 0.043 |  | 179 |  |
| 9450460 | Ketchikan | yes | yes | 703846 | 1.714 | exact | 1.714 |  | 180 |  |
| 9451054 | Port Alexander | yes | yes | 940709 | 0.203 | exact | 0.203 |  | 181 |  |
| 9451600 | Sitka | yes | yes | 896895 | 0.269 | exact | 0.269 |  | 182 |  |
| 9452210 | Juneau | yes | yes | 435831 | 0.638 | exact | 0.638 |  | 183 |  |
| 9452400 | Skagway, Taiya Inlet | yes | yes | 381632 | 0.239 | exact | 0.239 |  | 184 |  |
| 9452634 | Elfin Cove | yes | yes | 835083 | 0.746 | exact | 0.746 |  | 185 |  |
| 9453220 | Yakutat, Yakutat Bay | yes | yes | 843707 | 1.412 | exact | 1.412 |  | 186 |  |
| 9454050 | Cordova | yes | yes | 706462 | 0.480 | exact | 0.480 |  | 187 |  |
| 9454240 | Valdez | yes | yes | 533836 | 0.565 | exact | 0.565 |  | 188 |  |
| 9455090 | Seward | yes | yes | 490465 | 0.114 | exact | 0.114 |  | 189 |  |
| 9455500 | Seldovia | yes | yes | 472298 | 0.206 | exact | 0.206 |  | 190 |  |
| 9455760 | Nikiski | yes | yes | 343098 | 0.050 | exact | 0.050 |  | 191 |  |
| 9455920 | Anchorage | yes | yes | 169759 | 0.209 | exact | 0.209 |  | 192 |  |
| 9457292 | Kodiak Island | yes | yes | 584477 | 0.225 | exact | 0.225 |  | 193 |  |
| 9457804 | Alitak | yes | yes | 730823 | 0.045 | exact | 0.045 |  | 194 |  |
| 9459450 | Sand Point | yes | yes | 924305 | 0.068 | exact | 0.068 |  | 195 |  |
| 9459881 | King Cove | yes | yes | 909350 | 0.335 | exact | 0.335 |  | 196 |  |
| 9461380 | Adak Island | yes | yes | 3238200 | 0.112 | exact | 0.112 |  | 197 |  |
| 9461710 | Atka | yes | yes | 3080315 | 0.084 | exact | 0.084 |  | 198 |  |
| 9462450 | Nikolski | yes | yes | 3011359 | 0.621 | exact | 0.621 |  | 199 |  |
| 9462620 | Unalaska | yes | yes | 2843281 | 0.288 | exact | 0.288 |  | 200 |  |
| 9463502 | Port Moller | yes | yes | 1875324 | 0.129 | exact | 0.129 |  | 774 |  |
| 9464212 | Village Cove, St Paul Island | yes | yes | 3502729 | 0.658 | exact | 0.658 |  | 808 |  |
| 9468333 | Unalakleet | yes | yes | 3739659 | 0.522 | exact | 0.522 |  | 775 |  |
| 9468756 | Nome, Norton Sound | yes | yes | 4306014 | 0.125 | exact | 0.125 |  | 795 |  |
| 9491094 | Red Dog Dock | yes | yes | 4727311 | 0.082 | exact | 0.082 |  | 791 |  |
| 9497645 | Prudhoe Bay | yes | yes | 5778216 | 0.343 | exact | 0.343 |  | 761 |  |
| 9751364 | Christiansted Harbor, St Croix | yes | yes | 10974058 | 0.109 | exact | 0.018 | yes | 113 |  |
| 9751381 | Lameshur Bay, St John | yes | yes | 10682958 | 0.051 | exact | 0.051 |  | 114 |  |
| 9751401 | Limetree Bay | yes | yes | 10590193 | 0.018 | exact | 0.018 |  | 115 |  |
| 9751639 | Charlotte Amalie | yes | yes | 10682159 | 0.018 | exact | 0.018 |  | 116 |  |
| 9752235 | Culebra | yes | no | 11045628 | 0.065 | exact | 0.033 | yes |  | stations run: "No matching model station found" (not in points file station list) |
| 9752695 | Esperanza, Vieques Island | yes | yes | 10336446 | 0.058 | exact | 0.054 | yes | 118 |  |
| 9755371 | San Juan, La Puntilla, San Juan Bay | yes | yes | 11552885 | 0.042 | exact | 0.042 |  | 121 |  |
| 9759110 | Magueyes Island | yes | yes | 10480042 | 0.038 | exact | 0.038 |  | 124 |  |
| 9759394 | Mayaguez | yes | yes | 10994220 | 0.071 | exact | 0.026 | yes | 125 |  |
| 9759938 | Mona Island | yes | yes | 11142113 | 0.170 | exact | 0.042 | yes | 126 |  |

## Appendix F. Fields vs stations series, and skill tables, forecast_a, cycle 2026-10-14 00z

| ID | n | series RMSE (m) | series bias (m) | series corr | lag (h) | skill RMSE fields / stations | skill r fields / stations | skill bias fields / stations | central freq fields / stations | FLAG |
|---|---|---|---|---|---|---|---|---|---|---|
| 1611400 | 180 | 0.005 | -0.000 | 0.9995 | 0 | 0.04 / 0.05 | 0.97 / 0.97 | 0.03 / 0.03 | 100 / 100 |  |
| 1612340 | 180 | 0.006 | -0.006 | 1.0000 | 0 | 0.03 / 0.03 | 0.98 / 0.98 | 0.01 / 0.01 | 100 / 100 |  |
| 1612480 | 180 | 0.004 | -0.004 | 1.0000 | 0 | 0.04 / 0.04 | 0.98 / 0.98 | 0.02 / 0.02 | 100 / 100 |  |
| 1615680 | 180 | 0.015 | +0.015 | 1.0000 | 0 | 0.04 / 0.03 | 0.98 / 0.98 | 0.02 / 0.01 | 100 / 100 |  |
| 1617433 | 180 | 0.006 | +0.006 | 1.0000 | 0 | 0.04 / 0.04 | 0.97 / 0.97 | -0 / -0.01 | 100 / 100 |  |
| 1617760 | 180 | 0.001 | +0.000 | 1.0000 | 0 | 0.05 / 0.05 | 0.97 / 0.97 | -0.01 / -0.01 | 100 / 100 |  |
| 1619910 | 180 | 0.039 | -0.039 | 0.9999 | 0 | 0.02 / 0.04 | 0.98 / 0.98 | -0 / 0.04 | 100 / 100 |  |
| 1630000 | 180 | 0.043 | +0.043 | 1.0000 | 0 | 0.08 / 0.05 | 0.97 / 0.97 | 0.06 / 0.02 | 96.67 / 99.83 |  |
| 1631428 | 180 | 0.041 | +0.041 | 1.0000 | 0 | 0.23 / 0.25 | 0.58 / 0.58 | -0.08 / -0.12 | 76.67 / 67.94 |  |
| 1770000 | 180 | 0.001 | -0.001 | 1.0000 | 0 | 0.04 / 0.04 | 0.98 / 0.98 | 0.01 / 0.01 | 100 / 100 |  |
| 1820000 | 180 | 0.190 | +0.190 | 0.9994 | 0 | 0.22 / 0.04 | 0.99 / 0.99 | 0.21 / 0.02 | 2.78 / 100 | FLAG |
| 1890000 | 180 | 0.023 | -0.023 | 0.9999 | 0 | 0.07 / 0.05 | 0.98 / 0.98 | -0.06 / -0.04 | 100 / 100 |  |
| 8410140 | 180 | 0.012 | -0.012 | 1.0000 | 0 | 0.22 / 0.22 | 0.99 / 0.99 | -0.04 / -0.03 | 40 / 41.33 |  |
| 8411060 | 180 | 0.006 | -0.005 | 1.0000 | 0 | 0.15 / 0.14 | 0.99 / 1 | -0.06 / -0.05 | 63.89 / 67.39 |  |
| 8413320 | 180 | 0.017 | -0.017 | 1.0000 | 0 | 0.13 / 0.12 | 0.99 / 0.99 | -0.07 / -0.05 | 73.89 / 78 |  |
| 8418150 | 180 | 0.047 | -0.047 | 1.0000 | 0 | 0.16 / 0.13 | 0.99 / 0.99 | -0.11 / -0.06 | 59.44 / 80.67 |  |
| 8419870 | 180 | 0.003 | +0.002 | 1.0000 | 0 | 0.14 / 0.14 | 0.99 / 0.99 | -0.07 / -0.07 | 70.56 / 71.62 |  |
| 8443970 | 180 | 0.004 | -0.004 | 1.0000 | 0 | 0.15 / 0.15 | 0.99 / 0.99 | -0.09 / -0.08 | 68.33 / 68.89 |  |
| 8447386 | 180 | 0.002 | -0.002 | 1.0000 | 0 | 0.18 / 0.17 | 0.88 / 0.88 | -0.09 / -0.09 | 56.11 / 56.97 |  |
| 8447435 | 180 | 0.000 | -0.000 | 1.0000 | 0 | 0.17 / 0.17 | 0.96 / 0.96 | -0.12 / -0.11 | 68.89 / 67 |  |
| 8447930 | 180 | 0.044 | -0.044 | 1.0000 | 0 | 0.17 / 0.14 | 0.77 / 0.77 | -0.14 / -0.09 | 55.56 / 69.5 |  |
| 8449130 | 180 | 0.022 | -0.022 | 1.0000 | 0 | 0.18 / 0.16 | 0.91 / 0.91 | -0.13 / -0.11 | 65 / 66.61 |  |
| 8452660 | 180 | 0.000 | -0.000 | 1.0000 | 0 | 0.16 / 0.16 | 0.89 / 0.89 | -0.1 / -0.1 | 63.33 / 63.22 |  |
| 8452944 | 180 | 0.012 | -0.012 | 1.0000 | 0 | 0.18 / 0.17 | 0.89 / 0.89 | -0.11 / -0.1 | 56.11 / 58.17 |  |
| 8454000 | 180 | 0.001 | -0.000 | 1.0000 | 0 | 0.17 / 0.17 | 0.89 / 0.89 | -0.09 / -0.09 | 58.89 / 57.28 |  |
| 8461490 | 180 | 0.004 | -0.004 | 1.0000 | 0 | 0.18 / 0.18 | 0.86 / 0.86 | -0.13 / -0.13 | 60 / 61.83 |  |
| 8465705 | 180 | 0.044 | -0.044 | 1.0000 | 0 | 0.25 / 0.22 | 0.94 / 0.94 | -0.17 / -0.13 | 33.89 / 39.39 |  |
| 8467150 | 180 | 0.029 | -0.029 | 1.0000 | 0 | 0.24 / 0.23 | 0.95 / 0.95 | -0.16 / -0.13 | 35.56 / 38.33 |  |
| 8510560 | 180 | 0.000 | -0.000 | 1.0000 | 0 | 0.18 / 0.18 | 0.75 / 0.75 | -0.13 / -0.13 | 59.44 / 59.72 |  |
| 8516945 | 180 | 0.000 | -0.000 | 1.0000 | 0 | 0.26 / 0.26 | 0.95 / 0.95 | -0.15 / -0.15 | 38.33 / 38.28 |  |
| 8518750 | 180 | 0.030 | -0.029 | 0.9999 | 0 | 0.28 / 0.26 | 0.87 / 0.88 | -0.21 / -0.18 | 37.22 / 39.22 |  |
| 8531680 | 180 | 0.051 | -0.051 | 1.0000 | 0 | 0.31 / 0.26 | 0.84 / 0.86 | -0.23 / -0.16 | 40 / 47.39 | FLAG |
| 8534720 | 180 | 0.080 | -0.080 | 1.0000 | 0 | 0.37 / 0.3 | 0.8 / 0.8 | -0.31 / -0.23 | 26.67 / 41.39 | FLAG |
| 8536110 | 180 | 0.000 | +0.000 | 1.0000 | 0 | 0.31 / 0.31 | 0.85 / 0.86 | -0.23 / -0.23 | 38.89 / 39.89 |  |
| 8537121 | 180 | 0.022 | -0.022 | 1.0000 | 0 | 0.31 / 0.29 | 0.91 / 0.91 | -0.23 / -0.21 | 34.44 / 35.44 |  |
| 8539094 | 180 | 0.006 | -0.002 | 1.0000 | 0 | 0.33 / 0.34 | 0.92 / 0.92 | -0.18 / -0.17 | 30.56 / 29.17 |  |
| 8540433 | 180 | 0.004 | -0.003 | 1.0000 | 0 | 0.29 / 0.29 | 0.91 / 0.91 | -0.19 / -0.18 | 33.89 / 34.13 |  |
| 8545240 | 180 | 0.009 | +0.003 | 0.9998 | 0 | 0.3 / 0.3 | 0.91 / 0.91 | -0.17 / -0.17 | 31.67 / 32.11 |  |
| 8546252 | 180 | 0.013 | -0.011 | 0.9999 | 0 | 0.3 / 0.3 | 0.92 / 0.92 | -0.18 / -0.17 | 31.67 / 32.61 |  |
| 8548989 | 180 | 0.001 | +0.001 | 1.0000 | 0 | 0.33 / 0.33 | 0.93 / 0.93 | -0.16 / -0.17 | 31.11 / 32.78 |  |
| 8551762 | 180 | 0.041 | -0.039 | 0.9998 | 0 | 0.3 / 0.27 | 0.91 / 0.91 | -0.22 / -0.18 | 33.89 / 36.44 |  |
| 8555889 | 180 | 0.028 | -0.028 | 1.0000 | 0 | 0.33 / 0.31 | 0.86 / 0.87 | -0.25 / -0.22 | 31.67 / 40.5 |  |
| 8557380 | 180 | 0.000 | +0.000 | 1.0000 | 0 | 0.33 / 0.33 | 0.8 / 0.8 | -0.25 / -0.24 | 38.33 / 39.28 |  |
| 8570283 | 180 | 0.021 | -0.017 | 0.9992 | 0 | 0.3 / 0.28 | 0.51 / 0.51 | -0.22 / -0.21 | 46.11 / 49.5 |  |
| 8571421 | 166 | 0.042 | -0.042 | 0.9999 | 0 | 0.25 / 0.2 | 0.79 / 0.72 | -0.2 / -0.14 | 43.89 / 64.36 |  |
| 8571892 | 180 | 0.022 | -0.022 | 1.0000 | 0 | 0.2 / 0.18 | 0.82 / 0.82 | -0.14 / -0.12 | 56.11 / 64.39 |  |
| 8573364 | 176 | 0.002 | +0.000 | 1.0000 | 0 | 0.16 / 0.16 | 0.89 / 0.87 | -0.11 / -0.1 | 65 / 65.38 |  |
| 8573927 | 180 | 0.015 | +0.005 | 0.9988 | 0 | 0.17 / 0.18 | 0.93 / 0.93 | -0.12 / -0.13 | 60 / 57.17 |  |
| 8574680 | 180 | 0.001 | -0.001 | 1.0000 | 0 | 0.17 / 0.17 | 0.89 / 0.89 | -0.12 / -0.11 | 62.78 / 63.5 |  |
| 8575512 | 180 | 0.013 | -0.013 | 1.0000 | 0 | 0.18 / 0.17 | 0.85 / 0.86 | -0.13 / -0.12 | 61.11 / 66.06 |  |
| 8577330 | 180 | 0.018 | -0.018 | 1.0000 | 0 | 0.22 / 0.2 | 0.71 / 0.72 | -0.17 / -0.15 | 50.56 / 58.83 |  |
| 8594900 | 180 | 0.000 | +0.000 | 1.0000 | 0 | 0.2 / 0.2 | 0.91 / 0.91 | -0.16 / -0.16 | 41.67 / 42.06 |  |
| 8632200 | 180 | 0.025 | -0.025 | 1.0000 | 0 | 0.36 / 0.34 | 0.61 / 0.6 | -0.29 / -0.26 | 39.44 / 42.72 |  |
| 8635027 | 180 | 0.026 | -0.026 | 1.0000 | 0 | 0.24 / 0.22 | 0.75 / 0.76 | -0.2 / -0.17 | 52.22 / 55.39 |  |
| 8635750 | 180 | 0.031 | -0.031 | 1.0000 | 0 | 0.25 / 0.23 | 0.57 / 0.56 | -0.21 / -0.18 | 48.89 / 55.22 |  |
| 8636580 | 180 | 0.039 | -0.039 | 0.9993 | 0 | 0.29 / 0.26 | 0.27 / 0.27 | -0.24 / -0.2 | 47.22 / 52.67 |  |
| 8637689 | 180 | 0.310 | -0.310 | 1.0000 | 0 | 0.59 / 0.31 | 0.57 / 0.57 | -0.56 / -0.25 | 0 / 44 | FLAG |
| 8638610 | 180 | 0.046 | -0.046 | 1.0000 | 0 | 0.41 / 0.37 | 0.55 / 0.55 | -0.34 / -0.3 | 31.67 / 41.56 |  |
| 8638901 | 180 | 0.044 | -0.044 | 1.0000 | 0 | 0.37 / 0.34 | 0.63 / 0.63 | -0.31 / -0.26 | 37.22 / 43.39 |  |
| 8639348 | 180 | 0.003 | -0.003 | 1.0000 | 0 | 0.39 / 0.38 | 0.59 / 0.59 | -0.3 / -0.3 | 38.89 / 38.5 |  |
| 8651370 | 180 | 0.035 | -0.035 | 1.0000 | 0 | 0.42 / 0.39 | 0.65 / 0.65 | -0.33 / -0.3 | 35.56 / 41.61 |  |
| 8652587 | 175 | 0.006 | -0.006 | 1.0000 | 0 | 0.14 / 0.13 | 0.67 / 0.67 | -0.08 / -0.07 | 76.11 / 80.54 |  |
| 8654467 | 180 | 0.061 | -0.061 | 0.9985 | 0 | 0.33 / 0.29 | -0.46 / -0.47 | -0.26 / -0.2 | 46.93 / 46.85 | FLAG |
| 8656483 | 180 | 0.036 | -0.035 | 0.9999 | 0 | 0.21 / 0.18 | 0.84 / 0.84 | -0.14 / -0.1 | 52.22 / 60.5 |  |
| 8658120 | 180 | 0.006 | -0.004 | 0.9999 | 0 | 0.17 / 0.17 | 0.92 / 0.92 | -0.08 / -0.07 | 57.22 / 58.17 |  |
| 8658163 | 180 | 0.046 | -0.046 | 1.0000 | 0 | 0.24 / 0.21 | 0.9 / 0.9 | -0.19 / -0.14 | 42.22 / 47.11 |  |
| 8661070 | 180 | 0.000 | +0.000 | 1.0000 | 0 | 0.16 / 0.16 | 0.93 / 0.93 | -0.06 / -0.06 | 63.89 / 64.56 |  |
| 8665530 | 180 | 0.022 | -0.021 | 1.0000 | 0 | 0.16 / 0.15 | 0.94 / 0.94 | -0.06 / -0.04 | 61.67 / 63.06 |  |
| 8670870 | 180 | 0.015 | -0.015 | 1.0000 | 0 | 0.22 / 0.22 | 0.92 / 0.92 | -0.05 / -0.03 | 43.33 / 41.33 |  |
| 8679598 | 180 | 0.020 | +0.019 | 1.0000 | 0 | 0.44 / 0.45 | 0.91 / 0.91 | -0.38 / -0.4 | 18.33 / 14.56 |  |
| 8720030 | 180 | 0.024 | -0.023 | 0.9999 | 0 | 0.21 / 0.21 | 0.91 / 0.91 | -0.05 / -0.03 | 43.89 / 44.17 |  |
| 8720218 | 180 | 0.017 | -0.014 | 0.9998 | 0 | 0.12 / 0.13 | 0.95 / 0.94 | -0.02 / -0 | 77.78 / 72.61 |  |
| 8720219 | 180 | 0.007 | -0.005 | 0.9999 | 0 |  / 0.13 |  / 0.93 |  / 0.03 |  / 69.56 |  |
| 8720226 | 180 | 0.004 | -0.003 | 1.0000 | 0 | 0.1 / 0.1 | 0.93 / 0.93 | 0.07 / 0.07 | 85 / 85.33 |  |
| 8720357 | 180 | 0.007 | -0.007 | 1.0000 | 0 | 0.08 / 0.09 | 0.92 / 0.92 | 0.06 / 0.06 | 89.44 / 87.78 |  |
| 8721604 | 180 | 0.014 | -0.014 | 1.0000 | 0 | 0.1 / 0.1 | 0.95 / 0.95 | -0.02 / -0.01 | 87.22 / 88.67 |  |
| 8722670 | 180 | 0.026 | -0.026 | 1.0000 | 0 |  / 0.12 |  / 0.9 |  / -0.05 |  / 78.11 |  |
| 8722956 | 180 | 0.026 | -0.026 | 1.0000 | 0 | 0.12 / 0.11 | 0.91 / 0.91 | -0.07 / -0.04 | 76.67 / 79 |  |
| 8723214 | 180 | 0.012 | -0.012 | 0.9999 | 0 | 0.08 / 0.08 | 0.91 / 0.91 | -0.03 / -0.02 | 88.33 / 91.28 |  |
| 8723970 | 180 | 0.004 | +0.000 | 0.9994 | 0 | 0.07 / 0.07 | 0.91 / 0.91 | -0.05 / -0.04 | 98.33 / 98.61 |  |
| 8724580 | 180 | 0.023 | -0.023 | 0.9999 | 0 | 0.07 / 0.06 | 0.95 / 0.95 | -0.06 / -0.04 | 97.78 / 99.89 |  |
| 8725114 | 180 | 0.042 | -0.036 | 0.9949 | 0 | 0.11 / 0.13 | 0.87 / 0.84 | 0.01 / 0.04 | 85 / 74 |  |
| 8725520 | 180 | 0.000 | +0.000 | 1.0000 | 0 | 0.13 / 0.13 | 0.78 / 0.78 | 0.06 / 0.06 | 68.89 / 68.83 |  |
| 8726384 | 180 | 0.013 | -0.013 | 1.0000 | 0 | 0.08 / 0.07 | 0.94 / 0.94 | -0.01 / 0 | 93.89 / 96.44 |  |
| 8726520 | 180 | 0.003 | -0.003 | 1.0000 | 0 | 0.09 / 0.09 | 0.93 / 0.93 | 0.01 / 0.01 | 90.56 / 90.78 |  |
| 8726607 | 180 | 0.001 | +0.000 | 1.0000 | 0 | 0.09 / 0.09 | 0.93 / 0.93 | 0.01 / 0.01 | 87.22 / 86.94 |  |
| 8726724 | 180 | 0.000 | +0.000 | 1.0000 | 0 | 0.08 / 0.08 | 0.94 / 0.94 | 0.01 / 0.01 | 96.11 / 95.78 |  |
| 8727520 | 180 | 0.004 | -0.004 | 1.0000 | 0 | 0.14 / 0.14 | 0.91 / 0.91 | 0.03 / 0.03 | 81.67 / 82.39 |  |
| 8728690 | 180 | 0.001 | -0.000 | 1.0000 | 0 | 0.11 / 0.11 | 0.87 / 0.87 | 0.04 / 0.04 | 80 / 78.44 |  |
| 8729108 | 180 | 0.002 | -0.001 | 0.9999 | 0 | 0.09 / 0.09 | 0.89 / 0.89 | 0.03 / 0.03 | 88.89 / 87.56 |  |
| 8729210 | 180 | 0.000 | +0.000 | 1.0000 | 0 | 0.09 / 0.09 | 0.9 / 0.9 | 0.04 / 0.04 | 88.89 / 89.11 |  |
| 8729840 | 180 | 0.000 | +0.000 | 1.0000 | 0 | 0.08 / 0.08 | 0.9 / 0.9 | 0.04 / 0.04 | 92.78 / 92.44 |  |
| 8735180 | 180 | 0.017 | -0.016 | 0.9999 | 0 | 0.09 / 0.1 | 0.88 / 0.88 | 0.04 / 0.06 | 92.78 / 85.78 |  |
| 8735391 | 180 | 0.001 | -0.000 | 1.0000 | 0 | 0.14 / 0.14 | 0.85 / 0.85 | 0.07 / 0.07 | 70 / 70.28 |  |
| 8736897 | 180 | 0.000 | +0.000 | 1.0000 | 0 | 0.15 / 0.15 | 0.85 / 0.85 | 0.07 / 0.07 | 68.33 / 68.28 |  |
| 8737048 | 180 | 0.000 | -0.000 | 1.0000 | 0 | 0.14 / 0.14 | 0.85 / 0.85 | 0.07 / 0.07 | 68.33 / 68.94 |  |
| 8737138 | 180 | 0.000 | +0.000 | 1.0000 | 0 | 0.15 / 0.14 | 0.87 / 0.86 | 0.07 / 0.07 | 73.33 / 73.56 |  |
| 8741533 | 180 | 0.001 | +0.000 | 1.0000 | 0 | 0.1 / 0.1 | 0.91 / 0.91 | 0.05 / 0.05 | 88.89 / 87.89 |  |
| 8747437 | 180 | 0.001 | +0.000 | 1.0000 | 0 | 0.1 / 0.1 | 0.87 / 0.87 | 0.02 / 0.02 | 88.33 / 89.44 |  |
| 8760721 | 180 | 0.014 | +0.014 | 1.0000 | 0 | 0.1 / 0.09 | 0.88 / 0.88 | 0.07 / 0.05 | 86.11 / 91.28 |  |
| 8760922 | 180 | 0.042 | +0.042 | 0.9998 | 0 | 0.46 / 0.5 | 0.94 / 0.94 | -0.45 / -0.5 | 0 / 0 |  |
| 8761305 | 180 | 0.004 | -0.004 | 1.0000 | 0 | 0.13 / 0.13 | 0.76 / 0.76 | -0.03 / -0.03 | 70 / 70.72 |  |
| 8761724 | 180 | 0.017 | -0.015 | 0.9994 | 0 | 0.09 / 0.11 | 0.93 / 0.93 | 0.08 / 0.09 | 86.11 / 81.5 |  |
| 8761927 | 180 | 0.004 | +0.003 | 0.9996 | 0 | 0.07 / 0.07 | 0.76 / 0.75 | -0.02 / -0.02 | 98.33 / 98.17 |  |
| 8762075 | 180 | 0.002 | -0.002 | 1.0000 | 0 |  / 0.13 |  / 0.9 |  / 0.11 |  / 70.67 |  |
| 8764044 | 180 | 0.004 | +0.004 | 1.0000 | 0 |  / 0.12 |  / 0.89 |  / 0.09 |  / 73.5 |  |
| 8764227 | 180 | 0.007 | -0.000 | 0.9996 | 0 |  / 0.18 |  / 0.88 |  / 0.14 |  / 54 |  |
| 8764314 | 180 | 0.034 | -0.034 | 1.0000 | 0 | 0.16 / 0.19 | 0.87 / 0.87 | 0.11 / 0.15 | 74.44 / 55.89 |  |
| 8766072 | 180 | 0.007 | -0.005 | 0.9999 | 0 |  / 0.18 |  / 0.87 |  / 0.13 |  / 61 |  |
| 8767816 | 180 | 0.000 | +0.000 | 1.0000 | 0 |  / 0.16 |  / 0.9 |  / 0.11 |  / 66.5 |  |
| 8767961 | 180 | 0.000 | +0.000 | 1.0000 | 0 | 0.16 / 0.16 | 0.89 / 0.89 | 0.12 / 0.12 | 62.78 / 63.22 |  |
| 8768094 | 180 | 0.010 | -0.009 | 1.0000 | 0 | 0.14 / 0.15 | 0.89 / 0.89 | 0.1 / 0.11 | 67.78 / 63.33 |  |
| 8770475 | 180 | 0.005 | -0.005 | 1.0000 | 0 | 0.12 / 0.12 | 0.87 / 0.87 | 0.08 / 0.09 | 76.11 / 73.44 |  |
| 8770520 | 180 | 0.007 | -0.007 | 1.0000 | 0 | 0.12 / 0.13 | 0.91 / 0.9 | 0.09 / 0.1 | 77.78 / 75 |  |
| 8770613 | 180 | 0.005 | +0.005 | 1.0000 | 0 | 0.16 / 0.15 | 0.88 / 0.88 | 0.1 / 0.09 | 71.11 / 73.78 |  |
| 8770777 | 180 | 0.000 | -0.000 | 1.0000 | 0 | 0.17 / 0.17 | 0.88 / 0.87 | 0.11 / 0.11 | 71.11 / 69.61 |  |
| 8770808 | 180 | 0.001 | -0.001 | 1.0000 | 0 | 0.14 / 0.14 | 0.84 / 0.84 | 0.09 / 0.1 | 68.89 / 68.56 |  |
| 8770822 | 180 | 0.048 | +0.047 | 0.9996 | 0 | 1.03 / 1.08 | 0.89 / 0.89 | -1.03 / -1.08 | 0 / 0 |  |
| 8770971 | 180 | 0.017 | -0.017 | 0.9998 | 0 | 0.13 / 0.14 | 0.82 / 0.81 | 0.06 / 0.08 | 73.89 / 69.44 |  |
| 8771013 | 180 | 0.004 | -0.004 | 1.0000 | 0 | 0.12 / 0.12 | 0.87 / 0.87 | 0.07 / 0.08 | 79.44 / 77.17 |  |
| 8771341 | 180 | 0.022 | -0.022 | 1.0000 | 0 | 0.11 / 0.12 | 0.9 / 0.9 | 0.06 / 0.08 | 78.89 / 75.22 |  |
| 8771450 | 180 | 0.001 | -0.001 | 1.0000 | 0 | 0.13 / 0.13 | 0.88 / 0.88 | 0.08 / 0.08 | 76.67 / 76.72 |  |
| 8771486 | 180 | 0.003 | -0.003 | 0.9999 | 0 | 1.1 / 0.5 | 0.07 / 0.27 | -0.16 / 0.03 | 73.08 / 76.57 |  |
| 8771972 | 180 | 0.119 | +0.006 | 0.7825 | 3 | 0.12 / 0.1 | 0.74 / 0.89 | 0.07 / 0.06 | 78.86 / 83.84 | FLAG |
| 8773037 | 180 | 0.011 | +0.011 | 1.0000 | 0 | 0.13 / 0.14 | 0.94 / 0.94 | -0.13 / -0.14 | 71.11 / 53.56 |  |
| 8773259 | 180 | 0.007 | +0.006 | 0.9997 | 0 | 0.13 / 0.13 | 0.86 / 0.87 | 0.09 / 0.09 | 72.78 / 73.89 |  |
| 8773701 | 180 | 0.013 | -0.013 | 0.9999 | 0 | 0.09 / 0.1 | 0.88 / 0.88 | 0.06 / 0.08 | 85 / 82.67 |  |
| 8773767 | 180 | 0.011 | -0.010 | 0.9998 | 0 | 0.08 / 0.09 | 0.92 / 0.92 | 0.05 / 0.06 | 86.67 / 84.78 |  |
| 8774513 | 180 | 0.002 | -0.001 | 0.9999 | 0 | 0.07 / 0.07 | 0.93 / 0.93 | 0.05 / 0.05 | 100 / 100 |  |
| 8774770 | 180 | 0.000 | +0.000 | 1.0000 | 0 | 0.06 / 0.06 | 0.91 / 0.91 | 0.05 / 0.05 | 100 / 100 |  |
| 8775237 | 180 | 0.010 | -0.010 | 1.0000 | 0 | 0.09 / 0.1 | 0.92 / 0.91 | 0.06 / 0.08 | 90.56 / 89.33 |  |
| 8775241 | 180 | 0.118 | -0.115 | 0.9963 | 0 | 0.1 / 0.09 | 0.94 / 0.94 | -0.06 / 0.06 | 88.33 / 91.11 | FLAG |
| 8775283 | 180 | 0.004 | -0.004 | 1.0000 | 0 | 0.07 / 0.07 | 0.8 / 0.8 | 0.04 / 0.04 | 91.67 / 92.11 |  |
| 8775296 | 180 | 0.005 | -0.005 | 1.0000 | 0 | 0.08 / 0.08 | 0.77 / 0.76 | 0.05 / 0.05 | 91.11 / 91.17 |  |
| 8775792 | 180 | 0.012 | -0.011 | 0.9973 | 0 | 0.08 / 0.09 | 0.61 / 0.6 | 0.06 / 0.07 | 91.67 / 91.83 |  |
| 8779280 | 180 | 0.015 | -0.015 | 0.9999 | 0 | 0.12 / 0.13 | 0.74 / 0.74 | 0.09 / 0.11 | 66.67 / 64.04 |  |
| 8779748 | 174 | 0.022 | -0.022 | 0.9999 | 0 | 0.07 / 0.08 | 0.95 / 0.94 | 0.03 / 0.06 | 94.44 / 93.23 |  |
| 8779749 | 180 | 0.045 | -0.045 | 1.0000 | 0 | 0.06 / 0.08 | 0.95 / 0.95 | 0.01 / 0.06 | 96.11 / 92.44 |  |
| 8779770 | 180 | 0.002 | -0.002 | 1.0000 | 0 | 0.09 / 0.09 | 0.94 / 0.94 | 0.07 / 0.07 | 90 / 90.78 |  |
| 9410170 | 180 | 0.250 | -0.250 | 0.9998 | 0 | 0.27 / 0.06 | 0.99 / 0.99 | -0.26 / -0.01 | 0.56 / 97.28 | FLAG |
| 9410230 | 180 | 0.032 | -0.032 | 1.0000 | 0 | 0.08 / 0.06 | 0.99 / 0.99 | -0.05 / -0.02 | 93.89 / 96.11 |  |
| 9410660 | 180 | 0.023 | -0.023 | 1.0000 | 0 | 0.06 / 0.06 | 0.99 / 0.99 | -0.03 / -0.01 | 95.56 / 97.56 |  |
| 9410840 | 180 | 0.036 | -0.036 | 1.0000 | 0 | 0.06 / 0.05 | 0.99 / 0.99 | -0.04 / -0 | 96.67 / 98.67 |  |
| 9411340 | 180 | 0.048 | -0.048 | 1.0000 | 0 | 0.07 / 0.05 | 0.99 / 0.99 | -0.04 / 0.01 | 95.56 / 98.83 |  |
| 9412110 | 180 | 0.241 | -0.241 | 0.9997 | 0 | 0.27 / 0.06 | 0.99 / 0.99 | -0.26 / -0.02 | 0.56 / 97.89 | FLAG |
| 9413450 | 180 | 0.057 | -0.057 | 1.0000 | 0 | 0.1 / 0.06 | 0.99 / 0.99 | -0.09 / -0.03 | 88.89 / 98.94 | FLAG |
| 9414290 | 180 | 0.020 | -0.020 | 1.0000 | 0 | 0.07 / 0.06 | 0.99 / 0.99 | -0.05 / -0.03 | 97.22 / 99.06 |  |
| 9414523 | 180 | 0.000 | -0.000 | 1.0000 | 0 | 0.25 / 0.25 | 0.99 / 0.99 | -0.24 / -0.24 | 11.67 / 11.44 |  |
| 9414750 | 180 | 0.231 | -0.231 | 1.0000 | 0 | 0.27 / 0.07 | 0.99 / 0.99 | -0.27 / -0.04 | 0.56 / 96.83 | FLAG |
| 9414863 | 180 | 0.077 | -0.077 | 1.0000 | 0 | 0.12 / 0.07 | 0.99 / 0.99 | -0.11 / -0.03 | 79.44 / 96 | FLAG |
| 9415020 | 180 | 0.043 | -0.043 | 1.0000 | 0 | 0.09 / 0.06 | 1 / 0.99 | -0.08 / -0.03 | 93.33 / 99.55 |  |
| 9415102 | 180 | 0.009 | -0.009 | 1.0000 | 0 | 0.09 / 0.08 | 0.99 / 0.99 | -0.05 / -0.05 | 91.11 / 93.89 |  |
| 9415144 | 180 | 0.002 | +0.001 | 1.0000 | 0 | 0.09 / 0.09 | 0.98 / 0.98 | -0.04 / -0.04 | 91.11 / 90.33 |  |
| 9416841 | 180 | 0.024 | -0.024 | 1.0000 | 0 | 0.12 / 0.06 | 0.97 / 1 | -0.05 / -0.04 | 93.75 / 98.6 |  |
| 9418767 | 180 | 0.000 | +0.000 | 1.0000 | 0 | 0.06 / 0.07 | 0.99 / 0.99 | -0.03 / -0.03 | 95 / 94.72 |  |
| 9419750 | 180 | 0.003 | -0.002 | 1.0000 | 0 | 0.08 / 0.08 | 0.99 / 0.99 | -0.05 / -0.05 | 93.89 / 94.89 |  |
| 9431647 | 180 | 0.023 | +0.022 | 0.9999 | 0 | 0.05 / 0.07 | 1 / 1 | -0.02 / -0.04 | 97.22 / 95.39 |  |
| 9432780 | 180 | 0.004 | -0.003 | 1.0000 | 0 | 0.08 / 0.08 | 0.99 / 0.99 | -0.04 / -0.04 | 91.11 / 92.33 |  |
| 9435380 | 180 | 0.002 | -0.002 | 1.0000 | 0 | 0.07 / 0.07 | 1 / 1 | -0.04 / -0.04 | 93.89 / 94.44 |  |
| 9437540 | 180 | 0.003 | -0.000 | 1.0000 | 0 | 0.09 / 0.08 | 0.99 / 0.99 | -0.03 / -0.03 | 87.78 / 89 |  |
| 9439040 | 180 | 0.002 | -0.002 | 1.0000 | 0 | 0.08 / 0.08 | 0.99 / 0.99 | -0.02 / -0.02 | 91.11 / 90.94 |  |
| 9439099 | 180 | 0.010 | +0.009 | 1.0000 | 0 | 0.17 / 0.17 | 0.96 / 0.96 | 0.06 / 0.05 | 57.22 / 55.06 |  |
| 9439201 | 180 | 0.020 | +0.020 | 1.0000 | 0 | 0.28 / 0.26 | 0.81 / 0.82 | 0.2 / 0.18 | 40.56 / 42.22 |  |
| 9440083 | 180 | 0.001 | -0.000 | 1.0000 | 0 | 0.29 / 0.29 | 0.83 / 0.83 | 0.22 / 0.22 | 33.33 / 32.5 |  |
| 9440357 | 180 | 0.009 | -0.009 | 1.0000 | 0 | 0.51 / 0.52 | 0.87 / 0.87 | 0.48 / 0.49 | 2.78 / 1.78 |  |
| 9440422 | 180 | 0.012 | +0.012 | 1.0000 | 0 | 0.23 / 0.22 | 0.91 / 0.91 | 0.14 / 0.13 | 51.67 / 52.89 |  |
| 9440569 | 180 | 0.008 | +0.008 | 1.0000 | 0 | 0.15 / 0.15 | 0.97 / 0.97 | 0.03 / 0.02 | 57.78 / 57.39 |  |
| 9440581 | 180 | 0.011 | -0.011 | 1.0000 | 0 | 0.1 / 0.09 | 0.99 / 0.99 | -0.07 / -0.06 | 86.67 / 88.39 |  |
| 9440910 | 180 | 0.007 | -0.006 | 1.0000 | 0 | 0.12 / 0.12 | 0.98 / 0.98 | -0.03 / -0.02 | 77.22 / 77.56 |  |
| 9441102 | 180 | 0.002 | -0.002 | 1.0000 | 0 | 0.08 / 0.08 | 0.99 / 0.99 | -0.04 / -0.03 | 87.22 / 88.72 |  |
| 9442396 | 180 | 0.007 | -0.006 | 1.0000 | 0 | 0.08 / 0.08 | 1 / 1 | -0.06 / -0.06 | 90.56 / 90.83 |  |
| 9443090 | 180 | 0.006 | +0.006 | 1.0000 | 0 | 0.1 / 0.1 | 0.99 / 0.99 | -0.06 / -0.07 | 85.56 / 83.56 |  |
| 9444090 | 180 | 0.025 | +0.025 | 1.0000 | 0 | 0.14 / 0.15 | 0.98 / 0.98 | -0.01 / -0.04 | 70 / 72.61 |  |
| 9444900 | 180 | 0.008 | -0.008 | 1.0000 | 0 | 0.18 / 0.17 | 0.98 / 0.98 | -0.04 / -0.04 | 57.22 / 57.5 |  |
| 9445958 | 180 | 0.001 | +0.001 | 1.0000 | 0 | 0.25 / 0.25 | 0.98 / 0.98 | -0.14 / -0.15 | 50.56 / 49.94 |  |
| 9446484 | 180 | 0.113 | -0.113 | 1.0000 | 0 | 0.25 / 0.2 | 0.98 / 0.98 | -0.14 / -0.03 | 50.56 / 51.61 | FLAG |
| 9447130 | 180 | 0.113 | -0.113 | 1.0000 | 0 | 0.24 / 0.2 | 0.98 / 0.98 | -0.14 / -0.03 | 48.33 / 52.11 | FLAG |
| 9449424 | 180 | 0.014 | -0.014 | 1.0000 | 0 | 0.22 / 0.22 | 0.98 / 0.98 | -0.05 / -0.03 | 39.44 / 41.39 |  |
| 9449880 | 180 | 0.014 | -0.014 | 1.0000 | 0 | 0.2 / 0.19 | 0.98 / 0.98 | -0.04 / -0.02 | 46.11 / 46.67 |  |
| 9450460 | 180 | 0.138 | +0.134 | 0.9994 | 0 | 0.17 / 0.18 | 0.99 / 0.99 | 0.07 / -0.06 | 51.11 / 63.06 | FLAG |
| 9451054 | 180 | 0.002 | +0.002 | 1.0000 | 0 | 0.09 / 0.09 | 0.99 / 0.99 | -0.03 / -0.04 | 91.67 / 90.67 |  |
| 9451600 | 180 | 0.113 | -0.113 | 1.0000 | 0 | 0.16 / 0.27 | 0.99 / 0.99 | 0.14 / 0.26 | 53.89 / 8.72 | FLAG |
| 9452210 | 180 | 0.442 | +0.442 | 0.9999 | 0 | 0.38 / 0.21 | 0.99 / 0.99 | 0.34 / -0.1 | 17.78 / 52.72 | FLAG |
| 9452400 | 180 | 0.024 | +0.024 | 1.0000 | 0 | 0.21 / 0.21 | 0.99 / 0.99 | -0.02 / -0.04 | 46.11 / 48 |  |
| 9452634 | 180 | 0.338 | +0.338 | 0.9999 | 0 | 0.24 / 0.14 | 0.99 / 0.99 | 0.23 / -0.11 | 16.11 / 68.28 | FLAG |
| 9453220 | 180 | 0.066 | +0.065 | 0.9999 | 0 | 0.1 / 0.14 | 0.99 / 0.99 | -0.03 / -0.1 | 86.11 / 68.89 | FLAG |
| 9454050 | 180 | 0.013 | +0.013 | 1.0000 | 0 | 0.15 / 0.16 | 0.99 / 0.99 | -0.08 / -0.09 | 68.33 / 67.94 |  |
| 9454240 | 180 | 0.905 | +0.905 | 1.0000 | 0 | 0.82 / 0.15 | 0.99 / 0.99 | 0.81 / -0.1 | 0 / 68.33 | FLAG |
| 9455090 | 180 | 0.058 | +0.058 | 1.0000 | 0 | 0.07 / 0.09 | 0.99 / 1 | -0.01 / -0.06 | 95.56 / 89.17 | FLAG |
| 9455500 | 180 | 0.017 | +0.006 | 0.9999 | 0 | 0.11 / 0.11 | 1 / 1 | -0.04 / -0.05 | 82.78 / 81.33 |  |
| 9455760 | 180 | 0.234 | -0.219 | 0.9985 | 0 | 0.34 / 0.21 | 0.99 / 0.99 | -0.29 / -0.07 | 25 / 46.06 | FLAG |
| 9455920 | 180 | 0.035 | +0.034 | 1.0000 | 0 | 0.74 / 0.64 | 0.97 / 0.98 | -0.53 / -0.51 | 13.95 / 13.01 |  |
| 9457292 | 180 | 0.217 | +0.217 | 0.9998 | 0 | 0.21 / 0.09 | 0.99 / 0.99 | 0.19 / -0.03 | 30 / 90.54 | FLAG |
| 9457804 | 180 | 0.002 | -0.002 | 1.0000 | 0 | 0.09 / 0.08 | 0.99 / 1 | -0.02 / -0.02 | 91.67 / 91.33 |  |
| 9459450 | 180 | 0.092 | -0.092 | 1.0000 | 0 | 0.12 / 0.09 | 0.99 / 0.99 | -0.09 / 0.01 | 78.89 / 90.33 | FLAG |
| 9459881 | 180 | 0.108 | +0.108 | 1.0000 | 0 |  / 0.11 |  / 0.98 |  / 0.02 |  / 81.11 | FLAG |
| 9461380 | 180 | 0.154 | +0.154 | 0.9997 | 0 |  / 0.05 |  / 0.99 |  / -0 |  / 99.67 | FLAG |
| 9461710 | 180 | 0.010 | +0.010 | 1.0000 | 0 |  / 0.07 |  / 0.98 |  / 0 |  / 97.33 |  |
| 9462450 | 180 | 0.074 | +0.074 | 0.9999 | 0 |  / 0.1 |  / 0.97 |  / -0.02 |  / 86.11 | FLAG |
| 9462620 | 180 | 0.248 | +0.248 | 0.9998 | 0 |  / 0.12 |  / 0.97 |  / 0.02 |  / 75.94 | FLAG |
| 9463502 | 180 | 0.002 | -0.001 | 1.0000 | 0 |  / 0.21 |  / 0.97 |  / -0.03 |  / 49.33 |  |
| 9464212 | 180 | 0.006 | -0.002 | 0.9999 | 0 |  / 0.1 |  / 0.98 |  / 0.01 |  / 89.11 |  |
| 9468333 | 180 | 0.042 | -0.042 | 1.0000 | 0 |  / 0.21 |  / 0.94 |  / -0 |  / 44.39 |  |
| 9468756 | 180 | 0.017 | -0.017 | 1.0000 | 0 | 0.13 / 0.12 | 0.97 / 0.97 | -0.04 / -0.02 | 76.67 / 77.44 |  |
| 9491094 | 180 | 0.030 | -0.030 | 1.0000 | 0 | 0.1 / 0.09 | 0.96 / 0.96 | -0.03 / -0 | 90.56 / 92.22 |  |
| 9497645 | 180 | 0.024 | -0.023 | 0.9991 | 0 |  / 0.11 |  / 0.88 |  / 0.03 |  / 82.11 |  |
| 9751364 | 180 | 0.006 | -0.006 | 1.0000 | 0 |  / 0.04 |  / 0.93 |  / 0.02 |  / 100 |  |
| 9751381 | 180 | 0.010 | -0.010 | 1.0000 | 0 |  / 0.03 |  / 0.94 |  / 0 |  / 100 |  |
| 9751401 | 180 | 0.025 | -0.025 | 1.0000 | 0 |  / 0.03 |  / 0.94 |  / 0.01 |  / 100 |  |
| 9751639 | 180 | 0.047 | -0.047 | 0.9996 | 0 |  / 0.04 |  / 0.92 |  / -0 |  / 100 |  |
| 9752695 | 180 | 0.038 | -0.038 | 0.9999 | 0 |  / 0.04 |  / 0.92 |  / 0.01 |  / 100 |  |
| 9755371 | 180 | 0.007 | -0.007 | 1.0000 | 0 |  / 0.05 |  / 0.92 |  / -0 |  / 100 |  |
| 9759110 | 180 | 0.004 | -0.000 | 0.9989 | 0 | 0.11 / 0.1 | 0.75 / 0.76 | -0.1 / -0.09 | 89.68 / 97.25 |  |
| 9759394 | 180 | 0.009 | -0.009 | 1.0000 | 0 | 0.05 / 0.05 | 0.86 / 0.86 | 0 / 0.01 | 100 / 99.94 |  |
| 9759938 | 180 | 0.000 | -0.000 | 1.0000 | 0 | 0.07 / 0.07 | 0.88 / 0.89 | 0.06 / 0.06 | 99.44 / 99.22 |  |
