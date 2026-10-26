# Analysis scripts for the ADCIRC fields validation runs

These scripts produce the numbers and appendix tables in
`docs/adcirc_fields_validation.md` (feature branch). They read only run logs,
control files, `.prd`/`.obs` files and skill CSVs, and never open a STOFS
model file. Validation branch only; not for the PR.

## Inputs

- **Cluster run folders:** the four folders brought back from the cluster
  (`stofs2d-fields-nowcast-2day`, `stofs2d-stations-nowcast-2day`,
  `stofs2d-fields-forecasta-1cycle`, `stofs2d-stations-forecasta-1cycle`), all
  under one root directory.
- **Laptop run folder:** the laptop nowcast run home (`stofs2d_validation/`).
  It holds `snapshot_a/` (fields) and `snapshot_b/` (stations) with
  `control_files/` and `stats/`, plus `data/` with the `.prd` and `.obs`
  files. Only item h (cluster vs laptop) needs it.
- **`laptop_mesh_geometry_item4.csv`:** one row per observation station.
  - **Contents:** the nearest STOFS-2D-Global mesh node of any kind and the
    exact great-circle distances to it and to the laptop's matched node.
  - **Origin:** computed in the laptop analysis from the mesh `x`/`y` of
    `stofs_2d_glo.t00z.fields.cwl.nc` (cycle 2026-10-20 00z). The mesh is the
    same for every cycle.
  - **Why it is committed:** it supplies the "nearer node excluded as dry" test
    and the exact distances, without opening a model file.

## Scripts

Run from the repo root with the `ofs_dps` environment active.
`cluster_analysis.py` imports `ofs_skill`, so `src/` must be importable.

```bash
export PYTHONPATH=$PWD/src
RUNS=/path/to/adcirc_val_runs_20261026      # cluster run folders
LAPTOP=/path/to/stofs2d_validation          # laptop run home
OUT=/path/to/analysis_out

# Items 1, a, b: completion, variable handling, wall time, memory, stages
python dan_runs/analysis/cluster_stages.py "$RUNS" > "$OUT/stages.json"

# Items c-h, plus the extra numbers quoted in the doc text
python dan_runs/analysis/cluster_analysis.py "$RUNS" "$LAPTOP" \
    dan_runs/analysis/laptop_mesh_geometry_item4.csv "$OUT"

# Appendices C-F (Markdown), from the CSVs written by cluster_analysis.py
python dan_runs/analysis/cluster_doc_tables.py "$OUT" "$OUT/appendix.md"
```

## Outputs of `cluster_analysis.py`

| File | Content |
|---|---|
| `<cast>_per_station.csv` | Item c, per observation station: matched in each mode, node, km and its source, nearest node of any kind, whether a nearer node was excluded as dry, and the reason when only one mode matched. |
| `<cast>_series_compare.csv` | Item d: fields vs stations RMSE, bias, correlation, best lag and FLAG. |
| `<cast>_skill_side_by_side.csv` | Item d: the two runs' skill-table metrics. |
| `<cast>_series_checks.csv` | Item e: steps, monotonic, duplicates and span per fields series. |
| `<cast>_nan.csv` | Item f: NaN steps per fields series. |
| `summary.json` | Totals for items c-f per cast. Item h under `laptop_compare`, and the numbers quoted in the doc text under `extras`. |

`extras` holds:
- the stations whose node changes between nowcast and forecast_a;
- dry-shift ranges and the stations more than 1 km from their node;
- short overlaps and non-zero lags;
- the largest offsets, and FLAGs by correlation or overlapping the dry rule;
- which product is closer to the observations at flagged stations;
- the bias correlation between the two windows, and FLAGs found in one window
  only.

`<cast>` is `nowcast` or `forecast_a`.

## Notes

- **Distance precision:** when a forecast_a node is not in the laptop
  geometry, its distance comes from the 3-decimal node coordinates in the model
  control file, about ±0.1 km (`km_source` = `ctl 3-decimal coords`).
- **Line endings:** observation files are compared with line endings
  normalized, because laptop files are CRLF and cluster files LF.
