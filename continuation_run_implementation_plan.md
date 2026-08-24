# Issue #211 — continuation runs: implementation plan

Supersedes the line references in `continuation_run_plan.md` (written pre-#203).
All `file:line` below verified against `main @ 9bbc4de` (2026-08-24).

## 0. Behavior

`create_1dplot.py ... -cr` with the **full** window (`-s` original start, `-e` new end).
Per artifact the pipeline classifies on-disk coverage and, for a *prefix* artifact,
fetches/extracts only `[last_row - overlap, end]` and merges it in. `.int`, skill
CSVs and plots are always recomputed from the extended `.obs`/`.prd`.

Without `-cr` nothing changes.

## 1. Coverage classifier — `src/ofs_skill/utils/timeseries_coverage.py`

Add `COVERS` / `PREFIX` / `STALE` string constants and

```
classify_coverage(path, start_dt, end_dt, *, logger=None, now=None,
                  tolerance=STALENESS_TOLERANCE) -> str
```

* unreadable / no data rows / window shorter than tolerance -> `COVERS` (fail open,
  preserves today's `covers_run_window` contract)
* `first > start + tolerance` -> `STALE` (starts late / disjoint)
* `last < min(end, now) - tolerance` -> `PREFIX`
* otherwise `COVERS`

`covers_run_window` becomes `classify_coverage(...) == COVERS` — one predicate, three
existing call sites unchanged in behavior.

Also add `continuation_start(path, start_dt, end_dt, overlap, ...)` returning the tail
fetch start (`last - overlap`, clamped to `>= start_dt`) or `None` when not `PREFIX`.

## 2. Merge machinery — new `src/ofs_skill/utils/series_continuation.py`

`.obs` and `.prd` rows are self-contained fixed-width text produced by
`format_obs_timeseries.format_scalar/format_vector` (`:74-78`, `:158-163`), so the merge
is a **text-level** merge keyed on the `(Y,M,D,H,M)` columns — no reformatting, no
float round-tripping, no drift.

* `read_series_file(path)` -> `(header_or_None, [data_lines])`, text mode, utf-8.
* `merge_series_lines(existing, new)` -> `(merged, stats)` or `None`:
  * key = `datetime` parsed from cols 1-5 (reuses `_row_datetime`)
  * duplicate keys **within** the existing file -> return `None` (refuse to merge)
  * new rows win on collision (`keep='last'`): re-fetched/re-extracted values are the
    newer QC state
  * result sorted by key (providers other than CO-OPS do not sort)
* `seam_gap_seconds(merged_dts, seam_dt)` -> largest step within +/-1 sample of the
  seam, for the gap guard.
* `write_series_file(path, header, lines)` -> temp file in the same dir + `os.replace`
  (atomic; a crash can never leave a truncated artifact that `covers_run_window`
  fails *open* on). Empty `lines` writes a 0-byte file — the blank-file contract.
* `merge_and_write(path, new_lines, header, ...)` -> `True` merged / `False` caller
  should fall back to a normal full write.

## 3. Wiring

### 3.1 CLI + prop
* `create_1dplot.py` argparse: `-cr/--Continue_Run` (`store_true`),
  `-co/--Continue_Overlap` (float hours, default 24).
* `_run_pipeline`: `prop1.continue_run = getattr(run_args, 'Continue_Run', False)`,
  same for the overlap (GUI-safe).
* `model_properties.ModelProperties.__init__`: `continue_run = False`,
  `continue_overlap_hours = 24.0` (+ docstring attrs).
* `gui_helpers.GuiParams`: matching fields so a GUI launch never `AttributeError`s.
* `get_model_data.py`: accepts both flags; model downloads are already per-date
  incremental, so it logs that and otherwise behaves identically.

### 3.2 Guards (`create_1dplot.py` validation block)
Continuation is disabled with a warning, falling back to normal behavior, for:
* `forecast_a` — window is reshuffled per cycle and `.prd` names embed `forecast_hr`
* `-hs` horizon skill — bypasses `.prd` entirely, output is per-cycle CSV

### 3.3 Observations — `_ensure_obs_files` (`get_skill.py:847`)
Under `continue_run`, per station:
* `PREFIX` -> **do not delete**; record `obs_path -> tail_start` in a plan dict
* `STALE` -> delete as today
* missing / 0-byte -> full-window fetch as today (a 0-byte file carries no window
  information, so it cannot be extended)

Plan is passed as an explicit argument, not stashed on `prop` (the gates run on
`copy.copy(p)` in a 2-worker pool, `get_skill.py:1021/1024`):

`get_station_observations(prop, logger, continuation=None)`
-> `_process_variable_obs(..., continuation=...)`: at the skip gate
(`get_station_observations.py:892-917`), a station whose `.obs` is in the plan is
submitted with a narrowed window instead of being skipped
-> `_fetch_and_format_station(..., merge_into_existing=True)`: at the write site
(`:621-654`) the formatted tail is merged into the existing file instead of
truncating it.

**Splice-safety (the highest risk in this feature).** Several retrieval choices are
window-dependent and unrecorded in the file, so a short tail window can silently
resolve a *different* datum / sensor / ADCP bin than the head:
* CO-OPS multi-datum fallback loop (`:260-300`)
* CO-OPS currents unrestricted-bin retry (`:459-474`)

Both are **disabled** when `merge_into_existing` is set. If the tail's primary
retrieval comes back empty, the existing file is left untouched and the reason logged
— never spliced, never truncated.

### 3.4 Model extraction — `_ensure_prd_files` (`get_skill.py:890`)
Under `continue_run`, classify every expected `.prd` for the variable:
* **all `COVERS`** -> nothing to do (today's behavior)
* **some `PREFIX`, none `STALE`, none missing** -> continuation pass:
  `tail_start = min(last timestamps) - overlap`; call `get_node_ofs` on a
  `copy.copy(p)` whose `start_date_full` is `tail_start` (ISO — inside `get_skill`
  the dates are ISO, `get_skill.py:717-724`) with `continuation_prd_merge = True`.
  Then **re-classify against the full window**; anything still not `COVERS` is
  deleted and regenerated by the existing full-window path.
* **anything else** (mixed / missing / stale) -> today's behavior verbatim

So continuation is strictly an optimization layered in front of the existing gate,
and every failure mode degrades to a correct full regeneration.

Inside `get_node_ofs`:
* `_all_prd_files_complete` (`:1398`) returns `False` immediately in merge mode —
  its row-count and coverage checks are written against the *run* window, which in
  the tail pass is the tail, and it would otherwise short-circuit the extraction.
* both `.prd` write sites (`:1816-1844`, `:1884-1908`) go through `merge_and_write`.
* the filename-key CSV (`:1638-1657`) is merged (read existing, concat, dedup on
  `DateTime`, sort) instead of clobbered — a tail run would otherwise truncate the
  key that the plot hover text merges against.

### 3.5 Pairing / skill / plots
`_ensure_paired_data_exists` (`create_1dplot.py:611`): under `continue_run`, every
existing `.int` for the run is removed so pairing always re-runs over the full
concatenated series. `format_paired_one_d` needs no change — its window mask
(`:193-204`, `:389-400`) already crops to `[start - lookback, end]`. Skill CSVs and
plots are rewritten every run as they are today.

## 4. Known, documented limitations
* A continuation run does **not** refresh provider QC revisions older than the
  overlap window. Users wanting full re-verification run without `-cr`.
* Overlapping timestamps resolve to the newly fetched value, so continuation output
  is not guaranteed byte-identical to a from-scratch run when a provider has revised
  historical data. It is byte-identical when sources are stable (asserted by the
  offline fixture test).
* Non-date parameters (`-d`, `-t`, station ctl contents) are assumed constant
  between runs; the artifacts do not record them.
* Sub-minute observations collapse on the merge key (the file format has
  minute resolution, so they already collapse in the written file).

## 5. Tests (`tests/continuation_run_test.py`, pytest, `tmp_path`)
* `classify_coverage`: COVERS / PREFIX / STALE boundaries, tolerance edges,
  future window clamp, unreadable + 0-byte fail-open, injectable `now=`.
* `covers_run_window` unchanged (re-run the existing `stale_cache_coverage_test.py`).
* merge: dedup at the seam, new-wins-on-collision, sort of unsorted input,
  `nan` token round-trip, header/headerless mix, duplicate-key refusal,
  0-byte contract, atomic replace, trailing-partial-line tolerance.
* gates: `_ensure_obs_files` / `_ensure_prd_files` with stub props — PREFIX plans
  a tail, STALE deletes, missing does a full fetch, flag off = today's behavior.
* `_ensure_paired_data_exists` removes `.int` under the flag.
* forecast_a / horizon-skill guard falls back with a warning.
* End-to-end offline: build window-A `.obs`/`.prd` fixtures, run the merge path to
  window A+B, assert the result equals a fresh A+B write byte for byte.
* Live e2e (manual, recorded in the PR): cbofs 3-day -> continue to 5-day vs a
  from-scratch 5-day run; diff `.int`, skill CSVs, plot point counts.
