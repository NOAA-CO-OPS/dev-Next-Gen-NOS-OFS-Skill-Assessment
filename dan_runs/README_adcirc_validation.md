# ADCIRC (STOFS-2D-Global) validation runs on the cluster

This branch, `validation/adcirc-fields-nearest-node`, is
`feature/adcirc-fields-nearest-node` plus this `dan_runs/` folder. Nothing in
`dan_runs/` goes into the PR.

There are four runs. Every run uses `-o stofs_2d_glo -d MSL -so co-ops`.

| Folder | What it runs | Window or cycle |
|---|---|---|
| `stofs2d-fields-nowcast-2day` | (a) fields nowcast, default `-vs`, so it also tests that ADCIRC runs drop everything except water level | 2026-10-17T00:00:00Z to 2026-10-19T00:00:00Z |
| `stofs2d-stations-nowcast-2day` | (b) stations nowcast, `-vs water_level` | same as (a) |
| `stofs2d-fields-forecasta-1cycle` | (c1) fields forecast_a, `-vs water_level` | cycle 2026-10-14 00z, 180 h, to 2026-10-21T12:00:00Z |
| `stofs2d-stations-forecasta-1cycle` | (c2) stations forecast_a, `-vs water_level` | same cycle as (c1) |

The window for (a) and (b) is the same as the laptop run, so the results can
be compared with `adcirc_fields_validation_laptop_2026-10-22.md`.

## 1. Setup (once per session)

```bash
cd /lustre/code/devNG/dev-Next-Gen-NOS-OFS-Skill-Assessment
git fetch
git checkout validation/adcirc-fields-nearest-node
git pull
git log -1 --format='%H %s'
ls -l conf/api_keys.conf
conda activate ofs_dps
```

- **`git log -1`:** compare the hash it prints with the hash you were given for
  this branch.
- **`ls -l conf/api_keys.conf`:** the file must exist. It is not in git. If it
  is missing, see Troubleshooting.
- **`conda activate ofs_dps`:** the job scripts do not activate an environment.
  `sbatch` passes on the environment active when you submit.

## 2. Before each round of runs

1. Run `git pull` in the clone, then `git log -1 --format=%H`. Compare the hash
   with the one you were given.
2. After you submit, check the top of each `job_<run>.out`. Its `Commit:` line
   must show that same hash. If it differs, cancel the job
   (`scancel <jobid>`), pull, and submit again.

## 3. Submitting

Submit each job from inside its own folder. Slurm writes `job_<run>.out` and
`job_<run>.err` in the folder you submit from.

### Order: one after another (recommended)

I am not certain the four runs share no state, so run them one after another.
What they do share:

- **One log file.** Every run appends to `log/ofs_dps.log` in the clone
  (`conf/logging.conf`, line 51). Python does not support several processes
  logging to one file
  ([logging cookbook](https://docs.python.org/3/howto/logging-cookbook.html#logging-to-a-single-file-from-multiple-processes)).
  Each `.out` also gets the full log, so a garbled `ofs_dps.log` would lose
  nothing.
- **The model-file cache.** Points files for the stations runs go into
  `~/.ofs_cache/s3/` (`get_s3_cache_dir` in
  `src/ofs_skill/obs_retrieval/utils.py`). Fields files are streamed and not
  cached. Runs (b) and (c2) read files from different dates, so they should not
  write the same cache file.
- **Everything else** is written under each run's own folder, including the
  station inventory, which goes to `control_files/`.

Order: (a), (b), (c1), (c2). Run (a) is the main test. This queues all four,
each starting when the one before it ends, whether or not it succeeded:

```bash
cd /lustre/code/devNG/dev-Next-Gen-NOS-OFS-Skill-Assessment/dan_runs
cd stofs2d-fields-nowcast-2day
j1=$(sbatch --parsable job_stofs2d-fields-nowcast-2day.sh); echo "a  $j1"
cd ../stofs2d-stations-nowcast-2day
j2=$(sbatch --parsable --dependency=afterany:$j1 job_stofs2d-stations-nowcast-2day.sh); echo "b  $j2"
cd ../stofs2d-fields-forecasta-1cycle
j3=$(sbatch --parsable --dependency=afterany:$j2 job_stofs2d-fields-forecasta-1cycle.sh); echo "c1 $j3"
cd ../stofs2d-stations-forecasta-1cycle
j4=$(sbatch --parsable --dependency=afterany:$j3 job_stofs2d-stations-forecasta-1cycle.sh); echo "c2 $j4"
squeue -u $USER
```

Or submit one at a time in your usual way, waiting for each to finish:

```bash
cd /lustre/code/devNG/dev-Next-Gen-NOS-OFS-Skill-Assessment/dan_runs/stofs2d-fields-nowcast-2day
sbatch job_stofs2d-fields-nowcast-2day.sh
```

`--dependency=afterany` and `--parsable` are described in the
[sbatch manual](https://slurm.schedmd.com/sbatch.html).

## 4. What success looks like

For every run:

- **`.out` top:** `Branch: validation/adcirc-fields-nearest-node`, the
  `Commit:` hash, the start time and the full command.
- **`.out` end:** the last log line is `Finished create_1dplot!`, followed by
  `End:` and `Exit code: 0`.
- **`.err` end:** if `/usr/bin/time` exists on the node, the `.err` ends with
  its report. `Maximum resident set size (kbytes)` is the peak memory and
  `Elapsed (wall clock) time` is the run time. Note both down.
- **Expected `.out` lines that are not errors:**
  - `Model data path …/example_data/stofs_2d_glo/… not found, or is empty`,
    followed by `S3 fallback enabled`. These mean the model files come from
    NODD, as intended.
  - `CO-OPS water_level data not found for station … Unsupported vertical
    datum conversion '<x>'->'msl'`. This is a known existing problem; the
    laptop runs dropped 26–32 stations this way.

### Files expected in each run folder

| Run | `control_files/` | `data/` |
|---|---|---|
| (a) fields nowcast | `inventory_all_stofs_2d_glo.csv`, `station_ledger_stofs_2d_glo.csv`, `stofs_2d_glo_wl_station.ctl`, `stofs_2d_glo_wl_model.ctl`, `stofs_2d_glo_wl_datum_report.csv` | `observations/1d_station/*_wl_station.obs`; `model/1d_node/*_nowcast_fields_model.prd`; `skill/1d_pair/*_nowcast_fields_pair.int`; `skill/stats/skill_stofs_2d_glo_water_level_nowcast_fields.csv` (plus `_hw_`, `_lw_` and `all_stations`); `visual/*_timeseries_nowcast_fields.html`, summary barplots and `plotly_maps/` |
| (b) stations nowcast | as (a), but `stofs_2d_glo_wl_model_station.ctl` in place of `_wl_model.ctl` | as (a), with `stations` in place of `fields` in the names |
| (c1) fields forecast_a | as (a) | as (a), with `forecast_a` in place of `nowcast` in the names |
| (c2) stations forecast_a | as (b) | as (b), with `forecast_a` in place of `nowcast` in the names |

- **Variables:** only water-level files (`wl`, `water_level`) should appear.
  Any temperature, salinity or currents file in (a) is a failure; bring the
  folder back.
- **Station counts:** the laptop run of (a) produced 240 obs files and 240
  fields `.prd` files. Run (b) produced 215 stations `.prd` files.

### Rough run times

- **Laptop (a):** about 1 h 45 min for a fresh run. Each of five phases
  streams the 16 fields files from NODD (about 7 GB) and took about 11 min.
  Pairing, skill and plots took about 37 min, slowed by CO-OPS HTTP 403
  retries.
- **Laptop (b):** about 1 h 35 min, mostly pairing, skill and plots during the
  same CO-OPS 403s.
- **(c1) and (c2):** not run on the laptop. Run (c1) reads 180 hourly steps
  from one 10 GB fields file, against 96 steps for (a). I'm speculating that
  its file-reading phases take up to about twice as long as in (a).
- **Cluster reference:** a 3-day stofs_3d_atl stations run took about 26 min
  on the cluster. How fast the cluster streams from NODD is unknown.

## 5. Troubleshooting

| Problem | How to recognise it | What to do |
|---|---|---|
| `conf/api_keys.conf` missing | `ls -l` in setup fails. In `.out`: `API_USGS_PAT is not set. USGS API requests will be limited to 50/hour.` | These runs use `-so co-ops`, so the USGS key is not used and the warning is harmless. Run anyway. Never commit the file. |
| PROJ / TIFF read errors | `.err` lines with `TIFFFillTile` or other PROJ/TIFF messages | Harmless if the `.out` ends with `Finished create_1dplot!`. If the run did not finish, bring the folder back. |
| NODD file missing (404) or timeout | `.out`/`.err` lines with `404`, `Not Found`, `timed out`, `ReadTimeout` or `ClientConnectorError` near a `noaa-gestofs-pds` URL, often followed by an exit | Check the URL in the log with `curl -sI <url> \| head -1`. If it gives 200 now, resubmit that one run once. Otherwise bring the folder back. |
| API rate limit or block | `.out` lines like `CO-OPS water_level station=… HTTP 403 (attempt 1/10); retrying in …s`, or `429` | Retries are automatic. The run is slower and may drop some stations. Let it finish and bring the folder back. Do not resubmit straight away; that adds to the load. |
| Out of memory | `.err`: `oom-kill`, `Detected … oom_kill event`, `Out Of Memory` or `Killed`; `Exit code: 137` in `.out`; `sacct -j <jobid> --format=JobID,State,MaxRSS` shows `OUT_OF_MEMORY` | In that run's `job_<run>.sh`, change `##SBATCH --mem=16G` to `#SBATCH --mem=16G` (one `#`) and resubmit once. If it fails again, bring the folder back. Report the edit (see Rules). |
| Killed at the time limit | `.err`: `CANCELLED AT … DUE TO TIME LIMIT`; `sacct` shows `TIMEOUT` | Note the elapsed time and bring the folder back. Do not add a `--time` line on the spot. |
| Anything else (Python traceback, `Exit code` not 0) | `Traceback` in `.out`/`.err` | Do not fix it. Bring the folder back. |

## 6. Rules

- Do not commit or push from the cluster.
- If you edit anything there (for example the `--mem` line), keep the edit
  local and report it back, so it can be made on the branch.
- Do not edit anything outside `dan_runs/`.

## 7. Final steps, whatever happened

1. Bring every run folder back, as you normally do. Each must include its
   `.conf`, `job_<run>.sh`, `job_<run>.out`, `job_<run>.err`, `data/` and
   `control_files/`.
2. Check out your usual branch again, so later runs do not use the validation
   code:

   ```bash
   cd /lustre/code/devNG/dev-Next-Gen-NOS-OFS-Skill-Assessment
   git status --short           # shows any local edits to report back
   git checkout <your usual branch>
   git log -1 --format='%H %s'
   ```

   If `git status --short` shows edits that block the checkout, report them
   first. Then keep a copy (`git stash`) or discard them
   (`git checkout -- dan_runs`).
