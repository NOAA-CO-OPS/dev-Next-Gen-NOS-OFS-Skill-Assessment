"""Stage timestamps, time -v report and warning counts from the cluster run logs."""
import glob
import json
import os
import re
import sys

ROOT = sys.argv[1]
TS = re.compile(r"^'(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)'")
MARKERS = [
    ('run_start', 'Run command:', 'first'),
    ('inventory_start', '--- Starting Inventory Retrieval Process ---', 'first'),
    ('inventory_end', 'Finished OFS boundary filter.', 'first'),
    ('model_start', '--- Starting OFS Model process ---', 'first'),
    ('intake_start', 'Starting catalog', 'first'),
    ('intake_files', 'Skipping cache for|Resolved .* remote station file', 'first'),
    ('ctl_build_start', r'Model Control File \(.*\) not found', 'first'),
    ('dry_start', 'ADCIRC dry node scan: checking', 'first'),
    ('dry_end', 'ADCIRC dry node scan finished', 'first'),
    ('nearest_node', 'ADCIRC nearest-node:', 'first'),
    ('extract_start', 'Extracting data found in the Model Control File', 'first'),
    ('ctl_done', 'Model Control File for water_level created successfully', 'first'),
    ('model_end', 'Finished with model data processing!', 'first'),
    ('paired_search', 'Searching for paired dataset', 'first'),
    ('metrics_first', 'Start wl metrics', 'first'),
    ('metrics_last', 'Start wl metrics', 'last'),
    ('plot_first', 'Completed plot for station', 'first'),
    ('plot_last', 'Completed plot for station', 'last'),
    ('finished', 'Finished create_1dplot!', 'last'),
]
COUNTS = {
    'restrict_tempsalt': 'has no water temperature or salinity output',
    'restrict_currents': 'currents are not yet supported in this pipeline',
    'gomofs': '(?i)gomofs',
    'http_403': 'HTTP 403|403 Client Error',
    'http_429': 'HTTP 429|429 Client Error',
    'datum_msl_error': 'Unsupported vertical datum conversion',
    'batch_precompute_failed': 'Batch precomputation failed',
    'tracebacks': '^Traceback',
    'other_var_mentions': '(?i)water_temperature|salinity|currents',
}
out = {}
for run in sorted(glob.glob(f'{ROOT}/stofs2d-*')):
    name = os.path.basename(run)
    lines = open(f'{run}/job_{name}.out', encoding='utf-8', errors='replace').read().splitlines()
    err = open(f'{run}/job_{name}.err', encoding='utf-8', errors='replace').read().splitlines()
    m = {}
    for key, pat, which in MARKERS:
        hits = [ln for ln in lines if re.search(pat, ln)]
        if hits:
            ln = hits[0] if which == 'first' else hits[-1]
            t = TS.match(ln)
            m[key] = (t.group(1) if t else None, ln[25:200])
    c = {k: sum(1 for ln in lines if re.search(p, ln)) for k, p in COUNTS.items()}
    tv = {}
    for ln in err:
        for k in ('Percent of CPU', 'Elapsed (wall clock)', 'Maximum resident set size',
                  'Exit status', 'User time', 'System time'):
            if k in ln:
                tv[k] = ln.split(':', 1)[1].strip() if k != 'Elapsed (wall clock)' \
                    else ln.rsplit(': ', 1)[1].strip()
    head = {ln.split(':', 1)[0]: ln.split(':', 1)[1].strip() for ln in lines[:4] if ':' in ln}
    tail = [ln for ln in lines[-4:] if ln.startswith(('End:', 'Exit code'))]
    other = [ln[25:220] for ln in lines if re.search(COUNTS['other_var_mentions'], ln)]
    out[name] = {'head': head, 'tail': tail, 'markers': m, 'counts': c, 'time_v': tv,
                 'other_var_lines': sorted(set(other))[:15]}
print(json.dumps(out, indent=1))
