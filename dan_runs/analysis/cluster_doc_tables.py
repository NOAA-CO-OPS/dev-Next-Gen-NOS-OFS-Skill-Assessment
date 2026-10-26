"""Markdown appendix tables (per-station matching, series and skill) for the cluster runs."""
import sys

import pandas as pd

A, OUT = sys.argv[1], sys.argv[2]
LABELS = {'nowcast': ('C', 'D', 'nowcast, 2026-10-17T00Z to 2026-10-19T00Z'),
          'forecast_a': ('E', 'F', 'forecast_a, cycle 2026-10-14 00z')}


def yn(v):
    return 'yes' if v else 'no'


def fmt(v, spec):
    return '' if pd.isna(v) else format(v, spec)


def pair(r, k):
    return f'{fmt(r.get(f"{k}_fields"), "g")} / {fmt(r.get(f"{k}_stations"), "g")}'


lines = []
for lab, (ap1, ap2, title) in LABELS.items():
    a = pd.read_csv(f'{A}/{lab}_per_station.csv', dtype={'ID': str})
    c = pd.read_csv(f'{A}/{lab}_series_compare.csv', dtype={'ID': str})
    s = pd.read_csv(f'{A}/{lab}_skill_side_by_side.csv', dtype={'ID': str})
    lines += [f'## Appendix {ap1}. Per-station matching, {title}', '',
              '| ID | Name | fields | stations | fields node | km | km source | nearest node of any kind, km | nearer node dry | stations index | reason if one mode only |',
              '|---|---|---|---|---|---|---|---|---|---|---|']
    for _, r in a.sort_values('ID').iterrows():
        src = 'exact' if r.km_source.startswith('exact') else 'ctl'
        si = '' if pd.isna(r.stations_node) else str(int(r.stations_node))
        reason = '' if pd.isna(r.reason_one_mode) else r.reason_one_mode
        lines.append(f'| {r.ID} | {r.Name} | {yn(r.matched_fields)} | {yn(r.matched_stations)} | '
                     f'{int(r.node)} | {r.km:.3f} | {src} | {r.nearest_any_km:.3f} | '
                     f'{"yes" if r.farther_due_to_dry else ""} | {si} | {reason} |')
    lines += ['', f'## Appendix {ap2}. Fields vs stations series, and skill tables, {title}', '',
              '| ID | n | series RMSE (m) | series bias (m) | series corr | lag (h) | skill RMSE fields / stations | skill r fields / stations | skill bias fields / stations | central freq fields / stations | FLAG |',
              '|---|---|---|---|---|---|---|---|---|---|---|']
    m = c.merge(s, on='ID', how='left')
    for _, r in m.sort_values('ID').iterrows():
        lines.append(f'| {r.ID} | {int(r.n)} | {r.rmse:.3f} | {r.bias:+.3f} | {r["corr"]:.4f} | '
                     f'{int(r.best_lag_h)} | {pair(r, "rmse")} | {pair(r, "r")} | '
                     f'{pair(r, "bias")} | {pair(r, "central_freq")} | '
                     f'{"FLAG" if r.flag else ""} |')
    lines.append('')
open(OUT, 'w', encoding='utf-8', newline='\n').write('\n'.join(lines))
print(len(lines))
