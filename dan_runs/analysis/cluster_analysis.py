"""Analysis of the four cluster validation runs (items c-h), with no model-file access.

Grid geometry (nearest node of any kind, exact node distance) comes from the
laptop analysis CSV, which was computed from the STOFS-2D-Global mesh x/y; the
mesh is the same for every cycle. Distances for nodes the laptop never matched
are computed from the 3-decimal node coordinates in the model ctl file.

Usage: python cluster_analysis.py <cluster_root> <laptop_root> <laptop_item4_csv> <out_dir>
"""
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

from ofs_skill.model_processing.station_distance import calculate_station_distance

CR, LR, L4, OUT = sys.argv[1:5]
RUNS = {
    'fn': ('stofs2d-fields-nowcast-2day', 'nowcast', 'fields'),
    'sn': ('stofs2d-stations-nowcast-2day', 'nowcast', 'stations'),
    'fa': ('stofs2d-fields-forecasta-1cycle', 'forecast_a', 'fields'),
    'sa': ('stofs2d-stations-forecasta-1cycle', 'forecast_a', 'stations'),
}
WINDOWS = {'nowcast': (pd.Timestamp('2026-10-17 00:00'), pd.Timestamp('2026-10-19 00:00')),
           'forecast_a': (pd.Timestamp('2026-10-14 00:00'), pd.Timestamp('2026-10-21 12:00'))}
SKILL_COLS = ['rmse', 'r', 'bias', 'bias_standard_dev', 'central_freq', 'pos_outlier_freq',
              'neg_outlier_freq']


def obs_ctl(path):
    lines = open(path, encoding='utf-8').read().splitlines()[2:]
    rows = []
    for i in range(0, len(lines) - 1, 2):
        sid = lines[i].split()[0]
        m = re.search(r'"(.*)"', lines[i])
        lat, lon = map(float, lines[i + 1].split()[:2])
        rows.append((sid, m.group(1) if m else '', lat, lon))
    return pd.DataFrame(rows, columns=['ID', 'Name', 'lat', 'lon'])


def model_ctl(path):
    rows = [ln.split() for ln in open(path, encoding='utf-8').read().splitlines()[1:]
            if ln.strip()]
    return pd.DataFrame({'ID': [r[4] for r in rows], 'node': [int(r[0]) for r in rows],
                         'nlat': [float(r[2]) for r in rows],
                         'nlon': [float(r[3]) for r in rows]})


def prd(path):
    d = pd.read_csv(path, sep=r'\s+', skiprows=1, header=None,
                    names=['jd', 'Y', 'M', 'D', 'h', 'm', 'v'])
    t = pd.to_datetime(dict(year=d.Y, month=d.M, day=d.D, hour=d.h, minute=d.m))
    return pd.Series(d.v.values.astype(float), index=t)


def prd_file(run_dir, sid, cast, ftype):
    tag = 'forecast_a_00z' if cast == 'forecast_a' else 'nowcast'
    f = glob.glob(f'{run_dir}/data/model/1d_node/{sid}_stofs_2d_glo_wl_*_{tag}_{ftype}_model.prd')
    return f[0] if f else None


def skill(run_dir, cast, ftype):
    p = f'{run_dir}/data/skill/stats/skill_stofs_2d_glo_water_level_{cast}_{ftype}.csv'
    return pd.read_csv(p, dtype={'ID': str})


def no_match_ids(run_dir, name):
    txt = open(f'{run_dir}/job_{name}.out', encoding='utf-8', errors='replace').read()
    return set(re.findall(r'No matching model station found for obs station (\S+)\.', txt))


lap = pd.read_csv(L4, dtype={'ID': str})
lap_geo = lap.set_index('ID')[['node', 'km', 'nearest_any_node', 'nearest_any_km']]


def per_station(kf, ks):
    """Item c for one fields/stations pair."""
    fdir, sdir = (f'{CR}/{RUNS[k][0]}' for k in (kf, ks))
    of = obs_ctl(f'{fdir}/control_files/stofs_2d_glo_wl_station.ctl')
    os_ = obs_ctl(f'{sdir}/control_files/stofs_2d_glo_wl_station.ctl')
    mf = model_ctl(f'{fdir}/control_files/stofs_2d_glo_wl_model.ctl')
    ms = model_ctl(f'{sdir}/control_files/stofs_2d_glo_wl_model_station.ctl')
    nm = no_match_ids(sdir, RUNS[ks][0])
    a = of.merge(mf[['ID', 'node', 'nlat', 'nlon']], on='ID', how='left')
    a['km_ctl'] = [calculate_station_distance(la, lo, nla, nlo) if pd.notna(n) else np.nan
                   for la, lo, nla, nlo, n in zip(a.lat, a.lon, a.nlat, a.nlon, a.node)]
    ex = []
    for _, r in a.iterrows():
        g = lap_geo.loc[r.ID] if r.ID in lap_geo.index else None
        if g is not None and pd.notna(r.node) and int(g.node) == int(r.node):
            ex.append((r.km_ctl if np.isnan(g.km) else g.km, 'exact (laptop mesh)'))
        elif g is not None and pd.notna(r.node) and int(g.nearest_any_node) == int(r.node):
            ex.append((g.nearest_any_km, 'exact (laptop mesh)'))
        else:
            ex.append((r.km_ctl, 'ctl 3-decimal coords'))
    a['km'] = [e[0] for e in ex]
    a['km_source'] = [e[1] for e in ex]
    a['nearest_any_node'] = [lap_geo.nearest_any_node.get(i, np.nan) for i in a.ID]
    a['nearest_any_km'] = [lap_geo.nearest_any_km.get(i, np.nan) for i in a.ID]
    a['farther_due_to_dry'] = (a.node.notna() & a.nearest_any_node.notna()
                               & (a.node != a.nearest_any_node)
                               & (a.nearest_any_km < a.km - 1e-6))
    a['matched_fields'] = a.node.notna() & (a.km <= 4.0)
    a['matched_stations'] = a.ID.isin(ms.ID)
    a['stations_node'] = a.ID.map(ms.set_index('ID').node)
    a['reason_one_mode'] = ''
    a.loc[a.matched_fields & ~a.matched_stations, 'reason_one_mode'] = [
        'stations run: "No matching model station found" (not in points file station list)'
        if i in nm else 'stations run: absent from model ctl, reason not logged'
        for i in a.ID[a.matched_fields & ~a.matched_stations]]
    a.loc[~a.matched_fields & a.matched_stations, 'reason_one_mode'] = 'fields: no node within 4 km'
    return a, of, os_, mf, ms


def compare(kf, ks, a):
    """Items d, e, f for one fields/stations pair."""
    fdir, sdir = (f'{CR}/{RUNS[k][0]}' for k in (kf, ks))
    cast = RUNS[kf][1]
    w0, w1 = WINDOWS[cast]
    rows, nan_rows, checks = [], [], []
    for _, r in a[a.matched_fields].iterrows():
        fa = prd_file(fdir, r.ID, cast, 'fields')
        if not fa:
            checks.append({'ID': r.ID, 'note': 'no fields prd'})
            continue
        sa = prd(fa)
        d = np.diff(sa.index.values).astype('timedelta64[m]').astype(int)
        checks.append({'ID': r.ID, 'n': len(sa), 'monotonic': bool((d > 0).all()),
                       'duplicates': int(sa.index.duplicated().sum()),
                       'steps_min': sorted(set(d.tolist())),
                       'start': str(sa.index.min()), 'end': str(sa.index.max())})
        nan_rows.append((r.ID, int(sa.isna().sum()), int(sa[w0:w1].isna().sum()),
                         int(sa[w0:w1].size)))
        fb = prd_file(sdir, r.ID, cast, 'stations')
        if not fb:
            continue
        sb = prd(fb)
        sb = sb[~sb.index.duplicated()]
        common = sa[w0:w1].index.intersection(sb.index)
        va, vb = sa[common].values, sb[common].values
        ok = np.isfinite(va) & np.isfinite(vb)
        va, vb = va[ok], vb[ok]
        dif = va - vb
        hs, hb = sa[w0:w1], sb.reindex(sa[w0:w1].index)
        lags = []
        for lag in range(-3, 4):
            c = pd.Series(hs.values).corr(pd.Series(hb.shift(lag).values))
            lags.append((lag, -np.inf if np.isnan(c) else c))
        best = max(lags, key=lambda t: t[1])
        rows.append(dict(ID=r.ID, n=int(ok.sum()), rmse=float(np.sqrt(np.mean(dif ** 2))),
                         bias=float(dif.mean()), corr=float(np.corrcoef(va, vb)[0, 1]),
                         best_lag_h=best[0], fields_node=int(r.node), km=float(r.km)))
    cmp_ = pd.DataFrame(rows)
    cmp_['flag'] = (cmp_.bias.abs() > 0.05) | (cmp_['corr'] < 0.95)
    sf, ss = skill(fdir, cast, 'fields'), skill(sdir, cast, 'stations')
    sk = sf[['ID'] + SKILL_COLS].merge(ss[['ID'] + SKILL_COLS], on='ID', how='outer',
                                       suffixes=('_fields', '_stations'))
    nan = pd.DataFrame(nan_rows, columns=['ID', 'nan_all', 'nan_window', 'n_window'])
    return cmp_, sk, nan, pd.DataFrame(checks), sf, ss


def summarize(a, cmp_, nan, checks, sf, ss, ms):
    good = checks[checks.get('n').notna()] if 'n' in checks else checks.iloc[0:0]
    return {
        'obs': len(a), 'matched_fields': int(a.matched_fields.sum()),
        'matched_stations': int(a.matched_stations.sum()), 'model_ctl_stations': len(ms),
        'both': int((a.matched_fields & a.matched_stations).sum()),
        'only_fields': a.ID[a.matched_fields & ~a.matched_stations].tolist(),
        'only_stations': a.ID[~a.matched_fields & a.matched_stations].tolist(),
        'unmatched_fields': a.ID[~a.matched_fields].tolist(),
        'farther_due_to_dry': int(a.farther_due_to_dry.sum()),
        'farther_due_to_dry_ids': a.ID[a.farther_due_to_dry].tolist(),
        'nearest_any_unknown': a.ID[a.nearest_any_node.isna()].tolist(),
        'km_median': float(a.km.median()), 'km_max': float(a.km.max()),
        'km_ctl_source_count': int((a.km_source != 'exact (laptop mesh)').sum()),
        'cmp_n': len(cmp_), 'flagged': int(cmp_.flag.sum()),
        'flagged_ids': cmp_.ID[cmp_.flag].tolist(),
        'rmse_median': float(cmp_.rmse.median()), 'rmse_max': float(cmp_.rmse.max()),
        'bias_median': float(cmp_.bias.median()), 'corr_median': float(cmp_['corr'].median()),
        'corr_min': float(cmp_['corr'].min()),
        'n_common_set': sorted(set(cmp_.n.tolist())),
        'best_lag_counts': {str(k): int(v) for k, v in cmp_.best_lag_h.value_counts().items()},
        'series_ok_all': bool(all(m and dup == 0 and st == [60] for m, dup, st in
                                  zip(good.monotonic, good.duplicates, good.steps_min))),
        'series_n_set': sorted({int(x) for x in good.n}),
        'series_start_set': sorted(set(good.start)), 'series_end_set': sorted(set(good.end)),
        'series_no_prd': checks.ID[checks.get('n').isna()].tolist() if 'n' in checks else [],
        'nan_total_window': int(nan.nan_window.sum()), 'nan_total_all': int(nan.nan_all.sum()),
        'nan_stations': nan.ID[nan.nan_all > 0].tolist(),
        'skill_rows_fields': len(sf), 'skill_rows_stations': len(ss),
        'no_skill_fields': sorted(set(a.ID[a.matched_fields]) - set(sf.ID)),
        'no_skill_stations': sorted(set(a.ID[a.matched_stations]) - set(ss.ID)),
    }


def laptop_compare(a_now, mf, ms):
    """Item h: cluster nowcast vs laptop nowcast."""
    lmf = model_ctl(f'{LR}/snapshot_a/control_files/stofs_2d_glo_wl_model.ctl')
    lms = model_ctl(f'{LR}/snapshot_b/control_files/stofs_2d_glo_wl_model_station.ctl')
    lof = obs_ctl(f'{LR}/snapshot_a/control_files/stofs_2d_glo_wl_station.ctl')
    cof = obs_ctl(f'{CR}/{RUNS["fn"][0]}/control_files/stofs_2d_glo_wl_station.ctl')
    res = {}
    for tag, cl, lp in (('fields_ctl', mf, lmf), ('stations_ctl', ms, lms)):
        m = cl.merge(lp, on='ID', how='outer', suffixes=('_cluster', '_laptop'), indicator=True)
        diff = m[(m._merge != 'both') | (m.node_cluster != m.node_laptop)
                 | (m.nlat_cluster != m.nlat_laptop) | (m.nlon_cluster != m.nlon_laptop)]
        res[tag] = {'rows_cluster': len(cl), 'rows_laptop': len(lp),
                    'differences': diff.drop(columns='_merge').to_dict('records')}
    oc = cof.merge(lof, on='ID', how='outer', suffixes=('_cluster', '_laptop'), indicator=True)
    res['obs_ctl'] = {'rows_cluster': len(cof), 'rows_laptop': len(lof),
                      'differences': oc[(oc._merge != 'both') | (oc.lat_cluster != oc.lat_laptop)
                                        | (oc.lon_cluster != oc.lon_laptop)]
                      .drop(columns='_merge').to_dict('records')}
    kmd = a_now.merge(lap[['ID', 'km']], on='ID', suffixes=('', '_laptop'))
    res['km_maxabs_diff'] = float((kmd.km - kmd.km_laptop).abs().max())
    series = []
    for ftype, sub in (('fields', 'fn'), ('stations', 'sn')):
        for f in sorted(glob.glob(f'{CR}/{RUNS[sub][0]}/data/model/1d_node/*_nowcast_{ftype}_model.prd')):
            base = os.path.basename(f)
            lf = f'{LR}/data/model/1d_node/{base}'
            if not os.path.exists(lf):
                series.append({'file': base, 'status': 'missing on laptop'})
                continue
            c, l_ = prd(f), prd(lf)
            same_t = c.index.equals(l_.index)
            md = float(np.nanmax(np.abs(c.values - l_.values))) if same_t else np.nan
            if not same_t or md > 0:
                series.append({'file': base, 'status': 'differs', 'same_times': same_t,
                               'n_cluster': len(c), 'n_laptop': len(l_), 'maxabs_diff': md})
        lap_files = {os.path.basename(p) for p in
                     glob.glob(f'{LR}/data/model/1d_node/*_nowcast_{ftype}_model.prd')}
        cl_files = {os.path.basename(p) for p in
                    glob.glob(f'{CR}/{RUNS[sub][0]}/data/model/1d_node/*_nowcast_{ftype}_model.prd')}
        for base in sorted(lap_files - cl_files):
            series.append({'file': base, 'status': 'missing on cluster'})
    res['prd_files_compared'] = {
        'fields': len(glob.glob(f'{CR}/{RUNS["fn"][0]}/data/model/1d_node/*_nowcast_fields_model.prd')),
        'stations': len(glob.glob(f'{CR}/{RUNS["sn"][0]}/data/model/1d_node/*_nowcast_stations_model.prd'))}
    res['prd_differences'] = series
    obs_diff = []
    for f in sorted(glob.glob(f'{CR}/{RUNS["fn"][0]}/data/observations/1d_station/*.obs')):
        lf = f'{LR}/data/observations/1d_station/{os.path.basename(f)}'
        if not os.path.exists(lf):
            obs_diff.append({'file': os.path.basename(f), 'status': 'missing on laptop'})
        else:
            # compare text with line endings normalized (laptop files are CRLF, cluster LF)
            c_txt = open(f, encoding='utf-8').read().replace('\r\n', '\n').strip()
            l_txt = open(lf, encoding='utf-8').read().replace('\r\n', '\n').strip()
            if c_txt != l_txt:
                obs_diff.append({'file': os.path.basename(f), 'status': 'content differs'})
    res['obs_file_differences'] = obs_diff
    sk = {}
    for ftype, sub, snap in (('fields', 'fn', 'snapshot_a'), ('stations', 'sn', 'snapshot_b')):
        c = skill(f'{CR}/{RUNS[sub][0]}', 'nowcast', ftype).set_index('ID')
        l_ = pd.read_csv(f'{LR}/{snap}/stats/skill_stofs_2d_glo_water_level_nowcast_{ftype}.csv',
                         dtype={'ID': str}).set_index('ID')
        ids = c.index.union(l_.index)
        d = []
        for i in ids:
            if i not in c.index or i not in l_.index:
                d.append({'ID': i, 'status': 'cluster only' if i in c.index else 'laptop only'})
                continue
            for col in SKILL_COLS:
                x, y = c.at[i, col], l_.at[i, col]
                if not (pd.isna(x) and pd.isna(y)) and (pd.isna(x) or pd.isna(y)
                                                         or abs(float(x) - float(y)) > 1e-9):
                    d.append({'ID': i, 'col': col, 'cluster': x, 'laptop': y})
        sk[ftype] = {'rows_cluster': len(c), 'rows_laptop': len(l_), 'differences': d}
    res['skill_differences'] = sk
    return res


def extras(per, cmps, skills):
    """Numbers quoted in the doc text: node changes, dry shifts, offsets, flag overlaps."""
    n, f = per['nowcast'].set_index('ID'), per['forecast_a'].set_index('ID')
    chg = f[f.node != n.node]
    res = {'node_changes_nowcast_to_forecast_a': [
        {'ID': i, 'now_node': int(n.node[i]), 'now_km': round(float(n.km[i]), 3),
         'fa_node': int(r.node), 'fa_km': round(float(r.km), 3), 'fa_km_source': r.km_source,
         'nearest_any_node': int(r.nearest_any_node),
         'nearest_any_km': round(float(r.nearest_any_km), 3),
         'now_dry': bool(n.farther_due_to_dry[i]), 'fa_dry': bool(r.farther_due_to_dry)}
        for i, r in chg.iterrows()]}
    for lab in ('nowcast', 'forecast_a'):
        a, c, s = per[lab], cmps[lab], skills[lab]
        d = a[a.farther_due_to_dry]
        shift = d.km - d.nearest_any_km
        fl = c[c.flag].assign(absb=lambda x: x.bias.abs()).sort_values('absb', ascending=False)
        names = a.set_index('ID').Name
        m = c[c.flag].merge(s, on='ID', how='left')
        has = m.bias_fields.notna() & m.bias_stations.notna()
        f_closer = has & (m.bias_fields.abs() < m.bias_stations.abs())
        s_closer = has & (m.bias_stations.abs() < m.bias_fields.abs())
        res[lab] = {
            'dry_shift_km_min': round(float(shift.min()), 3),
            'dry_shift_km_max': round(float(shift.max()), 3),
            'km_over_1': a.ID[a.km > 1].tolist(),
            'short_overlap': c[c.n != c.n.max()][['ID', 'n']].to_dict('records'),
            'lag_not_zero': c[c.best_lag_h != 0][['ID', 'best_lag_h', 'corr', 'bias']]
            .to_dict('records'),
            'top_offsets': [{'ID': i, 'Name': names[i], 'bias': round(b, 3), 'corr': round(r, 4)}
                            for i, b, r in zip(fl.ID[:6], fl.bias[:6], fl['corr'][:6])],
            'flagged_by_corr': c.ID[c['corr'] < 0.95].tolist(),
            'flagged_and_dry': sorted(set(c.ID[c.flag]) & set(d.ID)),
            'flagged_with_both_skill_rows': int(has.sum()),
            'flagged_stations_closer_to_obs': int(s_closer.sum()),
            'flagged_fields_closer_to_obs': m.ID[f_closer].tolist(),
        }
    cn, cf = cmps['nowcast'].set_index('ID'), cmps['forecast_a'].set_index('ID')
    both = cn.join(cf, lsuffix='_now', rsuffix='_fa', how='inner')
    res['bias_corr_nowcast_vs_forecast_a'] = round(float(both.bias_now.corr(both.bias_fa)), 3)
    res['flag_nowcast_only'] = sorted(set(cn.index[cn.flag]) - set(cf.index[cf.flag]))
    res['flag_forecast_a_only'] = sorted(set(cf.index[cf.flag]) - set(cn.index[cn.flag]))
    return res


os.makedirs(OUT, exist_ok=True)
summary = {}
per, cmps, skills = {}, {}, {}
a_now = None
for label, kf, ks in (('nowcast', 'fn', 'sn'), ('forecast_a', 'fa', 'sa')):
    a, of, os_, mf, ms = per_station(kf, ks)
    cmp_, sk, nan, checks, sf, ss = compare(kf, ks, a)
    a.to_csv(f'{OUT}/{label}_per_station.csv', index=False)
    cmp_.to_csv(f'{OUT}/{label}_series_compare.csv', index=False)
    sk.to_csv(f'{OUT}/{label}_skill_side_by_side.csv', index=False)
    nan.to_csv(f'{OUT}/{label}_nan.csv', index=False)
    checks.to_csv(f'{OUT}/{label}_series_checks.csv', index=False)
    s = summarize(a, cmp_, nan, checks, sf, ss, ms)
    s['obs_sets_equal'] = set(of.ID) == set(os_.ID)
    summary[label] = s
    per[label], cmps[label], skills[label] = a, cmp_, sk
    if label == 'nowcast':
        a_now, mf_now, ms_now = a, mf, ms
summary['laptop_compare'] = laptop_compare(a_now, mf_now, ms_now)
summary['extras'] = extras(per, cmps, skills)
json.dump(summary, open(f'{OUT}/summary.json', 'w'), indent=1, default=str)
print(json.dumps(summary, indent=1, default=str)[:12000])
