# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    OISST V2.1 Anomaly Plotter v3.2                           ║
║  [ CACHING · REGION REGISTRY · TREND · PERCENTILE RANK                     ] ║
║  [            MULTI-YEAR PANEL · DISTRIBUTIONS · INSET MAPS                ] ║
╚══════════════════════════════════════════════════════════════════════════════╝

Modes of Operation:
───────────────────
  'single'          : Standard SST anomaly map for a specific date.
  'difference'      : Difference map between TARGET_DATE and BASELINE_DATE.
  'records'         : Anomaly map with record warm/cold stippling.
  'percentile_rank' : Color each ocean cell by historical percentile rank (0–100).
  'trend'           : Linear trend (°C/decade) map over the full OISST record.
  'multi_year_panel': Same-DOY anomaly maps tiled across COMPARISON_YEARS.

Changes in v3.2:
────────────────
  • Timeseries mode removed entirely (plot_timeseries, _fetch_one_year_ts,
    fetch_annual_regional_ts, _add_region_inset_ts, running_mean_1d,
    all TS_* config variables, CURRENT_YEAR_CACHE_DAYS).

Data Source:
────────────
  NOAA PSL OPeNDAP — noaa.oisst.v2.highres (1/4° daily, 1982–present)
"""

import os, sys, pickle, warnings, time
from datetime import datetime

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.util import add_cyclic_point
from scipy.ndimage import gaussian_filter
from scipy.stats import t as t_dist

warnings.filterwarnings('ignore')
plt.rcParams.update({'mathtext.default': 'regular'})

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG  ─ edit these before running
# ══════════════════════════════════════════════════════════════════════════════

MODE             = 'single'   # single | difference | records |
                                       # percentile_rank | trend |
                                       # multi_year_panel
TARGET_DATE      = '4/28/2026'         # MM/DD/YYYY — used by every mode
BASELINE_DATE    = '4/18/2015'         # difference mode only
REGION           = 'custom'            # key into REGION_REGISTRY  (all map modes)
THEME            = 'light'             # 'dark' | 'light'

REMOVE_GLOBAL_MEAN    = True           # Subtract global mean → RSST
SHOW_OCEANIC_INDICES  = False          # Oceanic index text box on map modes
SHOW_PCT_IN_INDICES   = False          # Append historical percentile to each index
SHOW_INSET_MAP        = False          # Global reference inset on all map modes
SHOW_PCT_OVERLAY      = False          # Overlay 10th/90th pct contours on anomaly maps

# ── multi_year_panel settings ────────────────────────────────────────────────
COMPARISON_YEARS   = [2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026]
PANEL_COLS         = 4

# ── trend settings ───────────────────────────────────────────────────────────
TREND_SMOOTH_SIGMA = 1.5
TREND_SHOW_SIG     = True

# ── cache settings ───────────────────────────────────────────────────────────
CACHE_DIR          = './oisst_cache'
USE_CACHE          = True
MAX_RETRIES        = 5

# ══════════════════════════════════════════════════════════════════════════════
#  REGION REGISTRY
# ══════════════════════════════════════════════════════════════════════════════

REGION_REGISTRY = {
    'global': {
        'label'      : 'Global Ocean',
        'extent'     : None,
        'central_lon': 180,
        'avg_box'    : {'lat': (-90,  90), 'lon': (  0, 360)},
        'inset_hl'   : None,
    },
    'north_atlantic': {
        'label'      : 'North Atlantic',
        'extent'     : [-100, 20, 0, 65],
        'central_lon': -40,
        'avg_box'    : {'lat': (  0,  65), 'lon': (260, 380)},
        'inset_hl'   : [-100, 20, 0, 65],
    },
    'tropics': {
        'label'      : 'Tropics (30°S–30°N)',
        'extent'     : [-180, 180, -30, 30],
        'central_lon': 180,
        'avg_box'    : {'lat': (-30,  30), 'lon': (  0, 360)},
        'inset_hl'   : [-180, 180, -30, 30],
    },
    'north_pacific': {
        'label'      : 'North Pacific',
        'extent'     : [100, 260, -20, 70],
        'central_lon': 180,
        'avg_box'    : {'lat': (-20,  70), 'lon': (100, 260)},
        'inset_hl'   : [100, 260, -20, 70],
    },
    'custom': {
        'label'      : 'ENP',
        'extent'     : [-180, 0,  -20,  70],
        'central_lon': -120,
        'avg_box'    : {'lat': (10,  33), 'lon': (240, 260)},
        'inset_hl'   : [-180, 0, -20, 70],
    },
    'MDR': {
        'label'      : 'Main Development Region',
        'extent'     : [-90, -10, 3, 28],
        'central_lon': -50,
        'avg_box'    : {'lat': ( 10,  20), 'lon': (300, 340)},
        'inset_hl'   : [-60, -20, 10, 20],
    },
    'nino34': {
        'label'      : 'Niño 3.4',
        'extent'     : [-180, -70, -15, 15],
        'central_lon': -125,
        'avg_box'    : {'lat': ( -5,   5), 'lon': (190, 240)},
        'inset_hl'   : [-170, -120, -5, 5],
    },
    'nino12': {
        'label'      : 'Niño 1+2',
        'extent'     : [-100, -60, -15, 5],
        'central_lon': -80,
        'avg_box'    : {'lat': (-10,   0), 'lon': (270, 280)},
        'inset_hl'   : [-90, -80, -10, 0],
    },
    'indian_ocean': {
        'label'      : 'Indian Ocean',
        'extent'     : [30, 120, -40, 30],
        'central_lon': 75,
        'avg_box'    : {'lat': (-40,  30), 'lon': ( 30, 120)},
        'inset_hl'   : [30, 120, -40, 30],
    },
    'gulf_caribbean': {
        'label'      : 'Gulf of Mexico & Caribbean',
        'extent'     : [-100, -55, 8, 32],
        'central_lon': -77,
        'avg_box'    : {'lat': (  8,  32), 'lon': (260, 305)},
        'inset_hl'   : [-100, -55, 8, 32],
    },
    'arctic': {
        'label'      : 'Arctic Ocean',
        'extent'     : [-180, 180, 55, 90],
        'central_lon': 0,
        'avg_box'    : {'lat': ( 55,  90), 'lon': (  0, 360)},
        'inset_hl'   : None,
    },
    'southern_ocean': {
        'label'      : 'Southern Ocean',
        'extent'     : [-180, 180, -90, -45],
        'central_lon': 0,
        'avg_box'    : {'lat': (-90, -45), 'lon': (  0, 360)},
        'inset_hl'   : None,
    },
}

# ══════════════════════════════════════════════════════════════════════════════
#  OCEANIC INDEX DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

SIMPLE_INDICES = {
    "Niño 1+2":       {"lat": (-10.0,  0.0), "lon": (270.0, 280.0), "subtract_gm": False},
    "Niño 3":         {"lat": ( -5.0,  5.0), "lon": (210.0, 270.0), "subtract_gm": False},
    "Niño 3.4 / ONI": {"lat": ( -5.0,  5.0), "lon": (190.0, 240.0), "subtract_gm": False},
    "Niño 4":         {"lat": ( -5.0,  5.0), "lon": (160.0, 210.0), "subtract_gm": False},
    "TNA":            {"lat": ( 5.5,  23.5), "lon": (302.5, 345.0), "subtract_gm": False},
    "MDR":            {"lat": (10.0,  20.0), "lon": (300.0, 340.0), "subtract_gm": False},
}

DIPOLE_INDICES = {
    "IOD* (W−E)": {
        "west": {"lat": (-10.0, 10.0), "lon": ( 50.0,  70.0)},
        "east": {"lat": (-10.0,  0.0), "lon": ( 90.0, 110.0)},
    },
}

INDEX_GROUPS = {
    "── ENSO ──":     ["Niño 1+2", "Niño 3", "Niño 3.4 / ONI", "Niño 4"],
    "── Atlantic ──": ["TNA", "MDR"],
    "── Indian ──":   [],
}

# ══════════════════════════════════════════════════════════════════════════════
#  THEMES & COLORMAPS
# ══════════════════════════════════════════════════════════════════════════════

THEMES = {
    'dark': {
        'fig_bg':        'black',
        'ax_bg':         '#0d0d1a',
        'title_color':   'white',
        'cbar_label':    'white',
        'cbar_ticks':    'white',
        'gridline':      'grey',
        'gridline_alpha': 0.35,
        'label_color':   'white',
        'coast_color':   'black',
        'border_color':  '#888888',
        'watermark_fg':  'white',
        'watermark_bg':  'black',
        'zero_line':     'white',
        'land_fill':     'dimgray',
        'ocean_fill':    '#1a1a3a',
        'inset_ocean':   '#0a0a1e',
    },
    'light': {
        'fig_bg':        'white',
        'ax_bg':         '#ffffff',
        'title_color':   'black',
        'cbar_label':    'black',
        'cbar_ticks':    'black',
        'gridline':      'black',
        'gridline_alpha': 0.5,
        'label_color':   'black',
        'coast_color':   'black',
        'border_color':  'black',
        'watermark_fg':  'black',
        'watermark_bg':  'white',
        'zero_line':     'black',
        'land_fill':     '#c8b99a',
        'ocean_fill':    '#c8e8f0',
        'inset_ocean':   '#c8e8f0',
    },
}


def get_sst_cmap():
    colors = ['#FEFFF6', '#8D00D6', '#3C29B6', '#0E4B8E', '#1F6EF1', '#2B85F5',
              '#53A6F4', '#9BD1FF', '#E0FFFF', '#FFFFFF', '#FFEB7E', '#FCC439',
              '#FEA100', '#F06600', '#FC3403', '#E3180A', '#BD0402', '#E9067C',
              '#FCFCFC']
    return mcolors.LinearSegmentedColormap.from_list("custom_sst", colors, N=256)

get_custom_cmap = get_sst_cmap


def get_percentile_cmap():
    anchors = [
        (0.00, '#08306B'),
        (0.05, '#2171B5'),
        (0.10, '#6BAED6'),
        (0.25, '#C6DBEF'),
        (0.50, '#FFFFFF'),
        (0.75, '#FCAE91'),
        (0.90, '#FB6A4A'),
        (0.95, '#CB181D'),
        (1.00, '#67000D'),
    ]
    return mcolors.LinearSegmentedColormap.from_list(
        "pct_rank", [(v, c) for v, c in anchors], N=256)


def _theme():
    return THEMES.get(THEME, THEMES['dark'])


# ══════════════════════════════════════════════════════════════════════════════
#  CACHE SYSTEM
# ══════════════════════════════════════════════════════════════════════════════

def _cache_path(key):
    os.makedirs(CACHE_DIR, exist_ok=True)
    safe = ''.join(c if c.isalnum() or c in '_-.' else '_' for c in key)
    return os.path.join(CACHE_DIR, safe + '.pkl')


def cache_save(key, data, quiet=False):
    if not USE_CACHE:
        return
    try:
        with open(_cache_path(key), 'wb') as f:
            pickle.dump({'data': data, 'ts': datetime.now()}, f, protocol=4)
        sz = os.path.getsize(_cache_path(key)) / 1e6
        if not quiet:
            print(f"  [cache] Saved  → {key}  ({sz:.1f} MB)")
    except Exception as e:
        if not quiet:
            print(f"  [cache] Save failed ({key}): {e}")


def cache_load(key, max_age_days=None):
    if not USE_CACHE:
        return None
    p = _cache_path(key)
    if not os.path.exists(p):
        return None
    try:
        with open(p, 'rb') as f:
            payload = pickle.load(f)
        if max_age_days is not None:
            age = (datetime.now() - payload.get('ts', datetime.min)).days
            if age > max_age_days:
                print(f"  [cache] Stale ({age}d > {max_age_days}d), ignoring: {key}")
                return None
        return payload['data']
    except Exception as e:
        print(f"  [cache] Load failed ({key}): {e}")
        return None


def cache_clear(pattern=None):
    if not os.path.isdir(CACHE_DIR):
        print("  [cache] No cache directory found.")
        return
    n = 0
    for fname in os.listdir(CACHE_DIR):
        if fname.endswith('.pkl'):
            if pattern is None or pattern in fname:
                os.remove(os.path.join(CACHE_DIR, fname))
                n += 1
    print(f"  [cache] Removed {n} file(s).")


def cache_list():
    if not os.path.isdir(CACHE_DIR):
        print("  [cache] Empty."); return
    files = sorted(f for f in os.listdir(CACHE_DIR) if f.endswith('.pkl'))
    if not files:
        print("  [cache] Empty."); return
    for fname in files:
        sz = os.path.getsize(os.path.join(CACHE_DIR, fname)) / 1e6
        print(f"  {fname:<60}  {sz:6.1f} MB")


# ══════════════════════════════════════════════════════════════════════════════
#  SHARED UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def _print_progress(iteration, total, prefix='', length=30):
    if total == 0:
        return
    percent = f"{100 * (iteration / float(total)):.1f}"
    filled = int(length * iteration // total)
    bar = '█' * filled + '░' * (length - filled)
    sys.stdout.write(f'\r{prefix} |{bar}| {percent}% ')
    sys.stdout.flush()
    if iteration == total:
        sys.stdout.write('\n')
        sys.stdout.flush()


def smooth_nan(arr, sigma=0.8):
    if sigma == 0:
        return arr
    V = arr.copy();  V[np.isnan(V)] = 0
    W = np.ones_like(arr); W[np.isnan(arr)] = 0
    with np.errstate(invalid='ignore'):
        out = gaussian_filter(V, sigma) / gaussian_filter(W, sigma)
    out[np.isnan(arr)] = np.nan
    return out


def _reg_info(key=None):
    return REGION_REGISTRY.get(key or REGION, REGION_REGISTRY['global'])


def _to_canonical_date(ts):
    if ts.month == 2 and ts.day == 29:
        return None
    return pd.Timestamp(2001, ts.month, ts.day)


# ══════════════════════════════════════════════════════════════════════════════
#  OISST DATA ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def _oisst_url(year, var='anom'):
    return (f'https://psl.noaa.gov/thredds/dodsC/Datasets/'
            f'noaa.oisst.v2.highres/sst.day.{var}.{year}.nc')


def _global_mean(da):
    w  = np.cos(np.deg2rad(da.lat.values))
    w2 = np.tile(w[:, None], (1, da.shape[1]))
    valid = np.where(~np.isnan(da.values), w2, np.nan)
    return float(np.nansum(da.values * w2) / np.nansum(valid))


def fetch_sst_anomaly(date_str, remove_gm=False, quiet=False):
    """
    Fetch one day's SST anomaly.  Individual-date results are cached.
    Returns: plot_arr, raw_arr, lats, lons, date, global_mean
    """
    date = pd.to_datetime(date_str)
    ck   = f"anom_{date.strftime('%Y%m%d')}"
    hit  = cache_load(ck)
    newly_fetched = False

    if hit is None:
        newly_fetched = True
        if not quiet:
            print(f"  [download] {date.strftime('%Y-%m-%d')} …", end=' ', flush=True)

        for attempt in range(MAX_RETRIES):
            try:
                ds = xr.open_dataset(_oisst_url(date.year, 'anom'))
                da = ds['anom'].sel(time=date, method='nearest').squeeze()
                fetched = pd.to_datetime(da.time.values)
                if fetched.date() != date.date():
                    ds.close()
                    raise ValueError(
                        f"Date mismatch: requested {date.date()}, nearest is {fetched.date()}")
                raw = da.values.copy()
                gm  = _global_mean(da)
                lats, lons = da.lat.values.copy(), da.lon.values.copy()
                ds.close()
                if not quiet:
                    print("done")
                hit = {'raw': raw, 'lats': lats, 'lons': lons, 'gm': gm}
                cache_save(ck, hit, quiet=quiet)
                break
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    print(f"\n  [warn] DAP failed for {date.strftime('%Y-%m-%d')}. "
                          f"Retrying ({attempt+1}/{MAX_RETRIES})...", end=' ')
                    time.sleep(2 ** attempt)
                else:
                    raise RuntimeError(
                        f"Failed to fetch {date_str} after {MAX_RETRIES} attempts: {e}")

    raw, lats, lons, gm = hit['raw'], hit['lats'], hit['lons'], hit['gm']
    plot_arr = raw - gm if remove_gm else raw
    if remove_gm and newly_fetched and not quiet:
        print(f"  → Global mean removed: {gm:+.4f}°C")
    return plot_arr, raw, lats, lons, date, gm


def fetch_doy_stack(target_date_str, remove_gm=False):
    """
    Build a (n_years × lat × lon) stack of anomalies for the same DOY across
    ALL available years 1982–present.  Results cached by (DOY, remove_gm).

    Returns dict: {'stack': ndarray, 'years': list[int], 'lats': ndarray, 'lons': ndarray}
    """
    target   = pd.to_datetime(target_date_str)
    doy      = target.dayofyear
    cur_yr   = datetime.now().year
    ck       = f"doy_stack_doy{doy:03d}_gm{int(remove_gm)}"
    cached   = cache_load(ck)

    if cached is not None:
        if cur_yr not in cached['years']:
            cur_date = pd.Timestamp(year=cur_yr, month=target.month, day=target.day)
            try:
                _, raw, lats, lons, _, gm = fetch_sst_anomaly(
                    cur_date.strftime('%m/%d/%Y'), remove_gm=False, quiet=True)
                arr = raw - gm if remove_gm else raw
                cached['stack'] = np.concatenate(
                    [cached['stack'], arr[np.newaxis]], axis=0)
                cached['years'].append(cur_yr)
                cache_save(ck, cached, quiet=True)
            except Exception as e:
                print(f"  [warn] Could not append {cur_yr}: {e}")
        return cached

    start_yr = 1982
    arrays, years, lats, lons = [], [], None, None
    years_to_fetch = list(range(start_yr, cur_yr + 1))
    total_yrs = len(years_to_fetch)

    print(f"\n  [download] DOY={doy} stack, {start_yr}–{cur_yr}…")

    for i, yr in enumerate(years_to_fetch):
        _print_progress(i, total_yrs, prefix="  [progress]")
        is_leap = pd.Timestamp(yr, 1, 1).is_leap_year
        if target.month == 2 and target.day == 29 and not is_leap:
            continue
        try:
            dt = pd.Timestamp(yr, target.month, target.day)
            _, raw, _lats, _lons, _, gm = fetch_sst_anomaly(
                dt.strftime('%m/%d/%Y'), remove_gm=False, quiet=True)
            arr = raw - gm if remove_gm else raw
            if lats is None:
                lats, lons = _lats, _lons
            arrays.append(arr)
            years.append(yr)
        except Exception:
            pass

    _print_progress(total_yrs, total_yrs, prefix="  [progress]")
    print(f"  → {len(years)} years loaded.")
    result = {'stack': np.stack(arrays, axis=0), 'years': years,
              'lats': lats, 'lons': lons}
    cache_save(ck, result, quiet=True)
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  OCEANIC INDEX ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def _region_mean(arr, lats, lons, lat_min, lat_max, lon_min, lon_max):
    lat_ok = (lats >= lat_min) & (lats <= lat_max)
    lon_ok = (lons >= lon_min) & (lons <= lon_max)
    mask   = lat_ok[:, None] & lon_ok[None, :]
    w      = np.cos(np.deg2rad(lats))
    w2d    = np.where(mask, np.tile(w[:, None], (1, len(lons))), np.nan)
    masked = np.where(mask, arr, np.nan)
    denom  = np.nansum(w2d)
    return np.nansum(masked * w2d) / denom if denom > 0 else np.nan


def compute_oceanic_indices(raw_arr, lats, lons, gm,
                            hist_stack=None, hist_years=None):
    results = {}

    for name, cfg in SIMPLE_INDICES.items():
        val = _region_mean(raw_arr, lats, lons,
                           cfg['lat'][0], cfg['lat'][1],
                           cfg['lon'][0], cfg['lon'][1])
        if cfg['subtract_gm']:
            val -= gm

        pct = None
        if hist_stack is not None and not np.isnan(val):
            hist_vals = np.array([
                _region_mean(hist_stack[i], lats, lons,
                             cfg['lat'][0], cfg['lat'][1],
                             cfg['lon'][0], cfg['lon'][1])
                for i in range(hist_stack.shape[0])
            ])
            valid_hist = hist_vals[~np.isnan(hist_vals)]
            if len(valid_hist) > 0:
                pct = float(np.sum(valid_hist < val) / len(valid_hist) * 100)

        results[name] = {'val': val, 'pct': pct}

    for name, cfg in DIPOLE_INDICES.items():
        w_val = _region_mean(raw_arr, lats, lons,
                             cfg['west']['lat'][0], cfg['west']['lat'][1],
                             cfg['west']['lon'][0], cfg['west']['lon'][1])
        e_val = _region_mean(raw_arr, lats, lons,
                             cfg['east']['lat'][0], cfg['east']['lat'][1],
                             cfg['east']['lon'][0], cfg['east']['lon'][1])
        results[name] = {'val': w_val - e_val, 'pct': None}

    if "Niño 3.4 / ONI" in results:
        results['RONI'] = {'val': results["Niño 3.4 / ONI"]['val'] - gm, 'pct': None}

    return results


def _build_index_text(index_results, gm):
    sep = '─' * 44
    if REMOVE_GLOBAL_MEAN:
        header = f"RSST Indices (°C)\n  Global mean removed : {gm:+.3f}°C\n{sep}"
    else:
        header = f"Raw SST Indices (°C)\n  Global mean : {gm:+.3f}°C\n{sep}"

    lines = [header]
    for group, keys in INDEX_GROUPS.items():
        grp_lines = []
        for k in keys:
            if k in index_results:
                r       = index_results[k]
                pct_str = (f"  ({r['pct']:5.1f}th pct)"
                           if SHOW_PCT_IN_INDICES and r['pct'] is not None
                           else "")
                grp_lines.append(f"  {k:<22}: {r['val']:+.2f}{pct_str}")
        if grp_lines:
            lines.append(f"\n{group}")
            lines.extend(grp_lines)

    for dk, dv in DIPOLE_INDICES.items():
        if dk in index_results:
            lines.append(f"\n{dk:<22}: {index_results[dk]['val']:+.2f}")

    if 'RONI' in index_results:
        lines.append(f"\n  {'RONI (3.4 − GM)':<22}: {index_results['RONI']['val']:+.2f}")

    lines.append("\n* = approximation")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  MAP LAYOUT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _base_fig(region_key=None):
    T   = _theme()
    reg = _reg_info(region_key)
    ext = reg['extent']
    clon = reg['central_lon']

    proj  = ccrs.PlateCarree(central_longitude=clon)
    trans = ccrs.PlateCarree()
    fig   = plt.figure(figsize=(16, 8), facecolor=T['fig_bg'])
    ax    = plt.axes(projection=proj)
    ax.set_facecolor(T['ax_bg'])

    if ext:
        ax.set_extent(ext, crs=trans)
    else:
        ax.set_global()

    for sp in ax.spines.values():
        sp.set_edgecolor(T['border_color'])
        sp.set_linewidth(1.0)

    return fig, ax, proj, trans, ext


def _add_map_features(ax, region_key=None):
    T   = _theme()
    reg = _reg_info(region_key)
    has_extent = reg['extent'] is not None

    ax.coastlines(color=T['coast_color'], linewidth=0.6, zorder=3)
    
    ax.add_feature(cfeature.STATES, linewidth=0.4)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3,
                   edgecolor=T['border_color'], alpha=0.5, zorder=3)
    ax.add_feature(cfeature.LAND, facecolor=T['land_fill'],
                   edgecolor=T['border_color'], zorder=2)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=has_extent,
                      linewidth=0.5, color=T['gridline'],
                      alpha=T['gridline_alpha'], linestyle='--', zorder=4)
    if has_extent:
        gl.top_labels = gl.right_labels = False
        gl.xlabel_style = {'color': T['label_color'], 'fontsize': 9}
        gl.ylabel_style = {'color': T['label_color'], 'fontsize': 9}
        gl.xformatter  = LONGITUDE_FORMATTER
        gl.yformatter  = LATITUDE_FORMATTER
    else:
        ax.set_xticks(np.arange(-180, 181, 40), crs=ccrs.PlateCarree())
        ax.set_yticks(np.arange( -90,  91, 30), crs=ccrs.PlateCarree())
        import cartopy.mpl.ticker as cticker
        ax.xaxis.set_major_formatter(cticker.LongitudeFormatter())
        ax.yaxis.set_major_formatter(cticker.LatitudeFormatter())
        ax.tick_params(labelsize=9, colors=T['label_color'])


def _add_watermark(ax, text='Data: NOAA OISST v2.1 (PSL)'):
    T = _theme()
    box = dict(boxstyle='round,pad=0.5', facecolor=T['watermark_bg'],
               alpha=0.85, edgecolor=T['border_color'], linewidth=0.8)
    ax.text(0.985, 0.03, text, transform=ax.transAxes,
            fontsize=9, fontweight='bold', color=T['watermark_fg'],
            va='bottom', ha='right', bbox=box, zorder=106)


def _styled_colorbar(fig, ax, cf, label, ticks=None):
    T   = _theme()
    fmt = ticker.FuncFormatter(lambda x, _: f'{x:.1f}')
    cbar = plt.colorbar(cf, ax=ax, orientation='horizontal',
                        pad=0.06, fraction=0.046, aspect=40, shrink=0.75,
                        format=fmt, extend='both')
    if ticks is not None:
        cbar.set_ticks(ticks)
    cbar.set_label(label, color=T['cbar_label'], fontsize=10,
                   labelpad=4, fontweight='bold')
    cbar.ax.tick_params(labelsize=9, colors=T['cbar_ticks'])
    return cbar


def _title(ax, main, sub):
    T = _theme()
    plt.title(f"{main}\n", loc='center', color=T['title_color'],
              fontsize=14, fontweight='bold')
    ax.text(0.5, 1.015, sub, transform=ax.transAxes,
            color=T['title_color'], fontsize=11, ha='center', va='bottom')


def _style_legend(leg):
    T = _theme()
    leg.get_frame().set_facecolor(T['watermark_bg'])
    leg.get_frame().set_edgecolor(T['border_color'])
    for txt in leg.get_texts():
        txt.set_color(T['label_color'])


def _contourf_map(ax, arr, lats, lons, levels, cmap, transform, sigma=0.8):
    arr_s, lons_c = add_cyclic_point(smooth_nan(arr, sigma), coord=lons)
    lon2d, lat2d  = np.meshgrid(lons_c, lats)
    return ax.contourf(lon2d, lat2d, arr_s, levels=levels, cmap=cmap,
                       extend='both', transform=transform, zorder=1), lon2d, lat2d


def _add_region_inset(fig, ax, highlight_lonlat=None):
    T = _theme()
    fig.canvas.draw()
    bb   = ax.get_position()
    iw, ih = 0.11, 0.16
    ax_in = fig.add_axes([bb.x0 + 0.008, bb.y0 + 0.008, iw, ih],
                          projection=ccrs.Robinson(central_longitude=0))
    ax_in.set_global()
    ax_in.set_facecolor(T['inset_ocean'])
    ax_in.add_feature(cfeature.LAND, facecolor=T['land_fill'],
                      edgecolor='none', zorder=1)
    ax_in.coastlines(resolution='110m', linewidth=0.25,
                     color=T['coast_color'], zorder=2)

    if highlight_lonlat:
        lon_w, lon_e, lat_s, lat_n = highlight_lonlat
        if lon_e > lon_w:
            lons_b = [lon_w, lon_e, lon_e, lon_w, lon_w]
            lats_b = [lat_s, lat_s, lat_n, lat_n, lat_s]
            ax_in.fill(lons_b, lats_b, transform=ccrs.PlateCarree(),
                       color='#FF3333', alpha=0.45, zorder=3)
            ax_in.plot(lons_b, lats_b, transform=ccrs.PlateCarree(),
                       color='#FF3333', linewidth=0.9, zorder=4)

    ax_in.set_xticks([]); ax_in.set_yticks([])
    for sp in ax_in.spines.values():
        sp.set_edgecolor(T['border_color']); sp.set_linewidth(0.6)
    return ax_in


def _add_pct_overlay(ax, arr_tgt, hist_stack, lats, lons, transform):
    p90 = np.nanpercentile(hist_stack, 90, axis=0)
    p10 = np.nanpercentile(hist_stack, 10, axis=0)
    for mask, color in [((arr_tgt > p90).astype(float), '#FF6666'),
                         ((arr_tgt < p10).astype(float), '#6666FF')]:
        mask[np.isnan(arr_tgt)] = np.nan
        mc, lc = add_cyclic_point(mask, coord=lons)
        lon2d, lat2d = np.meshgrid(lc, lats)
        try:
            ax.contour(lon2d, lat2d, mc, levels=[0.5], colors=[color],
                       linewidths=0.8, transform=transform, zorder=5)
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════════
#  PLOT MODES
# ══════════════════════════════════════════════════════════════════════════════

def plot_single():
    T  = _theme()
    arr, raw, lats, lons, dt, gm = fetch_sst_anomaly(TARGET_DATE, REMOVE_GLOBAL_MEAN)

    hist_stack = None
    if SHOW_PCT_OVERLAY or (SHOW_OCEANIC_INDICES and SHOW_PCT_IN_INDICES):
        sd = fetch_doy_stack(TARGET_DATE, remove_gm=False)
        hist_mask  = np.array([y != dt.year for y in sd['years']])
        hist_stack = sd['stack'][hist_mask]

    fig, ax, _, trans, _ = _base_fig()
    cmap   = get_sst_cmap()
    levels = np.linspace(-4.0, 4.0, 41)
    cf, lon2d, lat2d = _contourf_map(ax, arr, lats, lons, levels, cmap, trans)

    if SHOW_PCT_OVERLAY and hist_stack is not None:
        _add_pct_overlay(ax, raw, hist_stack, lats, lons, trans)

    _add_map_features(ax)

    if SHOW_OCEANIC_INDICES:
        idx = compute_oceanic_indices(raw, lats, lons, gm, hist_stack=hist_stack)
        box = dict(boxstyle='round,pad=0.5', facecolor=T['watermark_bg'],
                   alpha=0.88, edgecolor=T['border_color'], linewidth=0.8)
        ax.text(0.012, 0.02, _build_index_text(idx, gm),
                transform=ax.transAxes, fontsize=7.5,
                fontfamily='monospace', va='bottom',
                color=T['watermark_fg'], bbox=box, zorder=106)

    title_main = ('OISST Relative SST Anomaly (RSST)' if REMOVE_GLOBAL_MEAN
                  else 'OISST Sea Surface Temperature Anomaly')
    cb_label   = (f'RSST Anomaly (°C) [GM removed: {gm:+.3f}°C]'
                  if REMOVE_GLOBAL_MEAN else 'SST Anomaly (°C)')

    _title(ax, title_main, f"Valid: {dt.strftime('%B %d, %Y')}")
    _styled_colorbar(fig, ax, cf, cb_label, ticks=np.arange(-4, 5, 1))
    _add_watermark(ax)

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    if SHOW_INSET_MAP:
        _add_region_inset(fig, ax, _reg_info()['inset_hl'])

    out = f"OISST_Anom_{dt.strftime('%Y%m%d')}_{REGION}_{THEME}.png"
    plt.savefig(out, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f'\n  [saved] {out}');  plt.show()


def plot_difference():
    T  = _theme()
    arr_t, _, lats, lons, dt_t, gm_t = fetch_sst_anomaly(TARGET_DATE, REMOVE_GLOBAL_MEAN)
    arr_b, _, _,    _,    dt_b, _    = fetch_sst_anomaly(BASELINE_DATE, REMOVE_GLOBAL_MEAN)
    diff = arr_t - arr_b

    fig, ax, _, trans, _ = _base_fig()
    cmap   = plt.get_cmap('twilight_shifted')
    levels = np.linspace(-4.0, 4.0, 41)
    cf, *_ = _contourf_map(ax, diff, lats, lons, levels, cmap, trans)

    _add_map_features(ax)
    title_main = ('OISST RSST Difference' if REMOVE_GLOBAL_MEAN
                  else 'OISST SST Anomaly Difference')
    sub = (f"{dt_t.strftime('%B %d, %Y')}  minus  {dt_b.strftime('%B %d, %Y')}")
    _title(ax, title_main, sub)
    _styled_colorbar(fig, ax, cf, '∆ Anomaly (°C)', ticks=np.arange(-4, 4, 1))
    _add_watermark(ax)

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    if SHOW_INSET_MAP:
        _add_region_inset(fig, ax, _reg_info()['inset_hl'])

    out = f"OISST_Diff_{dt_t.strftime('%Y%m%d')}_{dt_b.strftime('%Y%m%d')}_{REGION}_{THEME}.png"
    plt.savefig(out, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f'\n  [saved] {out}');  plt.show()


def plot_records():
    T  = _theme()
    arr, raw, lats, lons, dt, gm = fetch_sst_anomaly(TARGET_DATE, REMOVE_GLOBAL_MEAN)

    sd = fetch_doy_stack(TARGET_DATE, remove_gm=False)
    hist_mask  = np.array([y != dt.year for y in sd['years']])
    hist_stack = sd['stack'][hist_mask]
    hist_max   = np.nanmax(hist_stack, axis=0)
    hist_min   = np.nanmin(hist_stack, axis=0)

    fig, ax, _, trans, _ = _base_fig()
    cmap   = get_sst_cmap()
    levels = np.linspace(-4.0, 4.0, 41)
    cf, lon2d, lat2d = _contourf_map(ax, arr, lats, lons, levels, cmap, trans)

    warm_mask = raw > hist_max
    cold_mask = raw < hist_min
    warm_c, _ = add_cyclic_point(warm_mask, coord=lons)
    cold_c, _ = add_cyclic_point(cold_mask, coord=lons)

    for mask_c, color in [(warm_c, '#800000'), (cold_c, '#2222FF')]:
        if np.any(mask_c):
            try:
                ax.contour(lon2d, lat2d, mask_c.astype(float),
                           levels=[0.5], colors=[color],
                           linewidths=0.8, transform=trans, zorder=4)
            except Exception:
                pass

    stride = 4
    for mask_c, color, label in [
            (warm_c, '#FF2222', 'Record Warm'),
            (cold_c, '#2222FF', 'Record Cold')]:
        pts = mask_c[::stride, ::stride]
        ax.scatter(lon2d[::stride, ::stride][pts],
                   lat2d[::stride, ::stride][pts],
                   s=2.5, color=color, marker='o', alpha=0.9,
                   edgecolors='none', transform=trans, zorder=5, label=label)

    _add_map_features(ax)
    leg = ax.legend(loc='lower left', framealpha=0.9, fontsize=9)
    _style_legend(leg)

    if SHOW_OCEANIC_INDICES:
        idx = compute_oceanic_indices(raw, lats, lons, gm, hist_stack=hist_stack)
        box = dict(boxstyle='round,pad=0.5', facecolor=T['watermark_bg'],
                   alpha=0.88, edgecolor=T['border_color'], linewidth=0.8)
        ax.text(0.012, 0.02, _build_index_text(idx, gm),
                transform=ax.transAxes, fontsize=7.5,
                fontfamily='monospace', va='bottom',
                color=T['watermark_fg'], bbox=box, zorder=106)

    title_main = ('OISST RSST with Daily Records' if REMOVE_GLOBAL_MEAN
                  else 'OISST SST Anomaly with Daily Records')
    n_hist = int(hist_mask.sum())
    sub = (f"Valid: {dt.strftime('%B %d, %Y')}  ·  "
           f"Records vs 1982–Present  (N = {n_hist} years)")
    _title(ax, title_main, sub)

    cb_label = (f'RSST Anomaly (°C) [GM removed: {gm:+.3f}°C]'
                if REMOVE_GLOBAL_MEAN else 'SST Anomaly (°C)')
    _styled_colorbar(fig, ax, cf, cb_label, ticks=np.arange(-4, 5, 1))
    _add_watermark(ax)

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    if SHOW_INSET_MAP:
        _add_region_inset(fig, ax, _reg_info()['inset_hl'])

    out = f"OISST_Records_{dt.strftime('%Y%m%d')}_{REGION}_{THEME}.png"
    plt.savefig(out, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f'\n  [saved] {out}');  plt.show()


def plot_percentile_rank():
    T  = _theme()
    arr, raw, lats, lons, dt, gm = fetch_sst_anomaly(TARGET_DATE, REMOVE_GLOBAL_MEAN)

    print("  [pct_rank] Building historical stack…")
    sd = fetch_doy_stack(TARGET_DATE, remove_gm=REMOVE_GLOBAL_MEAN)
    hist_mask  = np.array([y != dt.year for y in sd['years']])
    hist_stack = sd['stack'][hist_mask]
    n_hist     = int(hist_mask.sum())

    print(f"  [pct_rank] Computing rank over {n_hist} years…")
    below    = np.nansum(hist_stack < arr[np.newaxis], axis=0)
    pct_rank = (below / n_hist) * 100.0
    pct_rank[np.all(np.isnan(hist_stack), axis=0)] = np.nan

    fig, ax, _, trans, _ = _base_fig()
    cmap   = get_percentile_cmap()
    levels = np.linspace(0, 100, 101)
    cf, lon2d, lat2d = _contourf_map(ax, pct_rank, lats, lons, levels, cmap, trans, sigma=0.8)

    rank_s, lons_c = add_cyclic_point(smooth_nan(pct_rank, 0.8), coord=lons)
    for pv, lw, ls in [(10, 0.7, '--'), (25, 0.5, ':'),
                        (75, 0.5, ':'), (90, 0.7, '--'), (95, 0.9, '-')]:
        try:
            cs = ax.contour(lon2d, lat2d, rank_s, levels=[pv],
                            colors=['white'], linewidths=lw,
                            linestyles=ls, transform=trans, zorder=3, alpha=0.7)
            ax.clabel(cs, fmt=f'{pv}th', fontsize=6.5,
                      colors='white', inline=True)
        except Exception:
            pass

    _add_map_features(ax)

    title_main = 'OISST SST Anomaly — Percentile Rank'
    sub = (f"Valid: {dt.strftime('%B %d, %Y')}  ·  "
           f"Ranked vs 1982–Present excluding {dt.year}  (N = {n_hist} years)")
    _title(ax, title_main, sub)

    cbar = plt.colorbar(cf, ax=ax, orientation='horizontal',
                        pad=0.06, fraction=0.046, aspect=40, shrink=0.75,
                        extend='neither')
    cbar.set_ticks([0, 5, 10, 25, 50, 75, 90, 95, 100])
    cbar.set_label('Percentile Rank (%)', color=T['cbar_label'],
                   fontsize=10, fontweight='bold')
    cbar.ax.tick_params(labelsize=9, colors=T['cbar_ticks'])

    _add_watermark(ax)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    if SHOW_INSET_MAP:
        _add_region_inset(fig, ax, _reg_info()['inset_hl'])

    out = f"OISST_PctRank_{dt.strftime('%Y%m%d')}_{REGION}_{THEME}.png"
    plt.savefig(out, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f'\n  [saved] {out}');  plt.show()


def plot_trend():
    T    = _theme()
    print("  [trend] Fetching DOY stack…")
    sd   = fetch_doy_stack(TARGET_DATE, remove_gm=REMOVE_GLOBAL_MEAN)
    stk  = sd['stack']
    yrs  = np.array(sd['years'], dtype=float)
    lats = sd['lats']
    lons = sd['lons']

    print("  [trend] Vectorised OLS…")
    x   = yrs - yrs.mean()
    Sxx = float(np.sum(x ** 2))

    Sxy   = np.nansum(x[:, None, None] * stk, axis=0)
    slope = Sxy / Sxx * 10.0

    y_mean = np.nanmean(stk, axis=0)
    Syy    = np.nansum((stk - y_mean[None]) ** 2, axis=0)
    with np.errstate(invalid='ignore', divide='ignore'):
        r      = Sxy / np.sqrt(Sxx * Syy)
        r      = np.clip(r, -1.0, 1.0)
        n_eff  = np.sum(~np.isnan(stk), axis=0).astype(float)
        t_stat = r * np.sqrt(np.maximum(n_eff - 2, 1) / (1.0 - r ** 2 + 1e-12))
    p_vals = 2.0 * t_dist.sf(np.abs(t_stat), df=np.maximum(n_eff - 2, 1))

    land   = np.all(np.isnan(stk), axis=0)
    slope[land]  = np.nan
    p_vals[land] = np.nan

    vmax = max(0.5, float(np.round(np.nanpercentile(np.abs(slope), 98) * 2) / 2))
    levels = np.linspace(-vmax, vmax, 41)

    fig, ax, _, trans, _ = _base_fig()
    cmap = plt.get_cmap('RdBu_r')
    cf, lon2d, lat2d = _contourf_map(ax, slope, lats, lons,
                                      levels, cmap, trans, sigma=TREND_SMOOTH_SIGMA)

    if TREND_SHOW_SIG:
        sig_c, _ = add_cyclic_point((p_vals < 0.05) & ~land, coord=lons)
        stride   = 4
        sp_mask  = sig_c[::stride, ::stride]
        ax.scatter(lon2d[::stride, ::stride][sp_mask],
                   lat2d[::stride, ::stride][sp_mask],
                   s=0.6, color='black', alpha=0.35,
                   transform=trans, zorder=4)

    _add_map_features(ax)

    tgt_dt     = pd.to_datetime(TARGET_DATE)
    title_main = ('OISST RSST Trend' if REMOVE_GLOBAL_MEAN
                  else 'OISST SST Anomaly Trend')
    sig_note   = '  ·  dots: p < 0.05' if TREND_SHOW_SIG else ''
    sub = (f"Linear trend  ({chr(176)}C / decade)  ·  DOY {tgt_dt.dayofyear}"
           f"  ({tgt_dt.strftime('%b %d')})  ·  "
           f"{int(yrs.min())}–{int(yrs.max())}{sig_note}")

    _title(ax, title_main, sub)
    _styled_colorbar(fig, ax, cf, 'Trend (°C / decade)',
                     ticks=np.arange(-vmax, vmax + 0.1, 0.5))
    _add_watermark(ax)

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    if SHOW_INSET_MAP:
        _add_region_inset(fig, ax, _reg_info()['inset_hl'])

    out = f"OISST_Trend_DOY{tgt_dt.dayofyear:03d}_{REGION}_{THEME}.png"
    plt.savefig(out, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f'\n  [saved] {out}');  plt.show()


def plot_multi_year_panel():
    T        = _theme()
    tgt_dt   = pd.to_datetime(TARGET_DATE)
    n        = len(COMPARISON_YEARS)
    cols     = min(PANEL_COLS, n)
    rows     = (n + cols - 1) // cols
    reg      = _reg_info()
    clon     = reg['central_lon']
    ext      = reg['extent']

    fig = plt.figure(figsize=(5.2 * cols, 4.4 * rows + 1.0),
                     facecolor=T['fig_bg'])
    fig.subplots_adjust(hspace=0.12, wspace=0.06,
                        top=0.92, bottom=0.08, left=0.03, right=0.97)

    cmap   = get_sst_cmap()
    levels = np.linspace(-4.0, 4.0, 41)
    first_cf = None

    for i, yr in enumerate(COMPARISON_YEARS):
        ax = fig.add_subplot(rows, cols, i + 1,
                              projection=ccrs.PlateCarree(central_longitude=clon))
        ax.set_facecolor(T['ax_bg'])
        if ext:
            ax.set_extent(ext, crs=ccrs.PlateCarree())
        else:
            ax.set_global()

        date_str = f"{tgt_dt.month}/{tgt_dt.day}/{yr}"
        try:
            arr, _, lats, lons, _, _ = fetch_sst_anomaly(date_str, REMOVE_GLOBAL_MEAN)
            arr_s, lons_c = add_cyclic_point(smooth_nan(arr, 0.8), coord=lons)
            lon2d, lat2d  = np.meshgrid(lons_c, lats)
            cf = ax.contourf(lon2d, lat2d, arr_s, levels=levels,
                             cmap=cmap, extend='both',
                             transform=ccrs.PlateCarree(), zorder=1)
            if first_cf is None:
                first_cf = cf
        except Exception as e:
            print(f"  [warn] {date_str}: {e}")
            ax.text(0.5, 0.5, f'No data\n{yr}', ha='center', va='center',
                    color=T['label_color'], transform=ax.transAxes, fontsize=11)

        ax.coastlines(color=T['coast_color'], linewidth=0.4, zorder=3)
        ax.add_feature(cfeature.LAND, facecolor=T['land_fill'],
                       edgecolor='none', zorder=2)
        for sp in ax.spines.values():
            sp.set_edgecolor(T['border_color'])

        ax.set_title(f"{tgt_dt.strftime('%b %d')}, {yr}",
                     color=T['title_color'], fontsize=10, fontweight='bold', pad=3)

    if first_cf is not None:
        cbar_ax = fig.add_axes([0.15, 0.035, 0.70, 0.016])
        fmt     = ticker.FuncFormatter(lambda x, _: f'{x:.1f}')
        cbar    = fig.colorbar(first_cf, cax=cbar_ax, orientation='horizontal',
                               format=fmt, extend='both')
        cbar.set_ticks(np.arange(-4, 5, 1))
        cb_label = ('RSST Anomaly (°C)' if REMOVE_GLOBAL_MEAN
                    else 'SST Anomaly (°C)')
        cbar.set_label(cb_label, color=T['cbar_label'],
                       fontsize=10, fontweight='bold')
        cbar.ax.tick_params(labelsize=9, colors=T['cbar_ticks'])

    fig.suptitle(
        f"OISST {'RSST' if REMOVE_GLOBAL_MEAN else 'SST'} Anomaly  —  "
        f"{tgt_dt.strftime('%B %d')}  across years  |  {reg['label']}",
        color=T['title_color'], fontsize=14, fontweight='bold', y=0.97)

    out = f"OISST_Panel_{tgt_dt.strftime('%m%d')}_{REGION}_{THEME}.png"
    plt.savefig(out, dpi=180, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f'\n  [saved] {out}');  plt.show()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN EXECUTOR
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print('═' * 66)
    print(f"  OISST V2.1 Anomaly Plotter v3.2")
    print(f"  Mode   : {MODE:<18}  Region : {REGION}")
    print(f"  Target : {TARGET_DATE:<18}  Theme  : {THEME}")
    print(f"  RSST   : {str(REMOVE_GLOBAL_MEAN):<18}  Cache  : {CACHE_DIR}")
    if MODE == 'difference':
        print(f"  Base   : {BASELINE_DATE}")
    if MODE == 'multi_year_panel':
        print(f"  Years  : {COMPARISON_YEARS}")
    print('═' * 66)

    dispatch = {
        'single'          : plot_single,
        'difference'      : plot_difference,
        'records'         : plot_records,
        'percentile_rank' : plot_percentile_rank,
        'trend'           : plot_trend,
        'multi_year_panel': plot_multi_year_panel,
    }

    fn = dispatch.get(MODE)
    if fn is None:
        print(f"  [error] Unknown mode '{MODE}'.")
        print(f"  Valid modes: {', '.join(dispatch)}")
        sys.exit(1)

    fn()


if __name__ == '__main__':
    main()