import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker
import numpy as np
import os
import requests
import calendar

# Ensure that the math text renderer uses our standard modern font
plt.rcParams.update({'mathtext.default': 'regular'})

# --- SETTINGS & DIRECTORIES ---
BASE_URL = "https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/v6"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
DATA_DIR = os.path.join(SCRIPT_DIR, "ersst_v6_data")
os.makedirs(DATA_DIR, exist_ok=True)

# Define Custom Colorbar levels and labels
CBAR_LIMITS = (-3.0, 3.0)
CBAR_TICKS = [-3.0, -2.0, -1.0, 0, 1.0, 2.0, 3.0]
CBAR_LABELS = ['-3.0', '-2.0', '-1.0', '0', '1.0', '2.0', '3.0']

def get_custom_cmap():
    """Generates a custom colormap matching the provided aesthetic."""
    colors = ['#4A0074', '#063A79', '#1C74B5', '#88C1DE', '#FFFFFF', 
              '#FBD99E', '#ED6D24', '#A50016', '#D6007A']
    return mcolors.LinearSegmentedColormap.from_list("custom_sst", colors, N=256)

def download_file(year, month):
    """Downloads a specific monthly ERSSTv6 file if not already present."""
    month_str = str(month).zfill(2)
    filename = f"ersst.v6.{year}{month_str}.nc"
    local_path = os.path.join(DATA_DIR, filename)
    
    if not os.path.exists(local_path):
        url = f"{BASE_URL}/{filename}"
        print(f"Downloading: {filename}...")
        try:
            r = requests.get(url, timeout=15)
            if r.status_code == 200:
                with open(local_path, 'wb') as f:
                    f.write(r.content)
            else:
                print(f"Warning: Could not download {filename} (Status {r.status_code})")
        except Exception as e:
            print(f"Error: {e}")
    return local_path

def get_sst_data(year, month):
    """Opens a dataset and returns the SST for the first time step."""
    path = download_file(year, month)
    if os.path.exists(path):
        with xr.open_dataset(path) as ds:
            return ds['sst'].isel(lev=0).squeeze().load()
    return None

def get_sliding_climo(target_year, month):
    """Calculates the 30-year mean SST based on the 30 years PRIOR to the target year."""
    start_year = target_year - 30
    end_year = target_year - 1
    
    if start_year < 1850:
        start_year, end_year = 1850, 1879
        
    climo_years = range(start_year, end_year + 1)
    sst_list = []
    
    print(f"  --> Calculating {calendar.month_name[month]} sliding climatology ({start_year}-{end_year})...")
    for yr in climo_years:
        data = get_sst_data(yr, month)
        if data is not None:
            sst_list.append(data)
            
    if sst_list:
        return xr.concat(sst_list, dim='time').mean(dim='time')
    return None

def plot_composite(events, remove_global_mean=True, show_regional_text=True, show_regional_boxes=True, custom_cmap='RdBu_r', attribution="@TheSteveCop"):
    """
    events: List of tuples [(year, month), ...]
    remove_global_mean: Toggle to subtract the area-weighted spatial average.
    show_regional_text: Toggle to display regional SST anomalies in a text box.
    show_regional_boxes: Toggle to draw colored bounding boxes over the regions.
    """
    all_anoms = []
    processed_count = 0
    
    for yr, mon in events:
        print(f"Processing {calendar.month_name[mon]} {yr}...")
        current_sst = get_sst_data(yr, mon)
        climo_mean = get_sliding_climo(yr, mon)
        
        if current_sst is not None and climo_mean is not None:
            anom = current_sst - climo_mean
            all_anoms.append(anom)
            processed_count += 1
        else:
            print(f"  Warning: Data missing for {calendar.month_abbr[mon]} {yr}. Skipping.")
    
    if processed_count == 0:
        print("Error: No data successfully processed.")
        return

    # Create Composite
    composite_anom = xr.concat(all_anoms, dim='event').mean(dim='event')
    
    # --- REGIONAL INDICES CALCULATION (Calculated on raw anomalies) ---
    regions = {
        "Niño 3.4": {"lat": (-5, 5), "lon": (190, 240)},           
        "Global Tropics": {"lat": (-20, 20), "lon": (0, 350)},     
        "Gulf of Mexico": {"lat": (20, 30), "lon": (262, 280)},    
        "Caribbean": {"lat": (10, 20), "lon": (275, 300)},         
        "Tropical Atl MDR": {"lat": (10, 20), "lon": (300, 340)}   
    }
    
    region_values = {}
    index_text = "Raw Regional Anomalies (°C)\n" + "-"*30 + "\n"
    
    for name, bnds in regions.items():
        mask = ((composite_anom.lat >= bnds['lat'][0]) & (composite_anom.lat <= bnds['lat'][1]) & 
                (composite_anom.lon >= bnds['lon'][0]) & (composite_anom.lon <= bnds['lon'][1]))
        reg_anom = composite_anom.where(mask)
        w = np.cos(np.deg2rad(reg_anom.lat))
        val = reg_anom.weighted(w).mean(('lat', 'lon')).values
        region_values[name] = val
        index_text += f"{name:16}: {val:+.2f}\n"

    print(f"\n{index_text}")

    # --- GLOBAL MEAN REMOVAL LOGIC ---
    if remove_global_mean:
        weights = np.cos(np.deg2rad(composite_anom.lat))
        global_mean_val = composite_anom.weighted(weights).mean(('lat', 'lon')).values
        print(f"  --> [ACTION] Area-Weighted Global Mean Removed: {global_mean_val:.4f}°C")
        
        composite_anom = composite_anom - global_mean_val
        title_main = "SST Anomalies (Global Mean Removed)"
    else:
        print("  --> [SKIP] Global Mean Removal is OFF.")
        title_main = "SST Anomalies"

    # --- PLOTTING ---
    fig = plt.figure(figsize=(16, 8))
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
    
    ax.add_feature(cfeature.LAND, facecolor='#dddddd', zorder=2)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=3)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.5, zorder=3)
    
    ax.set_xticks(np.arange(-180, 181, 20), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(-90, 91, 20), crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(cticker.LongitudeFormatter())
    ax.yaxis.set_major_formatter(cticker.LatitudeFormatter())
    ax.tick_params(labelsize=9)
    ax.grid(linestyle='--', alpha=0.5, color='black', linewidth=0.5, zorder=1)
    
    clevs = np.arange(-2, 2.1, 0.1)
    cf = ax.contourf(composite_anom.lon, composite_anom.lat, composite_anom, 
                     levels=clevs, cmap=custom_cmap, extend='both', transform=ccrs.PlateCarree())
    
    cbar = plt.colorbar(cf, orientation='vertical', pad=0.02, aspect=40, shrink=0.8)
    cbar.ax.tick_params(labelsize=9)
    cbar.set_label('SST Anomaly (°C)', size=10)
    cbar.set_ticks(CBAR_TICKS)
    cbar.ax.set_yticklabels(CBAR_LABELS)

    # --- OVERLAYING REGIONAL BOXES ---
    if show_regional_boxes:
        norm = mcolors.Normalize(vmin=CBAR_LIMITS[0], vmax=CBAR_LIMITS[1])
        cmap_obj = plt.get_cmap(custom_cmap) if isinstance(custom_cmap, str) else custom_cmap

        for name, bnds in regions.items():
            val = region_values[name]
            box_color = cmap_obj(norm(val))
            
            if name == "Global Tropics":
                lw = 2
                for lat_val in bnds['lat']:
                    # Black shadow line
                    ax.plot([-180, 180], [lat_val, lat_val], color='black', linewidth=lw+1.5, 
                            linestyle='-', transform=ccrs.PlateCarree(), zorder=4)
                    # Colored dash line
                    ax.plot([-180, 180], [lat_val, lat_val], color=box_color, linewidth=lw, 
                            linestyle='-', transform=ccrs.PlateCarree(), zorder=5)
            else:
                width = bnds['lon'][1] - bnds['lon'][0]
                height = bnds['lat'][1] - bnds['lat'][0]
                lw = 2.5
                
                # Black shadow box
                bg_rect = mpatches.Rectangle((bnds['lon'][0], bnds['lat'][0]), width, height,
                                             linewidth=lw+1.5, edgecolor='black', facecolor='none', 
                                             linestyle='-', transform=ccrs.PlateCarree(), zorder=4)
                ax.add_patch(bg_rect)
                
                # Colored bounding box
                rect = mpatches.Rectangle((bnds['lon'][0], bnds['lat'][0]), width, height,
                                          linewidth=lw, edgecolor=box_color, facecolor='none', 
                                          linestyle='-', transform=ccrs.PlateCarree(), zorder=5)
                ax.add_patch(rect)

    # Shared box properties for text overlays
    box_props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.85, edgecolor='black', linewidth=0.8)

    # --- ADDING THE REGIONAL TEXT BOX (BOTTOM LEFT) ---
    if show_regional_text:
        ax.text(0.015, 0.03, index_text.strip(), transform=ax.transAxes, fontsize=9, 
                fontfamily='monospace', fontweight='bold', verticalalignment='bottom', 
                bbox=box_props, zorder=6)

    # --- ADDING THE ATTRIBUTION TEXT BOX (BOTTOM RIGHT) ---
    ax.text(0.985, 0.03, attribution, transform=ax.transAxes, fontsize=9, 
            fontfamily='sans-serif', fontweight='bold', verticalalignment='bottom', horizontalalignment='right',
            bbox=box_props, zorder=6)

    # --- TOP TITLES (Centered & Dynamic Font Sizing) ---
    
    # Dynamic logic: Make the text bigger if there are fewer dates!
    if len(events) == 1:
        date_fontsize = 12
    elif len(events) <= 5:
        date_fontsize = 12
    else:
        date_fontsize = 11

    # Build the date string
    if len(events) > 10:
        date_str = f"{processed_count} Events"
    else:
        date_str = ", ".join([f"{calendar.month_abbr[m]} {y}" for y, m in events])
    
    # Combine into a single string. \mathbf{} renders 'Composite:' as bold, the rest stays normal.
    mixed_centered_title = r"$\mathbf{Composite:}$ " + date_str
    
    # 1. Main Title (Bold) centered. The newline creates space for the subtitle.
    plt.title(f"ERSSTv6 {title_main}\n", loc='center', fontsize=12, fontweight='bold', fontname='sans-serif')
    
    # 2. Subtitle / Dates perfectly centered as one whole block!
    ax.text(0.5, 1.010, mixed_centered_title, transform=ax.transAxes, fontsize=date_fontsize, 
            fontname='sans-serif', ha='center', va='bottom')

    # 3. Methodology (Top Right - Italics) - ONLY showing if global mean is removed!
    if remove_global_mean:
        methodology_text = "*Based on a 30-Year\n Sliding Climatology"
        plt.title(methodology_text, loc='right', fontsize=7, fontstyle='italic', fontname='sans-serif')

    # Saving
    safe_date_str = "_".join([f"{y}_{m}" for y, m in events[:3]])
    if len(events) > 3: safe_date_str += "_comp"
    save_filename = f"ERSSTv6_{'Rel' if remove_global_mean else 'Anom'}_{safe_date_str}.png"
    save_path = os.path.join(SCRIPT_DIR, save_filename)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to: {save_path}")
    plt.show()

# ==============================================================================
# --- RUNNING THE SCRIPT: COMPOSITE EXAMPLES ---
# ==============================================================================

my_cmap = get_custom_cmap()

# EXAMPLE 1: Super El Niño (Decembers of 1982, 1997, 2015)
events_el_nino = [(1982, 12), (1997, 12), (2015, 12)]

# EXAMPLE 2: Strong La Niña (Decembers of 1988, 1999, 2010, 2020)
events_la_nina = [(1988, 12), (1999, 12), (2010, 12), (2020, 12)]

# EXAMPLE 3: Positive PDO Phase (Januaries of strong +PDO years)
events_pdo_pos = [(1983, 1), (1987, 1), (1992, 1), (1993, 1), (1997, 1)]

# EXAMPLE 4: 1930s Dust Bowl Era (Augusts from 1934-1936)
events_dust_bowl = [(1934, 8), (1935, 8), (1936, 8)]

    #(1965, 9), 
    #(1972, 9), 
    #(1982, 9),
    #(1987, 9), 
    #(1997, 9), 
    #(2015, 9),
    
# Change the 'events' variable below to whichever list you want to run!
plot_composite([

 #(1878, 8)
 #(1957, 3),
 #(1963, 9),
 #(1965, 3),
 #(1972, 3),
 #(1982, 3),
 #(1987, 3),
 #(1991, 3),
 #(1997, 3),
 (2015, 3),
 #(2023, 9),
 #(2026, 3)
 
 ],  
               remove_global_mean=True, 
               show_regional_text=True, 
               show_regional_boxes=False, 
               custom_cmap=my_cmap)