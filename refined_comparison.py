import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import datetime
import numpy as np

def to_doy(date_str):
    d = datetime.datetime.strptime(date_str, "%Y-%m-%d")
    oct1 = datetime.datetime(d.year if d.month >= 10 else d.year-1, 10, 1)
    return (d - oct1).days

def load_season(fname):
    with open(fname) as f:
        data = [d for d in json.load(f) if d.get("available")]
    return [(to_doy(d["date"]), d["rvi"]) for d in data]

def get_weather_for_period(lat, lng, start_date, end_date):
    """Get cumulative weather for a period"""
    import requests
    try:
        r = requests.get(
            "https://archive-api.open-meteo.com/v1/archive",
            params={
                "latitude": lat, "longitude": lng,
                "start_date": start_date, "end_date": end_date,
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
                "timezone": "Europe/London"
            }, timeout=20)
        if r.status_code == 200:
            data = r.json().get("daily", {})
            temps_max = data.get("temperature_2m_max", [])
            temps_min = data.get("temperature_2m_min", [])
            rain = data.get("precipitation_sum", [])
            
            # Growing degree days (base 0°C for wheat)
            gdd_total = sum([max(0, (mx+mn)/2) 
                           for mx,mn in zip(temps_max, temps_min) 
                           if mx and mn])
            rain_total = sum([r for r in rain if r])
            return {"gdd": round(gdd_total, 1), "rain": round(rain_total, 1)}
    except:
        pass
    return {"gdd": None, "rain": None}

# Load all seasons
s2122 = load_season("sar_2021_22.json")
s2223 = load_season("sar_2022_23.json")
s2324 = load_season("sar_2023_24.json")
s2425 = load_season("sar_timeseries_test.json")

historical = [s2122, s2223, s2324]

# Build historical mean and std at each day point
all_days = sorted(set([d[0] for d in s2425]))
hist_mean = []
hist_std = []
hist_min = []
hist_max = []

for day in all_days:
    vals = []
    for season in historical:
        closest = min(season, key=lambda x: abs(x[0]-day))
        if abs(closest[0]-day) <= 8:
            vals.append(closest[1])
    if vals:
        hist_mean.append(np.mean(vals))
        hist_std.append(np.std(vals))
        hist_min.append(np.min(vals))
        hist_max.append(np.max(vals))
    else:
        hist_mean.append(None)
        hist_std.append(None)
        hist_min.append(None)
        hist_max.append(None)

# Current season values at matching days
curr_vals = []
for day in all_days:
    closest = min(s2425, key=lambda x: abs(x[0]-day))
    curr_vals.append(closest[1] if abs(closest[0]-day) <= 8 else None)

# Calculate development alignment using std-normalised deviation
# Score = 100 when perfectly aligned
# Each 1 std deviation = -10 points
scores = []
deviations = []
z_scores = []
for i, day in enumerate(all_days):
    if hist_mean[i] and hist_std[i] and curr_vals[i] and hist_std[i] > 0:
        dev_pct = (curr_vals[i] - hist_mean[i]) / hist_mean[i] * 100
        # Z-score for anomaly detection only
        z = (curr_vals[i] - hist_mean[i]) / hist_std[i]
        # Simple score from deviation — capped 0-100
        score = max(0, min(100, round(100 - abs(dev_pct))))
        deviations.append(dev_pct)
        scores.append(score)
        z_scores.append(z)
    else:
        deviations.append(None)
        scores.append(None)
        z_scores.append(None)

# Get weather comparison
print("Fetching weather data...")
wx_current = get_weather_for_period(53.2307, -0.5406, "2024-10-01", "2025-06-10")
wx_2122 = get_weather_for_period(53.2307, -0.5406, "2021-10-01", "2022-06-10")
wx_2223 = get_weather_for_period(53.2307, -0.5406, "2022-10-01", "2023-06-10")
wx_2324 = get_weather_for_period(53.2307, -0.5406, "2023-10-01", "2024-06-10")

# Historical average weather
hist_gdds = [w['gdd'] for w in [wx_2122,wx_2223,wx_2324] if w['gdd']]
hist_rains = [w['rain'] for w in [wx_2122,wx_2223,wx_2324] if w['rain']]
hist_gdd_avg = round(np.mean(hist_gdds), 1) if hist_gdds else None
hist_rain_avg = round(np.mean(hist_rains), 1) if hist_rains else None

print(f"Current season: GDD={wx_current['gdd']} Rain={wx_current['rain']}mm")
print(f"3-year average: GDD={hist_gdd_avg} Rain={hist_rain_avg}mm")

# Weather attribution vs historical average
wx_factors = []
if wx_current['rain'] and hist_rain_avg:
    rain_diff = (wx_current['rain'] - hist_rain_avg) / hist_rain_avg * 100
    if rain_diff > 15:
        wx_factors.append(f"Rainfall +{rain_diff:.0f}% above 3-year average")
    elif rain_diff < -15:
        wx_factors.append(f"Rainfall {abs(rain_diff):.0f}% below 3-year average")
if wx_current['gdd'] and hist_gdd_avg:
    gdd_diff = (wx_current['gdd'] - hist_gdd_avg) / hist_gdd_avg * 100
    if gdd_diff < -10:
        wx_factors.append(f"Growing degree days {abs(gdd_diff):.0f}% below 3-year average")
    elif gdd_diff > 10:
        wx_factors.append(f"Growing degree days +{gdd_diff:.0f}% above 3-year average")

wx_hist_2324 = wx_2324  # keep for display

# Overall score — mean absolute deviation from baseline
valid_devs = [abs(d) for d in deviations if d is not None]
mean_dev = np.mean(valid_devs) if valid_devs else 0
avg_score = max(0, min(100, round(100 - mean_dev)))
mean_dev_signed = np.mean([d for d in deviations if d is not None]) if valid_devs else 0
status = "🟢 On Track" if avg_score >= 90 else "🟡 Slightly Behind" if avg_score >= 80 else "🔴 Behind Expected Development"

print(f"\nOverall Development Alignment: {avg_score}/100")
print(f"Status: {status}")
if wx_factors:
    print("Contributing weather factors:")
    for f in wx_factors:
        print(f"  → {f}")

# PLOT
fig = plt.figure(figsize=(16, 11))
fig.patch.set_facecolor('#0a1525')

# Create grid
gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3,
                      left=0.07, right=0.97, top=0.88, bottom=0.07)

ax1 = fig.add_subplot(gs[0:2, :])  # Main trajectory
ax2 = fig.add_subplot(gs[2, 0])    # Development score
ax3 = fig.add_subplot(gs[2, 1])    # Weather attribution

for ax in [ax1, ax2, ax3]:
    ax.set_facecolor('#0a1525')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('#333')
    ax.grid(alpha=0.1, color='white')

# --- TOP PANEL: Trajectory with confidence bands ---
valid_days = [d for d,m in zip(all_days, hist_mean) if m]
valid_mean = [m for m in hist_mean if m]
valid_std = [s for s in hist_std if s]
valid_min = [m for m in hist_min if m]
valid_max = [m for m in hist_max if m]

# Confidence band (mean ± 1 std)
upper = [m+s for m,s in zip(valid_mean, valid_std)]
lower = [m-s for m,s in zip(valid_mean, valid_std)]

ax1.fill_between(valid_days, valid_min, valid_max,
                alpha=0.12, color='#aaaaaa', label='Historical range (2021-24)')
ax1.fill_between(valid_days, lower, upper,
                alpha=0.25, color='#aaaaaa', label='Expected ± 1 std dev')
ax1.plot(valid_days, valid_mean, color='#aaaaaa', linewidth=2,
         linestyle='--', alpha=0.9, label='Historical mean (3 seasons)')

# Current season
curr_days = [d for d,v in zip(all_days, curr_vals) if v]
curr_rvi = [v for v in curr_vals if v]
ax1.plot(curr_days, curr_rvi, color='#4da6ff', linewidth=2.5,
         marker='o', markersize=4, label='2024/25 Current season')

# Highlight deviations
for i, (day, dev) in enumerate(zip(all_days, deviations)):
    if dev and abs(dev) > 10:
        color = '#ff444433' if dev < 0 else '#44ff4433'
        ax1.axvspan(day-6, day+6, alpha=0.3, color=color)

# Typical wheat growth windows (UK average — not field-detected)
stages = {"Sowing":0,"Emergence":24,"Tillering":61,
          "Stem ext.":166,"Heading":227,"Grain fill":257,"Harvest":293}
for stage, day in stages.items():
    ax1.axvline(x=day, color='#ffffff', alpha=0.2, linestyle=':', linewidth=1)
    ax1.text(day+1, 0.755, stage, color='#ffffff', fontsize=7,
             rotation=45, ha='left', alpha=0.5)
# Label as typical not detected
ax1.text(0.01, 0.02, 'Growth stage windows: UK typical averages (not field-detected)',
         transform=ax1.transAxes, color='#888888', fontsize=7, style='italic')

ax1.set_ylabel('SAR RVI', color='white', fontsize=11)
ax1.set_ylim(0.44, 0.78)
ax1.legend(facecolor='#1a2a3a', labelcolor='white', fontsize=9, loc='lower right')
ax1.set_title(
    'Lincolnshire Winter Wheat — Development Trajectory vs Historical Baseline\n'
    'Sentinel-1 SAR · 4-Season Analysis · Cube Earth Crop Intelligence',
    color='white', fontsize=12, fontweight='bold', pad=12)

# --- BOTTOM LEFT: Development Alignment ---
valid_score_days = [d for d,s in zip(all_days, scores) if s]
valid_score_vals = [s for s in scores if s]
score_colors = ['#44ff44' if s >= 90 else '#ffaa00' if s >= 75 else '#ff4444'
                for s in valid_score_vals]

ax2.bar(valid_score_days, valid_score_vals, color=score_colors, alpha=0.8, width=10)
ax2.axhline(y=100, color='white', linewidth=0.5, alpha=0.3, linestyle='--')
ax2.axhline(y=90, color='#44ff44', linewidth=0.5, alpha=0.4, linestyle='--')
ax2.axhline(y=75, color='#ffaa00', linewidth=0.5, alpha=0.4, linestyle='--')
ax2.set_ylim(50, 115)
ax2.set_ylabel('Development Alignment', color='white', fontsize=10)
ax2.set_xlabel('Days from Oct 1', color='white', fontsize=9)

# Score label
score_color = '#44ff44' if avg_score >= 90 else '#ffaa00' if avg_score >= 75 else '#ff4444'
ax2.text(0.05, 0.92, f'Season Score: {avg_score}/100',
         transform=ax2.transAxes, color=score_color,
         fontsize=12, fontweight='bold')
ax2.text(0.05, 0.82, status,
         transform=ax2.transAxes, color=score_color, fontsize=10)

# --- BOTTOM RIGHT: Anomaly detection + weather ---
ax3.axis('off')
ax3.set_title('Field Intelligence Summary', color='white', 
              fontsize=11, fontweight='bold')

lines = []
lines.append(('Field Status', 'white', 11, True))
lines.append(('', 'white', 8, False))

score_color2 = '#44ff44' if avg_score >= 90 else '#ffaa00' if avg_score >= 75 else '#ff4444'
lines.append((f'Development Alignment: {avg_score}/100', score_color2, 12, True))
lines.append((status, score_color2, 10, False))
lines.append((f'Mean deviation: {mean_dev_signed:+.1f}% from baseline', '#aaaaaa', 9, False))
lines.append(('Baseline: 2021/22 · 2022/23 · 2023/24', '#aaaaaa', 8, False))
lines.append(('', 'white', 8, False))

# Anomaly periods
anomaly_periods = []
for d, z in zip(all_days, z_scores):
    if z and z < -1.0:
        anomaly_periods.append(d)

if anomaly_periods:
    first_anomaly = anomaly_periods[0]
    lines.append(('Deviation Detected:', '#ff4444', 10, True))
    lines.append((f'  First flag: Day {first_anomaly} of season', '#ff4444', 9, False))
    lines.append((f'  Duration: {len(anomaly_periods)} observations', '#ff4444', 9, False))
    lines.append(('', 'white', 8, False))

lines.append(('Weather Attribution:', '#ffaa00', 10, True))
if wx_current['gdd'] and hist_gdd_avg:
    gdd_diff = round((wx_current['gdd'] - hist_gdd_avg) / hist_gdd_avg * 100)
    lines.append((f'  GDD: {wx_current["gdd"]} vs {hist_gdd_avg} avg ({gdd_diff:+d}%)',
                 '#ffaa00' if abs(gdd_diff)>10 else '#aaaaaa', 9, False))
if wx_current['rain'] and hist_rain_avg:
    rain_diff = round((wx_current['rain'] - hist_rain_avg) / hist_rain_avg * 100)
    lines.append((f'  Rain: {wx_current["rain"]}mm vs {hist_rain_avg}mm avg ({rain_diff:+d}%)',
                 '#ffaa00' if abs(rain_diff)>15 else '#aaaaaa', 9, False))

if wx_factors:
    lines.append(('', 'white', 8, False))
    lines.append(('Potential drivers:', '#ffaa00', 9, True))
    for factor in wx_factors:
        lines.append((f'  → {factor}', '#ffaa00', 8, False))

y_pos = 0.97
for text, color, size, bold in lines:
    weight = 'bold' if bold else 'normal'
    ax3.text(0.03, y_pos, text, transform=ax3.transAxes,
             color=color, fontsize=size, fontweight=weight, va='top')
    y_pos -= 0.09

plt.savefig('wheat_refined.png', dpi=150, bbox_inches='tight',
            facecolor='#0a1525')
print("\nSaved: wheat_refined.png")
plt.close()
