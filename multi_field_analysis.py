import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import datetime
import numpy as np

def to_doy(date_str):
    d = datetime.datetime.strptime(date_str, "%Y-%m-%d")
    oct1 = datetime.datetime(d.year if d.month >= 10 else d.year-1, 10, 1)
    return (d - oct1).days

def load_season(fname):
    try:
        with open(fname) as f:
            data = [d for d in json.load(f) if d.get("available")]
        return [(to_doy(d["date"]), d["rvi"]) for d in data]
    except:
        return []

def analyse_field(field_name, current_file, hist_files, label):
    current = load_season(current_file)
    historical = [load_season(f) for f in hist_files if load_season(f)]

    all_days = sorted(set([d[0] for d in current]))
    hist_mean, hist_std = [], []

    for day in all_days:
        vals = []
        for season in historical:
            if season:
                closest = min(season, key=lambda x: abs(x[0]-day))
                if abs(closest[0]-day) <= 8:
                    vals.append(closest[1])
        hist_mean.append(np.mean(vals) if vals else None)
        hist_std.append(np.std(vals) if len(vals)>1 else 0.02)

    curr_vals = []
    for day in all_days:
        closest = min(current, key=lambda x: abs(x[0]-day))
        curr_vals.append(closest[1] if abs(closest[0]-day) <= 8 else None)

    deviations = []
    scores = []
    for i in range(len(all_days)):
        if hist_mean[i] and curr_vals[i]:
            dev = (curr_vals[i] - hist_mean[i]) / hist_mean[i] * 100
            score = max(0, min(100, round(100 - abs(dev))))
            deviations.append(dev)
            scores.append(score)
        else:
            deviations.append(None)
            scores.append(None)

    valid_devs = [d for d in deviations if d is not None]
    mean_dev_signed = np.mean(valid_devs) if valid_devs else 0
    mean_dev_abs = np.mean([abs(d) for d in valid_devs]) if valid_devs else 0
    max_dev = max(valid_devs, key=abs) if valid_devs else 0
    obs_below = sum(1 for d in valid_devs if d < -5)
    obs_above = sum(1 for d in valid_devs if d > 5)
    total_obs = len(valid_devs)

    # First persistent deviation — first run of 3+ consecutive below -5%
    first_deviation_day = None
    deviation_duration = 0
    consecutive = 0
    for day, dev in zip(all_days, deviations):
        if dev is not None and dev < -5:
            consecutive += 1
            if consecutive >= 3 and first_deviation_day is None:
                # Find start of this run
                idx = all_days.index(day)
                first_deviation_day = all_days[max(0, idx-2)]
                deviation_duration = sum(1 for d in deviations[max(0,idx-2):] 
                                        if d is not None and d < -5)
        else:
            consecutive = 0
    # Development Performance Score
    # Formula: 100 - (mean_absolute_deviation × 1.0)
    # Simple, transparent, one sentence explainable:
    # "100 minus the average percentage deviation from the 3-year baseline"
    # Examples: 0% dev = 100, 5% dev = 95, 10% dev = 90, 15% dev = 85
    avg_score = max(0, min(100, round(100 - mean_dev_abs)))
    # Status based on signed deviation direction — not score
    if mean_dev_signed > 5:
        status = "🔵 Ahead of Baseline"
    elif mean_dev_signed >= -5:
        status = "🟢 On Track"
    elif mean_dev_signed >= -15:
        status = "🟡 Slightly Behind"
    else:
        status = "🔴 Significantly Behind"

    return {
        "label": label,
        "all_days": all_days,
        "curr_vals": curr_vals,
        "hist_mean": hist_mean,
        "hist_std": hist_std,
        "deviations": deviations,
        "scores": scores,
        "avg_score": avg_score,
        "mean_dev_signed": mean_dev_signed,
        "max_dev": max_dev,
        "obs_below": obs_below,
        "obs_above": obs_above,
        "total_obs": total_obs,
        "first_deviation_day": first_deviation_day,
        "deviation_duration": deviation_duration,
        "status": status
    }

# Analyse all three fields
fields = [
    {
        "name": "Field 1",
        "label": "Field 1 · 53.23°N 0.54°W",
        "current": "sar_timeseries_test.json",
        "hist": ["sar_2021_22.json","sar_2022_23.json","sar_2023_24.json"]
    },
    {
        "name": "Field 2",
        "label": "Field 2 · 53.12°N 0.35°W",
        "current": "sar_field_2_2425.json",
        "hist": ["sar_field_2_2021_22.json","sar_field_2_2022_23.json","sar_field_2_2023_24.json"]
    },
    {
        "name": "Field 3",
        "label": "Field 3 · 53.35°N 0.72°W",
        "current": "sar_field_3_2425.json",
        "hist": ["sar_field_3_2021_22.json","sar_field_3_2022_23.json","sar_field_3_2023_24.json"]
    }
]

results = []
for f in fields:
    print(f'Analysing {f["name"]}...')
    r = analyse_field(f["name"], f["current"], f["hist"], f["label"])
    results.append(r)
    print(f'  Score: {r["avg_score"]}/100 | {r["status"]} | Dev: {r["mean_dev_signed"]:+.1f}% | Max: {r["max_dev"]:+.1f}%')
    print(f'  Below baseline: {r["obs_below"]}/{r["total_obs"]} | First flag: Day {r["first_deviation_day"]}')

# MULTI-FIELD CHART
fig = plt.figure(figsize=(18, 13))
fig.patch.set_facecolor('#0a1525')
fig.suptitle('Lincolnshire Winter Wheat — Multi-Field Development Analysis 2024/25\n'
             'Sentinel-1 SAR · 3 Fields · 4-Season Baseline · Cube Earth',
             color='white', fontsize=13, fontweight='bold', y=0.98)

gs = gridspec.GridSpec(2, 3, hspace=0.4, wspace=0.3,
                       left=0.06, right=0.97, top=0.91, bottom=0.06)

colors = ['#4da6ff', '#ff9944', '#44ff99']

# Top row — individual field trajectories
for i, (r, color) in enumerate(zip(results, colors)):
    ax = fig.add_subplot(gs[0, i])
    ax.set_facecolor('#0a1525')
    ax.tick_params(colors='white', labelsize=7)
    for spine in ax.spines.values():
        spine.set_color('#333')
    ax.grid(alpha=0.1, color='white')

    valid_days = [d for d,m in zip(r["all_days"],r["hist_mean"]) if m]
    valid_mean = [m for m in r["hist_mean"] if m]
    valid_std = [s for s in r["hist_std"] if s is not None]
    upper = [m+s for m,s in zip(valid_mean,valid_std)]
    lower = [m-s for m,s in zip(valid_mean,valid_std)]

    ax.fill_between(valid_days, lower, upper, alpha=0.2, color='#aaaaaa')
    ax.plot(valid_days, valid_mean, color='#aaaaaa', linewidth=1.5,
            linestyle='--', alpha=0.7, label='Baseline mean')

    curr_x = [d for d,v in zip(r["all_days"],r["curr_vals"]) if v]
    curr_y = [v for v in r["curr_vals"] if v]
    ax.plot(curr_x, curr_y, color=color, linewidth=2,
            marker='o', markersize=3, label='2024/25')

    score_color = '#44ff44' if r["avg_score"] >= 90 else '#ffaa00' if r["avg_score"] >= 80 else '#ff4444'
    # Growth stage markers — UK typical averages, not field-detected
    stages = {"Sow":0,"Emer":24,"Till":61,"Stem":166,"Head":227,"Harv":293}
    for stage, day in stages.items():
        ax.axvline(x=day, color='#ffffff', alpha=0.15, linestyle=':', linewidth=0.8)
        ax.text(day+1, ax.get_ylim()[1]*0.97 if ax.get_ylim()[1] else 0.75,
                stage, color='#666666', fontsize=6, rotation=45, ha='left')
    ax.set_title(r["label"], color='white', fontsize=8, fontweight='bold')
    ax.text(0.05, 0.95, f'Performance: {r["avg_score"]}/100',
            transform=ax.transAxes, color=score_color, fontsize=9, fontweight='bold', va='top')
    ax.text(0.05, 0.85, r["status"],
            transform=ax.transAxes, color=score_color, fontsize=8, va='top')
    ax.text(0.05, 0.75, f'Dev: {r["mean_dev_signed"]:+.1f}%',
            transform=ax.transAxes, color='#aaaaaa', fontsize=7, va='top')
    ax.set_ylabel('SAR RVI', color='white', fontsize=8)
    ax.set_xlabel('Days from Oct 1', color='white', fontsize=7)
    ax.legend(facecolor='#1a2a3a', labelcolor='white', fontsize=6, loc='lower right')

# Bottom row — deviation comparison across all fields
ax_dev = fig.add_subplot(gs[1, :2])
ax_dev.set_facecolor('#0a1525')
ax_dev.tick_params(colors='white')
for spine in ax_dev.spines.values():
    spine.set_color('#333')
ax_dev.grid(alpha=0.1, color='white')

for r, color in zip(results, colors):
    dev_x = [d for d,v in zip(r["all_days"],r["deviations"]) if v is not None]
    dev_y = [v for v in r["deviations"] if v is not None]
    ax_dev.plot(dev_x, dev_y, color=color, linewidth=1.5,
               marker='o', markersize=3, label=r["label"].split('·')[0].strip(), alpha=0.85)

ax_dev.axhline(y=0, color='white', linewidth=1, alpha=0.5)
ax_dev.axhline(y=10, color='#44ff44', linewidth=0.5, alpha=0.3, linestyle='--')
ax_dev.axhline(y=-10, color='#ff4444', linewidth=0.5, alpha=0.3, linestyle='--')
ax_dev.set_ylabel('Deviation from baseline (%)', color='white', fontsize=10)
ax_dev.set_xlabel('Days from Oct 1 (season start)', color='white', fontsize=10)
ax_dev.set_title('Cross-Field Deviation Comparison', color='white', fontsize=11, fontweight='bold')
ax_dev.legend(facecolor='#1a2a3a', labelcolor='white', fontsize=9)

# Bottom right — summary panel
ax_sum = fig.add_subplot(gs[1, 2])
ax_sum.set_facecolor('#0a1525')
ax_sum.axis('off')
ax_sum.set_title('Field Summary 2024/25', color='white', fontsize=11, fontweight='bold')

y = 0.92
ax_sum.text(0.05, y, 'Baseline: 2021/22 · 2022/23 · 2023/24',
            transform=ax_sum.transAxes, color='#888888', fontsize=8, va='top', style='italic')
y -= 0.12

for r, color in zip(results, colors):
    score_color = '#44ff44' if r["avg_score"] >= 90 else '#ffaa00' if r["avg_score"] >= 80 else '#ff4444'
    ax_sum.text(0.05, y, r["label"].split('·')[0].strip(),
                transform=ax_sum.transAxes, color=color, fontsize=10, fontweight='bold', va='top')
    y -= 0.08
    ax_sum.text(0.05, y, f'  Performance: {r["avg_score"]}/100  {r["status"]}',
                transform=ax_sum.transAxes, color=score_color, fontsize=9, va='top')
    y -= 0.08
    ax_sum.text(0.05, y, f'  Mean: {r["mean_dev_signed"]:+.1f}%  Max: {r["max_dev"]:+.1f}%',
                transform=ax_sum.transAxes, color='#aaaaaa', fontsize=8, va='top')
    y -= 0.07
    ax_sum.text(0.05, y, f'  Below baseline: {r["obs_below"]}/{r["total_obs"]} obs',
                transform=ax_sum.transAxes, color='#aaaaaa', fontsize=8, va='top')
    y -= 0.07
    if r["first_deviation_day"] is not None:
        ax_sum.text(0.05, y, f'  First flag: Day {r["first_deviation_day"]} ({r["deviation_duration"]} obs)',
                    transform=ax_sum.transAxes, color='#ffaa00', fontsize=8, va='top')
    y -= 0.10

ax_sum.text(0.05, y, 'Method: Sentinel-1 SAR RVI',
            transform=ax_sum.transAxes, color='#888888', fontsize=7, va='top', style='italic')
y -= 0.07
ax_sum.text(0.05, y, 'NDVI integration pending',
            transform=ax_sum.transAxes, color='#888888', fontsize=7, va='top', style='italic')

plt.savefig('multi_field_analysis.png', dpi=150, bbox_inches='tight',
            facecolor='#0a1525')
print('\nSaved: multi_field_analysis.png')
plt.close()
