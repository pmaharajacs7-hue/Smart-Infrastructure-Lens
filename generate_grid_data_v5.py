"""
Power Grid Data Generator v4 — Realistic Noise
Fixes:
  1. Restores all original columns (timestamp, asset_id, busbar_delta_t, current_amps, etc.)
  2. Adds boundary ambiguity — values near thresholds get probabilistic labels
  3. Adds sensor drift, outliers, and missing-value simulation
  4. Injects correlated noise (not just independent Gaussian per sensor)
  5. Preserves inter-parameter correlation from v2

This will break the perfect 1.00 score by making the classification
problem genuinely hard near decision boundaries.
"""



import pandas as pd
import numpy as np
import random
import os
from datetime import datetime, timedelta

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

NUM_RECORDS = 5000
OUTPUT_DIR  = r"."   # change to your folder

# ─── Helpers ──────────────────────────────────────────────────────────────────

def timestamps(n, start="2024-01-01 00:00:00", interval_minutes=5):
    dt = datetime.strptime(start, "%Y-%m-%d %H:%M:%S")
    return [(dt + timedelta(minutes=i * interval_minutes)).strftime("%Y-%m-%d %H:%M:%S")
            for i in range(n)]

def sensor_noise(value, intensity=0.02):
    """Gaussian jitter proportional to value magnitude."""
    return value + np.random.normal(0, max(abs(value) * intensity, 1e-6))

def drift_noise(n, base, sigma, drift_rate=0.0003):
    """
    AR(1) series with slow drift — simulates sensor calibration drift over time.
    drift_rate: how much the mean shifts per step
    """
    values = []
    x = base
    mean = base
    for i in range(n):
        mean += np.random.normal(0, drift_rate * abs(base))  # slow random walk of mean
        x = 0.92 * x + 0.08 * mean + np.random.normal(0, sigma)
        values.append(x)
    return np.array(values)

def boundary_label(value, thresholds, base_label, flip_prob_scale=0.4):
    """
    Assigns label with uncertainty near boundaries.
    
    The closer a value is to a threshold, the higher the chance
    the label flips to the adjacent class — simulating:
      - Real-world sensor uncertainty
      - Human labeling inconsistency  
      - Threshold ambiguity in standards
    
    thresholds: dict with keys warn_lo, warn_hi, crit_lo, crit_hi
    """
    warn_lo = thresholds["warn_lo"]
    warn_hi = thresholds["warn_hi"]
    crit_lo = thresholds["crit_lo"]
    crit_hi = thresholds["crit_hi"]

    # Distance to nearest threshold (normalized)
    distances = [
        abs(value - warn_lo) / (abs(warn_lo) + 1e-6),
        abs(value - warn_hi) / (abs(warn_hi) + 1e-6),
        abs(value - crit_lo) / (abs(crit_lo) + 1e-6),
        abs(value - crit_hi) / (abs(crit_hi) + 1e-6),
    ]
    min_dist = min(distances)

    # Flip probability: high when close to boundary, near zero when far away
    # sigmoid-shaped: values within 3% of threshold have ~25% flip chance
    flip_prob = flip_prob_scale * np.exp(-min_dist * 30)

    if random.random() < flip_prob:
        # Flip to adjacent class
        adjacent = {"Normal": "Warning", "Warning": "Critical",
                    "Critical": "Warning"}
        return adjacent.get(base_label, base_label)

    return base_label

def overall_status(*labels):
    p = {"Normal": 0, "Warning": 1, "Critical": 2}
    return max(labels, key=lambda x: p[x])

# ─── 1. TRANSFORMERS ──────────────────────────────────────────────────────────
# Columns: timestamp, asset_id, oil_temp_c, load_pct, vibration_um, overall_status
# Correlation: load → oil_temp → vibration (same as v2)

XFMR_THRESH = {
    "oil":  {"warn_lo": 20,  "warn_hi": 90,  "crit_lo": -99, "crit_hi": 105},
    "load": {"warn_lo": 0,   "warn_hi": 90,  "crit_lo": -99, "crit_hi": 120},
    "vib":  {"warn_lo": 0,   "warn_hi": 5,   "crit_lo": -99, "crit_hi": 20},
}

def gen_transformer_data():
    ts_list   = timestamps(NUM_RECORDS)
    load_vals = drift_noise(NUM_RECORDS, base=58, sigma=12)

    # Inject multiple critical events — more frequent and deeper
    critical_windows = [
        (600,  750,  75),   # moderate overload
        (1800, 2000, 90),   # severe overload
        (3200, 3350, 80),   # moderate
        (4500, 4700, 95),   # worst case
    ]

    rows = []
    for i in range(NUM_RECORDS):
        asset_id = f"TRF-{(i % 5) + 1:02d}"
        load = float(load_vals[i])

        # Apply critical event drift
        for start, end, magnitude in critical_windows:
            if start <= i < end:
                load += magnitude * np.sin(np.pi * (i - start) / (end - start))

        load = float(np.clip(load, 0, 160))

        oil_base = 45 + 0.38 * load
        oil  = float(sensor_noise(oil_base, intensity=0.025))
        oil  = max(10.0, oil)

        vib_base = 2.0 + 0.22 * max(0, (oil - 65) / 15)
        vib  = float(sensor_noise(vib_base, intensity=0.04))
        vib  = max(0.0, vib)

        # Base labels
        if oil > 105 or load > 120 or vib > 20:
            base_lbl = "Critical"
        elif oil > 90 or load > 90 or vib > 5:
            base_lbl = "Warning"
        else:
            base_lbl = "Normal"

        # Boundary ambiguity — REDUCED flip_prob_scale near Critical (0.2 not 0.4)
        # so Critical events don't get randomly relabeled as Warning
        oil_lbl  = boundary_label(oil,  XFMR_THRESH["oil"],  base_lbl, flip_prob_scale=0.2)
        load_lbl = boundary_label(load, XFMR_THRESH["load"],
                                  "Critical" if load > 120 else "Warning" if load > 90 else "Normal",
                                  flip_prob_scale=0.2)
        vib_lbl  = boundary_label(vib,  XFMR_THRESH["vib"],
                                  "Critical" if vib > 20 else "Warning" if vib > 5 else "Normal",
                                  flip_prob_scale=0.2)

        ov = overall_status(oil_lbl, load_lbl, vib_lbl)

        rows.append({
            "timestamp":      ts_list[i],
            "asset_id":       asset_id,
            "asset_type":     "Transformer",
            "oil_temp_c":     round(oil,  2),
            "load_pct":       round(load, 2),
            "vibration_um":   round(vib,  3),
            "overall_status": ov,
        })
    return pd.DataFrame(rows)

# ─── 2. SUBSTATIONS ───────────────────────────────────────────────────────────
# Columns: timestamp, asset_id, sf6_pressure_bar, busbar_temp_c,
#          busbar_delta_t, voltage_stability_pu, overall_status
# Correlation: sf6 drop → voltage instability; busbar_delta_t → voltage drop

SUB_THRESH = {
    "sf6": {"warn_lo": 5.0, "warn_hi": 99,  "crit_lo": 4.5, "crit_hi": 99},
    "bbt": {"warn_lo": 0,   "warn_hi": 55,  "crit_lo": -99, "crit_hi": 65},
    "vs":  {"warn_lo": 0.95,"warn_hi": 1.05,"crit_lo": 0.90,"crit_hi": 1.10},
}

def gen_substation_data():
    ts_list  = timestamps(NUM_RECORDS)
    # Base SF6 at 5.9 — clearly Normal (>5.6), gives more headroom before Warning
    sf6_vals = drift_noise(NUM_RECORDS, base=5.90, sigma=0.05, drift_rate=0.00005)

    # Multiple SF6 leak events — gradual and deep enough to reach Critical
    leak_windows = [
        (1000, 1800, -0.6),   # moderate leak → Warning
        (2800, 3600, -1.1),   # severe leak → Critical
    ]

    rows = []
    for i in range(NUM_RECORDS):
        asset_id = f"SUB-{(i % 3) + 1:02d}"
        sf6 = float(sf6_vals[i])

        for start, end, magnitude in leak_windows:
            if start <= i < end:
                sf6 += magnitude * np.sin(np.pi * (i - start) / (end - start))

        sf6 = float(np.clip(sf6, 3.5, 7.5))

        delta_t_base = 7.0   # well within Normal (<10 threshold)
        if 3800 <= i < 4100:
            delta_t_base += 38 * np.sin(np.pi * (i - 3800) / 300)
        delta_t  = float(max(0, sensor_noise(delta_t_base, intensity=0.06)))
        busbar_t = round(25 + delta_t, 2)

        sf6_effect = max(0, (5.6 - sf6) * 0.045)
        bbt_effect = max(0, (delta_t - 10) * 0.0008)
        vs = float(np.clip(
            sensor_noise(1.00 - sf6_effect - bbt_effect, intensity=0.006),
            0.80, 1.15
        ))

        sf6_lbl = "Critical" if sf6 < 5.0 else "Warning" if sf6 < 5.6 else "Normal"
        bbt_lbl = "Critical" if delta_t > 35 else "Warning" if delta_t > 10 else "Normal"
        vs_lbl  = ("Critical" if vs < 0.90 or vs > 1.10
                   else "Warning" if vs < 0.95 or vs > 1.05
                   else "Normal")

        # Reduced flip scale (0.2) so Normal doesn't bleed into Warning as much
        sf6_lbl = boundary_label(sf6,     SUB_THRESH["sf6"], sf6_lbl, flip_prob_scale=0.2)
        bbt_lbl = boundary_label(delta_t, SUB_THRESH["bbt"], bbt_lbl, flip_prob_scale=0.2)
        vs_lbl  = boundary_label(vs,      SUB_THRESH["vs"],  vs_lbl,  flip_prob_scale=0.2)

        ov = overall_status(sf6_lbl, bbt_lbl, vs_lbl)

        rows.append({
            "timestamp":           ts_list[i],
            "asset_id":            asset_id,
            "asset_type":          "Substation",
            "sf6_pressure_bar":    round(sf6,     3),
            "busbar_temp_c":       busbar_t,
            "busbar_delta_t":      round(delta_t, 2),
            "voltage_stability_pu":round(vs,      4),
            "overall_status":      ov,
        })
    return pd.DataFrame(rows)

# ─── 3. POWER LINES ───────────────────────────────────────────────────────────
# Columns: timestamp, asset_id, current_amps, current_pct_rated,
#          ground_clearance_ft, overall_status
# Correlation: current → thermal sag → reduced clearance

RATED_AMPS = 800.0

PL_THRESH = {
    "curr_pct": {"warn_lo": 0,  "warn_hi": 80,  "crit_lo": -99, "crit_hi": 110},
    "clear":    {"warn_lo": 21, "warn_hi": 99,  "crit_lo": 20,  "crit_hi": 99},
}

def gen_powerline_data():
    ts_list      = timestamps(NUM_RECORDS)
    current_vals = drift_noise(NUM_RECORDS, base=440, sigma=28)

    rows = []
    for i in range(NUM_RECORDS):
        asset_id = f"PL-{(i % 8) + 1:02d}"

        # Inject load surge events
        base_current = current_vals[i]
        if 1400 <= i < 1600:
            base_current += 280 * np.sin(np.pi * (i - 1400) / 200)
        if 3600 <= i < 3900:
            base_current += 460 * np.sin(np.pi * (i - 3600) / 300)

        current     = float(max(0, sensor_noise(base_current, intensity=0.02)))
        current_pct = float(np.clip((current / RATED_AMPS) * 100, 0, 160))

        # Correlated thermal sag
        thermal_sag = max(0, (current_pct - 50) * 0.022)
        clearance   = float(max(15.0,
            sensor_noise(24.0 - thermal_sag, intensity=0.01)
        ))

        # Labels
        curr_lbl  = "Critical" if current_pct > 110 else "Warning" if current_pct > 80 else "Normal"
        clear_lbl = "Critical" if clearance < 20 else "Warning" if clearance < 21 else "Normal"

        curr_lbl  = boundary_label(current_pct, PL_THRESH["curr_pct"], curr_lbl)
        clear_lbl = boundary_label(clearance,   PL_THRESH["clear"],    clear_lbl)

        ov = overall_status(curr_lbl, clear_lbl)

        rows.append({
            "timestamp":           ts_list[i],
            "asset_id":            asset_id,
            "asset_type":          "PowerLine",
            "current_amps":        round(current,     1),
            "current_pct_rated":   round(current_pct, 2),
            "ground_clearance_ft": round(clearance,   3),
            "overall_status":      ov,
        })
    return pd.DataFrame(rows)

# ─── Write + report ───────────────────────────────────────────────────────────

def write(df, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    dist = df["overall_status"].value_counts()
    total = len(df)
    print(f"  {filename}  ({total} rows)")
    for label in ["Normal", "Warning", "Critical"]:
        n = dist.get(label, 0)
        print(f"     {label:8s}: {n:4d}  ({n/total*100:.1f}%)")

if __name__ == "__main__":
    print("=" * 55)
    print("Power Grid Data Generator v4 — Realistic Noise")
    print("=" * 55)
    print("\nKey changes vs your v3 noise script:")
    print("  + Restored: timestamp, asset_id, busbar_delta_t, current_amps")
    print("  + Boundary ambiguity: near-threshold values get probabilistic labels")
    print("  + Sensor drift via AR(1) — no instantaneous state jumps")
    print("  + Correlated noise chains preserved")
    print("  + This WILL break the 1.00 scores\n")

    write(gen_transformer_data(), "transformers_v5.csv")
    write(gen_substation_data(),  "substations_v5.csv")
    write(gen_powerline_data(),   "powerlines_v5.csv")

    print("\nDone. Update ASSETS in train_rf_model.py to use _v5 filenames.")
    print("Expected scores after retraining: F1-macro ~0.75-0.90")
    print("(Lower than 1.00 = more realistic = better for a real project)")