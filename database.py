"""
database.py — Supabase connector for Power Grid Command Center
==============================================================
Writes to the single `grid_analytics` table.

Setup:
    pip install supabase

    Set your credentials below or as environment variables:
        SUPABASE_URL = "https://xxxx.supabase.co"
        SUPABASE_KEY = "your-anon-or-service-role-key"
"""

import os
from datetime import datetime

# ── Credentials ───────────────────────────────────────────────────────────────
SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://stiqtzofcahrkyyxdfni.supabase.co")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "sb_publishable_UWIv-mEGSWy-kohs7rV_nw_PsECnyfe")

# ── Client (lazy init) ────────────────────────────────────────────────────────
_client = None

def get_client():
    global _client
    if _client is None:
        from supabase import create_client
        _client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _client


# ── Write one step ────────────────────────────────────────────────────────────

def write_step(sim_step: int, asset_type: str, asset_id: str,
               sim_timestamp: str, result: dict, metrics: dict):
    """
    Inserts one row into grid_analytics per asset per simulation step.
    Call this from dashboard.py inside the simulation loop.
    """
    ai     = result.get("ai_analysis", {})
    status = result["overall_status"]

    # Map engine status → table convention (engine uses HEALTHY/WARNING/CRITICAL)
    status_map = {"HEALTHY": "Normal", "WARNING": "Warning", "CRITICAL": "Critical"}
    status_clean = status_map.get(status, status)

    # Determine trigger source
    if ai.get("anomaly_detected") and status != "HEALTHY":
        source = "AI+RULE"
    elif ai.get("anomaly_detected"):
        source = "AI_ONLY"
    else:
        source = "RULE_ONLY"

    row = {
        "timestamp":      sim_timestamp,
        "asset_id":       asset_id,
        "asset_type":     asset_type,
        "overall_status": status_clean,
        "ai_anomaly":     ai.get("anomaly_detected", False),
        "ai_score":       ai.get("normality_score"),
        "trigger_source": source,
        "status_details": result.get("status_details", ""),
        # Sensor columns — None if not applicable to this asset type
        "oil_temp_c":           metrics.get("oil_temp_c"),
        "load_pct":             metrics.get("load_pct"),
        "vibration_um":         metrics.get("vibration_um"),
        "sf6_pressure_bar":     metrics.get("sf6_pressure_bar"),
        "busbar_temp_c":        metrics.get("busbar_temp_c"),
        "busbar_delta_t":       metrics.get("busbar_delta_t"),
        "voltage_stability_pu": metrics.get("voltage_stability_pu"),
        "current_amps":         metrics.get("current_amps"),
        "current_pct_rated":    metrics.get("current_pct_rated"),
        "ground_clearance_ft":  metrics.get("ground_clearance_ft"),
    }

    get_client().table("grid_analytics").insert(row).execute()


# ── Read helpers ──────────────────────────────────────────────────────────────

def fetch_recent_alerts(limit=50):
    """Returns recent Warning/Critical rows ordered by timestamp."""
    return (
        get_client()
        .table("grid_analytics")
        .select("*")
        .in_("overall_status", ["Warning", "Critical"])
        .order("timestamp", desc=True)
        .limit(limit)
        .execute()
        .data
    )


def fetch_readings(asset_type: str, limit=500):
    """Returns recent sensor readings for one asset type."""
    return (
        get_client()
        .table("grid_analytics")
        .select("*")
        .eq("asset_type", asset_type)
        .order("timestamp", desc=True)
        .limit(limit)
        .execute()
        .data
    )


# ── Test connection ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        rows = get_client().table("grid_analytics").select("id").limit(1).execute()
        print("Supabase connection OK")
        print(f"Connected to: {SUPABASE_URL}")
    except Exception as e:
        print(f"Connection failed: {e}")


# ── Pipeline write ────────────────────────────────────────────────────────────

def write_step_pipeline(sim_step: int, prediction: str,
                        sim_timestamp: str, metrics: dict, confidence: float):
    """
    Inserts one pipeline reading into grid_analytics.
    asset_type = 'PIPELINE', asset_id = 'PIPE-NET-01'
    """
    row = {
        "timestamp":      sim_timestamp,
        "asset_id":       "PIPE-NET-01",
        "asset_type":     "PIPELINE",
        "overall_status": prediction,          # Normal / Moderate / Critical
        "ai_anomaly":     prediction != "Normal",
        "ai_score":       round(float(confidence), 4),
        "trigger_source": "RF_CLASSIFIER",
        "status_details": f"confidence={round(confidence*100,1)}%",
        # Map pipeline sensors into available float columns
        "oil_temp_c":           metrics.get("J1_pressure"),
        "load_pct":             metrics.get("J2_pressure"),
        "vibration_um":         metrics.get("J3_pressure"),
        "sf6_pressure_bar":     metrics.get("Home_pressure"),
        "busbar_temp_c":        metrics.get("Flow_Sta_J1"),
        "busbar_delta_t":       metrics.get("Flow_J1_J2"),
        "voltage_stability_pu": metrics.get("Flow_J2_J3"),
        "current_amps":         metrics.get("Flow_J3_Home"),
        "current_pct_rated":    metrics.get("Flow_loss_J1J2"),
        "ground_clearance_ft":  metrics.get("Flow_loss_J2J3"),
    }
    get_client().table("grid_analytics").insert(row).execute()
