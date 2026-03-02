"""
simulate.py - REAL EPANET simulation using WNTR
Network: Main Station → J1 → J2 → J3 → Home
"""

import wntr
import pandas as pd
import numpy as np
import os

np.random.seed(42)
os.makedirs("data", exist_ok=True)
os.makedirs("network", exist_ok=True)

# ─────────────────────────────────────────────
# STEP 1: BUILD THE WATER NETWORK
# ─────────────────────────────────────────────
def build_network():
    print("Building water network in EPANET...")

    wn = wntr.network.WaterNetworkModel()

    # Add Reservoir (Main Water Source)
    wn.add_reservoir(
        name         = "Station",
        base_head    = 60.0,        # 60 meters head pressure
        coordinates  = (0, 0)
    )

    # Add Junctions (nodes in the network)
    # (name, base_demand, elevation, coordinates)
    wn.add_junction("J1",   base_demand=0.002, elevation=5.0,  coordinates=(100, 0))
    wn.add_junction("J2",   base_demand=0.002, elevation=8.0,  coordinates=(200, 0))
    wn.add_junction("J3",   base_demand=0.002, elevation=10.0, coordinates=(300, 0))
    wn.add_junction("Home", base_demand=0.005, elevation=12.0, coordinates=(400, 0))

    # Add Pipes connecting the nodes
    # (name, start, end, length_m, diameter_m, roughness)
    wn.add_pipe("Pipe_Sta_J1", "Station", "J1",   length=200, diameter=0.15, roughness=100)
    wn.add_pipe("Pipe_J1_J2",  "J1",     "J2",   length=180, diameter=0.12, roughness=100)
    wn.add_pipe("Pipe_J2_J3",  "J2",     "J3",   length=160, diameter=0.10, roughness=100)
    wn.add_pipe("Pipe_J3_Home","J3",     "Home", length=120, diameter=0.08, roughness=100)

    # Simulation settings
    wn.options.time.duration         = 3600 * 24   # 24 hours
    wn.options.time.hydraulic_timestep = 3600      # every 1 hour
    wn.options.time.report_timestep    = 3600

    # Save network file
    wntr.network.write_inpfile(wn, "network/water_network.inp")
    print("Network saved → network/water_network.inp")
    return wn

# ─────────────────────────────────────────────
# STEP 2: SIMULATE NORMAL CONDITIONS
# ─────────────────────────────────────────────
def simulate_normal(wn):
    sim     = wntr.sim.EpanetSimulator(wn)
    results = sim.run_sim()
    return results

# ─────────────────────────────────────────────
# STEP 3: SIMULATE LEAK CONDITIONS
# ─────────────────────────────────────────────
def simulate_with_leak(wn_original, leak_node, leak_area):
    """
    Adds a leak at a specific junction and simulates.
    leak_area: size of the leak in m²
      - Moderate: 0.005 to 0.015
      - Critical:  0.020 to 0.050
    """
    # Fresh copy of network for each simulation
    wn = wntr.network.WaterNetworkModel("network/water_network.inp")

    # Add leak at the specified junction
    node = wn.get_node(leak_node)
    node.add_leak(wn, area=leak_area, start_time=0)

    # Run simulation with leak
    sim     = wntr.sim.EpanetSimulator(wn)
    results = sim.run_sim()
    return results

# ─────────────────────────────────────────────
# STEP 4: EXTRACT FEATURES FROM RESULTS
# ─────────────────────────────────────────────
def extract_features(results, label, label_name):
    """
    Extracts pressure & flow readings from EPANET results
    and returns a list of rows (one per timestep).
    """
    rows = []

    pressure = results.node["pressure"]   # DataFrame: rows=time, cols=nodes
    flowrate = results.link["flowrate"]   # DataFrame: rows=time, cols=pipes

    for t in pressure.index:
        try:
            row = {
                # Pressures at each node
                "Station_pressure": abs(pressure.loc[t, "Station"]),
                "J1_pressure":      abs(pressure.loc[t, "J1"]),
                "J2_pressure":      abs(pressure.loc[t, "J2"]),
                "J3_pressure":      abs(pressure.loc[t, "J3"]),
                "Home_pressure":    abs(pressure.loc[t, "Home"]),

                # Flow rates in each pipe
                "Flow_Sta_J1":  abs(flowrate.loc[t, "Pipe_Sta_J1"]) * 1000,
                "Flow_J1_J2":   abs(flowrate.loc[t, "Pipe_J1_J2"])  * 1000,
                "Flow_J2_J3":   abs(flowrate.loc[t, "Pipe_J2_J3"])  * 1000,
                "Flow_J3_Home": abs(flowrate.loc[t, "Pipe_J3_Home"])* 1000,

                # Derived features (pressure drops between nodes)
                "Pressure_drop_J1": abs(pressure.loc[t, "Station"]) - abs(pressure.loc[t, "J1"]),
                "Pressure_drop_J2": abs(pressure.loc[t, "J1"])      - abs(pressure.loc[t, "J2"]),
                "Pressure_drop_J3": abs(pressure.loc[t, "J2"])      - abs(pressure.loc[t, "J3"]),

                # Flow losses between segments
                "Flow_loss_J1J2": (abs(flowrate.loc[t, "Pipe_Sta_J1"]) - abs(flowrate.loc[t, "Pipe_J1_J2"])) * 1000,
                "Flow_loss_J2J3": (abs(flowrate.loc[t, "Pipe_J1_J2"])  - abs(flowrate.loc[t, "Pipe_J2_J3"])) * 1000,

                "label":      label,
                "label_name": label_name,
            }
            rows.append(row)
        except Exception as e:
            continue

    return rows

# ─────────────────────────────────────────────
# STEP 5: BUILD FULL DATASET
# ─────────────────────────────────────────────
def build_dataset():
    print("\n" + "="*55)
    print("  EPANET WATER NETWORK — DATASET GENERATION")
    print("  Network: Station → J1 → J2 → J3 → Home")
    print("="*55)

    # Build and save the network
    wn = build_network()
    all_rows = []

    # ── NORMAL SIMULATIONS ──
    print("\n✅ Simulating Normal conditions (no leak)...")
    for i in range(50):
        # Slightly vary demands to get more variety
        wn2 = wntr.network.WaterNetworkModel("network/water_network.inp")
        for jname in ["J1", "J2", "J3", "Home"]:
                j = wn2.get_node(jname)
                j.demand_timeseries_list[0].base_value = j.demand_timeseries_list[0].base_value * np.random.uniform(0.85, 1.15)
        sim     = wntr.sim.EpanetSimulator(wn2)
        results = sim.run_sim()
        rows    = extract_features(results, 0, "Normal")
        all_rows.extend(rows)

    print(f"   Normal rows collected: {len(all_rows)}")

    # ── MODERATE LEAK SIMULATIONS ──
    print("\n⚠️  Simulating Moderate Leak conditions...")
    moderate_start = len(all_rows)
    leak_nodes = ["J1", "J2", "J3"]
    for i in range(50):
        node      = np.random.choice(leak_nodes)
        leak_area = np.random.uniform(0.005, 0.015)
        try:
            results = simulate_with_leak(wn, node, leak_area)
            rows    = extract_features(results, 1, "Moderate")
            all_rows.extend(rows)
        except Exception as e:
            print(f"   Skipped one simulation: {e}")

    print(f"   Moderate rows collected: {len(all_rows) - moderate_start}")

    # ── CRITICAL LEAK SIMULATIONS ──
    print("\n🚨 Simulating Critical Leak conditions...")
    critical_start = len(all_rows)
    for i in range(50):
        node      = np.random.choice(leak_nodes)
        leak_area = np.random.uniform(0.020, 0.050)
        try:
            results = simulate_with_leak(wn, node, leak_area)
            rows    = extract_features(results, 2, "Critical")
            all_rows.extend(rows)
        except Exception as e:
            print(f"   Skipped one simulation: {e}")

    print(f"   Critical rows collected: {len(all_rows) - critical_start}")

    # ── SAVE DATASET ──
    df = pd.DataFrame(all_rows).dropna()
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv("data/pipeline_dataset.csv", index=False)

    print(f"\n{'='*55}")
    print(f"✅ Dataset saved → data/pipeline_dataset.csv")
    print(f"   Total rows : {len(df)}")
    print(f"   Normal     : {len(df[df.label==0])}")
    print(f"   Moderate   : {len(df[df.label==1])}")
    print(f"   Critical   : {len(df[df.label==2])}")
    print(f"{'='*55}")
    return df

if __name__ == "__main__":
    df = build_dataset()
    print("\nSample data:")
    print(df[["J1_pressure","J2_pressure","J3_pressure",
              "Flow_J1_J2","Flow_J2_J3","label_name"]].head(8).to_string(index=False))