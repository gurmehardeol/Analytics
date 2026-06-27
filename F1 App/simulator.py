"""F1 Race Counterfactual Simulator — Streamlit app

Load a race via FastF1 or from saved CSVs, then tweak pit strategy,
tyre compounds, safety-car timing, and driver pace to see how the
result changes.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import fastf1
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ── constants ─────────────────────────────────────────────────────────────────
SC_SLOWDOWN = 1.28  # SC laps ~28 % slower than median pace
COMPOUND_COLOR = {
    "SOFT": "#E8002D",
    "MEDIUM": "#FFF500",
    "HARD": "#C8C8C8",
    "INTERMEDIATE": "#39B54A",
    "WET": "#0067FF",
}

st.set_page_config(page_title="F1 Simulator", page_icon="🏎", layout="wide")
st.title("F1 Race Counterfactual Simulator")


# ── helpers ───────────────────────────────────────────────────────────────────

@st.cache_resource
def _f1_module():
    """Import F1 API.py (filename has a space, so use importlib)."""
    spec = importlib.util.spec_from_file_location(
        "f1_api", Path(__file__).parent / "F1 API.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@st.cache_data(show_spinner="Loading circuit telemetry (cached after first load)…")
def load_circuit_map(year: int, event: str, cache_dir: str):
    """Return (track_xy, corners) DataFrames for the circuit layout."""
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir)
    ev = int(event) if event.isdigit() else event
    sess = fastf1.get_session(year, ev, "R")
    sess.load(laps=True, telemetry=True, weather=False, messages=False)
    lap = sess.laps.pick_fastest()
    tel = lap.get_telemetry()[["X", "Y"]].reset_index(drop=True)
    try:
        ci = sess.get_circuit_info()
        corners = ci.corners[["X", "Y", "Number", "Letter"]].copy()
    except Exception:
        corners = pd.DataFrame()
    return tel, corners


# ── simulation engine ─────────────────────────────────────────────────────────

def run_sim(
    stints: pd.DataFrame,
    degradation: pd.DataFrame,
    driver_pace: pd.DataFrame,
    results: pd.DataFrame,
    *,
    sc_laps: set[int],
    pace_deltas: dict[str, float],
    pit_loss_s: float,
) -> pd.DataFrame:
    """
    Lap-by-lap race simulation.

    lap_time = driver_base_pace + compound_deg_rate * tyre_age [+ pit_loss on out-lap]

    The driver's fuel-corrected median pace is used as the base (tyre-age ~0 proxy).
    Switching compound changes the degradation slope; pace_deltas add a flat offset.
    """
    total = int(stints["end_lap"].max())
    deg = degradation.set_index("compound")["deg_s_per_lap"].to_dict()
    pace_map = driver_pace.set_index("driver")["median_pace_s"].to_dict()
    sc_pace = driver_pace["median_pace_s"].median() * SC_SLOWDOWN

    # Per-driver stint plan and pit out-laps
    plan: dict[str, list] = {
        drv: grp.sort_values("start_lap")[["compound", "start_lap", "end_lap"]].to_dict("records")
        for drv, grp in stints.groupby("driver")
    }
    pit_laps: dict[str, set] = {
        drv: {int(r["start_lap"]) for r in rows if int(r["start_lap"]) > 1}
        for drv, rows in plan.items()
    }

    # Only simulate classified finishers (DNFs keep their actual result)
    active = results[results["status"].str.contains("Finished|Lap", na=False)]["driver"].tolist()

    rows = []
    for drv in active:
        if drv not in pace_map:
            continue
        base = pace_map[drv] + pace_deltas.get(drv, 0.0)
        pits = pit_laps.get(drv, set())
        race_t = 0.0

        for lap in range(1, total + 1):
            compound, age = "HARD", lap
            for s in plan.get(drv, []):
                if s["start_lap"] <= lap <= s["end_lap"]:
                    compound = s["compound"]
                    age = lap - int(s["start_lap"]) + 1
                    break

            if lap in sc_laps:
                lt = sc_pace
            else:
                lt = base + deg.get(compound, 0.0) * age

            if lap in pits:
                lt += pit_loss_s

            race_t += lt

        rows.append({"driver": drv, "sim_time_s": round(race_t, 3)})

    sim = pd.DataFrame(rows).sort_values("sim_time_s").reset_index(drop=True)
    sim["sim_pos"] = range(1, len(sim) + 1)
    out = sim.merge(results[["driver", "finish", "team", "grid"]], on="driver", how="left")
    out["pos_delta"] = out["finish"].astype(float) - out["sim_pos"]
    return out


# ── sidebar: data loading ─────────────────────────────────────────────────────
with st.sidebar:
    st.header("Load Race")
    year = st.number_input("Year", 2018, 2025, 2024, step=1)
    event = st.text_input("Event (name or round number)", "Monza")
    cache_dir = st.text_input("Cache dir", "cache")
    data_dir = st.text_input("Data dir", "data")
    c1, c2 = st.columns(2)
    btn_live = c1.button("Fetch Live", type="primary", use_container_width=True)
    btn_csv = c2.button("Load CSVs", use_container_width=True)
    st.caption("'Fetch Live' downloads from FastF1 and saves CSVs. 'Load CSVs' reads previously saved data.")

# ── session state ─────────────────────────────────────────────────────────────
_defaults = {"T": None, "label": "", "pace_d": {}, "year": 2024, "event": "Monza",
             "cache_dir": "cache", "show_circuit": False}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── live fetch ────────────────────────────────────────────────────────────────
if btn_live:
    try:
        f1 = _f1_module()
        f1.enable_cache(cache_dir)
        ev = int(event) if event.isdigit() else event
        with st.spinner(f"Fetching {year} {event} from FastF1…"):
            data = f1.collect_race_data(year, ev)
            T = {k: v for k, v in data.__dict__.items()}
            data.write(data_dir)
        st.session_state.T = T
        st.session_state.label = f"{year} {event}"
        st.session_state.pace_d = {}
        st.session_state.year = int(year)
        st.session_state.event = event
        st.session_state.cache_dir = cache_dir
        st.session_state.show_circuit = False
        st.sidebar.success(f"Loaded and saved to {data_dir}/")
    except Exception as exc:
        st.sidebar.error(str(exc))

# ── csv load ──────────────────────────────────────────────────────────────────
if btn_csv:
    path = Path(data_dir)
    names = ["laps", "stints", "pit_stops", "degradation", "driver_pace",
             "events", "weather", "results"]
    T = {n: pd.read_csv(path / f"{n}.csv") for n in names if (path / f"{n}.csv").exists()}
    if T:
        st.session_state.T = T
        st.session_state.label = f"From {data_dir}/"
        st.session_state.pace_d = {}
        st.session_state.year = int(year)
        st.session_state.event = event
        st.session_state.cache_dir = cache_dir
        st.session_state.show_circuit = False
        st.sidebar.success(f"Loaded {len(T)} tables")
    else:
        st.sidebar.error(f"No CSVs found in {data_dir}/")

# ── gate ──────────────────────────────────────────────────────────────────────
if st.session_state.T is None:
    st.info("Use the sidebar to load a race — fetch live from FastF1 or load previously saved CSVs.")
    st.stop()

# ── unpack ────────────────────────────────────────────────────────────────────
T = st.session_state.T
stints = T["stints"].copy()
stints["start_lap"] = stints["start_lap"].astype(int)
stints["end_lap"] = stints["end_lap"].astype(int)
pit_stops = T["pit_stops"]
degradation = T["degradation"]
driver_pace = T["driver_pace"]
results = T["results"]
events_df = T.get("events", pd.DataFrame())
total_laps = int(stints["end_lap"].max())
compounds = sorted(degradation["compound"].unique())
drivers = sorted(results["driver"].unique())
default_pit_s = float(pit_stops["stationary_plus_lane_s"].median()) if len(pit_stops) else 24.0

st.caption(f"Race: **{st.session_state.label}** — {total_laps} laps · {len(drivers)} drivers")

tab_ov, tab_sim = st.tabs(["Race Overview", "Simulation"])


# ═════════════════════════════════════════════════════════════════════════════
# OVERVIEW TAB
# ═════════════════════════════════════════════════════════════════════════════
with tab_ov:

    # Results + pace side by side
    col_r, col_p = st.columns(2)
    with col_r:
        st.markdown("**Actual Race Result**")
        st.dataframe(results, use_container_width=True, hide_index=True)

    with col_p:
        st.markdown("**Driver Pace (fuel-corrected median, s/lap)**")
        fig_pace = px.bar(
            driver_pace, x="driver", y="median_pace_s", color="team",
            labels={"median_pace_s": "Pace (s)", "driver": ""},
        )
        fig_pace.update_layout(showlegend=False, height=320, margin=dict(t=5, b=5))
        st.plotly_chart(fig_pace, use_container_width=True)

    # Degradation model
    col_dt, col_dc = st.columns([1, 2])
    with col_dt:
        st.markdown("**Tyre Degradation Model**")
        st.dataframe(degradation, use_container_width=True, hide_index=True)

    with col_dc:
        st.markdown("**Degradation Curves**")
        ages = np.arange(0, 51)
        fig_deg = go.Figure()
        for _, row in degradation.iterrows():
            y = row["base_pace_s"] + row["deg_s_per_lap"] * ages
            fig_deg.add_trace(go.Scatter(
                x=ages, y=y, mode="lines", name=row["compound"],
                line=dict(color=COMPOUND_COLOR.get(row["compound"], "#aaa"), width=2),
            ))
        fig_deg.update_layout(
            xaxis_title="Tyre Age (laps)", yaxis_title="Lap Time (s)",
            height=250, margin=dict(t=5, b=5), legend=dict(orientation="h"),
        )
        st.plotly_chart(fig_deg, use_container_width=True)

    # Stint timeline (Gantt via horizontal stacked bars)
    st.markdown("**Stint Timeline**")
    fig_gantt = go.Figure()
    seen_compounds: set[str] = set()
    for _, r in stints.iterrows():
        cmp = r["compound"]
        color = COMPOUND_COLOR.get(cmp, "#888")
        fig_gantt.add_trace(go.Bar(
            x=[r["laps_in_stint"]],
            y=[r["driver"]],
            orientation="h",
            base=r["start_lap"] - 1,
            marker_color=color,
            marker_line_color="rgba(0,0,0,0.5)",
            marker_line_width=0.8,
            text=cmp[0],
            textposition="inside",
            name=cmp,
            showlegend=cmp not in seen_compounds,
            legendgroup=cmp,
        ))
        seen_compounds.add(cmp)
    fig_gantt.update_layout(
        barmode="stack",
        xaxis_title="Lap",
        yaxis_title="",
        legend_title="Compound",
        height=max(320, len(drivers) * 26),
        margin=dict(l=10, t=5, b=30),
    )
    st.plotly_chart(fig_gantt, use_container_width=True)

    # Race control events
    if events_df is not None and len(events_df):
        st.markdown("**Race Control Events**")
        st.dataframe(events_df, use_container_width=True, hide_index=True)

    # Circuit map (lazy — requires extra telemetry load)
    st.markdown("**Circuit Map**")
    if not st.session_state.show_circuit:
        if st.button("Load Circuit Map", help="Fetches fastest-lap telemetry to draw the track layout. Cached after first load."):
            st.session_state.show_circuit = True
            st.rerun()
    else:
        try:
            tel, corners = load_circuit_map(
                st.session_state.year,
                st.session_state.event,
                st.session_state.cache_dir,
            )
            fig_circ = go.Figure()
            # Track outline
            fig_circ.add_trace(go.Scatter(
                x=tel["X"], y=tel["Y"],
                mode="lines",
                line=dict(color="#E8002D", width=4),
                name="Track",
                showlegend=False,
            ))
            # Corner markers
            if len(corners):
                fig_circ.add_trace(go.Scatter(
                    x=corners["X"], y=corners["Y"],
                    mode="markers+text",
                    text=(corners["Number"].astype(str) + corners["Letter"].fillna("")).tolist(),
                    textposition="top center",
                    textfont=dict(color="white", size=10),
                    marker=dict(color="white", size=7, line=dict(color="#333", width=1)),
                    name="Corners",
                    showlegend=False,
                ))
            fig_circ.update_layout(
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x"),
                plot_bgcolor="#1a1a2e",
                paper_bgcolor="#1a1a2e",
                height=520,
                margin=dict(l=10, r=10, t=10, b=10),
            )
            st.plotly_chart(fig_circ, use_container_width=True)
        except Exception as exc:
            st.error(f"Could not load circuit map: {exc}")
            st.session_state.show_circuit = False


# ═════════════════════════════════════════════════════════════════════════════
# SIMULATION TAB
# ═════════════════════════════════════════════════════════════════════════════
with tab_sim:
    st.markdown("### Pit Strategy")
    st.caption(
        "Edit **Compound** to simulate a tyre change. "
        "Edit **Out-Lap** to move a pit stop earlier or later. "
        "Add rows to simulate an extra stop; delete rows to go one-stop."
    )

    edited_stints = st.data_editor(
        stints[["driver", "stint", "compound", "start_lap", "end_lap"]],
        column_config={
            "driver":    st.column_config.TextColumn("Driver",   disabled=True),
            "stint":     st.column_config.NumberColumn("Stint",  disabled=True),
            "compound":  st.column_config.SelectboxColumn("Compound", options=compounds),
            "start_lap": st.column_config.NumberColumn("Out-Lap", min_value=1, max_value=total_laps),
            "end_lap":   st.column_config.NumberColumn("End Lap", min_value=1, max_value=total_laps),
        },
        use_container_width=True,
        num_rows="dynamic",
        key="stints_ed",
    )

    col_sc, col_pit = st.columns(2)
    sc_laps: set[int] = set()

    with col_sc:
        st.markdown("### Safety Car")
        sc_on = st.toggle("Add / reposition Safety Car")
        if sc_on:
            sc_start = st.slider("SC deploy lap", 1, total_laps, max(1, total_laps // 3))
            sc_dur   = st.slider("SC duration (laps)", 1, 10, 4)
            sc_laps  = set(range(sc_start, sc_start + sc_dur + 1))
            st.info(f"SC active on laps {sc_start}–{sc_start + sc_dur} ({sc_dur + 1} laps at {SC_SLOWDOWN:.0%} pace)")

    with col_pit:
        st.markdown("### Pit Stop Loss")
        pit_s = st.slider("Pit loss per stop (s)", 15.0, 45.0, default_pit_s, step=0.5)
        st.caption(f"Actual median from this race: **{default_pit_s:.1f} s**")

    # Driver pace adjustment
    st.markdown("### Driver Pace Adjustment")
    st.caption("Simulate a faster car, setup gain, or a driver error by shifting their per-lap base time.")
    col_drv, col_sl = st.columns([1, 2])
    sel_drv = col_drv.selectbox("Driver", drivers)
    delta = col_sl.slider(
        f"{sel_drv} pace delta (s / lap)",
        min_value=-3.0, max_value=3.0,
        value=float(st.session_state.pace_d.get(sel_drv, 0.0)),
        step=0.1,
    )
    if delta != 0.0:
        st.session_state.pace_d[sel_drv] = delta
    elif sel_drv in st.session_state.pace_d:
        del st.session_state.pace_d[sel_drv]

    if st.session_state.pace_d:
        active_str = ", ".join(f"**{d}** {v:+.1f}s" for d, v in st.session_state.pace_d.items())
        st.caption(f"Active: {active_str}")

    if st.button("Reset All Pace Adjustments", type="secondary"):
        st.session_state.pace_d = {}
        st.rerun()

    st.divider()

    run_btn = st.button("Run Simulation", type="primary", use_container_width=True)

    if run_btn:
        with st.spinner("Simulating race…"):
            result = run_sim(
                edited_stints, degradation, driver_pace, results,
                sc_laps=sc_laps,
                pace_deltas=st.session_state.pace_d,
                pit_loss_s=pit_s,
            )

        st.markdown("### Simulated Result")

        # Podium cards
        p1, p2, p3 = st.columns(3)
        for col, pos, label in [(p1, 1, "P1"), (p2, 2, "P2"), (p3, 3, "P3")]:
            row = result[result["sim_pos"] == pos]
            if len(row):
                r = row.iloc[0]
                actual = f"was P{int(r['finish'])}" if pd.notna(r["finish"]) else "DNF"
                col.metric(label, r["driver"], actual)

        # Full result table
        disp = result[["sim_pos", "driver", "team", "sim_time_s", "finish", "pos_delta"]].copy()
        disp.columns = ["Sim Pos", "Driver", "Team", "Sim Time (s)", "Actual Pos", "Δ Pos"]
        disp["Δ Pos"] = disp["Δ Pos"].apply(lambda x: f"{int(x):+d}" if pd.notna(x) else "—")
        st.dataframe(disp, use_container_width=True, hide_index=True)

        # Position delta bar chart
        deltas = result["pos_delta"].fillna(0)
        bar_colors = ["#2ecc71" if d > 0 else "#e74c3c" if d < 0 else "#95a5a6" for d in deltas]
        fig_bar = go.Figure(go.Bar(
            x=result["driver"],
            y=deltas,
            marker_color=bar_colors,
            text=[f"{int(d):+d}" for d in deltas],
            textposition="outside",
        ))
        fig_bar.update_layout(
            title="Position change vs actual result (green = gained places in simulation)",
            xaxis_title="Driver",
            yaxis_title="Positions gained in sim",
            height=380,
            margin=dict(t=40),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # Gap to leader
        leader_time = result["sim_time_s"].min()
        result["gap_to_leader_s"] = (result["sim_time_s"] - leader_time).round(3)
        fig_gap = px.bar(
            result, x="driver", y="gap_to_leader_s", color="team",
            labels={"gap_to_leader_s": "Gap (s)", "driver": ""},
            title="Simulated gap to race leader",
        )
        fig_gap.update_layout(showlegend=False, height=320, margin=dict(t=40))
        st.plotly_chart(fig_gap, use_container_width=True)
