"""Pull F1 race data into tidy tables for counterfactual simulation.

Run locally (this needs network access to the F1 timing servers).
The output is a set of DataFrames keyed by what a race simulator consumes:
base pace, tyre degradation, pit losses, stints, and race-control events.

Usage:
    python f1_data.py --year 2023 --event Monza --out data/
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import fastf1


def enable_cache(cache_dir: str = "cache") -> None:
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir)

def load_race(year: int, event: str | int) -> fastf1.core.Session:
    session = fastf1.get_session(year, event, "R")
    session.load(laps=True, telemetry=False, weather=True, messages=True)
    return session


# ---------------------------------------------------------------------------
# Core lap table
# ---------------------------------------------------------------------------

def build_lap_table(session: fastf1.core.Session) -> pd.DataFrame:
    laps = session.laps.copy()
    laps = laps.pick_wo_box() if False else laps  # keep all laps incl. pit

    table = pd.DataFrame({
        "driver": laps["Driver"],
        "team": laps["Team"],
        "lap_number": laps["LapNumber"].astype("Int64"),
        "position": laps["Position"].astype("Int64"),
        "lap_time_s": laps["LapTime"].dt.total_seconds(),
        "sector1_s": laps["Sector1Time"].dt.total_seconds(),
        "sector2_s": laps["Sector2Time"].dt.total_seconds(),
        "sector3_s": laps["Sector3Time"].dt.total_seconds(),
        "compound": laps["Compound"],
        "tyre_age": laps["TyreLife"].astype("Int64"),
        "stint": laps["Stint"].astype("Int64"),
        "is_pit_in": laps["PitInTime"].notna(),
        "is_pit_out": laps["PitOutTime"].notna(),
        "track_status": laps["TrackStatus"],
        "is_accurate": laps["IsAccurate"],
    })
    return table.sort_values(["driver", "lap_number"]).reset_index(drop=True)


def add_fuel_corrected_pace(laps: pd.DataFrame, per_lap_gain_s: float = 0.06) -> pd.DataFrame:
    """Heavier early-race fuel makes laps slower; correct to an empty-tank basis.

    per_lap_gain_s is the typical lap-time gain per lap as fuel burns off
    (circuit-dependent; ~0.05-0.08s is a common range).
    """
    out = laps.copy()
    total_laps = out["lap_number"].max()
    laps_remaining = total_laps - out["lap_number"]
    out["fuel_corrected_s"] = out["lap_time_s"] - laps_remaining * per_lap_gain_s
    return out


# ---------------------------------------------------------------------------
# Stints and pit stops
# ---------------------------------------------------------------------------

def build_stint_table(laps: pd.DataFrame) -> pd.DataFrame:
    grouped = laps.groupby(["driver", "stint"], dropna=True)
    stints = grouped.agg(
        team=("team", "first"),
        compound=("compound", "first"),
        start_lap=("lap_number", "min"),
        end_lap=("lap_number", "max"),
        laps_in_stint=("lap_number", "count"),
        median_lap_s=("lap_time_s", "median"),
    ).reset_index()
    return stints.sort_values(["driver", "start_lap"]).reset_index(drop=True)


def build_pit_table(session: fastf1.core.Session) -> pd.DataFrame:
    laps = session.laps
    pit_in = laps[laps["PitInTime"].notna()]
    rows = []
    for _, lap in pit_in.iterrows():
        driver = lap["Driver"]
        in_lap = int(lap["LapNumber"])
        nxt = laps[(laps["Driver"] == driver) & (laps["LapNumber"] == in_lap + 1)]
        out_time = nxt["PitOutTime"].iloc[0] if len(nxt) else pd.NaT
        duration = (out_time - lap["PitInTime"]).total_seconds() if pd.notna(out_time) else np.nan
        rows.append({
            "driver": driver,
            "in_lap": in_lap,
            "stationary_plus_lane_s": duration,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tyre degradation model inputs
# ---------------------------------------------------------------------------

def build_degradation_table(laps: pd.DataFrame) -> pd.DataFrame:
    """Per-compound lap-time vs tyre age, slope = deg in seconds per lap.

    Uses only green-flag, non-pit, accurate laps so SC/pit laps don't pollute
    the fit.
    """
    clean = laps[
        laps["is_accurate"]
        & ~laps["is_pit_in"]
        & ~laps["is_pit_out"]
        & (laps["track_status"] == "1")
        & laps["fuel_corrected_s"].notna()
        & laps["tyre_age"].notna()
    ]
    rows = []
    for compound, grp in clean.groupby("compound"):
        if len(grp) < 5:
            continue
        slope, intercept = np.polyfit(grp["tyre_age"], grp["fuel_corrected_s"], 1)
        rows.append({
            "compound": compound,
            "deg_s_per_lap": round(float(slope), 4),
            "base_pace_s": round(float(intercept), 3),
            "sample_laps": len(grp),
        })
    return pd.DataFrame(rows).sort_values("base_pace_s").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Race control events (SC / VSC / red flag / weather)
# ---------------------------------------------------------------------------

def build_event_table(session: fastf1.core.Session) -> pd.DataFrame:
    msgs = session.race_control_messages
    if msgs is None or len(msgs) == 0:
        return pd.DataFrame(columns=["lap_number", "category", "message"])
    keep = msgs[msgs["Category"].isin(["SafetyCar", "Flag"])]
    return pd.DataFrame({
        "lap_number": keep["Lap"].astype("Int64"),
        "category": keep["Category"],
        "message": keep["Message"],
    }).reset_index(drop=True)


def build_weather_table(session: fastf1.core.Session) -> pd.DataFrame:
    w = session.weather_data
    if w is None or len(w) == 0:
        return pd.DataFrame()
    return pd.DataFrame({
        "time_min": (w["Time"].dt.total_seconds() / 60).round(1),
        "air_temp_c": w["AirTemp"],
        "track_temp_c": w["TrackTemp"],
        "rainfall": w["Rainfall"],
        "wind_speed": w["WindSpeed"],
    })


# ---------------------------------------------------------------------------
# Reference pace per driver (simulator's per-driver base lap time)
# ---------------------------------------------------------------------------

def build_driver_pace_table(laps: pd.DataFrame) -> pd.DataFrame:
    clean = laps[
        laps["is_accurate"]
        & ~laps["is_pit_in"]
        & ~laps["is_pit_out"]
        & (laps["track_status"] == "1")
        & laps["fuel_corrected_s"].notna()
    ]
    pace = clean.groupby("driver").agg(
        team=("team", "first"),
        median_pace_s=("fuel_corrected_s", "median"),
        best_pace_s=("fuel_corrected_s", "min"),
        clean_laps=("fuel_corrected_s", "count"),
    ).reset_index()
    return pace.sort_values("median_pace_s").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Result and orchestration
# ---------------------------------------------------------------------------

def build_results_table(session: fastf1.core.Session) -> pd.DataFrame:
    res = session.results
    return pd.DataFrame({
        "driver": res["Abbreviation"],
        "team": res["TeamName"],
        "grid": res["GridPosition"].astype("Int64"),
        "finish": res["Position"].astype("Int64"),
        "status": res["Status"],
        "points": res["Points"],
    }).reset_index(drop=True)


@dataclass
class RaceData:
    laps: pd.DataFrame
    stints: pd.DataFrame
    pit_stops: pd.DataFrame
    degradation: pd.DataFrame
    driver_pace: pd.DataFrame
    events: pd.DataFrame
    weather: pd.DataFrame
    results: pd.DataFrame

    def write(self, out_dir: str) -> None:
        path = Path(out_dir)
        path.mkdir(parents=True, exist_ok=True)
        for name, frame in self.__dict__.items():
            frame.to_parquet(path / f"{name}.parquet", index=False)
            frame.to_csv(path / f"{name}.csv", index=False)


def collect_race_data(year: int, event: str | int) -> RaceData:
    session = load_race(year, event)
    laps = add_fuel_corrected_pace(build_lap_table(session))
    return RaceData(
        laps=laps,
        stints=build_stint_table(laps),
        pit_stops=build_pit_table(session),
        degradation=build_degradation_table(laps),
        driver_pace=build_driver_pace_table(laps),
        events=build_event_table(session),
        weather=build_weather_table(session),
        results=build_results_table(session),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Pull F1 race data for counterfactuals.")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--event", required=True, help="Event name or round number")
    parser.add_argument("--out", default="data")
    parser.add_argument("--cache", default="cache")
    args = parser.parse_args()

    enable_cache(args.cache)
    event = int(args.event) if args.event.isdigit() else args.event
    data = collect_race_data(args.year, event)
    data.write(args.out)

    print(f"Wrote {len(data.__dict__)} tables to {args.out}/")
    print("\nDegradation model:")
    print(data.degradation.to_string(index=False))
    print("\nTop pace:")
    print(data.driver_pace.head(5).to_string(index=False))


if __name__ == "__main__":
    main()