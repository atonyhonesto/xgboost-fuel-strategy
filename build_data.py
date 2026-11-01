"""Build the small data files the app reads, so it never has to download raw feeds.

For every Next Gen race at the app's tracks it writes, into data/:
  races.csv   one row per race: track, date, laps, stage breaks
  cars.csv    one row per car per race: driver, finish, status, laps completed
  stints.csv  one row per stint (see fuel_window.race_stints)
  flags.csv   the flag on every leader lap (1 green, 2 yellow)
  run_outs.csv  race control notes about cars running out of fuel

Run after a race at one of these tracks:  python build_data.py
"""
from pathlib import Path

import pandas as pd

import fetch_nascar as fn
import fuel_window as fw

DATA = Path(__file__).parent / "data"
# Ovals where the app runs the simulation; Martinsville is there to show a track where fuel doesn't matter.
APP_TRACKS = {"michigan": 133, "pocono": 198, "nashville": 52, "kansas": 41, "las-vegas": 42,
              "texas": 43, "homestead": 40, "martinsville": 22}


def stage_breaks(race):
    """Laps where a stage ends before the finish (each one brings a caution)."""
    ends, total = [], 0
    for k in ("stage_1_laps", "stage_2_laps", "stage_3_laps"):
        total += race.get(k) or 0
        if 0 < total < race["scheduled_laps"]:
            ends.append(total)
    return ends


def build():
    races, cars, stints, flags, notes = [], [], [], [], []
    for key, track_id in APP_TRACKS.items():
        for season, race_id in fw.track_races(track_id):
            race = fn.race_results(season, race_id)
            feeds = fn.fetch_race(season, race_id)
            if not all(feeds.values()):
                print(f"skipping {season} {race_id}: missing feeds")
                continue
            races.append({
                "race_id": race_id, "season": season, "track": key, "track_name": race["track_name"],
                "race_date": race["race_date"][:10],  # no race_name: those are sponsor names
                "scheduled_laps": race["scheduled_laps"], "actual_laps": race["actual_laps"],
                "stage_breaks": " ".join(map(str, stage_breaks(race))),
                "track_miles": round(race["scheduled_distance"] / race["scheduled_laps"], 3),
            })
            cars += [{"race_id": race_id, "car": str(r["car_number"]), "driver": r["driver_fullname"],
                      "finish_pos": r["finishing_position"], "status": r["finishing_status"],
                      "laps_completed": r["laps_completed"]} for r in race["results"]]
            stints.append(fw.race_stints(season, race_id))
            flags += [{"race_id": race_id, "lap": lap, "flag": state}
                      for lap, state in fw.flags_by_lap(feeds["lap-times"]).items()]
            notes += [{"race_id": race_id, "lap": lap, "car": car, "note": note}
                      for lap, car, note in fw.run_outs(season, race_id)]
            print(f"{key:<13} {season} {race_id}")
    DATA.mkdir(exist_ok=True)
    pd.DataFrame(races).sort_values(["track", "race_date"]).to_csv(DATA / "races.csv", index=False)
    pd.DataFrame(cars).to_csv(DATA / "cars.csv", index=False)
    pd.concat(stints, ignore_index=True).to_csv(DATA / "stints.csv", index=False)
    pd.DataFrame(flags).to_csv(DATA / "flags.csv", index=False)
    pd.DataFrame(notes, columns=["race_id", "lap", "car", "note"]).to_csv(DATA / "run_outs.csv", index=False)


if __name__ == "__main__":
    build()
