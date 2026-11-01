"""Infer a track's fuel window (laps on one tank) from what cars actually did.

Nobody publishes fuel burn, so this works in laps, not gallons. Each car's race is cut
into stints at its pit stops, and every lap of a stint is marked green or yellow from
the lap-times flag feed. A stint a car finished without stopping proves the tank lasts
at least that long, so the longest stints give a lower bound on the window. Cars that
ran dry (found in race control's notes) give an upper bound.

Run directly for a report on one or more tracks (default: Homestead, Michigan, Pocono):
    python fuel_window.py homestead kansas
"""
import re
import sys

import pandas as pd

import fetch_nascar as fn

NEXT_GEN = 2022           # first season of the Next Gen car
TRACKS = {"homestead": 40, "michigan": 133, "pocono": 198,
          # the other 1.5-mile tracks, to compare with Homestead
          "kansas": 41, "las-vegas": 42, "texas": 43, "charlotte": 162, "chicagoland": 39}
GREEN, YELLOW = 1, 2
MIN_GREEN_STOP = 10       # shorter green-flag stints are damage, penalties or loose wheels
SLOW_LAP = 1.25           # a green lap this much slower than the car's usual pace hides a stop or trouble
RUN_OUT = re.compile(r"out of fuel|ran out|running out|ran dry|out of gas|no fuel pressure|"
                     r"switch\w* to pump", re.I)


def track_races(track_id, first_season=NEXT_GEN):
    """(season, race_id) of every points race run at the track since `first_season`.

    Read from the cached results, which cover every race from 2018 on.
    """
    races = []
    for path in sorted(fn.CACHE.glob("20*_results.json")):
        season, race_id = (int(x) for x in path.stem.split("_")[:2])
        if season < first_season:
            continue
        race = fn.race_results(season, race_id)
        if race and race["track_id"] == track_id and race.get("race_type_id") == 1:
            races.append((season, race_id))
    return races


def flags_by_lap(lap_times):
    """Leader lap -> flag state (1 green, 2 yellow; 8 is the grid and 4 the checkered flag)."""
    return {f["LapsCompleted"]: f["FlagState"] for f in lap_times["flags"]}


def car_stops(pit_json, car):
    """A car's real pit stops in lap order, with several stops under one caution merged.

    The feed opens with placeholder rows from the grid (zeros and -1s); they're skipped.
    A car that pits twice under the same yellow leaves full after the second stop, so
    only that one starts a new stint.
    """
    stops = sorted((s for s in pit_json if str(s.get("vehicle_number")) == car
                    and s.get("lap_count", 0) > 0 and s.get("pit_stop_duration", 0) > 0),
                   key=lambda s: s["lap_count"])
    merged = []
    for s in stops:
        if merged and s["pit_in_flag_status"] == YELLOW and merged[-1]["pit_in_flag_status"] == YELLOW \
                and s["lap_count"] - merged[-1]["lap_count"] <= 3:
            merged[-1] = s
        else:
            merged.append(s)
    return merged


def slow_green_laps(lap_times, flags, offset, start, end):
    """Green laps inside a stint (skipping the out- and in-laps) that were far off the car's pace.

    One of these usually means a stop the pit feed missed, or a spin, so the stint
    can't be trusted as a clean run on one tank.
    """
    def green(lap):
        return flags.get(lap + offset) == GREEN and flags.get(lap + offset - 1) == GREEN
    pace = pd.Series([t for lap, t in lap_times.items() if green(lap)]).median()
    return sum(1 for lap in range(start + 3, end) if green(lap) and lap_times.get(lap, 0) > SLOW_LAP * pace)


def race_stints(season, race_id):
    """One row per stint for every car in the race.

    `laps` are the car's own laps; `green` and `yellow` split them by flag. `end_how` is
    pit_green, pit_yellow, finish (running at the checkered flag) or out (retired).
    `laps_behind` is how far the car trailed the leader when the stint ended, and
    `slow_green` counts suspicious laps (see slow_green_laps).
    """
    feeds = fn.fetch_race(season, race_id, feeds=("lap-times", "live-pit-data"))
    race = fn.race_results(season, race_id)
    if not feeds["lap-times"] or not feeds["live-pit-data"] or not race:
        return pd.DataFrame()
    flags = flags_by_lap(feeds["lap-times"])
    times = {c["Number"]: {lap["Lap"]: lap["LapTime"] for lap in c["Laps"] if lap["LapTime"]}
             for c in feeds["lap-times"]["laps"]}
    lead_laps = race["actual_laps"]
    rows = []
    for car in race["results"]:
        num = str(car["car_number"])
        # (end lap, how, leader lap minus car lap); the offset maps a lapped car's laps to the flag feed
        ends = [(s["lap_count"], "pit_green" if s["pit_in_flag_status"] == GREEN else "pit_yellow",
                 s["leader_lap"] - s["lap_count"], s["pit_stop_type"])
                for s in car_stops(feeds["live-pit-data"], num)]
        last = "finish" if car["finishing_status"] == "Running" else "out"
        ends.append((car["laps_completed"], last, lead_laps - car["laps_completed"], ""))
        start = 0
        for end, how, offset, stop_type in ends:
            if end <= start:
                continue
            states = [flags.get(lap + offset, GREEN) for lap in range(start + 1, end + 1)]
            rows.append({
                "season": season, "race_id": race_id, "car": num, "start": start, "end": end,
                "laps": len(states), "green": states.count(GREEN), "yellow": states.count(YELLOW),
                "end_how": how, "stop_type": stop_type, "laps_behind": offset,
                "slow_green": slow_green_laps(times.get(num, {}), flags, offset, start, end),
                "status": car["finishing_status"], "finish_pos": car["finishing_position"],
            })
            start = end
    return pd.DataFrame(rows)


def track_stints(track_id, first_season=NEXT_GEN):
    """Every stint in every race at the track since `first_season`."""
    frames = [race_stints(season, race_id) for season, race_id in track_races(track_id, first_season)]
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def green_runs(season, race_id):
    """Each unbroken green-flag run: first lap, last lap and length."""
    flags = flags_by_lap(fn.fetch_race(season, race_id, feeds=("lap-times",))["lap-times"])
    runs, first = [], None
    for lap in sorted(flags):
        if flags[lap] == GREEN and first is None:
            first = lap
        if flags[lap] != GREEN and first is not None:
            runs.append({"season": season, "race_id": race_id, "first": first, "last": lap - 1,
                         "laps": lap - first})
            first = None
    return pd.DataFrame(runs)


def run_outs(season, race_id):
    """Race control notes that say a car ran out of fuel: (lap, car, note).

    A note can name several cars, so the car is the one named just before the
    run-out phrase (or the first one after it).
    """
    notes = (fn.fetch_race(season, race_id, feeds=("lap-notes",))["lap-notes"] or {}).get("laps", {})
    found = []
    for lap, ns in notes.items():
        for n in ns:
            hit = RUN_OUT.search(n["Note"])
            if not hit:
                continue
            before = re.findall(r"#(\d+)", n["Note"][:hit.start()])
            after = re.findall(r"#(\d+)", n["Note"][hit.start():])
            found.append((int(lap), (before or after or [None])[-1 if before else 0], n["Note"]))
    return found


def fuel_window(stints):
    """Summary of what the stints say about the tank.

    Only clean stints count as evidence: run on the lead lap (lapped cars run slower
    and burn less) with no suspicious slow green laps. Green laps alone are the safe
    lower bound (true even if yellow laps burned nothing); total laps is the bound if a
    yellow lap burns as much as a green one.
    """
    done = stints[stints["end_how"].isin(["pit_green", "pit_yellow", "finish"])]
    clean = done[(done["laps_behind"] == 0) & (done["slow_green"] == 0)]
    green_stops = stints[(stints["end_how"] == "pit_green") & (stints["laps"] >= MIN_GREEN_STOP)]
    longest = clean.sort_values(["green", "laps"], ascending=False)
    return {
        "races": stints["race_id"].nunique(),
        "stints": len(done),
        "clean_stints": len(clean),
        "green_stop_median": float(green_stops["laps"].median()),
        "green_stop_iqr": (float(green_stops["laps"].quantile(.25)), float(green_stops["laps"].quantile(.75))),
        "min_window_green_laps": int(longest["green"].iloc[0]),
        "min_window_total_laps": int(longest["laps"].iloc[0]),
        "longest": longest.head(8),
    }


def stint_at(stints, race_id, car, lap):
    """The stint a car was on when a note at `lap` was written (e.g. when it ran dry).

    Notes are logged on the leader's lap, and a car that "pits out of fuel" has its
    stop on the lap before, so the lap before the note is used.
    """
    lap -= 1
    s = stints[(stints["race_id"] == race_id) & (stints["car"] == str(car))
               & (stints["start"] < lap) & (stints["end"] >= lap)]
    return s.iloc[0] if len(s) else None


def laps_into_stint(season, race_id, stint, lap):
    """Green and yellow laps a car had run on its current stint by a given lap."""
    flags = flags_by_lap(fn.fetch_race(season, race_id, feeds=("lap-times",))["lap-times"])
    states = [flags.get(l, GREEN) for l in range(stint["start"] + 1, min(lap, stint["end"]) + 1)]
    return states.count(GREEN), states.count(YELLOW)


def report(name, track_id):
    """Print what the stints and the run-out notes say about one track's fuel window."""
    cols = ["season", "car", "start", "end", "laps", "green", "yellow", "end_how", "finish_pos"]
    st = track_stints(track_id)
    if st.empty:
        print(f"{name}: no lap and pit data")
        return
    w = fuel_window(st)
    runs = pd.concat([green_runs(s, r) for s, r in track_races(track_id)])
    print(f"\n{name}, {NEXT_GEN}+: {w['races']} races, {w['stints']} completed stints ({w['clean_stints']} clean)")
    print(f"  green-flag stops come {w['green_stop_median']:.0f} laps into a stint "
          f"(middle half {w['green_stop_iqr'][0]:.0f}-{w['green_stop_iqr'][1]:.0f})")
    print(f"  a tank lasts at least {w['min_window_green_laps']} green laps "
          f"({w['min_window_total_laps']} laps counting yellows)")
    print(f"  longest green-flag runs: {sorted(runs['laps'], reverse=True)[:6]}")
    print(w["longest"].head(5)[cols].to_string(index=False))
    for season, race_id in track_races(track_id):
        for lap, car, note in run_outs(season, race_id):
            s = stint_at(st, race_id, car, lap) if car else None
            if s is None:
                where = "stint not found"
            else:
                g, y = laps_into_stint(season, race_id, s, lap)
                where = f"{g} green + {y} yellow laps after its stop on lap {s['start']}"
            print(f"  {season} lap {lap}: #{car or '?'} out of fuel, {where}")
            print(f"     \"{note[:140]}\"")


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    for name in sys.argv[1:] or ["homestead", "michigan", "pocono"]:
        report(name, TRACKS[name])
