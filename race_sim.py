"""Lap-by-lap fuel strategy simulation for one race, run thousands of times.

Every simulated race:
  - stage breaks bring a caution at their known laps; other cautions come at random,
    at the track's measured rate per green lap, and last as long as real ones did;
  - a car burns one lap of fuel per green lap and YELLOW_BURN of a lap per yellow lap;
  - under green, a car pits when its stint reaches a target drawn from the track's real
    green-stop laps (tires or fuel), unless it can make the finish on what it has (with
    saving where fuel sets the stops; at full pace where tires do);
  - under yellow, a car pits with the chance real cars did, given how old its stint was
    and whether it could make the finish without stopping;
  - each car's true range differs a little from what its team planned on, so cars that
    try to stretch it sometimes run dry.
Everything is measured in laps of fuel, never gallons (fuel burn isn't published).

The parameters come from data/*.csv, written by build_data.py.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import fuel_window as fw

DATA = Path(__file__).parent / "data"
YELLOW_BURN = 0.35     # share of a green lap's fuel a yellow lap uses; assumed (see docs/race_sim.md)
SAVE = 0.10            # hard saving stretches a tank about this much (Michigan ~8%, Pocono ~16%)
TEAM_SPREAD = 0.02     # car-to-car spread of the real range, as a share of the window
LATE_STOP = 0.3        # a green stop with under this share of a tank to go, by a car that couldn't
                       # have made the finish at full pace, is a late fuel stop
FUEL_LIMITED = 0.75    # green stops this close to the window mean fuel, not tires, sets them
LATEST_STOP_Q = 0.95   # with the models: past this share of real green stops, a car out of reach pits
FUEL_FINISH_SHARE = 1 / 3   # a race is a fuel-mileage finish when this share of the field is caught out
SINCE_BINS = np.array([10, 20, 30, 40])   # stint-age bins for the yellow-flag pit table
MAX_OVERTIMES = 3
N_SIMS = 2000


# ---------------------------------------------------------------- data

def load():
    """The tables written by build_data.py."""
    read = lambda name, **kw: pd.read_csv(DATA / f"{name}.csv", **kw)
    races = read("races", dtype={"stage_breaks": str})
    races["stage_breaks"] = races["stage_breaks"].fillna("").map(lambda s: [int(x) for x in s.split()])
    return {"races": races, "cars": read("cars", dtype={"car": str}),
            "stints": read("stints", dtype={"car": str}), "flags": read("flags"),
            "run_outs": read("run_outs", dtype={"car": str})}


def race_flags(data, race_id):
    """Leader lap -> flag state for one race."""
    f = data["flags"][data["flags"]["race_id"] == race_id]
    return dict(zip(f["lap"], f["flag"]))


def caution_periods(flags):
    """(first, last) yellow lap of every caution. A yellow from lap 1 is a weather start, not a caution."""
    periods, first = [], None
    for lap in sorted(flags):
        if lap > 0 and flags[lap] == fw.YELLOW and first is None:
            first = lap
        if flags[lap] != fw.YELLOW and first is not None:
            if first > 1:
                periods.append((first, lap - 1))
            first = None
    return periods


def fuel_used(flags, start, end, yellow_burn):
    """Laps of fuel used from lap start+1 to lap end."""
    states = [flags.get(lap, fw.GREEN) for lap in range(start + 1, end + 1)]
    return states.count(fw.GREEN) + yellow_burn * states.count(fw.YELLOW)


# ---------------------------------------------------------------- track parameters

@dataclass
class Track:
    name: str
    laps: int                 # scheduled laps of the race being simulated
    stage_breaks: list
    field: int
    hazard: float             # chance of a random caution on any green lap
    caution_len: np.ndarray   # yellow laps of each real random caution
    stage_len: np.ndarray     # yellow laps of each real stage-break caution
    green_targets: np.ndarray  # fuel used at each real (clean) green-flag stop
    longest: float            # longest clean stint, in laps of fuel
    fuel_limited: bool        # do green stops come close to the longest stint?
    pit_table: np.ndarray     # P(pit under yellow) by [stint-age bin, can finish without stopping]
    yellow_burn: float
    save: float

    @property
    def window(self):
        """Laps of fuel in a tank with hard saving.

        Where fuel sets the stops, cars run right to the limit, so the longest clean stint
        is the tank with saving. Where tires set them, nobody had to save to run the longest
        stint, so it's taken as a full-pace tank (a lower bound) and saving adds on top.
        """
        return self.longest if self.fuel_limited else self.longest * (1 + self.save)

    @property
    def full_window(self):
        return self.window * (1 - self.save) if self.fuel_limited else self.longest


def track_params(data, track, race_id=None, yellow_burn=YELLOW_BURN, save=SAVE):
    """Measure a track's caution rate, caution lengths, stop habits and fuel window.

    Every Next Gen race at the track is used. The race format (laps, stage breaks, field
    size) comes from `race_id`, or the track's latest race.
    """
    races = data["races"][data["races"]["track"] == track]
    fmt = races.iloc[-1] if race_id is None else races[races["race_id"] == race_id].iloc[0]
    stints = data["stints"][data["stints"]["race_id"].isin(races["race_id"])]
    fuel = stints["green"] + yellow_burn * stints["yellow"]
    clean = stints["end_how"].isin(["pit_green", "pit_yellow", "finish"]) \
        & (stints["laps_behind"] == 0) & (stints["slow_green"] == 0)
    longest = float(fuel[clean].max())
    green_stops = clean & (stints["end_how"] == "pit_green") & (stints["laps"] >= fw.MIN_GREEN_STOP)
    green_targets = fuel[green_stops].to_numpy() if green_stops.any() else np.array([longest * 0.8])
    fuel_limited = bool(np.median(green_targets) >= FUEL_LIMITED * longest)
    window = longest if fuel_limited else longest * (1 + save)

    random_len, stage_len, n_random, green_laps = [], [], 0, 0
    pits = np.zeros((len(SINCE_BINS) + 1, 2))
    seen = np.zeros_like(pits)
    for _, race in races.iterrows():
        flags = race_flags(data, race["race_id"])
        green_laps += sum(1 for lap, s in flags.items() if s == fw.GREEN)
        breaks = set(race["stage_breaks"])
        for first, last in caution_periods(flags):
            if first - 1 in breaks:
                stage_len.append(last - first + 1)
            else:
                random_len.append(last - first + 1)
                n_random += 1
        # who pitted under each caution, by stint age and whether they could finish without stopping
        # (cars that finished on the lead lap only: lapped and damaged cars pit for other reasons)
        rc = data["cars"][(data["cars"]["race_id"] == race["race_id"]) & (data["cars"]["status"] == "Running")]
        lead = rc[rc["laps_completed"] == rc["laps_completed"].max()]["car"]
        rs = stints[(stints["race_id"] == race["race_id"]) & stints["car"].isin(lead)]
        for first, last in caution_periods(flags):
            on = rs[(rs["start"] < first) & (rs["end"] >= first)]
            for _, s in on.iterrows():
                used = fuel_used(flags, s["start"], first - 1, yellow_burn)
                can_finish = used + (race["actual_laps"] - last) <= window
                b = np.searchsorted(SINCE_BINS, first - 1 - s["start"], side="right")
                seen[b, int(can_finish)] += 1
                pits[b, int(can_finish)] += s["end_how"] == "pit_yellow" and s["end"] <= last + 1
    pooled = pits.sum() / max(seen.sum(), 1)
    pit_table = (pits + 2 * pooled) / (seen + 2)       # shrink thin cells toward the overall rate

    return Track(
        name=track, laps=int(fmt["scheduled_laps"]), stage_breaks=list(fmt["stage_breaks"]),
        field=int((data["cars"]["race_id"] == fmt["race_id"]).sum()),
        hazard=n_random / max(green_laps, 1), caution_len=np.array(random_len or [5]),
        stage_len=np.array(stage_len or [6]), green_targets=green_targets, longest=longest,
        fuel_limited=fuel_limited, pit_table=pit_table, yellow_burn=yellow_burn, save=save,
    )


# ---------------------------------------------------------------- the simulation

def simulate(t, n_sims=N_SIMS, start_lap=0, state=None, seed=0, tables=None):
    """Run the rest of the race from `start_lap` n_sims times.

    `state` (from replay_state) gives each car's fuel used and stint age at start_lap, and
    the caution in progress; without it every car starts full on lap 0.
    `tables` (from models.tables) swaps the simple averages for the XGBoost models: the
    caution chance, green-stop chance and yellow-stop chance then come from the models,
    lap by lap. Without it, the step-2 averages are used.
    Returns a dict of per-sim, per-car arrays and per-lap pit counts.
    """
    rng = np.random.default_rng(seed)
    n_cars = t.field if state is None else len(state["used"])
    shape = (n_sims, n_cars)
    used = np.zeros(shape) if state is None else np.broadcast_to(state["used"], shape).astype(float)
    since = np.zeros(shape, int) if state is None else np.broadcast_to(state["since"], shape).astype(int)
    real = t.window * (1 + TEAM_SPREAD * rng.standard_normal(shape))     # each car's true range with saving
    target = np.minimum(rng.choice(t.green_targets, shape), t.full_window)
    stretch = t.window if t.fuel_limited else t.full_window   # how far a team will go to skip a green stop
    latest_stop = min(np.quantile(t.green_targets, LATEST_STOP_Q), t.full_window)
    end = np.full(n_sims, t.laps)
    overtimes = np.zeros(n_sims, int)
    caution_left = np.zeros(n_sims, int)
    pit_open = np.zeros(n_sims, bool)     # first lap of a caution: the moment cars decide to pit
    stage_caution = np.zeros(n_sims, int)  # is the current caution a stage break?
    since_green = np.full(n_sims, 0 if state is None else state.get("since_green", 0))
    if state is not None and state["caution_left"]:
        caution_left[:] = rng.choice(t.caution_len, n_sims) - state["caution_elapsed"]
        caution_left = np.maximum(caution_left, 1)
        pit_open[:] = state["caution_elapsed"] <= 1
        stage_caution[:] = (start_lap - state["caution_elapsed"]) in t.stage_breaks

    max_lap = t.laps + 40
    pit_laps = np.zeros((max_lap + 1, 2))           # stops per lap: [green, yellow]
    stops = np.zeros(shape, int)
    green_stops = np.zeros(shape, int)
    late_stops = np.zeros(shape, int)
    dry = np.zeros(shape, int)
    first_stop = np.full(shape, -1)
    cautions = np.zeros(n_sims, int)

    for lap in range(start_lap + 1, max_lap + 1):
        running = lap <= end
        if not running.any():
            break
        yellow = caution_left > 0
        used += np.where(yellow, t.yellow_burn, 1.0)[:, None] * running[:, None]
        since += running[:, None]
        since_green = np.where(yellow, 0, since_green + 1)
        to_go = (end - lap)[:, None]
        need_stop = used + to_go > t.window          # can't make the finish even with saving
        ran_dry = (used > real) & running[:, None]

        # a stage break is a free stop under caution, so teams stretch to it like to the finish
        next_break = min((b for b in t.stage_breaks if b >= lap), default=t.laps + 1000)
        horizon = np.minimum(next_break, end)[:, None] - lap
        can_reach = used + horizon <= stretch
        if tables is None:
            green_pit = ~yellow[:, None] & (used >= target) & ~can_reach
            bins = np.searchsorted(SINCE_BINS, since, side="right")
            p_yellow = t.pit_table[bins, (~need_stop).astype(int)]
        else:
            # The model decides when a team chooses to pit, within the range real teams used. Past the
            # latest real green stop at this track, a car that can't reach the stage break or the finish
            # pits: real cars almost never got there, so the model never learned that a stop is certain.
            fuel_idx = np.minimum(used, tables["green"].shape[1] - 1).astype(int)
            chosen = rng.random(shape) < tables["green"][lap, fuel_idx]
            out_of_room = (used >= latest_stop) & ~can_reach
            green_pit = ~yellow[:, None] & (chosen | out_of_room)
            p_yellow = tables["yellow"][lap, fuel_idx, stage_caution[:, None]]
        yellow_pit = (yellow & pit_open)[:, None] & (rng.random(shape) < p_yellow)
        pit = (ran_dry | green_pit | yellow_pit) & running[:, None]

        if pit.any():
            under_green = pit & ~yellow[:, None]
            pit_laps[lap, 0] += under_green.sum()
            pit_laps[lap, 1] += (pit & yellow[:, None]).sum()
            stops += pit
            green_stops += under_green
            late_stops += under_green & (to_go < LATE_STOP * t.window) & (used + to_go > t.full_window)
            dry += ran_dry & running[:, None]
            first_stop = np.where(pit & (first_stop < 0), lap, first_stop)
            used = np.where(pit, 0.0, used)
            since = np.where(pit, 0, since)
            target = np.where(pit, np.minimum(rng.choice(t.green_targets, shape), t.full_window), target)

        # flags for the next lap
        caution_left = np.maximum(caution_left - 1, 0)
        pit_open[:] = False
        # only a green lap can be followed by a new caution: the lap after a caution is the restart
        green_now = running & ~yellow & (lap < end)
        stage = green_now & (lap in t.stage_breaks)
        if tables is None:
            hazard = t.hazard
        else:
            hazard = tables["caution"][lap, np.minimum(since_green, tables["caution"].shape[1] - 1)]
        crash = green_now & ~stage & (rng.random(n_sims) < hazard)
        start = stage | crash
        stage_caution = np.where(start, stage.astype(int), stage_caution)
        caution_left = np.where(stage, rng.choice(t.stage_len, n_sims), caution_left)
        caution_left = np.where(crash, rng.choice(t.caution_len, n_sims), caution_left)
        pit_open |= start
        cautions += crash
        # a caution that runs to the finish sends the race to overtime: two green laps after it
        ot = start & (lap + caution_left >= end) & (overtimes < MAX_OVERTIMES)
        end = np.where(ot, np.minimum(lap + caution_left + 2, max_lap), end)
        overtimes += ot

    full_range = real * (t.full_window / t.window)     # each car's full-pace range, built like the track's
    must_save = (used > full_range) & ~(dry > 0)
    caught_out = must_save | (late_stops > 0) | (dry > 0)
    return {
        "n_sims": n_sims, "end": end, "cautions": cautions, "overtimes": overtimes,
        "pit_laps": pit_laps, "stops": stops, "green_stops": green_stops, "late_stops": late_stops,
        "dry": dry, "first_stop": first_stop, "final_fuel": used, "must_save": must_save,
        "fuel_finish": caught_out.mean(axis=1) >= FUEL_FINISH_SHARE,
    }


def summary(t, sim):
    """Race-level numbers from a batch of simulations."""
    n = sim["n_sims"] * sim["stops"].shape[1]
    return {
        "p_fuel_finish": float(sim["fuel_finish"].mean()),
        "cautions": float(sim["cautions"].mean() + len(t.stage_breaks)),
        "random_cautions": float(sim["cautions"].mean()),
        "p_overtime": float((sim["overtimes"] > 0).mean()),
        "stops_per_car": float(sim["stops"].mean()),
        "green_stops_per_car": float(sim["green_stops"].mean()),
        "p_run_dry": float((sim["dry"] > 0).mean()),
        "p_must_save": float(sim["must_save"].mean()),
        "pit_by_lap": pd.DataFrame(sim["pit_laps"] / n, columns=["green", "yellow"]).rename_axis("lap"),
    }


# ---------------------------------------------------------------- replaying a real race from lap N

def replay_state(data, race_id, lap, yellow_burn=YELLOW_BURN):
    """Every running car's fuel used and stint age at `lap` of a real race, plus the caution in progress."""
    flags = race_flags(data, race_id)
    cars = data["cars"][data["cars"]["race_id"] == race_id]
    stints = data["stints"][data["stints"]["race_id"] == race_id]
    rows = []
    for _, c in cars.iterrows():
        if c["laps_completed"] <= lap and c["status"] != "Running":
            continue        # out of the race by now
        s = stints[(stints["car"] == c["car"]) & (stints["start"] <= lap) & (stints["end"] > lap)]
        start = int(s["start"].iloc[0]) if len(s) else int(stints[stints["car"] == c["car"]]["end"].max())
        rows.append({"car": c["car"], "driver": c["driver"], "finish_pos": c["finish_pos"],
                     "stopped_on": start, "since": lap - start,
                     "used": fuel_used(flags, start, lap, yellow_burn)})
    cars_now = pd.DataFrame(rows)
    elapsed = 0
    while flags.get(lap - elapsed) == fw.YELLOW:
        elapsed += 1
    since_green = 0
    while flags.get(lap - since_green) == fw.GREEN:
        since_green += 1
    return {"cars": cars_now, "used": cars_now["used"].to_numpy(), "since": cars_now["since"].to_numpy(),
            "caution_left": elapsed > 0, "caution_elapsed": elapsed, "since_green": since_green}


def replay_table(t, state, sim, lap):
    """Per car: can it make the finish from here, and what the simulations say it'll need."""
    cars = state["cars"].copy()
    to_go = t.laps - lap
    short = cars["used"] + to_go - t.window
    cars["laps_short"] = short.clip(lower=0).round(1)
    cars["verdict"] = np.select(
        [cars["used"] + to_go <= t.full_window, short <= 0],
        ["Makes it at full pace", "Makes it if it saves"], "Needs another stop")
    stopped = sim["first_stop"] > 0
    cars["p_stop"] = stopped.mean(axis=0)
    cars["p_green_stop"] = (sim["green_stops"] > 0).mean(axis=0)
    cars["p_save"] = sim["must_save"].mean(axis=0)
    cars["p_dry"] = (sim["dry"] > 0).mean(axis=0)
    fs = np.where(stopped, sim["first_stop"], np.nan)
    with np.errstate(all="ignore"):
        cars["next_stop_p10"] = np.nanpercentile(fs, 10, axis=0)
        cars["next_stop_mid"] = np.nanpercentile(fs, 50, axis=0)
        cars["next_stop_p90"] = np.nanpercentile(fs, 90, axis=0)
    return cars


def actual_next_stop(data, race_id, cars, lap):
    """What really happened after `lap`: each car's next stop lap and the flag it came under."""
    stints = data["stints"][data["stints"]["race_id"] == race_id]
    out = []
    for car in cars["car"]:
        nxt = stints[(stints["car"] == car) & (stints["end"] > lap) & stints["end_how"].str.startswith("pit")]
        nxt = nxt.sort_values("end").head(1)
        out.append((int(nxt["end"].iloc[0]), nxt["end_how"].iloc[0][4:]) if len(nxt) else (None, None))
    return pd.DataFrame(out, columns=["actual_next_stop", "actual_flag"], index=cars.index)


# ---------------------------------------------------------------- did it happen for real?

def actual_fuel_finish(data, race_id, t):
    """The same fuel-mileage test as the simulation, applied to a real race's final stints."""
    race = data["races"][data["races"]["race_id"] == race_id].iloc[0]
    stints = data["stints"][(data["stints"]["race_id"] == race_id)]
    finishers = stints[stints["end_how"] == "finish"]
    if finishers.empty:
        return False, 0.0
    must_save = finishers["green"] + t.yellow_burn * finishers["yellow"] > t.full_window
    to_go = race["actual_laps"] - stints["end"]
    late = stints[(stints["end_how"] == "pit_green") & (to_go < LATE_STOP * t.window)
                  & (stints["green"] + t.yellow_burn * stints["yellow"] + to_go > t.full_window)]["car"]
    dry_cars = data["run_outs"][data["run_outs"]["race_id"] == race_id]["car"].dropna()
    caught = must_save | finishers["car"].isin(late) | finishers["car"].isin(dry_cars)
    return bool(caught.mean() >= FUEL_FINISH_SHARE), float(caught.mean())


def actual_counts(data, race_id):
    """Real cautions (stage breaks included) and stops per car that finished on the lead lap."""
    flags = race_flags(data, race_id)
    cars = data["cars"][(data["cars"]["race_id"] == race_id) & (data["cars"]["status"] == "Running")]
    lead = cars[cars["laps_completed"] == cars["laps_completed"].max()]["car"]
    stints = data["stints"][(data["stints"]["race_id"] == race_id) & data["stints"]["car"].isin(lead)]
    stops = stints["end_how"].str.startswith("pit").groupby(stints["car"]).sum()
    return len(caution_periods(flags)), float(stops.mean())


def race_format(data, track, race_id=None):
    """The race to simulate: track, length and stage breaks (the track's latest race if no race_id)."""
    races = data["races"][data["races"]["track"] == track]
    row = races.iloc[-1] if race_id is None else races[races["race_id"] == race_id].iloc[0]
    return row.to_dict()


def check_track(data, track, n_sims=1000, models=None, **kw):
    """Simulated race next to the real one, race by race, using every race at the track.

    The parameters include the race being checked, so this is a consistency check, not a
    forecast test (that's the walk-forward backtest in train_models.py). With `models`
    (from models.load), the XGBoost models drive the simulation.
    """
    if models is not None:
        import models as m      # imported here: models.py itself imports this module
    rows, by_format = [], {}
    for _, race in data["races"][data["races"]["track"] == track].iterrows():
        t = track_params(data, track, race["race_id"], **kw)
        tables = None
        if models is not None:  # races at a track usually share a format, so build each format's tables once
            key = (race["scheduled_laps"], tuple(race["stage_breaks"]))
            if key not in by_format:
                by_format[key] = m.tables(models, race_format(data, track, race["race_id"]))
            tables = by_format[key]
        s = summary(t, simulate(t, n_sims=n_sims, seed=int(race["race_id"]), tables=tables))
        happened, share = actual_fuel_finish(data, race["race_id"], t)
        cautions, stops = actual_counts(data, race["race_id"])
        rows.append({"season": race["season"], "race_id": race["race_id"], "date": race["race_date"],
                     "p_fuel_finish": s["p_fuel_finish"], "fuel_finish": happened, "share_caught": share,
                     "sim_cautions": s["cautions"], "cautions": cautions,
                     "sim_stops": s["stops_per_car"], "stops": stops})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import time
    data = load()
    for track in ("michigan", "pocono", "nashville", "kansas", "las-vegas", "texas", "homestead", "martinsville"):
        t0 = time.time()
        t = track_params(data, track)
        s = summary(t, simulate(t))
        print(f"\n{track}: {t.laps} laps, breaks {t.stage_breaks}, window {t.window:.0f} "
              f"(full pace {t.full_window:.0f}, {'fuel' if t.fuel_limited else 'tire'}-limited), "
              f"caution every {1 / t.hazard:.0f} green laps, "
              f"green stops at {np.median(t.green_targets):.0f}  [{time.time() - t0:.1f}s]")
        print(f"  P(fuel-mileage finish) {s['p_fuel_finish']:.0%}  cautions {s['cautions']:.1f}  "
              f"stops/car {s['stops_per_car']:.1f} (green {s['green_stops_per_car']:.1f})  "
              f"run dry {s['p_run_dry']:.1%}  must save {s['p_must_save']:.0%}")
        print("  pit table (rows: stint age <10,<20,<30,<40,40+; cols: can't finish, can):")
        print("  ", np.round(t.pit_table, 2).tolist())
        print(check_track(data, track, n_sims=500).to_string(index=False))
