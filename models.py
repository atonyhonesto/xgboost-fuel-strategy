"""The XGBoost models behind the simulation: caution risk, green-flag stops, yellow-flag stops.

Three classifiers, each answering a yes/no question the simulation asks every lap:
  caution  Will a caution come out on the next lap?          one row per green lap
  green    Does this car pit under green on this lap?        one row per car per green lap
  yellow   Does this car pit under this caution?             one row per car per caution
Every feature is something the simulation knows as it goes (lap, laps to go, stage,
fuel used, laps since the last restart), so a model can be turned into a lookup table
the simulation reads (see tables()).

Each is backtested walk-forward in train_models.py against the simple averages the
simulation used before (step 2), and explained with SHAP values from XGBoost itself.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

import fuel_window as fw
import race_sim as rs

ROOT = Path(__file__).parent
FOLDER = ROOT / "models"
REPORTS = ROOT / "reports"
TRACKS = ["michigan", "pocono", "nashville", "kansas", "las-vegas", "texas", "homestead", "martinsville"]
MAX_SINCE = 150          # laps since the last restart are capped here
MAX_USED = 200           # laps of fuel used are capped here

FEATURES = {
    "caution": ["track", "track_miles", "lap_frac", "laps_to_go", "stage", "laps_to_stage_end",
                "laps_since_green"],
    "green": ["track", "track_miles", "fuel_used", "laps_to_go", "laps_to_stage_end", "stage", "lap_frac"],
    "yellow": ["track", "track_miles", "fuel_used", "laps_to_go", "laps_to_stage_end", "stage", "lap_frac",
               "stage_break"],
}
NAMES = {   # plain-English feature names for the app
    "track": "Which track", "track_miles": "Track length", "lap_frac": "How far into the race",
    "laps_to_go": "Laps to go", "stage": "Which stage", "laps_to_stage_end": "Laps to the end of the stage",
    "laps_since_green": "Laps since the last restart", "fuel_used": "Fuel used this stint (laps)",
    "stage_break": "Caution is a stage break",
}
PARAMS = {
    "caution": dict(n_estimators=250, learning_rate=0.03, max_depth=3, min_child_weight=30),
    "green": dict(n_estimators=300, learning_rate=0.05, max_depth=5, min_child_weight=50),
    "yellow": dict(n_estimators=200, learning_rate=0.05, max_depth=3, min_child_weight=20),
}
COMMON = dict(objective="binary:logistic", subsample=0.8, colsample_bytree=0.9, reg_lambda=5.0,
              tree_method="hist", enable_categorical=True, random_state=42, n_jobs=-1)


# ---------------------------------------------------------------- race context, lap by lap

def race_context(race, lap):
    """Stage, laps to the end of the stage and laps to go at `lap` of a race's schedule."""
    laps = race["scheduled_laps"]
    ends = list(race["stage_breaks"]) + [laps]
    stage = next((i + 1 for i, e in enumerate(ends) if lap <= e), len(ends))
    return stage, max(ends[stage - 1] - lap, 0), max(laps - lap, 0)


def _context_arrays(race, laps):
    """race_context for many laps at once."""
    ends = np.array(list(race["stage_breaks"]) + [race["scheduled_laps"]])
    stage_idx = np.minimum(np.searchsorted(ends, laps, side="left"), len(ends) - 1)
    return (stage_idx + 1, np.maximum(ends[stage_idx] - laps, 0),
            np.maximum(race["scheduled_laps"] - laps, 0))


# ---------------------------------------------------------------- training rows

def caution_rows(data):
    """One row per green lap: does a (non-stage-break) caution start on the next lap?"""
    rows = []
    for _, race in data["races"].iterrows():
        flags = rs.race_flags(data, race["race_id"])
        breaks = set(race["stage_breaks"])
        last = max(flags)
        since = 0
        for lap in range(1, last):
            if flags.get(lap) != fw.GREEN:
                since = 0
                continue
            since += 1
            if lap in breaks:
                continue        # the stage break brings a caution for sure: nothing to predict
            stage, to_stage, to_go = race_context(race, lap)
            rows.append({"race_id": race["race_id"], "race_date": race["race_date"], "track": race["track"],
                         "track_miles": race["track_miles"], "lap_frac": lap / race["scheduled_laps"],
                         "laps_to_go": to_go, "stage": stage, "laps_to_stage_end": to_stage,
                         "laps_since_green": min(since, MAX_SINCE),
                         "y": int(flags.get(lap + 1) == fw.YELLOW)})
    return pd.DataFrame(rows)


def _running_cars(data, race_id):
    cars = data["cars"][(data["cars"]["race_id"] == race_id) & (data["cars"]["status"] == "Running")]
    return set(cars["car"])


def green_rows(data, yellow_burn=rs.YELLOW_BURN):
    """One row per car per green lap of a stint: does the car pit under green at the end of this lap?

    Cars still running at the finish only, and stints with no sign of trouble (no
    suspiciously slow green laps), so damage stops don't teach the model about strategy.
    """
    out = []
    stints = data["stints"]
    for _, race in data["races"].iterrows():
        flags = rs.race_flags(data, race["race_id"])
        running = _running_cars(data, race["race_id"])
        rs_ = stints[(stints["race_id"] == race["race_id"]) & stints["car"].isin(running)
                     & (stints["slow_green"] == 0) & stints["end_how"].isin(["pit_green", "pit_yellow", "finish"])]
        for _, s in rs_.iterrows():
            laps = np.arange(s["start"] + 1, s["end"] + 1)
            lead = laps + s["laps_behind"]                      # the leader's lap, for the flag
            state = np.array([flags.get(l, fw.GREEN) for l in lead])
            used = np.cumsum(np.where(state == fw.YELLOW, yellow_burn, 1.0))
            green = state == fw.GREEN
            if not green.any():
                continue
            stage, to_stage, to_go = _context_arrays(race, lead)
            y = np.zeros(len(laps), int)
            if s["end_how"] == "pit_green":
                y[-1] = 1
            out.append(pd.DataFrame({
                "race_id": race["race_id"], "race_date": race["race_date"], "track": race["track"],
                "track_miles": race["track_miles"], "fuel_used": np.minimum(used, MAX_USED),
                "laps_to_go": to_go, "laps_to_stage_end": to_stage, "stage": stage,
                "lap_frac": lead / race["scheduled_laps"], "y": y})[green])
    return pd.concat(out, ignore_index=True)


def yellow_rows(data, yellow_burn=rs.YELLOW_BURN):
    """One row per car per caution: did a car that finished on the lead lap pit under it?"""
    rows = []
    stints = data["stints"]
    for _, race in data["races"].iterrows():
        flags = rs.race_flags(data, race["race_id"])
        breaks = set(race["stage_breaks"])
        cars = data["cars"][(data["cars"]["race_id"] == race["race_id"]) & (data["cars"]["status"] == "Running")]
        lead = cars[cars["laps_completed"] == cars["laps_completed"].max()]["car"]
        rs_ = stints[(stints["race_id"] == race["race_id"]) & stints["car"].isin(lead)]
        for first, last in rs.caution_periods(flags):
            stage, to_stage, to_go = race_context(race, first)
            for _, s in rs_[(rs_["start"] < first) & (rs_["end"] >= first)].iterrows():
                rows.append({"race_id": race["race_id"], "race_date": race["race_date"], "track": race["track"],
                             "track_miles": race["track_miles"],
                             "fuel_used": min(rs.fuel_used(flags, s["start"], first - 1, yellow_burn), MAX_USED),
                             "laps_to_go": to_go, "laps_to_stage_end": to_stage, "stage": stage,
                             "lap_frac": first / race["scheduled_laps"], "stage_break": int(first - 1 in breaks),
                             "since": first - 1 - s["start"],
                             "y": int(s["end_how"] == "pit_yellow" and s["end"] <= last + 1)})
    return pd.DataFrame(rows)


def build_rows(data, yellow_burn=rs.YELLOW_BURN):
    return {"caution": caution_rows(data), "green": green_rows(data, yellow_burn),
            "yellow": yellow_rows(data, yellow_burn)}


# ---------------------------------------------------------------- fitting, predicting, explaining

def prepare(df, kind):
    x = df[FEATURES[kind]].copy()
    x["track"] = pd.Categorical(x["track"], categories=TRACKS)
    return x


def fit(rows, kind):
    return xgb.XGBClassifier(**PARAMS[kind], **COMMON).fit(prepare(rows, kind), rows["y"])


def predict(model, rows, kind):
    return model.predict_proba(prepare(rows, kind))[:, 1]


def shap_values(model, rows, kind):
    """SHAP values (log-odds) per feature for each row: how much each feature pushed the answer up or down."""
    dm = xgb.DMatrix(prepare(rows, kind), enable_categorical=True)
    values = model.get_booster().predict(dm, pred_contribs=True)
    return pd.DataFrame(values[:, :-1], columns=FEATURES[kind], index=rows.index), values[:, -1]


# ---------------------------------------------------------------- lookup tables for the simulation

def tables(models, fmt, extra_laps=40):
    """Turn the models into lookup tables for one race format, so the simulation can read
    a probability per lap instead of calling XGBoost 2,000 times a lap.

    `fmt` needs track, track_miles, scheduled_laps and stage_breaks. Returns
      caution  [lap, laps since the last restart]           chance a caution starts next lap
      green    [lap, laps of fuel used]                      chance a car pits under green
      yellow   [lap, laps of fuel used, stage break 0/1]     chance a car pits under this caution
    """
    laps = np.arange(fmt["scheduled_laps"] + extra_laps + 1)
    stage, to_stage, to_go = _context_arrays(fmt, laps)
    base = pd.DataFrame({"lap": laps, "track": fmt["track"], "track_miles": fmt["track_miles"],
                         "lap_frac": laps / fmt["scheduled_laps"], "laps_to_go": to_go, "stage": stage,
                         "laps_to_stage_end": to_stage})

    def grid(col, values, kind, **fixed):
        g = base.merge(pd.DataFrame({col: values}), how="cross").assign(**fixed)
        return predict(models[kind], g, kind).reshape(len(laps), len(values))

    out = {"caution": grid("laps_since_green", np.arange(MAX_SINCE + 1), "caution"),
           "green": grid("fuel_used", np.arange(MAX_USED + 1, dtype=float), "green")}
    out["yellow"] = np.stack([grid("fuel_used", np.arange(MAX_USED + 1, dtype=float), "yellow", stage_break=sb)
                              for sb in (0, 1)], axis=-1)
    return out


def save(models, info):
    FOLDER.mkdir(exist_ok=True)
    for kind, m in models.items():
        m.save_model(FOLDER / f"{kind}.json")
    (FOLDER / "meta.json").write_text(json.dumps(info, indent=2))


def load():
    """The trained models and their notes, or (None, None) if train_models.py hasn't been run."""
    if not (FOLDER / "meta.json").exists():
        return None, None
    models = {}
    for kind in FEATURES:
        m = xgb.XGBClassifier()
        m.load_model(FOLDER / f"{kind}.json")
        models[kind] = m
    return models, json.loads((FOLDER / "meta.json").read_text())
