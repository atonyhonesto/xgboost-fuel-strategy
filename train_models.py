"""Backtest the three models walk-forward, then train them on every race and save them.

Walk-forward: every race from BACKTEST_FROM on is predicted by models trained only on
races run before it, exactly as if it were the next race on the calendar. Each model is
scored against the simple average the simulation used before (step 2):
  caution  each track's caution rate per green lap
  green    each track's green-stop rate by how much fuel the car has used (5-lap bins)
  yellow   each track's yellow-stop rate by stint age, split by whether the car could
           make the finish without stopping (the step-2 pit table)

Writes:
  models/caution.json, green.json, yellow.json, meta.json   the final models
  reports/backtest_metrics.csv     model vs baseline, per model
  reports/backtest_races.csv       expected vs actual counts, race by race
  reports/shap_importance.csv      average size of each feature's SHAP value
  reports/shap_dependence.csv      how each feature's SHAP value changes across its range

Run:  python train_models.py      (about 10 minutes)
"""
import time

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

import models as m
import race_sim as rs

BACKTEST_FROM = "2024-01-01"
SHRINK = 50            # pseudo-rows pulling thin baseline cells toward the overall rate
USED_BIN = 5
SHAP_SAMPLE = 20_000


# ---------------------------------------------------------------- baselines (step 2's simple averages)

def _rate_table(train, keys):
    """Event rate per group of `keys`, shrunk toward the same rate without the track split."""
    overall_keys = [k for k in keys if k != "track"]
    if overall_keys:
        prior = train.groupby(overall_keys)["y"].mean().rename("prior")
    g = train.groupby(keys)["y"].agg(["sum", "size"]).reset_index()
    g = g.merge(prior.reset_index(), on=overall_keys, how="left") if overall_keys else g.assign(prior=train["y"].mean())
    g["rate"] = (g["sum"] + SHRINK * g["prior"]) / (g["size"] + SHRINK)
    return g[keys + ["rate"]], (prior if overall_keys else train["y"].mean())


def _apply(table, prior, test, keys):
    out = test[keys].merge(table, on=keys, how="left")["rate"].to_numpy()
    if np.isnan(out).any():
        fill = (test[[k for k in keys if k != "track"]].merge(prior.reset_index(), how="left")["prior"].to_numpy()
                if isinstance(prior, pd.Series) else np.full(len(test), prior))
        out = np.where(np.isnan(out), fill, out)
    return np.nan_to_num(out, nan=0.0)


def windows(data, race_ids, yellow_burn=rs.YELLOW_BURN):
    """Each track's longest clean stint (laps of fuel), from the given races only."""
    st = data["stints"][data["stints"]["race_id"].isin(race_ids)]
    st = st.merge(data["races"][["race_id", "track"]], on="race_id")
    clean = st["end_how"].isin(["pit_green", "pit_yellow", "finish"]) & (st["laps_behind"] == 0) & (st["slow_green"] == 0)
    st = st[clean]
    return (st["green"] + yellow_burn * st["yellow"]).groupby(st["track"]).max()


def baseline(kind, train, test, data=None):
    if kind == "caution":
        table, prior = _rate_table(train, ["track"])
        return _apply(table, prior, test, ["track"])
    if kind == "green":
        tr, te = (d.assign(used_bin=(d["fuel_used"] // USED_BIN).astype(int)) for d in (train, test))
        table, prior = _rate_table(tr, ["track", "used_bin"])
        return _apply(table, prior, te, ["track", "used_bin"])
    # yellow: stint age bin x can finish, the step-2 pit table
    win = windows(data, train["race_id"].unique())
    tr, te = (d.assign(age_bin=np.searchsorted(rs.SINCE_BINS, d["since"], side="right"),
                       can=(d["fuel_used"] + d["laps_to_go"] <= d["track"].map(win).fillna(1e9)).astype(int))
              for d in (train, test))
    table, prior = _rate_table(tr, ["track", "age_bin", "can"])
    return _apply(table, prior, te, ["track", "age_bin", "can"])


# ---------------------------------------------------------------- walk-forward backtest

def backtest(data, rows):
    races = data["races"].sort_values("race_date")
    test_races = races[races["race_date"] >= BACKTEST_FROM]
    preds = {k: [] for k in rows}
    for i, (_, race) in enumerate(test_races.iterrows(), 1):
        t0 = time.time()
        for kind, df in rows.items():
            train = df[df["race_date"] < race["race_date"]]
            test = df[df["race_id"] == race["race_id"]]
            if test.empty:
                continue
            model = m.fit(train, kind)
            preds[kind].append(test.assign(p_model=m.predict(model, test, kind),
                                           p_base=baseline(kind, train, test, data)))
        print(f"  {i:>2}/{len(test_races)} {race['race_date']} {race['track']:<13} {time.time() - t0:5.1f}s")
    return {k: pd.concat(v, ignore_index=True) for k, v in preds.items()}


def score(preds):
    rows = []
    for kind, p in preds.items():
        for who, col in (("XGBoost model", "p_model"), ("Simple average (step 2)", "p_base")):
            q = np.clip(p[col], 1e-6, 1 - 1e-6)
            rows.append({"model": kind, "who": who, "rows": len(p), "events": int(p["y"].sum()),
                         "log_loss": log_loss(p["y"], q, labels=[0, 1]), "brier": brier_score_loss(p["y"], q),
                         "auc": roc_auc_score(p["y"], q)})
    out = pd.DataFrame(rows)
    base = out[out["who"].str.startswith("Simple")].set_index("model")["log_loss"]
    out["skill_vs_baseline"] = 1 - out["log_loss"] / out["model"].map(base)
    return out


def race_counts(preds, data):
    """Expected vs actual events, race by race: cautions, green stops, yellow stops."""
    parts = []
    for kind, p in preds.items():
        g = p.groupby("race_id").agg(actual=("y", "sum"), model=("p_model", "sum"), baseline=("p_base", "sum"))
        parts.append(g.assign(model_kind=kind).reset_index())
    out = pd.concat(parts, ignore_index=True)
    return out.merge(data["races"][["race_id", "season", "track", "race_date"]], on="race_id")


# ---------------------------------------------------------------- SHAP summaries of the final models

def shap_summaries(models, rows):
    imp, dep = [], []
    for kind, model in models.items():
        df = rows[kind]
        sample = df.sample(min(SHAP_SAMPLE, len(df)), random_state=0)
        values, _ = m.shap_values(model, sample, kind)
        for feat in m.FEATURES[kind]:
            imp.append({"model": kind, "feature": feat, "mean_abs_shap": float(values[feat].abs().mean())})
            x = sample[feat]
            if feat == "track":
                grp = x.astype(str)
            else:
                grp = pd.qcut(x.rank(method="first"), q=min(12, x.nunique()), labels=False)
            d = pd.DataFrame({"x": x if feat == "track" else x.astype(float), "shap": values[feat], "g": grp})
            d = d.groupby("g").agg(x=("x", "first" if feat == "track" else "mean"), shap=("shap", "mean"),
                                   n=("shap", "size")).reset_index(drop=True)
            dep.append(d.assign(model=kind, feature=feat))
    return pd.DataFrame(imp), pd.concat(dep, ignore_index=True)


def main():
    data = rs.load()
    print("Building training rows...")
    rows = m.build_rows(data)
    for kind, df in rows.items():
        print(f"  {kind:<8} {len(df):>8,} rows, {int(df['y'].sum()):>6,} events")

    print(f"Walk-forward backtest (every race from {BACKTEST_FROM}, trained only on earlier races)...")
    preds = backtest(data, rows)
    metrics = score(preds)
    counts = race_counts(preds, data)
    m.REPORTS.mkdir(exist_ok=True)
    metrics.to_csv(m.REPORTS / "backtest_metrics.csv", index=False)
    counts.to_csv(m.REPORTS / "backtest_races.csv", index=False)
    print(metrics.round(4).to_string(index=False))

    print("Training the final models on every race...")
    final = {kind: m.fit(df, kind) for kind, df in rows.items()}
    imp, dep = shap_summaries(final, rows)
    imp.to_csv(m.REPORTS / "shap_importance.csv", index=False)
    dep.to_csv(m.REPORTS / "shap_dependence.csv", index=False)
    m.save(final, {"trained_on_races": int(data["races"]["race_id"].nunique()),
                   "last_race": str(data["races"]["race_date"].max()), "backtest_from": BACKTEST_FROM,
                   "yellow_burn": rs.YELLOW_BURN, "features": m.FEATURES})
    print("Saved models/ and reports/.")


if __name__ == "__main__":
    main()
