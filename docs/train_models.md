# Testing and training the models (`train_models.py`)

This script answers the question that matters for any model: **is it actually better than
something simple?** Then it trains the final models the app uses.

```powershell
.venv\Scripts\python train_models.py
```
It takes about 10 minutes. Run it again after `build_data.py` adds new races.

## The test: walk-forward

Every race from **2024 on** (31 races) is predicted as if it were the next race on the
calendar. The models are trained only on races run **before** it, then scored on that race.
Nothing from the future leaks in. This is the honest way to test a model that will be used
on races that haven't happened yet.

Each model is scored against the **simple average it replaces**, which is what the
simulation used in step 2:

| Model | The simple average it's up against |
|---|---|
| Caution risk | the track's caution rate per green lap |
| Green-flag stops | the track's green-stop rate by fuel used (5-lap bins) |
| Yellow-flag stops | the track's yellow-stop rate by stint age, split by whether the car could make the finish |

The averages are also worked out only from earlier races, so the comparison is fair.

## How they're scored

- **Log loss** punishes confident wrong answers. Lower is better. "Skill" is how much lower
  the model's log loss is than the simple average's.
- **Brier score** is the average squared error of the chances. Lower is better.
- **AUC** is how often a real event gets a higher chance than a non-event: 0.5 is a coin
  flip, 1.0 is perfect.
- **Off by per race:** the expected number of events in a race (cautions, green stops,
  yellow stops) against the real number, averaged over races.

## The results (31 races, 2024 to October 2026)

| Model | Log loss: model vs average | Skill | AUC: model vs average | Off by per race: model vs average |
|---|---|---|---|---|
| Caution risk | 0.108 vs 0.114 | **5% better** | 0.68 vs 0.61 | 2.4 vs 2.8 cautions |
| Green-flag stops | 0.015 vs 0.022 | **29% better** | 0.96 vs 0.89 | 10.9 vs 11.6 stops |
| Yellow-flag stops | 0.429 vs 0.481 | **11% better** | 0.88 vs 0.85 | 10.0 vs 14.9 stops |

All three beat their simple average. The caution model's win is the smallest: cautions are
mostly crashes, and crashes are mostly luck. What it does catch is that cautions bunch up
right after restarts (about 12% on the restart lap, against about 2% mid-run).

Both the model and the simple average expect slightly more cautions than 2024–2026 actually
had (6.3 against 5.8 per race). Recent races have simply been a bit calmer than 2022–2023.

## What it writes

- `models/caution.json`, `green.json`, `yellow.json`: the final models, trained on all 53
  races. `meta.json` records what they were trained on.
- `reports/backtest_metrics.csv`: the scorecard above.
- `reports/backtest_races.csv`: expected against real counts, race by race.
- `reports/shap_importance.csv`: how much each feature moves each model's answers, on
  average.
- `reports/shap_dependence.csv`: how each feature's push changes across its range. These
  feed the SHAP charts in the app.

The app reads only these files. It never trains anything itself.
