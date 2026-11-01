# The XGBoost models (`models.py`)

Before these models, the simulation used one simple average for each decision. One caution
rate per track, the usual lap for a green-flag stop, and a table of how often cars pit
under yellow. Those averages ignore the situation. A caution is likelier right after a
restart. A car with 40 laps of fuel used and 8 laps to the stage break behaves
differently from one with 8 used and 40 to go.

This file holds three XGBoost classifiers. Each answers one yes/no question the simulation
asks every lap, and gives a chance rather than a flat average.

| Model | The question | One training row per | What it sees |
|---|---|---|---|
| `caution` | Will a caution come out on the next lap? | green lap | track, track length, how far into the race, laps to go, stage, laps to the end of the stage, laps since the last restart |
| `green` | Does this car pit under green on this lap? | car per green lap of a stint | track, track length, fuel used this stint, laps to go, laps to the end of the stage, stage, how far into the race |
| `yellow` | Does this car pit under this caution? | car per caution | the same as `green`, plus whether the caution is a stage break |

## What goes into the training rows

- **`caution_rows()`** has one row for every green lap of every race. The answer is
  1 if a caution starts on the next lap. Stage breaks are left out, because those
  cautions are certain: there's nothing to predict.
- **`green_rows()`** has one row for every green lap of every stint, for cars still
  running at the finish. Stints with a suspiciously slow green lap are skipped, so
  damage stops don't teach the model about strategy. The answer is 1 on the lap the
  car pitted under green.
- **`yellow_rows()`** has one row per car per caution, for cars that finished on the
  lead lap. The answer is 1 if the car pitted under that caution.

"Fuel used" counts each yellow lap as 0.35 of a green one, the same assumption the
simulation uses.

## Why tank size isn't a model

The plan had XGBoost estimating laps per tank by track. With only eight tracks, a model
would just memorize eight numbers. So the tank is still measured directly from the
longest clean stints (see `docs/fuel_window.md`). The `green` model covers the other half
of that question: when teams actually choose to pit.

## Turning a model into something the simulation can use

The simulation plays 2,000 races at once. Calling XGBoost for every car on every lap
would be far too slow. So **`tables()`** asks each model once for every situation the
race can be in, and stores the answers as lookup tables:

- `caution[lap, laps since the last restart]`
- `green[lap, laps of fuel used]`
- `yellow[lap, laps of fuel used, stage break or not]`

Each lap of the simulation just reads its chance from these tables.

## Explaining the models: SHAP

**`shap_values()`** splits each prediction into a push from each feature: how much it
raised or lowered the chance. It uses XGBoost's built-in SHAP values, so no extra
package is needed. The values are on the log-odds scale: positive means the feature made
the event more likely, negative less likely.

## The other functions

- **`build_rows()`** builds all three sets of training rows.
- **`fit()`** trains one model, and **`predict()`** returns its chances.
- **`save()`** writes the models to `models/`, and **`load()`** reads them back.
  `load()` returns nothing if the models haven't been trained yet; the app then falls
  back to the simple averages.

## Settings

`PARAMS` keeps every model small and cautious: shallow trees, many rows required per
leaf, and a modest learning rate. With 317 cautions and about 1,000 green-flag stops to
learn from, a bigger model would learn noise.
