# The app (`app.py`)

A Streamlit web app that simulates fuel strategy in Cup races. Run it on this computer
with `run_app.bat`, or:
```powershell
.venv\Scripts\streamlit run app.py
```

## At the top

- **The hook:** one line under the banner for fans: "Will they make it on fuel?",
  with the real number of races, and what the simulations show.
- **"Under the hood" button:** opens a panel on how the app is built, for the
  technical reader. It has:
  - live counts from the data (races, pit stops, car-laps),
  - tech badges, with XGBoost and SHAP marked "next",
  - four short steps: the data pipeline, inference with pandas, Monte Carlo in NumPy,
    and the app itself.
  The same write-up is repeated at the bottom of the How it works tab, so it isn't
  hidden behind a click. The text is built in `tech_story()`, and the counts in
  `get_stats()`.
- **Track:** the eight ovals in the app. It opens on Homestead-Miami; change
  `DEFAULT_TRACK` in `app.py` to open somewhere else.
- **Simulation engine:** "XGBoost models" (the default) or "Simple averages". The models
  decide cautions and pit stops lap by lap; the simple averages are what they were tested
  against. If the models haven't been trained on this computer, the switch is greyed out
  and the simple averages are used.
- **Assumptions:** two sliders for the things public data can't pin down:
  - how much fuel a yellow lap uses compared with a green one, and
  - how much further hard saving stretches a tank.
  Change them and everything reruns.
- **Chips and a note** describe the track, including whether fuel or tires set the
  stops. At tire-limited tracks the note says plainly that fuel rarely matters. At
  Martinsville it adds that stage breaks come before anyone runs low.

## The tabs

**Race forecast.** 2,000 simulations of the next race at the track, using its latest
format.
- **Stat tiles:** the chance of a fuel-mileage finish, cautions, stops per car, and the
  chance a car runs dry.
- **The pit chart:** the share of the field pitting on each lap, under green (blue)
  and under yellow (gold). Stage breaks send most of the field down pit road at once.
  Those spikes are clipped and labelled with their real height, so the rest stays
  readable. "Show as a table" lists the same numbers.
- **"How did the real races compare?"** The same test for each real race, next to its
  simulated chance, with real and simulated cautions and stops.

**Replay a race.** Pick a real race and a lap. For every car still running:
- when it last stopped and how much fuel it has used,
- whether it can make the finish at full pace, only by saving, or not at all if no
  caution comes,
- and, from 2,000 simulations of the rest of the race, its chances of stopping again,
  stopping under green, having to save and running dry.

A chart shows each car's likely next-stop window (the middle 80% of simulations),
with a diamond where it really stopped.

**Fuel windows.** The evidence behind the tank:
- full-pace and saving tank sizes,
- a histogram of where real green-flag stops came, with the tank marked,
- the longest clean stints,
- cars that ran out (from race control's notes),
- every track side by side.

**The models.** The XGBoost side, with results as they came out:
- **Three stat tiles:** how much better each model scored than its simple average in the
  walk-forward backtest, with its AUC against the average's.
- **The scorecard:** log loss, Brier score, AUC, and how far off the per-race counts were.
- **Caution risk lap by lap** at the chosen track: the model's curve, with its spikes after
  each restart, against the flat average.
- **SHAP:** pick a model to see which features move its answers most. Then pick a feature
  to see how it pushes the chance up or down across its range.

It reads `reports/` and `models/`, written by `train_models.py`.

**How it works.** The method and its limits, in plain words. It opens with the honest
constraint: nobody publishes fuel burn, so the app works in laps of fuel.

## Speed

Simulations are cached. The first view of a track takes a few seconds (up to about 10
at Martinsville), mostly for the real-race check. After that it's instant. A replay
takes about half a second per lap you pick.

## How it was checked

- **`streamlit.testing.v1.AppTest`:** every track, a replay, and a change of
  assumption, with no errors.
- **Screenshots** in light and dark, desktop and phone, taken with Playwright driving
  the installed Chrome. Playwright is a test tool on this PC only, not in
  `requirements.txt`.

The app never names the racing series or shows its marks. Team names, logos and
sponsors are left out too.
