# Python & XGBoost Fuel Strategy Simulator

When will each car pit? Will the race come down to fuel mileage? Who has to stop late,
and who can stretch it to the end? This app answers those questions for Cup Series
races by simulating them lap by lap, thousands of times over.

Built by **Tony Honesto** · [LinkedIn](https://www.linkedin.com/in/tony-honesto-4195023) ·
[GitHub](https://github.com/atonyhonesto)

## The honest constraint

Nobody publishes how much fuel a car burns. Teams keep it secret. So the app never
talks about gallons. It works out **laps of fuel** from what cars actually did: how
long they ran between stops, under green and under yellow, and where some ran dry.

The method was checked at Michigan and Pocono, where fuel decides races. There, the
tank size worked out from the longest stints matches the cars that really ran out of
fuel.

## What's in the app

- **Race forecast:** the chance of a fuel-mileage finish, cautions, stops per car, and
  when the field is likely to pit. Each is checked against every real race at the track
  since 2022.
- **The models:** the XGBoost backtest scorecard (wins and losses alike), caution risk
  lap by lap, and SHAP charts of what drives each model.
- **Replay a race:** pick a real race and a lap. See who can make it to the end at
  full pace, who has to save, and who needs another stop. Then compare with what
  really happened.
- **Fuel windows:** the evidence behind each track's tank size.
- **How it works:** the method and its limits.

Eight ovals are covered:
- Michigan and Pocono, where fuel sets the stops.
- Nashville, Kansas, Las Vegas, Texas and Homestead-Miami, where tires do.
- Martinsville, where fuel hardly matters.

## Under the hood

1. **Data pipeline (Python, Requests).** The app pulls public race timing feeds: every
   lap time, every pit stop and race control's notes, for 53 races since 2022. Each
   feed is downloaded once and cached, then boiled down to five small CSV tables
   (about 1 MB).
2. **Inference (pandas).** About 530,000 car-laps are cut into 13,000+ stints between
   11,000+ pit stops, and each lap is marked green or yellow. A lap-time check
   catches stops the pit data missed. The tank size is tested against cars that
   really ran dry.
3. **XGBoost models, explained with SHAP.** Three classifiers make the calls the
   simulation needs every lap: will a caution come out, does this car pit under green,
   does it pit under yellow? Each was backtested **walk-forward**: 31 races from 2024 on,
   each predicted only from races before it, against the simple average it replaces.

   | Model | Better than the simple average (log loss) | AUC: model vs average |
   |---|---|---|
   | Caution risk | 5% | 0.68 vs 0.61 |
   | Green-flag stops | 29% | 0.96 vs 0.89 |
   | Yellow-flag stops | 11% | 0.88 vs 0.85 |

   SHAP shows what drives them. Cautions bunch right after restarts (about 12% on the
   restart lap, against about 2% mid-run). Pit stops are driven most by fuel used, and by
   laps left to the stage break.
4. **Monte Carlo (NumPy).** The models become lap-by-lap lookup tables. Then 2,000 races
   are played out lap by lap, every car at once, in one vectorized loop: one to two
   seconds per track.
5. **App (Streamlit, Plotly).** Simulations are cached, and there are light and dark
   themes. The chart palette is checked for colour-blind readers. It's tested with
   AppTest and Playwright screenshots.

## The files

| File | What it does | Plain-English guide |
|---|---|---|
| `app.py` | the Streamlit app | [docs/app.md](docs/app.md) |
| `app_style.py` | colours, fonts, banner and footer | [docs/app_style.md](docs/app_style.md) |
| `race_sim.py` | the lap-by-lap simulation | [docs/race_sim.md](docs/race_sim.md) |
| `models.py` | the three XGBoost models, their SHAP values, and the lookup tables the simulation reads | [docs/models.md](docs/models.md) |
| `train_models.py` | walk-forward backtest against simple averages, then trains and saves the models | [docs/train_models.md](docs/train_models.md) |
| `fuel_window.py` | works out how far a tank goes at each track | [docs/fuel_window.md](docs/fuel_window.md) |
| `build_data.py` | turns the raw race feeds into the small tables in `data/` | [docs/build_data.md](docs/build_data.md) |
| `fetch_nascar.py` | downloads and caches the public race feeds | [docs/fetch_nascar.md](docs/fetch_nascar.md) |
| `check_colours.py` | checks chart colours for colour-blind safety | [docs/check_colours.md](docs/check_colours.md) |
| `simulation.py` | whole-race finishing-order helpers, kept for the models to come | [docs/simulation.md](docs/simulation.md) |

## Running it yourself

```powershell
setup.bat          # builds .venv on this computer
run_app.bat        # opens the app in your browser
```

To refresh the data after a new race: `.venv\Scripts\python build_data.py`.

## Next

- **Measure how much less fuel a yellow lap burns.** It's still an assumption (0.35 of a
  green lap), because nearly every car that ran dry did it under green.
- **More tracks:** Charlotte, Darlington, road courses and superspeedways.
- **Laps per tank stays measured, not modeled.** With eight tracks, a model would only
  memorize eight numbers. More tracks would change that.

---
An independent fan project built on public race timing data. Not affiliated with any
racing series, team or sponsor. For fun, not betting advice.
