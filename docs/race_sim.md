# Simulating a race lap by lap (`race_sim.py`)

This is the engine behind the app. It plays out a race one lap at a time, thousands
of times over, and keeps track of every car's fuel. Each run is different because
cautions come at random, so the results are chances: how likely a fuel-mileage
finish is, when cars are likely to pit, who might run dry.

Everything is in **laps of fuel**, never gallons, because nobody publishes fuel burn.

To print a summary for every track, and compare each one with the real races:
```powershell
.venv\Scripts\python race_sim.py
```

## What happens in one simulated race

Every car starts with a full tank. Then, lap by lap:

1. **Fuel burns.** A green lap uses one lap of fuel. A yellow lap uses
   `YELLOW_BURN` of one (0.35 by default; see "Assumptions" below).
2. **Cars that run out pit.** Each car's real range is a little different from what
   its team planned on (a spread of about 2%). So a car trying to stretch it can come
   up short.
3. **Green-flag stops.** When a car's stint reaches its target, it pits, unless it can
   reach the next stage break or the finish on what's left. Targets are drawn from
   the real green-flag stops at the track. "What's left" means with hard saving where
   fuel sets the stops, and at full pace where tires do.
4. **Yellow-flag stops.** On the first lap of a caution, each car decides whether to
   pit. It does so with the same chance real cars on the lead lap did, given how old
   their stint was and whether they could make the finish without stopping.
5. **Cautions.** Stage breaks always bring one, at their known laps. Other cautions
   come at random, at the track's real rate per green lap, and last as long as real
   ones did.
6. **Overtime.** A caution that would run to the finish sends the race to overtime:
   two more green laps once it ends (up to three tries).

At the end, a car **had to save** if its last stint used more than a full-pace tank.
A race is a **fuel-mileage finish** when at least a third of the field had to save,
made a late green-flag stop for fuel, or ran dry.

## Two engines: XGBoost models or simple averages

`simulate()` can run two ways:

- **With `tables`** (from `models.tables()`, the app's default), the three XGBoost models
  make the calls: the chance of a caution each lap, the chance each car pits under green,
  and the chance it pits under a caution. See `docs/models.md`.
- **Without `tables`**, it uses the simple per-track averages described below. These are
  what the models were tested against.

With the models, one safety rule stays: **the model decides when a team chooses to pit,
but the tank decides the latest it can.** Real teams almost never let a car get past the
latest green stop seen at the track, so the model has hardly any examples there. Its chances
for that range would be guesses. So once a car's stint passes the latest real green stop
there (the 95th percentile), and it can't reach the next stage break or the finish, it pits.

Checked race by race against the real races, the model engine is at least as close as the
simple averages on cautions per race, and closer on stops per car. At Homestead, for
example, it gives 6.7 stops per car against 6.4 real; the simple averages give 7.2.

## Fixes made while adding the models

- **A caution could start on the restart lap.** The check for a new caution also ran on
  the last lap of a caution. That never happens in real data, because a restart lap is
  always green. With the models, it showed up as about one extra caution per race. Now
  only a green lap can be followed by a new caution.
- **Late-stop rule.** A late green-flag stop only counts toward a "fuel-mileage finish"
  if the car couldn't have reached the finish at full pace. A late stop for tires isn't a
  fuel problem. The real-race check uses the same rule.
- **Full-pace range at tire-limited tracks.** It's now worked out the same way as the
  track's tank. Before, a car that ran exactly the longest stint seen was flagged as
  having to save fuel.

## Where the numbers come from

`track_params()` measures everything from the real races at the track since 2022:

- **The tank.** The longest clean stint (see `docs/fuel_window.md`), in laps of fuel.
- **Fuel-limited or tire-limited?** If green-flag stops come at 75% or more of that
  longest stint, fuel sets the stops (Michigan, Pocono). The longest stint is then
  the tank *with* saving, and full pace is 10% less.

  Otherwise tires set the stops (Homestead, Kansas, Martinsville, and others). Nobody
  needed to save to run the longest stint, so it's treated as a full-pace tank, and
  saving adds 10% on top. That's only a lower bound. It matters little, because at
  those tracks cars rarely run anywhere near empty.
- **Caution rate and lengths.** Cautions per green lap, and how many laps each one
  lasted. Stage-break cautions are counted separately.
- **Green-flag stop laps.** How much fuel cars had used when they pitted under green.
- **The yellow-flag pit table.** For cars that finished on the lead lap: the share
  that pitted under each caution, split by stint age (under 10, 10–19, 20–29, 30–39,
  40+ laps) and by whether they could make the finish without stopping. Thin cells
  are pulled toward the overall rate.

## Assumptions (changeable in the app)

- **`YELLOW_BURN = 0.35`.** How much fuel a yellow lap uses, compared with a green
  one. It can't be measured from public data yet: nearly every dry run came under
  green. One clue is Pocono 2026: the winner made 46 green + 11 yellow laps, while
  another car ran dry after 50 green. That only fits if a yellow lap burns about a
  third of a green one or less.
- **`SAVE = 0.10`.** How much further hard saving stretches a tank. Michigan 2025
  suggests about 8% and Pocono about 16%.

## Replaying a real race from lap N

- **`replay_state(data, race_id, lap)`** works out, for every car still running at
  that lap, when it last stopped and how much fuel it has used since.
- **`simulate(..., start_lap=lap, state=...)`** plays out the rest of the race from
  there.
- **`replay_table()`** gives each car a plain verdict if no caution comes:
  - makes it at full pace,
  - makes it if it saves, or
  - needs another stop.
  It also gives the simulated chances: another stop, a green-flag stop, having to
  save, running dry, and the likely window for its next stop.
- **`actual_next_stop()`** adds what really happened, to compare.

## The check against real races

`check_track()` (with or without `models`) runs the simulation for each real race and
puts it next to the real thing: chance of a fuel-mileage finish against whether it was one, plus cautions and
stops per car (simulated and real). The parameters come from every race at the track,
including the one being checked. So it's a consistency check, not a forecast test.
The walk-forward backtest comes with the models in step 3.

What it showed:
- **Cautions:** within about one per race at every track.
- **Stops per car:** within about one of the real races (lead-lap finishers).
- **Fuel-mileage finishes**, using the simple averages: about 10% at Michigan (1 real one
  in 5 races), about 6% at Pocono, and under 1% at the tire-limited tracks (none real).
  With the models: about 15% at Michigan, 13% at Pocono, 6% at Kansas, and 1–2% at
  Homestead and Martinsville. Pocono's 2025 and 2026 races and Kansas's spring 2024 race
  were close fuel-saving finishes that fell just short of the one-third line.

## The data

`load()` reads the tables in `data/` that `build_data.py` writes, so the app never
downloads raw feeds.
