# How far a tank goes (`fuel_window.py`)

Teams never say how much fuel their cars burn. But we can see what the cars *did*:
every lap, every pit stop, and whether the track was under green or yellow. If a car
ran 60 laps without stopping, its tank holds at least 60 laps. This file turns that
idea into numbers for a track. It works in **laps of fuel, not gallons**.

To see the results for Homestead-Miami, Michigan and Pocono:
```powershell
.venv\Scripts\python fuel_window.py
```
Or name the tracks you want (any of `homestead`, `michigan`, `pocono`, `kansas`,
`las-vegas`, `texas`, `charlotte`, `chicagoland`):
```powershell
.venv\Scripts\python fuel_window.py kansas las-vegas
```

## How it works

1. **Cut each car's race into stints.** A stint runs from one pit stop to the next
   (or to the checkered flag). If a car pits twice under the same caution, only the
   second stop counts, because that's when it leaves with a full tank.
2. **Mark every lap green or yellow** from the flag record in the lap-times feed.
   Lapped cars are lined up with the leader's lap, so their yellows land in the right place.
3. **Throw out stints we can't trust.** Only stints run on the lead lap count, since
   lapped cars go slower and use less fuel. A stint is also dropped if one of its
   green laps was 25% slower than the car's usual pace. That usually means a stop
   the pit feed missed, or a spin.
4. **The longest clean stint is the lower bound.** The tank lasts *at least* that long.
   It's reported two ways: green laps only (safe even if yellow laps burned nothing),
   and total laps (if a yellow lap burned as much as a green one). The truth sits
   somewhere between, because cars use less fuel behind the pace car.
5. **Cars that ran dry give the upper bound.** Race control's notes are searched for
   phrases like "ran out of fuel" or "no fuel pressure."

It also measures **when cars make green-flag stops**. If that's well short of the
tank's range, stops are being driven by tires, not fuel.

## The functions

- **`track_races(track_id)`** lists every points race at a track since 2022 (the
  Next Gen car), read from the saved results.
- **`race_stints(season, race_id)`** gives one row per stint for every car: start
  and end lap, green and yellow laps, how it ended (green stop, yellow stop, finish,
  or out of the race), how far behind the leader the car was, and its count of
  suspicious slow laps.
- **`track_stints(track_id)`** does the same for every race at a track.
- **`green_runs(season, race_id)`** lists each unbroken stretch of green-flag laps.
- **`run_outs(season, race_id)`** finds race control notes about cars running out of
  fuel. A note can mention several cars ("#11 extends his lead, #24 pits out of
  fuel"), so it picks the car named just before the "out of fuel" phrase.
- **`fuel_window(stints)`** sums it all up: typical green-stop lap, the lower bound on
  the window, and the longest clean stints behind it.
- **`stint_at(...)`** and **`laps_into_stint(...)`** find the stint a car was on when it
  ran dry, and how many green and yellow laps it had run by then.
- **`report(name, track_id)`** prints all of the above for one track.

## What it found at Homestead-Miami (2022 to 2025, four races)

- **Green-flag stops come about 35 laps into a stint** (the middle half fall between
  32 and 40 laps). That happens even in 80-lap green runs.
- **A tank lasts at least 61 green laps** (64 counting yellow laps). The 2024 winner
  ran 64 laps (61 green, 3 yellow) on one tank to a green-flag stop.
- So cars pit for **tires** long before fuel runs low. Homestead's surface wears
  tires hard, so fresh tires are worth more than the time a stop costs. Fuel only
  matters there if a late caution leaves cars needing to run 60+ laps to the end.
- **Nobody ran out of fuel at Homestead** in these races, so there's no upper bound
  from this track itself.

**Check against the other 1.5-mile tracks:** the longest clean stints at Kansas, Las
Vegas, Texas, Charlotte and Chicagoland run 57 to 70 green laps, close to
Homestead's 61. Two fuel problems turned up in the notes, but neither is a clean
"ran out of a full tank":
- Kansas 2026, #45 stumbled 46 green laps after a stop and switched to the backup
  pump, but kept going and won. That suggests low fuel, maybe a short fill.
- Las Vegas 2026, #11 lost fuel pressure 4 laps after a stop. That's a fill or pump
  problem, not range.

## The real test: Michigan and Pocono (2022 to 2026, five races each)

These are the tracks where fuel decides races, and cars actually ran dry there. So
they're where the method can be checked.

**Michigan (2 miles).** Green-flag stops come about **46 laps** into a stint (middle
half 40 to 51). That's right up against the tank, so fuel, not tires, sets the stops.
In 2025 the whole field pitted under caution on lap 147 and needed to go 53 laps to
the end:
- The top 15 made it, on 49 green + 3 yellow laps, by saving fuel.
- #24 ran dry after 48 green + 3 yellow laps. #71 ran dry after 42 green + 3 yellow.
- #77's crew told him at lap 160 that he was 4 laps short at full pace, which
  puts a full-pace tank at about 49 laps.
- The longest clean stint at Michigan is 53 green + 4 yellow laps (2024).

So **a Michigan tank is about 45 to 49 laps at full pace, stretching to about 53 with
hard saving.** The cars that ran dry fall right in that band. The check passes.

**Pocono (2.5 miles).** Green-flag stops come about **35 laps** in (middle half 26 to 42).
- In 2026, #20 ran 50 straight green laps while saving fuel, then came in out of
  fuel one lap from the end.
- In 2025 the winner was told, 35 laps into his last stint, that he was half a lap
  short of the 42 he needed. He saved enough to make it.
- The longest clean stint is 46 green + 11 yellow laps (2026 winner).

So **a Pocono tank is about 42 laps at normal pace and up to about 50 with heavy
saving.** The check passes, but saving fuel changes the range more here than at
Michigan. That fits the notes about drivers staying in one gear to save fuel.

**What this means for the simulation:** a track's window should be a range, from full
pace to all-out saving, not one number. Homestead is the contrast: there, cars pit
for tires about 25 laps before the tank would run dry.

**Still unknown:** how much less fuel a yellow lap burns. The dry runs above happened
almost entirely under green, so they can't separate the two. This needs a dry run
after a long caution, or a model fitted across many tracks.
