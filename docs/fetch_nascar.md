# Getting the race data (`fetch_nascar.py`)

The series publishes a surprising amount of race timing data as public JSON files
(the addresses are at the top of the file). You don't need an account or an API key.
This file downloads them and saves a copy, so each one is only fetched once. It came
from an earlier project, which is why a few of its helpers aren't used here.

To see which data exists for every Talladega race:
```powershell
.venv\Scripts\python fetch_nascar.py
```

## What's available

- **The schedule** (`/{year}/race_list_basic.json`) lists every race in a season,
  with the race IDs needed for everything else. It's available from 2015.
- **Lap times** (`/{year}/1/{race_id}/lap-times.json`) give every car's time and
  position on every single lap.
- **Pit stops** (`/{year}/1/{race_id}/live-pit-data.json`) cover each stop: how long
  it took, what was done, and places gained or lost.
- **Race notes** (`/{year}/1/{race_id}/lap-notes.json`) are race control's running
  commentary: cautions, incidents, penalties.
- **The weekend feed** (`/{year}/1/{race_id}/weekend-feed.json`) starts as an entry
  list, becomes the starting lineup after qualifying, and ends up as **every car's
  result**: finish, status, laps, team, make, qualifying and stage results. Results
  are available from 2018.
- **Loop data** (`/loopstats/prod/{year}/1/{race_id}.json`) gives every driver's
  rating, running positions and passing numbers, keyed by the feed's `driver_id`
  rather than car number. It's available from 2019.

Lap times, pit stops and race notes only exist from **2020 onwards**. Ask for an
older race and the server answers "403 Forbidden," which the code treats as "no data."
The `1` in each address means the Cup Series, and Talladega's track ID is 82.

## The functions

- **`get_json(url, cache_path)`** fetches a feed, or reads the saved copy if there
  is one. It returns `None` if the feed doesn't exist.
- **`talladega_races(year)`** lists that year's Talladega Cup races, with race ID,
  name, date and whether it's the spring or fall race.
- **`fetch_race(year, race_id)`** gets the lap, pit and notes feeds for one race in one go.
- **`weekend_feed(year, race_id)`** gets the weekend feed. It always downloads a
  fresh copy, because the feed keeps changing over a race weekend.
- **`race_results(year, race_id)`** gets every car's result for a finished race. It's
  saved once the race has been run, and gives `None` for a race that hasn't happened.
  Only the parts that are used (the results, stage results and race details) are kept,
  because the full weekend feed is several times bigger.
- **`loop_stats(year, race_id)`** gets every driver's loop data, or `None`.
- **`starting_info(year, race_id, car)`** (not used in this app): once the lineup is
  posted, it returns the car's start position, qualifying speed,
  the pole speed and the field size. Before that, or when the race has no ID yet, it
  returns `None`.

## The saved copies

A finished race never changes, so its feeds are saved in `data/cache/` and reused.
The folder isn't in the GitHub repo, so a fresh copy of the project downloads it on
first use.
If you ever want a fresh download, delete the file. The live weekend feed is never
saved, because it keeps changing over a race weekend. Once a race is over,
`race_results()` saves its final results as `*_results.json`.

## Quirks worth knowing

Real-world data always has a few surprises. These are the ones found here:
- **The 2017 schedule file actually lists 2018's races.** Races are matched by
  date, so this doesn't cause problems.
- **Before qualifying,** the weekend feed lists every car with a start position of
  0. That's how the code knows the lineup isn't out yet.
- **Races without time trials,** like spring 2026, have a lineup but a pole speed
  of 0. The qualifying gap is then left blank rather than calculated from a zero.
- **The pit feed opens with a block of placeholder rows** from the starting grid,
  full of zeros and −1s. These are skipped.
