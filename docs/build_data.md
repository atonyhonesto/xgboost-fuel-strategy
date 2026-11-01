# Building the app's data (`build_data.py`)

The raw race feeds run to hundreds of files and many megabytes. They live in
`data/cache/`, which isn't uploaded to GitHub. The app doesn't need them. This
script boils them down to five small tables in `data/`, about 1.2 MB in all, and
those are uploaded.

Run it after a new race at one of the app's tracks:
```powershell
.venv\Scripts\python build_data.py
```
Any feed it doesn't already have is downloaded and saved in the cache first.

## The tables

| File | One row per | What's in it |
|---|---|---|
| `races.csv` | race | track, date, scheduled and actual laps, the laps where stages end, track length |
| `cars.csv` | car per race | driver, finishing position, status (running or why it retired), laps completed |
| `stints.csv` | stint | from `fuel_window.race_stints()`: start and end lap, green and yellow laps, how it ended, and whether it's clean |
| `flags.csv` | lap per race | the flag on the leader's lap: 1 green, 2 yellow |
| `run_outs.csv` | note | race control notes about cars running out of fuel, with the car and lap |

## The tracks

`APP_TRACKS` lists the eight ovals in the app:
- Michigan and Pocono, where fuel sets the stops.
- Nashville, Kansas, Las Vegas, Texas and Homestead-Miami, where tires do.
- Martinsville, to show a short track where fuel hardly matters.

Charlotte and Darlington are left out for now, because their two races a year have
different lengths.

`stage_breaks()` turns the stage lengths into the laps where a stage ends before the
finish. A three-stage race has two; a four-stage race would have three.
