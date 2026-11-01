"""Download and cache the public race timing feeds for Cup Series races.

Probing on 2026-10-21 found that
lap-times, live-pit-data and lap-notes exist for 2021 onward; 2019 and earlier
return HTTP 403 (no file). The 2014 schedule file also returns 403.

Run directly to print a coverage table:  python fetch_nascar.py
"""
import json
from pathlib import Path

import requests

BASE = "https://cf.nascar.com/cacher"
LOOP_BASE = "https://cf.nascar.com/loopstats/prod"
CACHE = Path(__file__).parent / "data" / "cache"
CUP = 1
TALLADEGA_TRACK_ID = 82
RACE_FEEDS = ("lap-times", "live-pit-data", "lap-notes")


def get_json(url, cache_path, refresh=False):
    """Return parsed JSON from cache or network; None if the feed doesn't exist."""
    if cache_path.exists() and not refresh:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    resp = requests.get(url, timeout=30)
    if resp.status_code in (403, 404):
        return None
    resp.raise_for_status()
    data = resp.json()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(data), encoding="utf-8")
    return data


def talladega_races(year):
    """Cup races at Talladega in `year`, each with race_id, race_date and season_half."""
    schedule = get_json(f"{BASE}/{year}/race_list_basic.json", CACHE / f"{year}_race_list.json")
    if not schedule:
        return []
    races = []
    for race in schedule.get(f"series_{CUP}", []):
        if race.get("track_id") != TALLADEGA_TRACK_ID and "Talladega" not in race.get("track_name", ""):
            continue
        # The 2017 schedule file actually serves 2018's races.
        if not race["race_date"].startswith(str(year)):
            continue
        month = int(race["race_date"][5:7])
        races.append({
            "season": year,
            "race_id": race["race_id"],
            "race_name": race["race_name"],
            "feed_race_date": race["race_date"][:10],
            "season_half": "spring" if month <= 6 else "fall",
        })
    return races


def fetch_race(year, race_id, feeds=RACE_FEEDS, refresh=False):
    """Dict of feed name -> parsed JSON (None where the feed doesn't exist)."""
    return {
        feed: get_json(f"{BASE}/{year}/{CUP}/{race_id}/{feed}.json",
                       CACHE / f"{year}_{race_id}_{feed}.json", refresh=refresh)
        for feed in feeds
    }


def weekend_feed(year, race_id):
    """Entry list / qualifying / lineup for a race. Always refetched, since it changes all weekend."""
    return get_json(f"{BASE}/{year}/{CUP}/{race_id}/weekend-feed.json",
                    CACHE / f"{year}_{race_id}_weekend-feed.json", refresh=True)


def race_results(year, race_id, refresh=False):
    """The weekend feed's race block (every car's result, stages, qualifying) for a finished race.

    Cached once the race has been run; returns None for a race that hasn't happened yet.
    Available from 2018 on.
    """
    path = CACHE / f"{year}_{race_id}_results.json"
    feed = get_json(f"{BASE}/{year}/{CUP}/{race_id}/weekend-feed.json", path, refresh=refresh)
    race = (feed or {}).get("weekend_race", [None])[0]
    if not race or not any(r.get("finishing_position") for r in race.get("results", [])):
        path.unlink(missing_ok=True)  # don't keep a pre-race copy
        return None
    if "weekend_runs" in feed:
        # Keep only the race block's scalars, results and stages: the full feed (practice,
        # pit reports, schedule) is several times larger and isn't used.
        race = {k: v for k, v in race.items()
                if not isinstance(v, (list, dict)) or k in ("results", "stage_results")}
        path.write_text(json.dumps({"weekend_race": [race]}), encoding="utf-8")
    return race


def loop_stats(year, race_id, refresh=False):
    """Loop data (driver rating, passes, running positions) for every driver, or None.

    Keyed by driver_id, not car number. Available from 2019 on.
    """
    path = CACHE / f"{year}_{race_id}_loopstats.json"
    feed = get_json(f"{LOOP_BASE}/{year}/{CUP}/{race_id}.json", path, refresh=refresh)
    drivers = feed[0].get("drivers", []) if feed else []
    if not drivers:
        path.unlink(missing_ok=True)
        return None
    return drivers


def starting_info(year, race_id, car):
    """Start position and qualifying numbers for `car` once the lineup is posted, else None.

    Before qualifying the feed lists the entry with zeros in these fields. Races
    without time trials (e.g. spring 2026) have a lineup but a pole speed of 0.
    """
    if race_id is None:  # the race isn't on the published schedule yet
        return None
    feed = weekend_feed(year, race_id)
    if not feed or not feed.get("weekend_race"):
        return None
    race = feed["weekend_race"][0]
    entry = next((r for r in race.get("results", []) if str(r.get("car_number")) == str(car)), None)
    if not entry or not entry.get("starting_position"):
        return None
    return {
        "start_position": int(entry["starting_position"]),
        "qualifying_speed": entry.get("qualifying_speed") or None,
        "pole_speed": race.get("pole_winner_speed") or None,
        "field_size": race.get("number_of_cars_in_field") or len(race.get("results", [])),
    }


if __name__ == "__main__":
    print(f"{'season':<7}{'half':<8}{'race_id':<9}" + "".join(f"{f:<15}" for f in RACE_FEEDS))
    for year in range(2014, 2027):
        for race in talladega_races(year):
            feeds = fetch_race(year, race["race_id"])
            flags = "".join(f"{'yes' if feeds[f] else 'no':<15}" for f in RACE_FEEDS)
            print(f"{year:<7}{race['season_half']:<8}{race['race_id']:<9}{flags}")
