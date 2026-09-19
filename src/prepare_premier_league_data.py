"""Convert OpenFootball Premier League JSON into one consistent CSV."""

import csv
import json
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "premier_league" / "raw"
OUTPUT = ROOT / "data" / "premier_league" / "processed" / "matches.csv"

SEASONS = [
    "2020-21",
    "2021-22",
    "2022-23",
    "2023-24",
    "2024-25",
    "2025-26",
    "2026-27",
]

COLUMNS = [
    "season",
    "date",
    "round",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
    "status",
]


def full_time_score(match):
    score = match.get("score")

    if isinstance(score, dict):
        result = score.get("ft")
    elif isinstance(score, list):
        result = score
    elif score is None:
        return None
    else:
        raise ValueError(f"Unexpected score format: {score!r}")

    if result is None:
        return None

    if (
        not isinstance(result, list)
        or len(result) != 2
        or any(type(goals) is not int or goals < 0 for goals in result)
    ):
        raise ValueError(f"Invalid full-time score: {result!r}")

    return result


def main():
    rows = []
    seen_fixtures = set()

    for season in SEASONS:
        path = RAW_DIR / f"{season}.json"
        matches = json.loads(path.read_text(encoding="utf-8"))["matches"]

        if len(matches) != 380:
            raise ValueError(f"{season}: expected 380 fixtures, got {len(matches)}")

        for match in matches:
            match_date = date.fromisoformat(match["date"])
            home_team = match["team1"]
            away_team = match["team2"]
            fixture_key = (season, home_team, away_team)

            if home_team == away_team or fixture_key in seen_fixtures:
                raise ValueError(f"Invalid or duplicate fixture: {fixture_key}")
            seen_fixtures.add(fixture_key)

            result = full_time_score(match)

            if result is not None and match_date > date.today():
                raise ValueError(f"Future match already has a result: {fixture_key}")

            rows.append({
                "season": season,
                "date": match["date"],
                "round": match["round"],
                "home_team": home_team,
                "away_team": away_team,
                "home_goals": result[0] if result else "",
                "away_goals": result[1] if result else "",
                "status": "played" if result else "scheduled",
            })

    rows.sort(key=lambda row: (row["date"], row["home_team"], row["away_team"]))

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} matches to {OUTPUT}")
    for season in SEASONS:
        season_rows = [row for row in rows if row["season"] == season]
        played = sum(row["status"] == "played" for row in season_rows)
        print(f"{season}: {played} played, {len(season_rows) - played} scheduled")


if __name__ == "__main__":
    main()
