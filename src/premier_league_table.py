"""Calculate Premier League standings from completed matches."""

import csv
from pathlib import Path

from premier_league_teams import PREMIER_LEAGUE_2026_27_TEAMS


DATA_FILE = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "premier_league"
    / "processed"
    / "matches.csv"
)


def current_table():
    table = {
        team: {
            "team": team, "played": 0, "won": 0, "drawn": 0,
            "lost": 0, "goals_for": 0, "goals_against": 0,
            "goal_difference": 0, "points": 0,
        }
        for team in PREMIER_LEAGUE_2026_27_TEAMS
    }

    with DATA_FILE.open(newline="", encoding="utf-8") as file:
        matches = csv.DictReader(file)

        for match in matches:
            if match["season"] != "2026-27" or match["status"] != "played":
                continue

            home = table[match["home_team"]]
            away = table[match["away_team"]]
            home_goals = int(match["home_goals"])
            away_goals = int(match["away_goals"])

            home["played"] += 1
            away["played"] += 1
            home["goals_for"] += home_goals
            home["goals_against"] += away_goals
            away["goals_for"] += away_goals
            away["goals_against"] += home_goals

            if home_goals > away_goals:
                home["won"] += 1
                home["points"] += 3
                away["lost"] += 1
            elif home_goals < away_goals:
                away["won"] += 1
                away["points"] += 3
                home["lost"] += 1
            else:
                home["drawn"] += 1
                away["drawn"] += 1
                home["points"] += 1
                away["points"] += 1

    for team in table.values():
        team["goal_difference"] = team["goals_for"] - team["goals_against"]

    return sorted(
        table.values(),
        key=lambda team: (
            -team["points"],
            -team["goal_difference"],
            -team["goals_for"],
            team["team"],
        ),
    )


if __name__ == "__main__":
    standings = current_table()
    print("Pos  Team                         P   W   D   L   GD  Pts")
    for position, team in enumerate(standings, start=1):
        print(
            f"{position:>3}  {team['team']:<27}"
            f"{team['played']:>2}  {team['won']:>2}  "
            f"{team['drawn']:>2}  {team['lost']:>2}  "
            f"{team['goal_difference']:>3}  {team['points']:>3}"
        )
    print("Completed matches:", sum(team["played"] for team in standings) // 2)
