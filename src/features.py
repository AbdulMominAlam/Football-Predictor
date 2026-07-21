from collections import defaultdict, deque
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "clean_results.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "model_data.csv"

FORM_WINDOW = 5

INITIAL_ELO = 1500
ELO_K_FACTOR = 30
HOME_ADVANTAGE = 100


def load_data():
    """
    Load the cleaned football match dataset.
    """

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found at: {INPUT_FILE}"
        )

    data = pd.read_csv(INPUT_FILE, parse_dates=["date"])

    data = data.sort_values("date").reset_index(drop=True)

    return data


def create_target(data):
    """
    Create the target column.

    0 = Away win
    1 = Draw
    2 = Home win
    """

    data = data.copy()

    data["result"] = 1

    data.loc[
        data["home_score"] > data["away_score"],
        "result"
    ] = 2

    data.loc[
        data["home_score"] < data["away_score"],
        "result"
    ] = 0

    return data


def calculate_team_form(team_history):
    """
    Calculate recent statistics from a team's previous five matches.
    """

    matches = list(team_history)

    number_of_matches = len(matches)

    wins = sum(match["won"] for match in matches)
    draws = sum(match["drawn"] for match in matches)

    goals_scored = sum(
        match["goals_scored"] for match in matches
    )

    goals_conceded = sum(
        match["goals_conceded"] for match in matches
    )

    points = sum(
        match["points"] for match in matches
    )

    return {
        "win_rate": wins / number_of_matches,
        "draw_rate": draws / number_of_matches,
        "average_goals_scored": goals_scored / number_of_matches,
        "average_goals_conceded": goals_conceded / number_of_matches,
        "points_per_match": points / number_of_matches,
        "goal_difference_per_match":
            (goals_scored - goals_conceded) / number_of_matches,
    }


def expected_score(rating_a, rating_b):
    """
    Calculate the expected Elo score for team A.
    """

    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


def update_elo(
    home_rating,
    away_rating,
    home_score,
    away_score,
    neutral,
):
    """
    Update both teams' Elo ratings after a match.
    """

    adjusted_home_rating = home_rating

    if not neutral:
        adjusted_home_rating += HOME_ADVANTAGE

    expected_home = expected_score(
        adjusted_home_rating,
        away_rating,
    )

    expected_away = 1 - expected_home

    if home_score > away_score:
        actual_home = 1.0
        actual_away = 0.0

    elif home_score < away_score:
        actual_home = 0.0
        actual_away = 1.0

    else:
        actual_home = 0.5
        actual_away = 0.5

    new_home_rating = (
        home_rating
        + ELO_K_FACTOR * (actual_home - expected_home)
    )

    new_away_rating = (
        away_rating
        + ELO_K_FACTOR * (actual_away - expected_away)
    )

    return new_home_rating, new_away_rating


def create_features(data):
    """
    Create pre-match form and Elo features.

    Current match results are added only after the feature row
    has been created, preventing data leakage.
    """

    histories = defaultdict(
        lambda: deque(maxlen=FORM_WINDOW)
    )

    elo_ratings = defaultdict(
        lambda: INITIAL_ELO
    )

    feature_rows = []

    for _, match in data.iterrows():

        home_team = match["home_team"]
        away_team = match["away_team"]

        home_history = histories[home_team]
        away_history = histories[away_team]

        home_elo = elo_ratings[home_team]
        away_elo = elo_ratings[away_team]

        neutral = bool(match["neutral"])

        if (
            len(home_history) == FORM_WINDOW
            and len(away_history) == FORM_WINDOW
        ):
            home_form = calculate_team_form(home_history)
            away_form = calculate_team_form(away_history)

            feature_rows.append({
                "date": match["date"],
                "home_team": home_team,
                "away_team": away_team,
                "neutral": int(neutral),

                "home_elo": home_elo,
                "away_elo": away_elo,
                "elo_difference": home_elo - away_elo,

                "home_win_rate_last_5":
                    home_form["win_rate"],

                "away_win_rate_last_5":
                    away_form["win_rate"],

                "home_draw_rate_last_5":
                    home_form["draw_rate"],

                "away_draw_rate_last_5":
                    away_form["draw_rate"],

                "home_avg_goals_scored_last_5":
                    home_form["average_goals_scored"],

                "away_avg_goals_scored_last_5":
                    away_form["average_goals_scored"],

                "home_avg_goals_conceded_last_5":
                    home_form["average_goals_conceded"],

                "away_avg_goals_conceded_last_5":
                    away_form["average_goals_conceded"],

                "home_points_per_match_last_5":
                    home_form["points_per_match"],

                "away_points_per_match_last_5":
                    away_form["points_per_match"],

                "home_goal_difference_last_5":
                    home_form["goal_difference_per_match"],

                "away_goal_difference_last_5":
                    away_form["goal_difference_per_match"],

                "win_rate_difference":
                    home_form["win_rate"]
                    - away_form["win_rate"],

                "points_difference":
                    home_form["points_per_match"]
                    - away_form["points_per_match"],

                "scoring_difference":
                    home_form["average_goals_scored"]
                    - away_form["average_goals_scored"],

                "recent_goal_difference_difference":
                    home_form["goal_difference_per_match"]
                    - away_form["goal_difference_per_match"],

                "result": match["result"],
            })

        if match["home_score"] > match["away_score"]:
            home_won = 1
            away_won = 0

            home_drawn = 0
            away_drawn = 0

            home_points = 3
            away_points = 0

        elif match["home_score"] < match["away_score"]:
            home_won = 0
            away_won = 1

            home_drawn = 0
            away_drawn = 0

            home_points = 0
            away_points = 3

        else:
            home_won = 0
            away_won = 0

            home_drawn = 1
            away_drawn = 1

            home_points = 1
            away_points = 1

        histories[home_team].append({
            "won": home_won,
            "drawn": home_drawn,
            "points": home_points,
            "goals_scored": match["home_score"],
            "goals_conceded": match["away_score"],
        })

        histories[away_team].append({
            "won": away_won,
            "drawn": away_drawn,
            "points": away_points,
            "goals_scored": match["away_score"],
            "goals_conceded": match["home_score"],
        })

        new_home_elo, new_away_elo = update_elo(
            home_rating=home_elo,
            away_rating=away_elo,
            home_score=match["home_score"],
            away_score=match["away_score"],
            neutral=neutral,
        )

        elo_ratings[home_team] = new_home_elo
        elo_ratings[away_team] = new_away_elo

    return pd.DataFrame(feature_rows)


def inspect_features(data):
    """
    Display information about the feature dataset.
    """

    print("\n========== FIRST 5 ROWS ==========\n")
    print(data.head())

    print("\n========== DATASET SHAPE ==========\n")
    print(data.shape)

    print("\n========== MISSING VALUES ==========\n")
    print(data.isnull().sum())

    print("\n========== RESULT COUNTS ==========\n")
    print(data["result"].value_counts().sort_index())

    print("\n========== ELO SUMMARY ==========\n")
    print(
        data[
            [
                "home_elo",
                "away_elo",
                "elo_difference",
            ]
        ].describe()
    )

    print("\nResult meanings:")
    print("0 = Away win")
    print("1 = Draw")
    print("2 = Home win")


def save_features(data):
    """
    Save the model-ready dataset.
    """

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(f"\nModel dataset saved to:\n{OUTPUT_FILE}")


def main():
    football_data = load_data()

    football_data = create_target(football_data)

    model_data = create_features(football_data)

    inspect_features(model_data)

    save_features(model_data)


if __name__ == "__main__":
    main()