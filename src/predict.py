from collections import defaultdict, deque
from pathlib import Path

import joblib
import pandas as pd

from features import (
    FORM_WINDOW,
    INITIAL_ELO,
    calculate_team_form,
    update_elo,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = PROJECT_ROOT / "data" / "processed" / "clean_results.csv"
MODEL_FILE = PROJECT_ROOT / "models" / "random_forest_model.joblib"


def load_model():
    """
    Load the trained Random Forest model and its feature list.
    """

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Trained model not found at:\n{MODEL_FILE}\n\n"
            "Run python src/train_model.py first."
        )

    model_package = joblib.load(MODEL_FILE)

    model = model_package["model"]
    feature_columns = model_package["features"]

    return model, feature_columns


def load_match_data():
    """
    Load all historical match results.
    """

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Clean match data not found at:\n{DATA_FILE}\n\n"
            "Run python src/clean_data.py first."
        )

    data = pd.read_csv(
        DATA_FILE,
        parse_dates=["date"],
    )

    data = data.sort_values("date").reset_index(drop=True)

    return data


def convert_to_boolean(value):
    """
    Safely convert a value from the neutral column into True or False.
    """

    if isinstance(value, bool):
        return value

    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
        "y",
    }









def build_current_team_states(data):
    """
    Process all historical matches to calculate each team's:

    - Current Elo rating
    - Previous five-match history

    This should be called once before running many tournament simulations.
    """

    histories = defaultdict(lambda: deque(maxlen=FORM_WINDOW))
    elo_ratings = defaultdict(lambda: INITIAL_ELO)
    known_teams = set()

    for match in data.itertuples(index=False):

        home_team = match.home_team
        away_team = match.away_team

        home_score = match.home_score
        away_score = match.away_score

        neutral = convert_to_boolean(match.neutral)

        known_teams.add(home_team)
        known_teams.add(away_team)

        home_elo = elo_ratings[home_team]
        away_elo = elo_ratings[away_team]

        if home_score > away_score:
            home_won = 1
            away_won = 0

            home_drawn = 0
            away_drawn = 0

            home_points = 3
            away_points = 0

        elif home_score < away_score:
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
            "goals_scored": home_score,
            "goals_conceded": away_score,
        })

        histories[away_team].append({
            "won": away_won,
            "drawn": away_drawn,
            "points": away_points,
            "goals_scored": away_score,
            "goals_conceded": home_score,
        })

        new_home_elo, new_away_elo = update_elo(
            home_rating=home_elo,
            away_rating=away_elo,
            home_score=home_score,
            away_score=away_score,
            neutral=neutral,
        )

        elo_ratings[home_team] = new_home_elo
        elo_ratings[away_team] = new_away_elo

    return histories, elo_ratings, known_teams






def find_team_name(user_input, known_teams):
    """
    Match a user's team input without requiring exact capitalization.

    Example:
    brazil -> Brazil
    UNITED STATES -> United States
    """

    cleaned_input = user_input.strip().lower()

    team_lookup = {
        team.lower(): team
        for team in known_teams
    }

    return team_lookup.get(cleaned_input)


def ask_for_team(prompt, known_teams):
    """
    Ask the user for a valid team name.
    """

    while True:
        user_input = input(prompt).strip()

        matched_team = find_team_name(
            user_input,
            known_teams,
        )

        if matched_team is not None:
            return matched_team

        print(
            f"\nTeam '{user_input}' was not found in the dataset."
        )
        print(
            "Check the spelling and try again.\n"
        )


def ask_for_neutral():
    """
    Ask whether the match is being played at a neutral venue.
    """

    while True:
        answer = input(
            "Neutral venue? (y/n): "
        ).strip().lower()

        if answer in {"y", "yes"}:
            return True

        if answer in {"n", "no"}:
            return False

        print("Please enter y or n.\n")


def create_prediction_features(
    home_team,
    away_team,
    neutral,
    histories,
    elo_ratings,
):
    """
    Create the exact same features used during model training.
    """

    home_history = histories[home_team]
    away_history = histories[away_team]

    if len(home_history) < FORM_WINDOW:
        raise ValueError(
            f"{home_team} does not have at least "
            f"{FORM_WINDOW} historical matches."
        )

    if len(away_history) < FORM_WINDOW:
        raise ValueError(
            f"{away_team} does not have at least "
            f"{FORM_WINDOW} historical matches."
        )

    home_form = calculate_team_form(home_history)
    away_form = calculate_team_form(away_history)

    home_elo = elo_ratings[home_team]
    away_elo = elo_ratings[away_team]

    feature_values = {
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
    }

    return feature_values


def predict_match(
    model,
    feature_columns,
    feature_values,
):
    """
    Predict the result and probability of each outcome.
    """

    feature_data = pd.DataFrame(
        [feature_values]
    )

    # Ensure columns are in exactly the same order
    # as they were during training.
    feature_data = feature_data[feature_columns]

    probabilities = model.predict_proba(
        feature_data
    )[0]

    class_probabilities = {
        class_label: probability
        for class_label, probability
        in zip(model.classes_, probabilities)
    }

    predicted_class = model.predict(
        feature_data
    )[0]

    return predicted_class, class_probabilities


def display_prediction(
    home_team,
    away_team,
    neutral,
    predicted_class,
    probabilities,
    feature_values,
):
    """
    Display the prediction clearly.
    """

    result_names = {
        0: f"{away_team} win",
        1: "Draw",
        2: f"{home_team} win",
    }

    away_probability = probabilities.get(0, 0)
    draw_probability = probabilities.get(1, 0)
    home_probability = probabilities.get(2, 0)

    print("\n===================================")
    print("          MATCH PREDICTION")
    print("===================================\n")

    print(f"Home team: {home_team}")
    print(f"Away team: {away_team}")
    print(
        f"Venue: {'Neutral' if neutral else 'Home venue'}"
    )

    print("\nCurrent team strength:")

    print(
        f"{home_team} Elo: "
        f"{feature_values['home_elo']:.1f}"
    )

    print(
        f"{away_team} Elo: "
        f"{feature_values['away_elo']:.1f}"
    )

    print("\nOutcome probabilities:\n")

    print(
        f"{home_team} win: "
        f"{home_probability * 100:.2f}%"
    )

    print(
        f"Draw: "
        f"{draw_probability * 100:.2f}%"
    )

    print(
        f"{away_team} win: "
        f"{away_probability * 100:.2f}%"
    )

    print("\nPredicted result:")

    print(
        result_names[predicted_class]
    )

    print("\n===================================")


def main():
    print("\nLoading trained model and match history...")

    model, feature_columns = load_model()

    match_data = load_match_data()

    (
        histories,
        elo_ratings,
        known_teams,
    ) = build_current_team_states(match_data)

    print("Ready.\n")

    home_team = ask_for_team(
        "Enter home team: ",
        known_teams,
    )

    away_team = ask_for_team(
        "Enter away team: ",
        known_teams,
    )

    if home_team == away_team:
        print(
            "\nThe home and away teams cannot be the same."
        )
        return

    neutral = ask_for_neutral()

    feature_values = create_prediction_features(
        home_team=home_team,
        away_team=away_team,
        neutral=neutral,
        histories=histories,
        elo_ratings=elo_ratings,
    )

    predicted_class, probabilities = predict_match(
        model=model,
        feature_columns=feature_columns,
        feature_values=feature_values,
    )

    display_prediction(
        home_team=home_team,
        away_team=away_team,
        neutral=neutral,
        predicted_class=predicted_class,
        probabilities=probabilities,
        feature_values=feature_values,
    )


if __name__ == "__main__":
    main()