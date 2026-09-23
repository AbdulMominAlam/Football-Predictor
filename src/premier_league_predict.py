from datetime import date, timedelta
from functools import lru_cache
from math import exp, factorial
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

try:
    from .premier_league_features import (
        PremierLeagueFeatureBuilder,
        load_completed_matches,
    )
    from .premier_league_teams import (
        PREMIER_LEAGUE_2026_27_TEAMS,
    )
except ImportError:
    from premier_league_features import (
        PremierLeagueFeatureBuilder,
        load_completed_matches,
    )
    from premier_league_teams import (
        PREMIER_LEAGUE_2026_27_TEAMS,
    )

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "premier_league_model.joblib"
)

MATCHES_FILE = (
    PROJECT_ROOT
    / "data"
    / "premier_league"
    / "processed"
    / "matches.csv"
)

CURRENT_SEASON = "2026-27"
MAX_SCORE = 8


OUTCOME_LABELS = {
    "home_win": "Home Win",
    "draw": "Draw",
    "away_win": "Away Win",
}


@lru_cache(maxsize=1)
def load_model_package():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            "Premier League model not found. Run "
            "src/train_premier_league_model.py first."
        )

    return joblib.load(MODEL_FILE)


@lru_cache(maxsize=1)
def build_current_feature_state():
    """
    Replay all completed matches chronologically so that Elo,
    recent form, season form, home/away form, and rest history
    represent the latest available state.
    """
    matches = load_completed_matches()
    builder = PremierLeagueFeatureBuilder()

    for _, match in matches.iterrows():
        builder.process_match(match)

    return builder


@lru_cache(maxsize=1)
def load_fixtures():
    fixtures = pd.read_csv(MATCHES_FILE)

    fixtures["date"] = pd.to_datetime(
        fixtures["date"],
        errors="raise",
    )

    return fixtures


def validate_teams(home_team, away_team):
    valid_teams = set(
        PREMIER_LEAGUE_2026_27_TEAMS
    )

    if home_team not in valid_teams:
        raise ValueError(
            f"Unknown home team: {home_team}"
        )

    if away_team not in valid_teams:
        raise ValueError(
            f"Unknown away team: {away_team}"
        )

    if home_team == away_team:
        raise ValueError(
            "A team cannot play against itself."
        )


def find_fixture_date(home_team, away_team):
    fixtures = load_fixtures()

    matching_fixture = fixtures[
        (fixtures["season"] == CURRENT_SEASON)
        & (fixtures["home_team"] == home_team)
        & (fixtures["away_team"] == away_team)
        & (fixtures["status"] == "scheduled")
    ]

    if not matching_fixture.empty:
        return pd.Timestamp(
            matching_fixture.iloc[0]["date"]
        )

    completed = fixtures[
        fixtures["status"] == "played"
    ]

    latest_result_date = pd.Timestamp(
        completed["date"].max()
    )

    fallback_date = max(
        date.today(),
        (
            latest_result_date
            + timedelta(days=1)
        ).date(),
    )

    return pd.Timestamp(fallback_date)


def create_match_feature_row(
    home_team,
    away_team,
    match_date,
):
    builder = build_current_feature_state()

    builder.start_season(CURRENT_SEASON)

    features = builder.create_pre_match_features(
        home_team,
        away_team,
        match_date,
    )

    return {
        "season": CURRENT_SEASON,
        "date": match_date,
        "home_team": home_team,
        "away_team": away_team,
        **features,
    }


def align_classifier_probabilities(
    classifier,
    probabilities,
    class_order,
):
    positions = {
        class_name: position
        for position, class_name
        in enumerate(classifier.classes_)
    }

    return np.array(
        [
            probabilities[
                positions[class_name]
            ]
            for class_name in class_order
        ]
    )


def create_poisson_rows(feature_row):
    home_row = pd.DataFrame(
        [
            {
                "team": feature_row["home_team"],
                "opponent": feature_row["away_team"],
                "is_home": 1,
                "team_elo": feature_row["home_elo"],
                "opponent_elo": feature_row["away_elo"],
                "team_recent_goals_scored": (
                    feature_row["home_goals_scored"]
                ),
                "team_recent_goals_conceded": (
                    feature_row[
                        "home_goals_conceded"
                    ]
                ),
                "opponent_recent_goals_scored": (
                    feature_row["away_goals_scored"]
                ),
                "opponent_recent_goals_conceded": (
                    feature_row[
                        "away_goals_conceded"
                    ]
                ),
            }
        ]
    )

    away_row = pd.DataFrame(
        [
            {
                "team": feature_row["away_team"],
                "opponent": feature_row["home_team"],
                "is_home": 0,
                "team_elo": feature_row["away_elo"],
                "opponent_elo": feature_row["home_elo"],
                "team_recent_goals_scored": (
                    feature_row["away_goals_scored"]
                ),
                "team_recent_goals_conceded": (
                    feature_row[
                        "away_goals_conceded"
                    ]
                ),
                "opponent_recent_goals_scored": (
                    feature_row["home_goals_scored"]
                ),
                "opponent_recent_goals_conceded": (
                    feature_row[
                        "home_goals_conceded"
                    ]
                ),
            }
        ]
    )

    return home_row, away_row


def poisson_probability(goals, expected_goals):
    return (
        exp(-expected_goals)
        * expected_goals**goals
        / factorial(goals)
    )


def create_score_matrix(
    expected_home_goals,
    expected_away_goals,
):
    home_probabilities = np.array(
        [
            poisson_probability(
                goals,
                expected_home_goals,
            )
            for goals in range(MAX_SCORE + 1)
        ]
    )

    away_probabilities = np.array(
        [
            poisson_probability(
                goals,
                expected_away_goals,
            )
            for goals in range(MAX_SCORE + 1)
        ]
    )

    score_matrix = np.outer(
        home_probabilities,
        away_probabilities,
    )

    return score_matrix / score_matrix.sum()


def most_likely_conditional_score(
    score_matrix,
    outcome,
):
    valid_scores = []

    for home_goals in range(MAX_SCORE + 1):
        for away_goals in range(MAX_SCORE + 1):
            if (
                outcome == "home_win"
                and home_goals > away_goals
            ):
                valid = True
            elif (
                outcome == "away_win"
                and away_goals > home_goals
            ):
                valid = True
            elif (
                outcome == "draw"
                and home_goals == away_goals
            ):
                valid = True
            else:
                valid = False

            if valid:
                valid_scores.append(
                    (
                        score_matrix[
                            home_goals,
                            away_goals,
                        ],
                        home_goals,
                        away_goals,
                    )
                )

    _, home_goals, away_goals = max(
        valid_scores
    )

    return home_goals, away_goals


def predict_match(
    home_team,
    away_team,
    match_date=None,
):
    validate_teams(home_team, away_team)

    if match_date is None:
        match_date = find_fixture_date(
            home_team,
            away_team,
        )
    else:
        match_date = pd.Timestamp(match_date)

    feature_row = create_match_feature_row(
        home_team,
        away_team,
        match_date,
    )

    package = load_model_package()

    classifier = package["outcome_classifier"]
    poisson_model = package["poisson_model"]
    classifier_features = package[
        "classifier_features"
    ]
    class_order = package["class_order"]

    classifier_input = pd.DataFrame(
        [feature_row]
    )[classifier_features]

    raw_probabilities = (
        classifier.predict_proba(
            classifier_input
        )[0]
    )

    probabilities = (
        align_classifier_probabilities(
            classifier,
            raw_probabilities,
            class_order,
        )
    )

    probability_map = {
        class_name: float(probability)
        for class_name, probability
        in zip(class_order, probabilities)
    }

    most_likely_outcome = max(
        probability_map,
        key=probability_map.get,
    )

    home_poisson_row, away_poisson_row = (
        create_poisson_rows(feature_row)
    )

    expected_home_goals = float(
        poisson_model.predict(
            home_poisson_row
        )[0]
    )

    expected_away_goals = float(
        poisson_model.predict(
            away_poisson_row
        )[0]
    )

    expected_home_goals = float(
        np.clip(
            expected_home_goals,
            0.05,
            5.0,
        )
    )

    expected_away_goals = float(
        np.clip(
            expected_away_goals,
            0.05,
            5.0,
        )
    )

    score_matrix = create_score_matrix(
        expected_home_goals,
        expected_away_goals,
    )

    predicted_home_goals, predicted_away_goals = (
        most_likely_conditional_score(
            score_matrix,
            most_likely_outcome,
        )
    )

    return {
        "season": CURRENT_SEASON,
        "match_date": match_date.date().isoformat(),
        "home_team": home_team,
        "away_team": away_team,
        "home_win_probability": probability_map[
            "home_win"
        ],
        "draw_probability": probability_map[
            "draw"
        ],
        "away_win_probability": probability_map[
            "away_win"
        ],
        "most_likely_outcome": (
            most_likely_outcome
        ),
        "most_likely_outcome_label": (
            OUTCOME_LABELS[
                most_likely_outcome
            ]
        ),
        "expected_home_goals": (
            expected_home_goals
        ),
        "expected_away_goals": (
            expected_away_goals
        ),
        "predicted_home_goals": (
            predicted_home_goals
        ),
        "predicted_away_goals": (
            predicted_away_goals
        ),
        "predicted_score": (
            f"{predicted_home_goals}-"
            f"{predicted_away_goals}"
        ),
    }


def main():
    prediction = predict_match(
        "Arsenal",
        "Manchester City",
    )

    print(
        f"{prediction['home_team']} vs "
        f"{prediction['away_team']}"
    )
    print(
        f"Fixture date: "
        f"{prediction['match_date']}"
    )
    print(
        "Home win: "
        f"{prediction['home_win_probability']:.2%}"
    )
    print(
        "Draw: "
        f"{prediction['draw_probability']:.2%}"
    )
    print(
        "Away win: "
        f"{prediction['away_win_probability']:.2%}"
    )
    print(
        "Most likely outcome: "
        f"{prediction['most_likely_outcome_label']}"
    )
    print(
        "Expected goals: "
        f"{prediction['expected_home_goals']:.2f} - "
        f"{prediction['expected_away_goals']:.2f}"
    )
    print(
        "Predicted score: "
        f"{prediction['predicted_score']}"
    )


if __name__ == "__main__":
    main()