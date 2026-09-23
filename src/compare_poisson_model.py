from math import exp, factorial
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import (
    LogisticRegression,
    PoissonRegressor,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "premier_league"
    / "processed"
    / "model_features.csv"
)

RESULTS_FILE = (
    PROJECT_ROOT
    / "data"
    / "premier_league"
    / "processed"
    / "poisson_comparison_results.csv"
)

CLASS_ORDER = [
    "away_win",
    "draw",
    "home_win",
]

SEASON_ORDER = [
    "2015-16",
    "2016-17",
    "2017-18",
    "2018-19",
    "2019-20",
    "2020-21",
    "2021-22",
    "2022-23",
    "2023-24",
    "2024-25",
    "2025-26",
    "2026-27",
]

VALIDATION_SEASONS = [
    "2022-23",
    "2023-24",
    "2024-25",
]

TEST_SEASON = "2025-26"

POISSON_ALPHAS = [
    0.05,
    0.25,
    0.50,
    1.00,
    2.00,
]

POISSON_WEIGHTS = [
    1.00,
    0.75,
    0.50,
    0.25,
]

MAX_GOALS = 10


CLASSIFIER_FEATURES = [
    "home_elo",
    "away_elo",
    "elo_difference",
    "home_win_rate",
    "away_win_rate",
    "win_rate_difference",
    "home_draw_rate",
    "away_draw_rate",
    "draw_rate_difference",
    "home_points_per_match",
    "away_points_per_match",
    "points_per_match_difference",
    "home_goals_scored",
    "away_goals_scored",
    "goals_scored_difference",
    "home_goals_conceded",
    "away_goals_conceded",
    "goals_conceded_difference",
    "home_goal_difference",
    "away_goal_difference",
    "recent_goal_difference_difference",
]

POISSON_CATEGORICAL_FEATURES = [
    "team",
    "opponent",
]

POISSON_NUMERIC_FEATURES = [
    "is_home",
    "team_elo",
    "opponent_elo",
    "team_recent_goals_scored",
    "team_recent_goals_conceded",
    "opponent_recent_goals_scored",
    "opponent_recent_goals_conceded",
]


def load_data():
    data = pd.read_csv(DATA_FILE)

    data["date"] = pd.to_datetime(
        data["date"],
        errors="raise",
    )

    return data.sort_values(
        "date"
    ).reset_index(drop=True)


def seasons_before(season):
    position = SEASON_ORDER.index(season)
    return SEASON_ORDER[:position]


def create_classifier():
    return Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                LogisticRegression(
                    C=0.05,
                    max_iter=4000,
                    random_state=42,
                ),
            ),
        ]
    )


def create_poisson_model(alpha):
    preprocessing = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                ),
                POISSON_CATEGORICAL_FEATURES,
            ),
            (
                "numeric",
                StandardScaler(),
                POISSON_NUMERIC_FEATURES,
            ),
        ]
    )

    return Pipeline(
        steps=[
            (
                "preprocessing",
                preprocessing,
            ),
            (
                "model",
                PoissonRegressor(
                    alpha=alpha,
                    max_iter=2000,
                ),
            ),
        ]
    )


def create_goal_training_rows(matches):
    home_rows = pd.DataFrame(
        {
            "team": matches["home_team"],
            "opponent": matches["away_team"],
            "is_home": 1,
            "team_elo": matches["home_elo"],
            "opponent_elo": matches["away_elo"],
            "team_recent_goals_scored": (
                matches["home_goals_scored"]
            ),
            "team_recent_goals_conceded": (
                matches["home_goals_conceded"]
            ),
            "opponent_recent_goals_scored": (
                matches["away_goals_scored"]
            ),
            "opponent_recent_goals_conceded": (
                matches["away_goals_conceded"]
            ),
            "goals": matches["home_goals"],
        }
    )

    away_rows = pd.DataFrame(
        {
            "team": matches["away_team"],
            "opponent": matches["home_team"],
            "is_home": 0,
            "team_elo": matches["away_elo"],
            "opponent_elo": matches["home_elo"],
            "team_recent_goals_scored": (
                matches["away_goals_scored"]
            ),
            "team_recent_goals_conceded": (
                matches["away_goals_conceded"]
            ),
            "opponent_recent_goals_scored": (
                matches["home_goals_scored"]
            ),
            "opponent_recent_goals_conceded": (
                matches["home_goals_conceded"]
            ),
            "goals": matches["away_goals"],
        }
    )

    return pd.concat(
        [
            home_rows,
            away_rows,
        ],
        ignore_index=True,
    )


def create_goal_prediction_rows(matches, home_team):
    if home_team:
        return pd.DataFrame(
            {
                "team": matches["home_team"],
                "opponent": matches["away_team"],
                "is_home": 1,
                "team_elo": matches["home_elo"],
                "opponent_elo": matches["away_elo"],
                "team_recent_goals_scored": (
                    matches["home_goals_scored"]
                ),
                "team_recent_goals_conceded": (
                    matches["home_goals_conceded"]
                ),
                "opponent_recent_goals_scored": (
                    matches["away_goals_scored"]
                ),
                "opponent_recent_goals_conceded": (
                    matches["away_goals_conceded"]
                ),
            }
        )

    return pd.DataFrame(
        {
            "team": matches["away_team"],
            "opponent": matches["home_team"],
            "is_home": 0,
            "team_elo": matches["away_elo"],
            "opponent_elo": matches["home_elo"],
            "team_recent_goals_scored": (
                matches["away_goals_scored"]
            ),
            "team_recent_goals_conceded": (
                matches["away_goals_conceded"]
            ),
            "opponent_recent_goals_scored": (
                matches["home_goals_scored"]
            ),
            "opponent_recent_goals_conceded": (
                matches["home_goals_conceded"]
            ),
        }
    )


def poisson_probability(goals, expected_goals):
    return (
        exp(-expected_goals)
        * expected_goals**goals
        / factorial(goals)
    )


def outcome_probabilities(
    home_expected_goals,
    away_expected_goals,
):
    probabilities = []

    for home_lambda, away_lambda in zip(
        home_expected_goals,
        away_expected_goals,
    ):
        home_goal_probabilities = np.array(
            [
                poisson_probability(
                    goals,
                    home_lambda,
                )
                for goals in range(MAX_GOALS + 1)
            ]
        )

        away_goal_probabilities = np.array(
            [
                poisson_probability(
                    goals,
                    away_lambda,
                )
                for goals in range(MAX_GOALS + 1)
            ]
        )

        score_matrix = np.outer(
            home_goal_probabilities,
            away_goal_probabilities,
        )

        total_probability = score_matrix.sum()

        if total_probability > 0:
            score_matrix = (
                score_matrix / total_probability
            )

        home_win = np.tril(
            score_matrix,
            k=-1,
        ).sum()

        draw = np.trace(score_matrix)

        away_win = np.triu(
            score_matrix,
            k=1,
        ).sum()

        probabilities.append(
            [
                away_win,
                draw,
                home_win,
            ]
        )

    return np.array(probabilities)


def predict_poisson_probabilities(
    model,
    matches,
):
    home_rows = create_goal_prediction_rows(
        matches,
        home_team=True,
    )

    away_rows = create_goal_prediction_rows(
        matches,
        home_team=False,
    )

    home_expected_goals = model.predict(home_rows)
    away_expected_goals = model.predict(away_rows)

    home_expected_goals = np.clip(
        home_expected_goals,
        0.05,
        5.0,
    )

    away_expected_goals = np.clip(
        away_expected_goals,
        0.05,
        5.0,
    )

    probabilities = outcome_probabilities(
        home_expected_goals,
        away_expected_goals,
    )

    return (
        probabilities,
        home_expected_goals,
        away_expected_goals,
    )


def align_classifier_probabilities(
    model,
    probabilities,
):
    positions = {
        class_name: position
        for position, class_name
        in enumerate(model.classes_)
    }

    return np.column_stack(
        [
            probabilities[
                :,
                positions[class_name],
            ]
            for class_name in CLASS_ORDER
        ]
    )


def calculate_metrics(
    actual,
    probabilities,
):
    predicted_indexes = probabilities.argmax(axis=1)

    predictions = [
        CLASS_ORDER[index]
        for index in predicted_indexes
    ]

    return {
        "accuracy": accuracy_score(
            actual,
            predictions,
        ),
        "macro_f1": f1_score(
            actual,
            predictions,
            labels=CLASS_ORDER,
            average="macro",
            zero_division=0,
        ),
        "log_loss": log_loss(
            actual,
            probabilities,
            labels=CLASS_ORDER,
        ),
        "predictions": predictions,
    }


def run_validation(data):
    results = []

    for validation_season in VALIDATION_SEASONS:
        training_seasons = seasons_before(
            validation_season
        )

        train_data = data[
            data["season"].isin(
                training_seasons
            )
        ].copy()

        validation_data = data[
            data["season"]
            == validation_season
        ].copy()

        classifier = create_classifier()

        classifier.fit(
            train_data[CLASSIFIER_FEATURES],
            train_data["target"],
        )

        classifier_probabilities = (
            classifier.predict_proba(
                validation_data[
                    CLASSIFIER_FEATURES
                ]
            )
        )

        classifier_probabilities = (
            align_classifier_probabilities(
                classifier,
                classifier_probabilities,
            )
        )

        goal_training_data = (
            create_goal_training_rows(
                train_data
            )
        )

        for alpha in POISSON_ALPHAS:
            poisson_model = create_poisson_model(
                alpha
            )

            poisson_model.fit(
                goal_training_data[
                    POISSON_CATEGORICAL_FEATURES
                    + POISSON_NUMERIC_FEATURES
                ],
                goal_training_data["goals"],
            )

            poisson_probabilities, _, _ = (
                predict_poisson_probabilities(
                    poisson_model,
                    validation_data,
                )
            )

            for poisson_weight in POISSON_WEIGHTS:
                blended_probabilities = (
                    poisson_weight
                    * poisson_probabilities
                    + (1 - poisson_weight)
                    * classifier_probabilities
                )

                metrics = calculate_metrics(
                    validation_data["target"],
                    blended_probabilities,
                )

                results.append(
                    {
                        "validation_season": (
                            validation_season
                        ),
                        "alpha": alpha,
                        "poisson_weight": (
                            poisson_weight
                        ),
                        "accuracy": (
                            metrics["accuracy"]
                        ),
                        "macro_f1": (
                            metrics["macro_f1"]
                        ),
                        "log_loss": (
                            metrics["log_loss"]
                        ),
                    }
                )

    return pd.DataFrame(results)


def summarize_results(results):
    summary = (
        results.groupby(
            [
                "alpha",
                "poisson_weight",
            ]
        )
        .agg(
            mean_accuracy=("accuracy", "mean"),
            mean_macro_f1=("macro_f1", "mean"),
            mean_log_loss=("log_loss", "mean"),
        )
        .reset_index()
        .sort_values(
            [
                "mean_log_loss",
                "mean_accuracy",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .reset_index(drop=True)
    )

    return summary


def evaluate_test(data, selected):
    training_seasons = seasons_before(TEST_SEASON)

    train_data = data[
        data["season"].isin(
            training_seasons
        )
    ].copy()

    test_data = data[
        data["season"] == TEST_SEASON
    ].copy()

    classifier = create_classifier()

    classifier.fit(
        train_data[CLASSIFIER_FEATURES],
        train_data["target"],
    )

    classifier_probabilities = (
        classifier.predict_proba(
            test_data[CLASSIFIER_FEATURES]
        )
    )

    classifier_probabilities = (
        align_classifier_probabilities(
            classifier,
            classifier_probabilities,
        )
    )

    goal_training_data = create_goal_training_rows(
        train_data
    )

    poisson_model = create_poisson_model(
        selected["alpha"]
    )

    poisson_model.fit(
        goal_training_data[
            POISSON_CATEGORICAL_FEATURES
            + POISSON_NUMERIC_FEATURES
        ],
        goal_training_data["goals"],
    )

    (
        poisson_probabilities,
        home_expected_goals,
        away_expected_goals,
    ) = predict_poisson_probabilities(
        poisson_model,
        test_data,
    )

    poisson_weight = selected["poisson_weight"]

    final_probabilities = (
        poisson_weight * poisson_probabilities
        + (1 - poisson_weight)
        * classifier_probabilities
    )

    metrics = calculate_metrics(
        test_data["target"],
        final_probabilities,
    )

    actual_counts = (
        test_data["target"]
        .value_counts()
        .reindex(
            CLASS_ORDER,
            fill_value=0,
        )
    )

    predicted_counts = (
        pd.Series(metrics["predictions"])
        .value_counts()
        .reindex(
            CLASS_ORDER,
            fill_value=0,
        )
    )

    print("\nFinal 2025-26 test")
    print("------------------")
    print(f"Poisson alpha: {selected['alpha']}")
    print(
        "Poisson probability weight: "
        f"{poisson_weight:.2f}"
    )
    print(
        "Classifier probability weight: "
        f"{1 - poisson_weight:.2f}"
    )
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Log Loss: {metrics['log_loss']:.4f}")

    print("\nOutcome counts:")
    print(
        pd.DataFrame(
            {
                "actual": actual_counts,
                "predicted": predicted_counts,
            }
        ).to_string()
    )

    print("\nExpected-goal averages:")
    print(
        "Predicted home goals: "
        f"{home_expected_goals.mean():.3f}"
    )
    print(
        "Actual home goals: "
        f"{test_data['home_goals'].mean():.3f}"
    )
    print(
        "Predicted away goals: "
        f"{away_expected_goals.mean():.3f}"
    )
    print(
        "Actual away goals: "
        f"{test_data['away_goals'].mean():.3f}"
    )


def main():
    data = load_data()

    print(f"Loaded {len(data)} completed matches.")
    print(
        "Testing Poisson models and blended "
        "probabilities..."
    )

    results = run_validation(data)

    summary = summarize_results(results)

    RESULTS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        RESULTS_FILE,
        index=False,
    )

    print("\nTop 10 validation configurations")
    print("--------------------------------")
    print(
        summary.head(10).to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    selected = summary.iloc[0]

    print("\nSelected configuration")
    print("----------------------")
    print(f"Poisson alpha: {selected['alpha']}")
    print(
        "Poisson weight: "
        f"{selected['poisson_weight']:.2f}"
    )
    print(
        "Validation accuracy: "
        f"{selected['mean_accuracy']:.4f}"
    )
    print(
        "Validation macro F1: "
        f"{selected['mean_macro_f1']:.4f}"
    )
    print(
        "Validation log loss: "
        f"{selected['mean_log_loss']:.4f}"
    )

    evaluate_test(
        data,
        selected,
    )

    print(
        f"\nSaved results to: {RESULTS_FILE}"
    )


if __name__ == "__main__":
    main()