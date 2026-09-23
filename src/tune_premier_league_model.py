from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from premier_league_features import FEATURE_COLUMNS


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_DATA_FILE = (
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
    / "model_tuning_results.csv"
)

TARGET_COLUMN = "target"

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


ORIGINAL_FEATURES = [
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

TEN_MATCH_FEATURES = [
    "home_10_match_win_rate",
    "away_10_match_win_rate",
    "ten_match_win_rate_difference",
    "home_10_match_points_per_match",
    "away_10_match_points_per_match",
    "ten_match_points_difference",
    "home_10_match_goal_difference",
    "away_10_match_goal_difference",
    "ten_match_goal_difference_difference",
]

SEASON_FEATURES = [
    "home_season_matches",
    "away_season_matches",
    "season_matches_difference",
    "home_season_points_per_match",
    "away_season_points_per_match",
    "season_points_difference",
    "home_season_goal_difference",
    "away_season_goal_difference",
    "season_goal_difference_difference",
    "home_home_points_per_match",
    "away_away_points_per_match",
    "venue_points_difference",
    "home_home_goal_difference",
    "away_away_goal_difference",
    "venue_goal_difference_difference",
    "home_rest_days",
    "away_rest_days",
    "rest_days_difference",
]

COMPACT_FEATURES = [
    "elo_difference",
    "elo_expected_home",
    "win_rate_difference",
    "draw_rate_difference",
    "points_per_match_difference",
    "goals_scored_difference",
    "goals_conceded_difference",
    "recent_goal_difference_difference",
    "ten_match_win_rate_difference",
    "ten_match_points_difference",
    "ten_match_goal_difference_difference",
    "season_matches_difference",
    "season_points_difference",
    "season_goal_difference_difference",
    "venue_points_difference",
    "venue_goal_difference_difference",
    "rest_days_difference",
]

FEATURE_SETS = {
    "original_21": ORIGINAL_FEATURES,
    "original_plus_10_match": (
        ORIGINAL_FEATURES
        + ["elo_expected_home"]
        + TEN_MATCH_FEATURES
    ),
    "original_plus_season": (
        ORIGINAL_FEATURES
        + ["elo_expected_home"]
        + SEASON_FEATURES
    ),
    "compact_differences": COMPACT_FEATURES,
    "all_49": FEATURE_COLUMNS,
}


def logistic_model(c_value):
    return Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                LogisticRegression(
                    C=c_value,
                    max_iter=4000,
                    random_state=42,
                ),
            ),
        ]
    )


def random_forest_model():
    return RandomForestClassifier(
        n_estimators=400,
        max_depth=9,
        min_samples_split=18,
        min_samples_leaf=8,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )


def extra_trees_model():
    return ExtraTreesClassifier(
        n_estimators=400,
        max_depth=12,
        min_samples_split=15,
        min_samples_leaf=8,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )


def ensemble_model():
    return VotingClassifier(
        estimators=[
            (
                "logistic",
                logistic_model(0.25),
            ),
            (
                "random_forest",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=9,
                    min_samples_split=18,
                    min_samples_leaf=8,
                    max_features="sqrt",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
            (
                "extra_trees",
                ExtraTreesClassifier(
                    n_estimators=300,
                    max_depth=12,
                    min_samples_split=15,
                    min_samples_leaf=8,
                    max_features="sqrt",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ],
        voting="soft",
        weights=[2, 1, 1],
        n_jobs=-1,
    )


def create_models():
    return {
        "logistic_c_0.05": logistic_model(0.05),
        "logistic_c_0.25": logistic_model(0.25),
        "logistic_c_1.00": logistic_model(1.0),
        "random_forest": random_forest_model(),
        "extra_trees": extra_trees_model(),
        "soft_voting_ensemble": ensemble_model(),
    }


def load_data():
    data = pd.read_csv(FEATURE_DATA_FILE)

    data["date"] = pd.to_datetime(
        data["date"],
        errors="raise",
    )

    missing_features = (
        set(FEATURE_COLUMNS) - set(data.columns)
    )

    if missing_features:
        raise ValueError(
            "Missing model features: "
            + ", ".join(sorted(missing_features))
        )

    return data.sort_values(
        "date"
    ).reset_index(drop=True)


def seasons_before(season):
    position = SEASON_ORDER.index(season)
    return SEASON_ORDER[:position]


def align_probabilities(model, probabilities):
    positions = {
        class_name: position
        for position, class_name in enumerate(
            model.classes_
        )
    }

    return np.column_stack(
        [
            probabilities[:, positions[class_name]]
            for class_name in CLASS_ORDER
        ]
    )


def calculate_metrics(model, x_data, y_data):
    predictions = model.predict(x_data)

    probabilities = align_probabilities(
        model,
        model.predict_proba(x_data),
    )

    return {
        "accuracy": accuracy_score(
            y_data,
            predictions,
        ),
        "macro_f1": f1_score(
            y_data,
            predictions,
            labels=CLASS_ORDER,
            average="macro",
            zero_division=0,
        ),
        "log_loss": log_loss(
            y_data,
            probabilities,
            labels=CLASS_ORDER,
        ),
    }


def run_validation(data):
    models = create_models()
    results = []

    total_combinations = (
        len(FEATURE_SETS) * len(models)
    )

    print(
        f"Testing {total_combinations} "
        "model and feature combinations."
    )

    for feature_set_name, features in FEATURE_SETS.items():
        print(
            f"\nFeature set: {feature_set_name} "
            f"({len(features)} features)"
        )

        for model_name, model_template in models.items():
            fold_metrics = []

            for validation_season in VALIDATION_SEASONS:
                training_seasons = seasons_before(
                    validation_season
                )

                train_data = data[
                    data["season"].isin(
                        training_seasons
                    )
                ]

                validation_data = data[
                    data["season"]
                    == validation_season
                ]

                model = clone(model_template)

                model.fit(
                    train_data[features],
                    train_data[TARGET_COLUMN],
                )

                metrics = calculate_metrics(
                    model,
                    validation_data[features],
                    validation_data[TARGET_COLUMN],
                )

                fold_metrics.append(metrics)

            mean_accuracy = np.mean(
                [
                    metrics["accuracy"]
                    for metrics in fold_metrics
                ]
            )

            mean_macro_f1 = np.mean(
                [
                    metrics["macro_f1"]
                    for metrics in fold_metrics
                ]
            )

            mean_log_loss = np.mean(
                [
                    metrics["log_loss"]
                    for metrics in fold_metrics
                ]
            )

            results.append(
                {
                    "feature_set": feature_set_name,
                    "feature_count": len(features),
                    "model": model_name,
                    "mean_accuracy": mean_accuracy,
                    "mean_macro_f1": mean_macro_f1,
                    "mean_log_loss": mean_log_loss,
                }
            )

            print(
                f"{model_name:<24} "
                f"Accuracy: {mean_accuracy:.4f}  "
                f"Macro F1: {mean_macro_f1:.4f}  "
                f"Log Loss: {mean_log_loss:.4f}"
            )

    return pd.DataFrame(results)


def select_model(results):
    baseline_log_loss = 1.0620

    eligible = results[
        results["mean_log_loss"]
        < baseline_log_loss
    ].copy()

    eligible = eligible.sort_values(
        [
            "mean_accuracy",
            "mean_log_loss",
        ],
        ascending=[
            False,
            True,
        ],
    ).reset_index(drop=True)

    return eligible.iloc[0]


def evaluate_on_test(data, selected):
    feature_set_name = selected["feature_set"]
    model_name = selected["model"]

    features = FEATURE_SETS[feature_set_name]
    model = create_models()[model_name]

    train_data = data[
        data["season"].isin(
            seasons_before(TEST_SEASON)
        )
    ]

    test_data = data[
        data["season"] == TEST_SEASON
    ]

    model.fit(
        train_data[features],
        train_data[TARGET_COLUMN],
    )

    metrics = calculate_metrics(
        model,
        test_data[features],
        test_data[TARGET_COLUMN],
    )

    predictions = model.predict(
        test_data[features]
    )

    counts = pd.DataFrame(
        {
            "actual": (
                test_data[TARGET_COLUMN]
                .value_counts()
                .reindex(
                    CLASS_ORDER,
                    fill_value=0,
                )
            ),
            "predicted": (
                pd.Series(predictions)
                .value_counts()
                .reindex(
                    CLASS_ORDER,
                    fill_value=0,
                )
            ),
        }
    )

    print("\nFinal chronological test")
    print("------------------------")
    print(f"Feature set: {feature_set_name}")
    print(f"Feature count: {len(features)}")
    print(f"Model: {model_name}")
    print(f"Test season: {TEST_SEASON}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Log Loss: {metrics['log_loss']:.4f}")

    print("\nOutcome counts:")
    print(counts.to_string())


def main():
    data = load_data()

    print(f"Loaded {len(data)} completed matches.")

    results = run_validation(data)

    RESULTS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        RESULTS_FILE,
        index=False,
    )

    ranked = results.sort_values(
        [
            "mean_accuracy",
            "mean_log_loss",
        ],
        ascending=[
            False,
            True,
        ],
    ).reset_index(drop=True)

    print("\nTop 10 validation combinations")
    print("------------------------------")
    print(
        ranked.head(10).to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    selected = select_model(results)

    print("\nSelected configuration")
    print("----------------------")
    print(
        f"Feature set: "
        f"{selected['feature_set']}"
    )
    print(f"Model: {selected['model']}")
    print(
        f"Validation accuracy: "
        f"{selected['mean_accuracy']:.4f}"
    )
    print(
        f"Validation log loss: "
        f"{selected['mean_log_loss']:.4f}"
    )

    evaluate_on_test(
        data,
        selected,
    )

    print(
        f"\nSaved tuning results to: "
        f"{RESULTS_FILE}"
    )


if __name__ == "__main__":
    main()