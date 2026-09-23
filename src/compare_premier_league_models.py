from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "premier_league"
    / "processed"
    / "model_features.csv"
)

FEATURE_COLUMNS = [
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

TARGET_COLUMN = "target"

CLASS_ORDER = [
    "away_win",
    "draw",
    "home_win",
]

SEASON_ORDER = [
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


def load_data():
    data = pd.read_csv(FEATURE_DATA_FILE)

    data["date"] = pd.to_datetime(
        data["date"],
        errors="raise",
    )

    data = data.sort_values("date").reset_index(drop=True)

    return data


def create_models():
    return {
        "Baseline": DummyClassifier(
            strategy="prior",
        ),
        "Logistic Regression": Pipeline(
            steps=[
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "model",
                    LogisticRegression(
                        C=0.25,
                        max_iter=3000,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=500,
            max_depth=10,
            min_samples_split=15,
            min_samples_leaf=8,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=500,
            max_depth=12,
            min_samples_split=15,
            min_samples_leaf=8,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        ),
        "Histogram Gradient Boosting": (
            HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=250,
                max_leaf_nodes=15,
                min_samples_leaf=20,
                l2_regularization=2.0,
                random_state=42,
            )
        ),
    }


def align_probabilities(model, probabilities):
    """
    Reorder probability columns so that every model uses:
    away_win, draw, home_win.
    """
    class_positions = {
        class_name: position
        for position, class_name in enumerate(model.classes_)
    }

    return np.column_stack(
        [
            probabilities[:, class_positions[class_name]]
            for class_name in CLASS_ORDER
        ]
    )


def calculate_metrics(model, x_data, y_data):
    predictions = model.predict(x_data)

    probabilities = model.predict_proba(x_data)
    probabilities = align_probabilities(
        model,
        probabilities,
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


def seasons_before(season):
    season_position = SEASON_ORDER.index(season)

    return SEASON_ORDER[:season_position]


def run_rolling_validation(data, models):
    results = []

    print("Rolling chronological validation")
    print("--------------------------------")

    for validation_season in VALIDATION_SEASONS:
        training_seasons = seasons_before(
            validation_season
        )

        train_data = data[
            data["season"].isin(training_seasons)
        ].copy()

        validation_data = data[
            data["season"] == validation_season
        ].copy()

        print(
            f"\nValidation season: {validation_season}"
        )
        print(
            "Training seasons: "
            + ", ".join(training_seasons)
        )
        print(
            f"Training matches: {len(train_data)}"
        )
        print(
            f"Validation matches: "
            f"{len(validation_data)}"
        )

        x_train = train_data[FEATURE_COLUMNS]
        y_train = train_data[TARGET_COLUMN]

        x_validation = validation_data[FEATURE_COLUMNS]
        y_validation = validation_data[TARGET_COLUMN]

        for model_name, model_template in models.items():
            model = clone(model_template)

            model.fit(
                x_train,
                y_train,
            )

            metrics = calculate_metrics(
                model,
                x_validation,
                y_validation,
            )

            results.append(
                {
                    "model": model_name,
                    "validation_season": validation_season,
                    **metrics,
                }
            )

            print(
                f"{model_name:<30} "
                f"Accuracy: {metrics['accuracy']:.3f}  "
                f"Macro F1: {metrics['macro_f1']:.3f}  "
                f"Log Loss: {metrics['log_loss']:.3f}"
            )

    return pd.DataFrame(results)


def summarize_validation(results):
    summary = (
        results.groupby("model")
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

    print("\nAverage validation performance")
    print("------------------------------")

    display = summary.copy()

    for column in [
        "mean_accuracy",
        "mean_macro_f1",
        "mean_log_loss",
    ]:
        display[column] = display[column].map(
            lambda value: f"{value:.4f}"
        )

    print(display.to_string(index=False))

    return summary


def evaluate_selected_model(
    data,
    models,
    selected_model_name,
):
    training_seasons = seasons_before(TEST_SEASON)

    train_data = data[
        data["season"].isin(training_seasons)
    ].copy()

    test_data = data[
        data["season"] == TEST_SEASON
    ].copy()

    model = clone(models[selected_model_name])

    model.fit(
        train_data[FEATURE_COLUMNS],
        train_data[TARGET_COLUMN],
    )

    metrics = calculate_metrics(
        model,
        test_data[FEATURE_COLUMNS],
        test_data[TARGET_COLUMN],
    )

    predictions = model.predict(
        test_data[FEATURE_COLUMNS]
    )

    prediction_counts = (
        pd.Series(predictions)
        .value_counts()
        .reindex(CLASS_ORDER, fill_value=0)
    )

    actual_counts = (
        test_data[TARGET_COLUMN]
        .value_counts()
        .reindex(CLASS_ORDER, fill_value=0)
    )

    comparison = pd.DataFrame(
        {
            "actual": actual_counts,
            "predicted": prediction_counts,
        }
    )

    print("\nSelected model test")
    print("-------------------")
    print(f"Selected model: {selected_model_name}")
    print(
        "Training seasons: "
        + ", ".join(training_seasons)
    )
    print(f"Test season: {TEST_SEASON}")
    print(f"Test matches: {len(test_data)}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Log Loss: {metrics['log_loss']:.4f}")

    print("\nActual and predicted outcomes")
    print("-----------------------------")
    print(comparison.to_string())

    return model, metrics


def main():
    data = load_data()
    models = create_models()

    print(f"Loaded {len(data)} completed matches.")
    print(
        "Models: "
        + ", ".join(models.keys())
    )

    validation_results = run_rolling_validation(
        data,
        models,
    )

    summary = summarize_validation(
        validation_results
    )

    non_baseline_summary = summary[
        summary["model"] != "Baseline"
    ]

    selected_model_name = (
        non_baseline_summary.iloc[0]["model"]
    )

    print(
        "\nBest model based on average validation "
        f"log loss: {selected_model_name}"
    )

    evaluate_selected_model(
        data,
        models,
        selected_model_name,
    )


if __name__ == "__main__":
    main()