from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "premier_league"
    / "processed"
    / "model_features.csv"
)

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "premier_league_model.joblib"
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

TRAINING_SEASONS = [
    "2020-21",
    "2021-22",
    "2022-23",
    "2023-24",
    "2024-25",
]

TEST_SEASON = "2025-26"


def load_feature_data():
    data = pd.read_csv(FEATURE_DATA_FILE)

    required_columns = set(
        FEATURE_COLUMNS
        + [
            TARGET_COLUMN,
            "season",
            "date",
        ]
    )

    missing_columns = required_columns - set(data.columns)

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    data["date"] = pd.to_datetime(
        data["date"],
        errors="raise",
    )

    data = data.sort_values("date").reset_index(drop=True)

    return data


def create_random_forest():
    return RandomForestClassifier(
        n_estimators=500,
        max_depth=12,
        min_samples_split=12,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )


def evaluate_model(model, test_data):
    x_test = test_data[FEATURE_COLUMNS]
    y_test = test_data[TARGET_COLUMN]

    predictions = model.predict(x_test)

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    baseline_prediction = test_data[TARGET_COLUMN].mode()[0]
    baseline_predictions = [
        baseline_prediction
    ] * len(test_data)

    baseline_accuracy = accuracy_score(
        y_test,
        baseline_predictions,
    )

    print("\nEvaluation results")
    print("------------------")
    print(f"Test season: {TEST_SEASON}")
    print(f"Test matches: {len(test_data)}")
    print(f"Model accuracy: {accuracy:.2%}")
    print(
        f"Most-common-result baseline: "
        f"{baseline_accuracy:.2%}"
    )

    print("\nClassification report")
    print("---------------------")
    print(
        classification_report(
            y_test,
            predictions,
            labels=CLASS_ORDER,
            digits=3,
            zero_division=0,
        )
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=CLASS_ORDER,
    )

    matrix_dataframe = pd.DataFrame(
        matrix,
        index=[
            f"Actual {label}"
            for label in CLASS_ORDER
        ],
        columns=[
            f"Predicted {label}"
            for label in CLASS_ORDER
        ],
    )

    print("Confusion matrix")
    print("----------------")
    print(matrix_dataframe.to_string())

    return accuracy


def display_feature_importance(model):
    importance = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance": model.feature_importances_,
        }
    )

    importance = importance.sort_values(
        "importance",
        ascending=False,
    ).reset_index(drop=True)

    print("\nFeature importance")
    print("------------------")
    print(importance.to_string(index=False))

    return importance


def train_evaluation_model(data):
    train_data = data[
        data["season"].isin(TRAINING_SEASONS)
    ].copy()

    test_data = data[
        data["season"] == TEST_SEASON
    ].copy()

    if train_data.empty:
        raise ValueError("The training dataset is empty.")

    if test_data.empty:
        raise ValueError("The test dataset is empty.")

    latest_training_date = train_data["date"].max()
    earliest_test_date = test_data["date"].min()

    if latest_training_date >= earliest_test_date:
        raise ValueError(
            "The chronological split is invalid. "
            "Training data overlaps with test data."
        )

    print("Chronological evaluation split")
    print("------------------------------")
    print(
        "Training seasons: "
        + ", ".join(TRAINING_SEASONS)
    )
    print(f"Training matches: {len(train_data)}")
    print(f"Test season: {TEST_SEASON}")
    print(f"Test matches: {len(test_data)}")
    print(
        "Latest training match: "
        f"{latest_training_date.date()}"
    )
    print(
        "Earliest test match: "
        f"{earliest_test_date.date()}"
    )

    x_train = train_data[FEATURE_COLUMNS]
    y_train = train_data[TARGET_COLUMN]

    model = create_random_forest()

    model.fit(
        x_train,
        y_train,
    )

    accuracy = evaluate_model(
        model,
        test_data,
    )

    display_feature_importance(model)

    return accuracy


def train_final_model(data, evaluation_accuracy):
    """
    After evaluation, train the production model on every completed
    match available, including completed 2026-27 matches.
    """
    x_all = data[FEATURE_COLUMNS]
    y_all = data[TARGET_COLUMN]

    final_model = create_random_forest()

    final_model.fit(
        x_all,
        y_all,
    )

    model_package = {
        "model": final_model,
        "feature_columns": FEATURE_COLUMNS,
        "class_order": list(final_model.classes_),
        "evaluation_test_season": TEST_SEASON,
        "evaluation_accuracy": evaluation_accuracy,
        "training_matches": len(data),
        "training_seasons": sorted(
            data["season"].unique().tolist()
        ),
        "latest_training_date": str(
            data["date"].max().date()
        ),
    }

    MODEL_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model_package,
        MODEL_FILE,
    )

    print("\nFinal production model")
    print("----------------------")
    print(f"Training matches: {len(data)}")
    print(
        "Included seasons: "
        + ", ".join(
            sorted(data["season"].unique())
        )
    )
    print(
        "Latest result used: "
        f"{data['date'].max().date()}"
    )
    print(f"Saved model to: {MODEL_FILE}")


def main():
    data = load_feature_data()

    print(f"Loaded {len(data)} completed matches.")
    print(
        "Available seasons: "
        + ", ".join(
            sorted(data["season"].unique())
        )
    )

    evaluation_accuracy = train_evaluation_model(data)

    train_final_model(
        data,
        evaluation_accuracy,
    )


if __name__ == "__main__":
    main()