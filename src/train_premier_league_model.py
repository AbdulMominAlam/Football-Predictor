from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from compare_poisson_model import (
    CLASS_ORDER,
    POISSON_CATEGORICAL_FEATURES,
    POISSON_NUMERIC_FEATURES,
    create_goal_training_rows,
    create_poisson_model,
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

TEST_SEASON = "2025-26"
POISSON_ALPHA = 0.05


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


def load_data():
    data = pd.read_csv(FEATURE_DATA_FILE)

    data["date"] = pd.to_datetime(
        data["date"],
        errors="raise",
    )

    required_columns = set(
        CLASSIFIER_FEATURES
        + [
            "season",
            "date",
            "home_team",
            "away_team",
            "home_goals",
            "away_goals",
            "target",
        ]
    )

    missing_columns = (
        required_columns - set(data.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    return data.sort_values(
        "date"
    ).reset_index(drop=True)


def create_outcome_classifier():
    return Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                LogisticRegression(
                    C=0.25,
                    max_iter=4000,
                    random_state=42,
                ),
            ),
        ]
    )


def align_probabilities(model, probabilities):
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


def evaluate_classifier(data):
    train_data = data[
        data["season"] < TEST_SEASON
    ].copy()

    test_data = data[
        data["season"] == TEST_SEASON
    ].copy()

    if train_data.empty or test_data.empty:
        raise ValueError(
            "Training or test data is empty."
        )

    classifier = create_outcome_classifier()

    classifier.fit(
        train_data[CLASSIFIER_FEATURES],
        train_data["target"],
    )

    predictions = classifier.predict(
        test_data[CLASSIFIER_FEATURES]
    )

    probabilities = align_probabilities(
        classifier,
        classifier.predict_proba(
            test_data[CLASSIFIER_FEATURES]
        ),
    )

    accuracy = accuracy_score(
        test_data["target"],
        predictions,
    )

    probability_log_loss = log_loss(
        test_data["target"],
        probabilities,
        labels=CLASS_ORDER,
    )

    matrix = confusion_matrix(
        test_data["target"],
        predictions,
        labels=CLASS_ORDER,
    )

    matrix_frame = pd.DataFrame(
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

    print("Held-out evaluation")
    print("-------------------")
    print(
        "Training seasons: "
        + ", ".join(
            sorted(
                train_data["season"].unique()
            )
        )
    )
    print(f"Test season: {TEST_SEASON}")
    print(f"Training matches: {len(train_data)}")
    print(f"Test matches: {len(test_data)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Log Loss: {probability_log_loss:.4f}")

    print("\nClassification report")
    print("---------------------")
    print(
        classification_report(
            test_data["target"],
            predictions,
            labels=CLASS_ORDER,
            digits=3,
            zero_division=0,
        )
    )

    print("Confusion matrix")
    print("----------------")
    print(matrix_frame.to_string())

    return {
        "accuracy": accuracy,
        "log_loss": probability_log_loss,
    }


def train_final_models(data):
    classifier = create_outcome_classifier()

    classifier.fit(
        data[CLASSIFIER_FEATURES],
        data["target"],
    )

    goal_training_data = create_goal_training_rows(
        data
    )

    poisson_model = create_poisson_model(
        POISSON_ALPHA
    )

    poisson_features = (
        POISSON_CATEGORICAL_FEATURES
        + POISSON_NUMERIC_FEATURES
    )

    poisson_model.fit(
        goal_training_data[poisson_features],
        goal_training_data["goals"],
    )

    return classifier, poisson_model


def save_model_package(
    data,
    classifier,
    poisson_model,
    evaluation,
):
    model_package = {
        "outcome_classifier": classifier,
        "poisson_model": poisson_model,
        "classifier_features": CLASSIFIER_FEATURES,
        "poisson_categorical_features": (
            POISSON_CATEGORICAL_FEATURES
        ),
        "poisson_numeric_features": (
            POISSON_NUMERIC_FEATURES
        ),
        "class_order": CLASS_ORDER,
        "poisson_alpha": POISSON_ALPHA,
        "test_season": TEST_SEASON,
        "test_accuracy": evaluation["accuracy"],
        "test_log_loss": evaluation["log_loss"],
        "training_matches": len(data),
        "training_seasons": sorted(
            data["season"].unique().tolist()
        ),
        "latest_training_date": str(
            data["date"].max().date()
        ),
        "methodology": (
            "Logistic Regression predicts match outcome "
            "probabilities. Poisson regression estimates "
            "expected goals and generates realistic scores."
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

    print("\nFinal production package")
    print("------------------------")
    print(f"Training matches: {len(data)}")
    print(
        "Training seasons: "
        + ", ".join(
            sorted(data["season"].unique())
        )
    )
    print(
        "Latest training result: "
        f"{data['date'].max().date()}"
    )
    print(f"Poisson alpha: {POISSON_ALPHA}")
    print(f"Saved to: {MODEL_FILE}")


def main():
    data = load_data()

    print(
        f"Loaded {len(data)} completed matches."
    )

    evaluation = evaluate_classifier(data)

    classifier, poisson_model = train_final_models(
        data
    )

    save_model_package(
        data,
        classifier,
        poisson_model,
        evaluation,
    )


if __name__ == "__main__":
    main()