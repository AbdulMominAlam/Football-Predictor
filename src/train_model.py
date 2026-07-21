from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = PROJECT_ROOT / "data" / "processed" / "model_data.csv"
MODEL_FILE = PROJECT_ROOT / "models" / "random_forest_model.joblib"


FEATURE_COLUMNS = [
    "neutral",

    "home_elo",
    "away_elo",
    "elo_difference",

    "home_win_rate_last_5",
    "away_win_rate_last_5",

    "home_draw_rate_last_5",
    "away_draw_rate_last_5",

    "home_avg_goals_scored_last_5",
    "away_avg_goals_scored_last_5",

    "home_avg_goals_conceded_last_5",
    "away_avg_goals_conceded_last_5",

    "home_points_per_match_last_5",
    "away_points_per_match_last_5",

    "home_goal_difference_last_5",
    "away_goal_difference_last_5",

    "win_rate_difference",
    "points_difference",
    "scoring_difference",
    "recent_goal_difference_difference",
]


def load_data():
    """
    Load the model-ready dataset.
    """

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Model dataset not found at: {DATA_FILE}"
        )

    data = pd.read_csv(DATA_FILE, parse_dates=["date"])

    # Make sure matches are ordered chronologically
    data = data.sort_values("date").reset_index(drop=True)

    return data


def split_data(data):
    """
    Split the data chronologically.

    The first 80% is used for training.
    The latest 20% is used for testing.
    """

    split_index = int(len(data) * 0.80)

    training_data = data.iloc[:split_index]
    testing_data = data.iloc[split_index:]

    X_train = training_data[FEATURE_COLUMNS]
    y_train = training_data["result"]

    X_test = testing_data[FEATURE_COLUMNS]
    y_test = testing_data["result"]

    return X_train, X_test, y_train, y_test, training_data, testing_data


def train_model(X_train, y_train):
    """
    Train a Random Forest classifier.
    """

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model using unseen test data.
    """

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("\n========== MODEL ACCURACY ==========\n")
    print(f"{accuracy:.4f}")
    print(f"{accuracy * 100:.2f}%")

    print("\n========== CLASSIFICATION REPORT ==========\n")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "Away win",
                "Draw",
                "Home win",
            ],
            zero_division=0,
        )
    )

    print("\n========== CONFUSION MATRIX ==========\n")
    print(confusion_matrix(y_test, predictions))


def inspect_split(training_data, testing_data):
    """
    Display information about the chronological split.
    """

    print("\n========== TRAINING DATA ==========\n")
    print(f"Matches: {len(training_data)}")
    print(
        f"Date range: {training_data['date'].min()} "
        f"to {training_data['date'].max()}"
    )

    print("\n========== TESTING DATA ==========\n")
    print(f"Matches: {len(testing_data)}")
    print(
        f"Date range: {testing_data['date'].min()} "
        f"to {testing_data['date'].max()}"
    )


def show_feature_importance(model):
    """
    Show which features influenced the model most.
    """

    feature_importance = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "importance": model.feature_importances_,
    })

    feature_importance = feature_importance.sort_values(
        "importance",
        ascending=False,
    )

    print("\n========== FEATURE IMPORTANCE ==========\n")
    print(feature_importance.to_string(index=False))


def save_model(model):
    """
    Save the trained model inside the models folder.
    """

    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)

    model_package = {
        "model": model,
        "features": FEATURE_COLUMNS,
    }

    joblib.dump(model_package, MODEL_FILE)

    print(f"\nModel saved to:\n{MODEL_FILE}")


def main():
    football_data = load_data()

    (
        X_train,
        X_test,
        y_train,
        y_test,
        training_data,
        testing_data,
    ) = split_data(football_data)

    inspect_split(training_data, testing_data)

    model = train_model(X_train, y_train)

    evaluate_model(model, X_test, y_test)

    show_feature_importance(model)

    save_model(model)


if __name__ == "__main__":
    main()