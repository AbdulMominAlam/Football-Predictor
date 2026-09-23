from copy import deepcopy
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .premier_league_features import (
        PremierLeagueFeatureBuilder,
        load_completed_matches,
    )
    from .premier_league_predict import (
        CURRENT_SEASON,
        align_classifier_probabilities,
        create_poisson_rows,
        create_score_matrix,
        load_model_package,
    )
    from .premier_league_teams import (
        PREMIER_LEAGUE_2026_27_TEAMS,
    )
except ImportError:
    from premier_league_features import (
        PremierLeagueFeatureBuilder,
        load_completed_matches,
    )
    from premier_league_predict import (
        CURRENT_SEASON,
        align_classifier_probabilities,
        create_poisson_rows,
        create_score_matrix,
        load_model_package,
    )
    from premier_league_teams import (
        PREMIER_LEAGUE_2026_27_TEAMS,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MATCHES_FILE = (
    PROJECT_ROOT
    / "data"
    / "premier_league"
    / "processed"
    / "matches.csv"
)


def empty_table_row(team):
    return {
        "Team": team,
        "P": 0,
        "W": 0,
        "D": 0,
        "L": 0,
        "GF": 0,
        "GA": 0,
        "GD": 0,
        "Pts": 0,
    }


def create_empty_table():
    return {
        team: empty_table_row(team)
        for team in PREMIER_LEAGUE_2026_27_TEAMS
    }


def update_table(
    table,
    home_team,
    away_team,
    home_goals,
    away_goals,
):
    home = table[home_team]
    away = table[away_team]

    home["P"] += 1
    away["P"] += 1

    home["GF"] += home_goals
    home["GA"] += away_goals

    away["GF"] += away_goals
    away["GA"] += home_goals

    if home_goals > away_goals:
        home["W"] += 1
        home["Pts"] += 3
        away["L"] += 1
    elif away_goals > home_goals:
        away["W"] += 1
        away["Pts"] += 3
        home["L"] += 1
    else:
        home["D"] += 1
        away["D"] += 1
        home["Pts"] += 1
        away["Pts"] += 1

    home["GD"] = home["GF"] - home["GA"]
    away["GD"] = away["GF"] - away["GA"]


@lru_cache(maxsize=1)
def load_all_fixtures():
    fixtures = pd.read_csv(MATCHES_FILE)

    fixtures["date"] = pd.to_datetime(
        fixtures["date"],
        errors="raise",
    )

    return fixtures.sort_values(
        [
            "date",
            "home_team",
            "away_team",
        ]
    ).reset_index(drop=True)


@lru_cache(maxsize=1)
def build_base_state():
    """
    Build the model state using every real completed match.
    The returned state is copied before each simulation.
    """
    completed_matches = load_completed_matches()
    builder = PremierLeagueFeatureBuilder()

    for _, match in completed_matches.iterrows():
        builder.process_match(match)

    return builder


def build_current_table():
    table = create_empty_table()
    fixtures = load_all_fixtures()

    completed_current_matches = fixtures[
        (fixtures["season"] == CURRENT_SEASON)
        & (fixtures["status"] == "played")
    ]

    for _, match in completed_current_matches.iterrows():
        update_table(
            table,
            match["home_team"],
            match["away_team"],
            int(match["home_goals"]),
            int(match["away_goals"]),
        )

    return table


def create_match_features(
    builder,
    fixture,
):
    builder.start_season(CURRENT_SEASON)

    features = builder.create_pre_match_features(
        fixture["home_team"],
        fixture["away_team"],
        pd.Timestamp(fixture["date"]),
    )

    return {
        "season": CURRENT_SEASON,
        "date": pd.Timestamp(fixture["date"]),
        "home_team": fixture["home_team"],
        "away_team": fixture["away_team"],
        **features,
    }


def predict_probabilities(
    feature_row,
    model_package,
):
    classifier = model_package[
        "outcome_classifier"
    ]

    classifier_features = model_package[
        "classifier_features"
    ]

    class_order = model_package["class_order"]

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

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    return probabilities / probabilities.sum()


def predict_expected_goals(
    feature_row,
    model_package,
):
    poisson_model = model_package["poisson_model"]

    home_row, away_row = create_poisson_rows(
        feature_row
    )

    expected_home_goals = float(
        poisson_model.predict(home_row)[0]
    )

    expected_away_goals = float(
        poisson_model.predict(away_row)[0]
    )

    return (
        float(
            np.clip(
                expected_home_goals,
                0.05,
                5.0,
            )
        ),
        float(
            np.clip(
                expected_away_goals,
                0.05,
                5.0,
            )
        ),
    )


def sample_score_for_outcome(
    score_matrix,
    outcome,
    rng,
):
    valid_scores = []
    score_probabilities = []

    rows, columns = score_matrix.shape

    for home_goals in range(rows):
        for away_goals in range(columns):
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
                        home_goals,
                        away_goals,
                    )
                )
                score_probabilities.append(
                    score_matrix[
                        home_goals,
                        away_goals,
                    ]
                )

    score_probabilities = np.asarray(
        score_probabilities,
        dtype=float,
    )

    score_probabilities = (
        score_probabilities
        / score_probabilities.sum()
    )

    selected_index = rng.choice(
        len(valid_scores),
        p=score_probabilities,
    )

    return valid_scores[selected_index]


def update_simulation_state(
    builder,
    fixture,
    home_goals,
    away_goals,
):
    simulated_match = pd.Series(
        {
            "season": CURRENT_SEASON,
            "date": pd.Timestamp(
                fixture["date"]
            ),
            "home_team": fixture["home_team"],
            "away_team": fixture["away_team"],
            "home_goals": home_goals,
            "away_goals": away_goals,
        }
    )

    builder.process_match(simulated_match)


def create_ranked_table(table, rng):
    table_frame = pd.DataFrame(
        table.values()
    )

    table_frame["_random_tiebreaker"] = (
        rng.random(len(table_frame))
    )

    table_frame = table_frame.sort_values(
        [
            "Pts",
            "GD",
            "GF",
            "_random_tiebreaker",
        ],
        ascending=[
            False,
            False,
            False,
            False,
        ],
    ).reset_index(drop=True)

    table_frame.insert(
        0,
        "Pos",
        range(1, len(table_frame) + 1),
    )

    return table_frame.drop(
        columns="_random_tiebreaker"
    )


def simulate_season(seed=None):
    rng = np.random.default_rng(seed)

    builder = deepcopy(build_base_state())
    table = build_current_table()
    model_package = load_model_package()

    fixtures = load_all_fixtures()

    remaining_fixtures = fixtures[
        (fixtures["season"] == CURRENT_SEASON)
        & (fixtures["status"] == "scheduled")
    ].copy()

    remaining_fixtures = (
        remaining_fixtures.sort_values(
            [
                "date",
                "home_team",
                "away_team",
            ]
        )
    )

    simulated_results = []

    for _, fixture in remaining_fixtures.iterrows():
        feature_row = create_match_features(
            builder,
            fixture,
        )

        probabilities = predict_probabilities(
            feature_row,
            model_package,
        )

        class_order = model_package[
            "class_order"
        ]

        outcome = rng.choice(
            class_order,
            p=probabilities,
        )

        (
            expected_home_goals,
            expected_away_goals,
        ) = predict_expected_goals(
            feature_row,
            model_package,
        )

        score_matrix = create_score_matrix(
            expected_home_goals,
            expected_away_goals,
        )

        home_goals, away_goals = (
            sample_score_for_outcome(
                score_matrix,
                outcome,
                rng,
            )
        )

        update_table(
            table,
            fixture["home_team"],
            fixture["away_team"],
            home_goals,
            away_goals,
        )

        update_simulation_state(
            builder,
            fixture,
            home_goals,
            away_goals,
        )

        simulated_results.append(
            {
                "date": pd.Timestamp(
                    fixture["date"]
                ).date().isoformat(),
                "home_team": fixture["home_team"],
                "away_team": fixture["away_team"],
                "home_goals": home_goals,
                "away_goals": away_goals,
                "outcome": outcome,
                "home_win_probability": (
                    probabilities[
                        class_order.index(
                            "home_win"
                        )
                    ]
                ),
                "draw_probability": (
                    probabilities[
                        class_order.index(
                            "draw"
                        )
                    ]
                ),
                "away_win_probability": (
                    probabilities[
                        class_order.index(
                            "away_win"
                        )
                    ]
                ),
                "expected_home_goals": (
                    expected_home_goals
                ),
                "expected_away_goals": (
                    expected_away_goals
                ),
            }
        )

    final_table = create_ranked_table(
        table,
        rng,
    )

    simulated_results = pd.DataFrame(
        simulated_results
    )

    return {
        "champion": final_table.iloc[0]["Team"],
        "table": final_table,
        "simulated_matches": simulated_results,
    }


def main():
    result = simulate_season(seed=42)

    print(
        f"Simulated matches: "
        f"{len(result['simulated_matches'])}"
    )
    print(
        f"Champion: {result['champion']}"
    )

    print("\nFinal table")
    print("-----------")
    print(
        result["table"].to_string(
            index=False
        )
    )

    print("\nFirst 10 simulated fixtures")
    print("---------------------------")
    print(
        result["simulated_matches"]
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()