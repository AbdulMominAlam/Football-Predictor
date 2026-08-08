from collections import Counter
from pathlib import Path
import time

import pandas as pd

from world_cup_simulator import (
    build_current_team_states,
    ensure_minimum_team_history,
    load_match_data,
    load_model,
    simulate_tournament,
)

from world_cup_teams import WORLD_CUP_2026_TEAMS


PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "world_cup_1000_simulations.csv"
)


def main():
    print(
        "\nLoading model and historical match data..."
    )

    model, feature_columns = load_model()

    match_data = load_match_data()

    (
        histories,
        elo_ratings,
        known_teams,
    ) = build_current_team_states(
        match_data
    )

    tournament_teams = (
        WORLD_CUP_2026_TEAMS.copy()
    )

    ensure_minimum_team_history(
        tournament_teams=tournament_teams,
        histories=histories,
        elo_ratings=elo_ratings,
    )

    print("Ready.")

    simulations = 1000

    champion_counts = Counter()
    runner_up_counts = Counter()

    print(
        f"\nRunning {simulations} "
        "tournament simulations..."
    )

    start = time.perf_counter()

    for simulation_number in range(
        simulations
    ):

        champion, runner_up = (
            simulate_tournament(
                tournament_teams=(
                    tournament_teams
                ),
                model=model,
                feature_columns=(
                    feature_columns
                ),
                base_histories=histories,
                base_elo_ratings=(
                    elo_ratings
                ),
                show_output=False,
            )
        )

        champion_counts[champion] += 1

        runner_up_counts[
            runner_up
        ] += 1

        if (
            simulation_number + 1
        ) % 100 == 0:

            print(
                f"Completed "
                f"{simulation_number + 1} "
                f"/ {simulations}"
            )

    end = time.perf_counter()

    results = []

    for team in tournament_teams:

        championship_wins = (
            champion_counts[team]
        )

        runner_up_finishes = (
            runner_up_counts[team]
        )

        championship_probability = (
            championship_wins
            / simulations
            * 100
        )

        runner_up_probability = (
            runner_up_finishes
            / simulations
            * 100
        )

        final_probability = (
            (
                championship_wins
                + runner_up_finishes
            )
            / simulations
            * 100
        )

        results.append(
            {
                "team": team,
                "championship_wins": (
                    championship_wins
                ),
                "championship_probability": (
                    championship_probability
                ),
                "runner_up_finishes": (
                    runner_up_finishes
                ),
                "runner_up_probability": (
                    runner_up_probability
                ),
                "final_probability": (
                    final_probability
                ),
            }
        )

    results_df = pd.DataFrame(
        results
    )

    results_df = (
        results_df.sort_values(
            by=(
                "championship_probability"
            ),
            ascending=False,
        )
        .reset_index(drop=True)
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        "\n=========================================="
    )
    print(
        "      WORLD CUP CHAMPIONSHIP ODDS"
    )
    print(
        "==========================================\n"
    )

    for _, row in results_df.iterrows():

        print(
            f"{row['team']:<22}"
            f"{int(row['championship_wins']):>4} wins "
            f"("
            f"{row['championship_probability']:6.2f}"
            f"%)"
        )

    print(
        "\n=========================================="
    )
    print(
        "         RUNNER-UP FINISHES"
    )
    print(
        "==========================================\n"
    )

    runner_up_df = (
        results_df.sort_values(
            by="runner_up_probability",
            ascending=False,
        )
    )

    for _, row in runner_up_df.iterrows():

        print(
            f"{row['team']:<22}"
            f"{int(row['runner_up_finishes']):>4} finals "
            f"("
            f"{row['runner_up_probability']:6.2f}"
            f"%)"
        )

    print(
        "\n=========================================="
    )

    print(
        f"Simulations : {simulations}"
    )

    print(
        f"Time Taken  : "
        f"{end - start:.2f} seconds"
    )

    print(
        f"Results saved to:\n"
        f"{OUTPUT_FILE}"
    )

    print(
        "=========================================="
    )


if __name__ == "__main__":
    main()