from collections import Counter
import time
from world_cup_simulator import (
    ask_for_teams,
    build_current_team_states,
    load_match_data,
    load_model,
    simulate_tournament,
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
    ) = build_current_team_states(match_data)

    print("Ready.")

    tournament_teams = ask_for_teams(
        known_teams
    )

    simulations = 1000
    champion_counts = Counter()

    print(
        f"\nRunning {simulations} tournament simulations..."
    )
    start = time.perf_counter()

    for simulation_number in range(simulations):
        champion, runner_up = simulate_tournament(
            tournament_teams=tournament_teams,
            model=model,
            feature_columns=feature_columns,
            base_histories=histories,
            base_elo_ratings=elo_ratings,
            show_output=False,
        )
        end = time.perf_counter()

        champion_counts[champion] += 1

        if (simulation_number + 1) % 100 == 0:
            print(
                f"Completed {simulation_number + 1} "
                "simulations"
            )

    for team in tournament_teams:
        champion_counts.setdefault(team, 0)

    print("\n===================================")
    print("      CHAMPIONSHIP PROBABILITIES")
    print("===================================\n")

    for team, wins in champion_counts.most_common():
        probability = (
            wins / simulations
        ) * 100

        print(
            f"{team:<20} "
            f"{wins:>4} wins "
            f"({probability:>6.2f}%)"
        )

    print(f"\nTime taken: {end - start:.2f} seconds")

if __name__ == "__main__":
    main()