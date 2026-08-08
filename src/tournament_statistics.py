from collections import Counter
import time

from world_cup_simulator import (
    build_current_team_states,
    ensure_minimum_team_history,
    load_match_data,
    load_model,
    simulate_tournament,
)

from world_cup_teams import WORLD_CUP_2026_TEAMS


def main():
    print("\nLoading model and historical match data...")

    model, feature_columns = load_model()

    match_data = load_match_data()

    (
        histories,
        elo_ratings,
        known_teams,
    ) = build_current_team_states(
        match_data
    )

    # Use the official 2026 World Cup teams
    tournament_teams = WORLD_CUP_2026_TEAMS.copy()

    # Give new teams enough form history and Elo
    ensure_minimum_team_history(
        tournament_teams=tournament_teams,
        histories=histories,
        elo_ratings=elo_ratings,
    )

    print("Ready.")

    simulations = 300

    champion_counts = Counter()
    runner_up_counts = Counter()

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

        champion_counts[champion] += 1
        runner_up_counts[runner_up] += 1

        if (simulation_number + 1) % 100 == 0:
            print(
                f"Completed {simulation_number + 1} / {simulations}"
            )

    end = time.perf_counter()

    # Make sure every team appears
    for team in tournament_teams:
        champion_counts.setdefault(team, 0)
        runner_up_counts.setdefault(team, 0)

    print("\n==========================================")
    print("      WORLD CUP CHAMPIONSHIP ODDS")
    print("==========================================\n")

    for team, wins in champion_counts.most_common():

        probability = wins / simulations * 100

        print(
            f"{team:<22}"
            f"{wins:>4} wins"
            f" ({probability:6.2f}%)"
        )

    print("\n==========================================")
    print("         RUNNER-UP FINISHES")
    print("==========================================\n")

    for team, finishes in runner_up_counts.most_common():

        probability = finishes / simulations * 100

        print(
            f"{team:<22}"
            f"{finishes:>4} finals"
            f" ({probability:6.2f}%)"
        )

    print("\n==========================================")
    print(f"Simulations : {simulations}")
    print(f"Time Taken  : {end - start:.2f} seconds")
    print("==========================================")


if __name__ == "__main__":
    main()