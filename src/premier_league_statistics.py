import argparse
import time
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

try:
    from .premier_league_simulator import (
        simulate_season,
    )
except ImportError:
    from premier_league_simulator import (
        simulate_season,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "premier_league"
    / "processed"
)


def simulate_once(
    simulation_number,
    base_seed,
):
    result = simulate_season(
        seed=base_seed + simulation_number
    )

    table = result["table"]

    records = []

    for _, row in table.iterrows():
        records.append(
            {
                "simulation": simulation_number,
                "team": row["Team"],
                "position": int(row["Pos"]),
                "points": int(row["Pts"]),
                "wins": int(row["W"]),
                "draws": int(row["D"]),
                "losses": int(row["L"]),
                "goals_for": int(row["GF"]),
                "goals_against": int(row["GA"]),
                "goal_difference": int(row["GD"]),
                "champion": int(
                    row["Pos"] == 1
                ),
                "runner_up": int(
                    row["Pos"] == 2
                ),
                "top_four": int(
                    row["Pos"] <= 4
                ),
                "top_six": int(
                    row["Pos"] <= 6
                ),
                "relegated": int(
                    row["Pos"] >= 18
                ),
            }
        )

    return records


def run_monte_carlo(
    simulations,
    base_seed=42,
    jobs=1,
):
    if simulations <= 0:
        raise ValueError(
            "Simulations must be greater than zero."
        )

    print(
        f"Running {simulations} Premier League "
        "simulations..."
    )
    print(f"Parallel jobs: {jobs}")

    start_time = time.perf_counter()

    simulation_records = Parallel(
        n_jobs=jobs,
        backend="loky",
        verbose=10 if simulations >= 100 else 0,
    )(
        delayed(simulate_once)(
            simulation_number,
            base_seed,
        )
        for simulation_number
        in range(simulations)
    )

    flattened_records = [
        record
        for simulation in simulation_records
        for record in simulation
    ]

    raw_results = pd.DataFrame(
        flattened_records
    )

    summary = (
        raw_results.groupby("team")
        .agg(
            championships=("champion", "sum"),
            runner_up_finishes=(
                "runner_up",
                "sum",
            ),
            top_four_finishes=(
                "top_four",
                "sum",
            ),
            top_six_finishes=(
                "top_six",
                "sum",
            ),
            relegations=("relegated", "sum"),
            average_position=(
                "position",
                "mean",
            ),
            average_points=("points", "mean"),
            average_wins=("wins", "mean"),
            average_draws=("draws", "mean"),
            average_losses=("losses", "mean"),
            average_goals_for=(
                "goals_for",
                "mean",
            ),
            average_goals_against=(
                "goals_against",
                "mean",
            ),
            average_goal_difference=(
                "goal_difference",
                "mean",
            ),
            best_finish=("position", "min"),
            worst_finish=("position", "max"),
        )
        .reset_index()
    )

    summary["title_probability"] = (
        summary["championships"]
        / simulations
    )

    summary["runner_up_probability"] = (
        summary["runner_up_finishes"]
        / simulations
    )

    summary["top_four_probability"] = (
        summary["top_four_finishes"]
        / simulations
    )

    summary["top_six_probability"] = (
        summary["top_six_finishes"]
        / simulations
    )

    summary["relegation_probability"] = (
        summary["relegations"]
        / simulations
    )

    summary["simulations"] = simulations

    summary = summary.sort_values(
        [
            "title_probability",
            "top_four_probability",
            "average_position",
        ],
        ascending=[
            False,
            False,
            True,
        ],
    ).reset_index(drop=True)

    summary.insert(
        0,
        "rank",
        range(1, len(summary) + 1),
    )

    elapsed_time = (
        time.perf_counter() - start_time
    )

    print(
        f"Completed in {elapsed_time:.1f} seconds."
    )

    return summary, raw_results


def validate_results(
    summary,
    raw_results,
    simulations,
):
    expected_rows = simulations * 20

    if len(summary) != 20:
        raise ValueError(
            f"Expected 20 teams, got "
            f"{len(summary)}."
        )

    if len(raw_results) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} raw rows, "
            f"got {len(raw_results)}."
        )

    title_total = summary[
        "title_probability"
    ].sum()

    top_four_total = summary[
        "top_four_probability"
    ].sum()

    relegation_total = summary[
        "relegation_probability"
    ].sum()

    if not abs(title_total - 1.0) < 1e-9:
        raise ValueError(
            "Title probabilities do not sum to 1."
        )

    if not abs(top_four_total - 4.0) < 1e-9:
        raise ValueError(
            "Top-four probabilities do not sum to 4."
        )

    if not abs(relegation_total - 3.0) < 1e-9:
        raise ValueError(
            "Relegation probabilities do not sum to 3."
        )

    print("Validation passed.")
    print(
        f"Title probability total: "
        f"{title_total:.2f}"
    )
    print(
        f"Top-four probability total: "
        f"{top_four_total:.2f}"
    )
    print(
        f"Relegation probability total: "
        f"{relegation_total:.2f}"
    )


def save_summary(
    summary,
    simulations,
):
    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        OUTPUT_DIRECTORY
        / (
            "premier_league_"
            f"{simulations}_simulations.csv"
        )
    )

    summary.to_csv(
        output_file,
        index=False,
    )

    print(f"Saved results to: {output_file}")

    return output_file


def display_summary(summary):
    display = summary[
        [
            "rank",
            "team",
            "championships",
            "title_probability",
            "top_four_probability",
            "top_six_probability",
            "relegation_probability",
            "average_position",
            "average_points",
        ]
    ].copy()

    probability_columns = [
        "title_probability",
        "top_four_probability",
        "top_six_probability",
        "relegation_probability",
    ]

    for column in probability_columns:
        display[column] = display[column].map(
            lambda value: f"{value:.1%}"
        )

    display["average_position"] = (
        display["average_position"].map(
            lambda value: f"{value:.2f}"
        )
    )

    display["average_points"] = (
        display["average_points"].map(
            lambda value: f"{value:.2f}"
        )
    )

    print("\nMonte Carlo summary")
    print("-------------------")
    print(display.to_string(index=False))


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Run Premier League Monte Carlo "
            "simulations."
        )
    )

    parser.add_argument(
        "--simulations",
        type=int,
        default=1000,
        help=(
            "Number of complete seasons to "
            "simulate."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed.",
    )

    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help=(
            "Parallel processes. Use -1 for all "
            "available CPU cores."
        ),
    )

    return parser.parse_args()


def main():
    arguments = parse_arguments()

    summary, raw_results = run_monte_carlo(
        simulations=arguments.simulations,
        base_seed=arguments.seed,
        jobs=arguments.jobs,
    )

    validate_results(
        summary,
        raw_results,
        arguments.simulations,
    )

    save_summary(
        summary,
        arguments.simulations,
    )

    display_summary(summary)


if __name__ == "__main__":
    main()