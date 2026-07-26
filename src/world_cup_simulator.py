from collections import defaultdict, deque
import random
from itertools import combinations

from predict import (
    build_current_team_states,
    create_prediction_features,
    find_team_name,
    load_match_data,
    load_model,
    predict_match,
)
from features import FORM_WINDOW, INITIAL_ELO


NUMBER_OF_TEAMS = 48
TEAMS_PER_GROUP = 4

GROUP_NAMES = [
    "A", "B", "C", "D",
    "E", "F", "G", "H",
    "I", "J", "K", "L"
]


def ask_for_teams(known_teams):
    """
    Ask the user to enter 16 valid and unique teams.
    """

    selected_teams = []

    print(
        f"\nEnter {NUMBER_OF_TEAMS} teams for the tournament.\n"
    )

    while len(selected_teams) < NUMBER_OF_TEAMS:
        team_number = len(selected_teams) + 1

        user_input = input(
            f"Enter team {team_number}: "
        ).strip()

        matched_team = find_team_name(
            user_input,
            known_teams,
        )

        if matched_team is None:
            print(
                f"Team '{user_input}' was not found "
                "in the dataset.\n"
            )
            continue

        if matched_team in selected_teams:
            print(
                f"{matched_team} has already been added.\n"
            )
            continue

        selected_teams.append(matched_team)

    return selected_teams


def create_groups(teams):
    """
    Randomly divide 16 teams into four groups.
    """

    shuffled_teams = teams.copy()
    random.shuffle(shuffled_teams)

    groups = {}

    for group_index, group_name in enumerate(GROUP_NAMES):
        start_index = group_index * TEAMS_PER_GROUP
        end_index = start_index + TEAMS_PER_GROUP

        groups[group_name] = shuffled_teams[
            start_index:end_index
        ]

    return groups


def display_groups(groups):
    """
    Display the tournament groups.
    """

    print("\n===================================")
    print("          TOURNAMENT GROUPS")
    print("===================================")

    for group_name, teams in groups.items():
        print(f"\nGroup {group_name}")

        for position, team in enumerate(
            teams,
            start=1,
        ):
            print(f"{position}. {team}")


def create_group_table(teams):
    """
    Create an empty standings table for one group.
    """

    table = {}

    for team in teams:
        table[team] = {
            "played": 0,
            "wins": 0,
            "draws": 0,
            "losses": 0,
            "goals_for": 0,
            "goals_against": 0,
            "goal_difference": 0,
            "points": 0,
        }

    return table


def generate_winning_score(
    winner_probability,
    loser_probability,
):
    """
    Generate a winning scoreline based on how strongly
    the model favours the winning team.

    A strong favourite has a higher chance of winning
    by multiple goals.

    A close matchup is more likely to finish 1-0 or 2-1.
    """

    probability_difference = (
        winner_probability - loser_probability
    )

    # Very close match
    if probability_difference < 0.15:
        possible_scores = [
            (1, 0),
            (2, 1),
            (2, 0),
            (3, 2),
            (3, 1),
        ]

        score_weights = [
            35,
            35,
            15,
            10,
            5,
        ]

    # Clear favourite
    elif probability_difference < 0.35:
        possible_scores = [
            (1, 0),
            (2, 0),
            (2, 1),
            (3, 0),
            (3, 1),
            (3, 2),
            (4, 0),
            (4, 1),
        ]

        score_weights = [
            22,
            22,
            25,
            8,
            12,
            5,
            2,
            4,
        ]

    # Very strong favourite
    else:
        possible_scores = [
            (1, 0),
            (2, 0),
            (2, 1),
            (3, 0),
            (3, 1),
            (4, 0),
            (4, 1),
            (4, 2),
        ]

        score_weights = [
            10,
            25,
            12,
            20,
            15,
            8,
            7,
            3,
        ]

    return random.choices(
        population=possible_scores,
        weights=score_weights,
        k=1,
    )[0]


def generate_draw_score(
    draw_probability,
    home_probability,
    away_probability,
):
    """
    Generate a realistic draw score.

    Large mismatches mostly produce 0-0 or 1-1.

    Evenly matched teams are more likely to produce
    2-2 or occasionally 3-3.
    """

    strength_difference = abs(
        home_probability - away_probability
    )

    possible_scores = [
        (0, 0),
        (1, 1),
        (2, 2),
        (3, 3),
    ]

    # Huge mismatch
    if strength_difference >= 0.40:

        score_weights = [
            45,
            45,
            9,
            1,
        ]

    # Moderate mismatch
    elif strength_difference >= 0.20:

        score_weights = [
            30,
            50,
            18,
            2,
        ]

    # Very even teams
    else:

        if draw_probability >= 0.35:

            score_weights = [
                20,
                50,
                25,
                5,
            ]

        else:

            score_weights = [
                15,
                50,
                28,
                7,
            ]

    return random.choices(
        population=possible_scores,
        weights=score_weights,
        k=1,
    )[0]

def simulate_group_match(
    home_team,
    away_team,
    model,
    feature_columns,
    histories,
    elo_ratings,
    show_output=True,
):
    """
    Simulate one group-stage match.

    Group matches may end in a draw.
    """

    feature_values = create_prediction_features(
        home_team=home_team,
        away_team=away_team,
        neutral=True,
        histories=histories,
        elo_ratings=elo_ratings,
    )

    _, probabilities = predict_match(
        model=model,
        feature_columns=feature_columns,
        feature_values=feature_values,
    )

    away_probability = probabilities.get(0, 0)
    draw_probability = probabilities.get(1, 0)
    home_probability = probabilities.get(2, 0)

    sampled_outcome = random.choices(
        population=[0, 1, 2],
        weights=[
            away_probability,
            draw_probability,
            home_probability,
        ],
        k=1,
    )[0]

    if sampled_outcome == 2:
        home_score, away_score = generate_winning_score(
            winner_probability=home_probability,
            loser_probability=away_probability,
        )

    elif sampled_outcome == 0:
        away_score, home_score = generate_winning_score(
            winner_probability=away_probability,
            loser_probability=home_probability,
        )

    else:
        home_score, away_score = generate_draw_score(
            draw_probability=draw_probability,
            home_probability=home_probability,
            away_probability=away_probability,
        )

    if show_output:
        print(
            f"{home_team} {home_score}–{away_score} {away_team}"
        )

    return home_score, away_score


def update_group_table(
    table,
    home_team,
    away_team,
    home_score,
    away_score,
):
    """
    Update the standings after one match.
    """

    table[home_team]["played"] += 1
    table[away_team]["played"] += 1

    table[home_team]["goals_for"] += home_score
    table[home_team]["goals_against"] += away_score

    table[away_team]["goals_for"] += away_score
    table[away_team]["goals_against"] += home_score

    if home_score > away_score:
        table[home_team]["wins"] += 1
        table[home_team]["points"] += 3

        table[away_team]["losses"] += 1

    elif home_score < away_score:
        table[away_team]["wins"] += 1
        table[away_team]["points"] += 3

        table[home_team]["losses"] += 1

    else:
        table[home_team]["draws"] += 1
        table[away_team]["draws"] += 1

        table[home_team]["points"] += 1
        table[away_team]["points"] += 1

    for team in [home_team, away_team]:
        table[team]["goal_difference"] = (
            table[team]["goals_for"]
            - table[team]["goals_against"]
        )











def update_tournament_state(
    home_team,
    away_team,
    home_score,
    away_score,
    histories,
    elo_ratings,
):
    """
    Update recent form and Elo ratings after a
    simulated tournament match.

    The dictionary keys must match the format used
    by features.py.
    """

    # Determine match statistics for both teams
    if home_score > away_score:
        home_won = 1
        away_won = 0

        home_drawn = 0
        away_drawn = 0

        home_points = 3
        away_points = 0

        actual_home_score = 1.0

    elif home_score < away_score:
        home_won = 0
        away_won = 1

        home_drawn = 0
        away_drawn = 0

        home_points = 0
        away_points = 3

        actual_home_score = 0.0

    else:
        home_won = 0
        away_won = 0

        home_drawn = 1
        away_drawn = 1

        home_points = 1
        away_points = 1

        actual_home_score = 0.5

    # Add the simulated match to the home team's history
    histories[home_team].append(
        {
            "won": home_won,
            "drawn": home_drawn,
            "goals_scored": home_score,
            "goals_conceded": away_score,
            "points": home_points,
        }
    )

    # Add the simulated match to the away team's history
    histories[away_team].append(
        {
            "won": away_won,
            "drawn": away_drawn,
            "goals_scored": away_score,
            "goals_conceded": home_score,
            "points": away_points,
        }
    )

    # Store Elo values before updating them
    home_elo = elo_ratings[home_team]
    away_elo = elo_ratings[away_team]

    # Calculate the expected result
    expected_home_score = (
        1
        /
        (
            1
            + 10 ** (
                (away_elo - home_elo) / 400
            )
        )
    )

    expected_away_score = 1 - expected_home_score
    actual_away_score = 1 - actual_home_score

    elo_k_factor = 30

    # Update both Elo ratings
    elo_ratings[home_team] = (
        home_elo
        + elo_k_factor
        * (
            actual_home_score
            - expected_home_score
        )
    )

    elo_ratings[away_team] = (
        away_elo
        + elo_k_factor
        * (
            actual_away_score
            - expected_away_score
        )
    )







def rank_group(table):
    """
    Rank teams using:

    1. Points
    2. Goal difference
    3. Goals scored
    4. Team name
    """

    ranked_teams = sorted(
        table.items(),
        key=lambda item: (
            item[1]["points"],
            item[1]["goal_difference"],
            item[1]["goals_for"],
            item[0],
        ),
        reverse=True,
    )

    return ranked_teams




def get_qualified_teams(group_results):
    """
    Select the 32 teams that qualify for the knockout stage.

    Qualification:
    - Top 2 teams from each of the 12 groups
    - Best 8 third-place teams
    """

    group_winners = []
    runners_up = []
    third_place_teams = []

    for group_name, ranked_teams in group_results.items():
        group_winners.append(ranked_teams[0][0])
        runners_up.append(ranked_teams[1][0])

        third_place_team, third_place_stats = ranked_teams[2]

        third_place_teams.append(
            (
                third_place_team,
                third_place_stats,
                group_name,
            )
        )

    third_place_teams.sort(
        key=lambda item: (
            item[1]["points"],
            item[1]["goal_difference"],
            item[1]["goals_for"],
            item[0],
        ),
        reverse=True,
    )

    best_third_place_teams = [
        team
        for team, stats, group_name
        in third_place_teams[:8]
    ]

    qualified_teams = (
        group_winners
        + runners_up
        + best_third_place_teams
    )

    return {
        "group_winners": group_winners,
        "runners_up": runners_up,
        "best_third_place": best_third_place_teams,
        "all_qualified": qualified_teams,
    }




def display_group_table(group_name, ranked_teams):
    """
    Display the final standings for one group.
    """

    print(f"\nGroup {group_name} Standings\n")

    print(
        f"{'Pos':<4}"
        f"{'Team':<18}"
        f"{'P':>3}"
        f"{'W':>4}"
        f"{'D':>4}"
        f"{'L':>4}"
        f"{'GF':>5}"
        f"{'GA':>5}"
        f"{'GD':>5}"
        f"{'Pts':>6}"
    )

    print("-" * 58)

    for position, (team, stats) in enumerate(
        ranked_teams,
        start=1,
    ):
        print(
            f"{position:<4}"
            f"{team:<18}"
            f"{stats['played']:>3}"
            f"{stats['wins']:>4}"
            f"{stats['draws']:>4}"
            f"{stats['losses']:>4}"
            f"{stats['goals_for']:>5}"
            f"{stats['goals_against']:>5}"
            f"{stats['goal_difference']:>5}"
            f"{stats['points']:>6}"
        )








def simulate_group(
    group_name,
    teams,
    model,
    feature_columns,
    histories,
    elo_ratings,
    show_output=True,
):
    """
    Simulate every match in one group.

    Four teams produce six matches.
    """

    if show_output:
        print("\n===================================")
        print(f"              GROUP {group_name}")
        print("===================================\n")

    table = create_group_table(teams)

    group_matches = combinations(teams, 2)

    for home_team, away_team in group_matches:
        home_score, away_score = simulate_group_match(
            home_team=home_team,
            away_team=away_team,
            model=model,
            feature_columns=feature_columns,
            histories=histories,
            elo_ratings=elo_ratings,
            show_output=show_output,
        )

        update_group_table(
            table=table,
            home_team=home_team,
            away_team=away_team,
            home_score=home_score,
            away_score=away_score,
        )

        update_tournament_state(
            home_team=home_team,
            away_team=away_team,
            home_score=home_score,
            away_score=away_score,
            histories=histories,
            elo_ratings=elo_ratings,
        )

    ranked_teams = rank_group(table)

    if show_output:
        display_group_table(
            group_name=group_name,
            ranked_teams=ranked_teams,
        )

    return ranked_teams









def choose_winner_after_draw(
    home_team,
    away_team,
    feature_values,
):
    """
    Resolve a knockout match that is drawn.

    Elo ratings determine each team's chance of winning
    extra time or penalties.
    """

    home_elo = feature_values["home_elo"]
    away_elo = feature_values["away_elo"]

    home_win_probability = (
        1
        /
        (
            1
            + 10 ** ((away_elo - home_elo) / 400)
        )
    )

    if random.random() < home_win_probability:
        return home_team

    return away_team










def simulate_knockout_match(
    home_team,
    away_team,
    model,
    feature_columns,
    histories,
    elo_ratings,
    show_output=True,
):
    """
    Simulate one knockout match.

    A knockout match must produce a winner.
    """

    feature_values = create_prediction_features(
        home_team=home_team,
        away_team=away_team,
        neutral=True,
        histories=histories,
        elo_ratings=elo_ratings,
    )

    _, probabilities = predict_match(
        model=model,
        feature_columns=feature_columns,
        feature_values=feature_values,
    )

    away_probability = probabilities.get(0, 0)
    draw_probability = probabilities.get(1, 0)
    home_probability = probabilities.get(2, 0)

    sampled_outcome = random.choices(
        population=[0, 1, 2],
        weights=[
            away_probability,
            draw_probability,
            home_probability,
        ],
        k=1,
    )[0]

    if show_output:
        print(f"\n{home_team} vs {away_team}")

        print(
            f"{home_team} win: "
            f"{home_probability * 100:.2f}%"
        )

        print(
            f"Draw: "
            f"{draw_probability * 100:.2f}%"
        )

        print(
            f"{away_team} win: "
            f"{away_probability * 100:.2f}%"
        )

    if sampled_outcome == 2:
        home_score, away_score = generate_winning_score(
            winner_probability=home_probability,
            loser_probability=away_probability,
        )

        winner = home_team

        if show_output:
            print(
                f"Score: {home_team} {home_score}"
                f"–{away_score} {away_team}"
            )
            print(f"Winner: {winner}")

    elif sampled_outcome == 0:
        away_score, home_score = generate_winning_score(
            winner_probability=away_probability,
            loser_probability=home_probability,
        )

        winner = away_team

        if show_output:
            print(
                f"Score: {home_team} {home_score}"
                f"–{away_score} {away_team}"
            )
            print(f"Winner: {winner}")

    else:
        home_score, away_score = generate_draw_score(
            draw_probability=draw_probability,
            home_probability=home_probability,
            away_probability=away_probability,
        )

        winner = choose_winner_after_draw(
            home_team=home_team,
            away_team=away_team,
            feature_values=feature_values,
        )

        if show_output:
            print(
                f"Score after normal time: "
                f"{home_team} {home_score}"
                f"–{away_score} {away_team}"
            )
            print(
                f"{winner} wins after extra time or penalties"
            )

    update_tournament_state(
        home_team=home_team,
        away_team=away_team,
        home_score=home_score,
        away_score=away_score,
        histories=histories,
        elo_ratings=elo_ratings,
    )

    return winner


def create_quarterfinal_matches(group_results):
    """
    Create quarterfinal matchups.

    Group winners play runners-up from other groups.
    """

    group_a_winner = group_results["A"][0][0]
    group_a_runner_up = group_results["A"][1][0]

    group_b_winner = group_results["B"][0][0]
    group_b_runner_up = group_results["B"][1][0]

    group_c_winner = group_results["C"][0][0]
    group_c_runner_up = group_results["C"][1][0]

    group_d_winner = group_results["D"][0][0]
    group_d_runner_up = group_results["D"][1][0]

    quarterfinal_matches = [
        (
            group_a_winner,
            group_b_runner_up,
        ),
        (
            group_c_winner,
            group_d_runner_up,
        ),
        (
            group_b_winner,
            group_a_runner_up,
        ),
        (
            group_d_winner,
            group_c_runner_up,
        ),
    ]

    return quarterfinal_matches











def simulate_knockout_round(
    matches,
    round_name,
    model,
    feature_columns,
    histories,
    elo_ratings,
    show_output=True,
):
    """
    Simulate every match in one knockout round.
    """

    if show_output:
        print("\n===================================")
        print(f"             {round_name.upper()}")
        print("===================================")

    winners = []

    for home_team, away_team in matches:
        winner = simulate_knockout_match(
            home_team=home_team,
            away_team=away_team,
            model=model,
            feature_columns=feature_columns,
            histories=histories,
            elo_ratings=elo_ratings,
            show_output=show_output,
        )

        winners.append(winner)

    return winners







def create_semifinal_matches(quarterfinal_winners):
    """
    Create two semifinal matches.
    """

    return [
        (
            quarterfinal_winners[0],
            quarterfinal_winners[1],
        ),
        (
            quarterfinal_winners[2],
            quarterfinal_winners[3],
        ),
    ]








def create_final_match(semifinal_winners):
    """
    Create the final match.
    """

    return [
        (
            semifinal_winners[0],
            semifinal_winners[1],
        )
    ]








def simulate_tournament(
    tournament_teams,
    model,
    feature_columns,
    base_histories,
    base_elo_ratings,
    show_output=True,
):
    """
    Simulate one complete tournament.

    Historical team states are copied at the beginning so every
    simulation starts from the same real-world historical state.

    Returns:
        champion
        runner_up
    """

    # Copy the historical match histories.
    # Each tournament can then modify its own copy.
    histories = defaultdict(
        lambda: deque(maxlen=FORM_WINDOW),
        {
            team: deque(
                team_history,
                maxlen=FORM_WINDOW,
            )
            for team, team_history
            in base_histories.items()
        },
    )

    # Copy the historical Elo ratings.
    elo_ratings = defaultdict(
        lambda: INITIAL_ELO,
        dict(base_elo_ratings),
    )

    groups = create_groups(
        tournament_teams
    )

    if show_output:
        display_groups(groups)

    group_results = {}

    for group_name, teams in groups.items():
        ranked_teams = simulate_group(
            group_name=group_name,
            teams=teams,
            model=model,
            feature_columns=feature_columns,
            histories=histories,
            elo_ratings=elo_ratings,
            show_output=show_output,
        )

        group_results[group_name] = ranked_teams

    if show_output:
        print("\n===================================")
        print("          GROUP STAGE COMPLETE")
        print("===================================")

        print("\nQualified teams:\n")

        for group_name, ranked_teams in group_results.items():
            group_winner = ranked_teams[0][0]
            runner_up = ranked_teams[1][0]

            print(
                f"Group {group_name}: "
                f"{group_winner}, {runner_up}"
            )

    quarterfinal_matches = create_quarterfinal_matches(
        group_results
    )

    quarterfinal_winners = simulate_knockout_round(
        matches=quarterfinal_matches,
        round_name="Quarterfinals",
        model=model,
        feature_columns=feature_columns,
        histories=histories,
        elo_ratings=elo_ratings,
        show_output=show_output,
    )

    semifinal_matches = create_semifinal_matches(
        quarterfinal_winners
    )

    semifinal_winners = simulate_knockout_round(
        matches=semifinal_matches,
        round_name="Semifinals",
        model=model,
        feature_columns=feature_columns,
        histories=histories,
        elo_ratings=elo_ratings,
        show_output=show_output,
    )

    final_match = create_final_match(
        semifinal_winners
    )

    final_winners = simulate_knockout_round(
        matches=final_match,
        round_name="Final",
        model=model,
        feature_columns=feature_columns,
        histories=histories,
        elo_ratings=elo_ratings,
        show_output=show_output,
    )

    champion = final_winners[0]

    runner_up = (
        semifinal_winners[1]
        if champion == semifinal_winners[0]
        else semifinal_winners[0]
    )

    if show_output:
        print("\n===================================")
        print("         TOURNAMENT COMPLETE")
        print("===================================\n")

        print(f"Champion: {champion}")
        print(f"Runner-up: {runner_up}")

        print("\n===================================")

    return champion, runner_up










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

    simulate_tournament(
        tournament_teams=tournament_teams,
        model=model,
        feature_columns=feature_columns,
        base_histories=histories,
        base_elo_ratings=elo_ratings,
        show_output=True,
    )


if __name__ == "__main__":
    main()