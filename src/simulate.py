import random

from predict import (
    build_current_team_states,
    create_prediction_features,
    find_team_name,
    load_match_data,
    load_model,
    predict_match,
)


NUMBER_OF_TEAMS = 8


def ask_for_tournament_teams(known_teams):
    """
    Ask the user to enter eight valid and unique teams.
    """

    tournament_teams = []

    print(
        f"\nEnter {NUMBER_OF_TEAMS} teams for the tournament.\n"
    )

    while len(tournament_teams) < NUMBER_OF_TEAMS:
        team_number = len(tournament_teams) + 1

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

        if matched_team in tournament_teams:
            print(
                f"{matched_team} has already been added.\n"
            )
            continue

        tournament_teams.append(matched_team)

    return tournament_teams


def create_random_bracket(teams):
    """
    Shuffle the teams so the bracket is different
    each time the tournament is run.
    """

    shuffled_teams = teams.copy()

    random.shuffle(shuffled_teams)

    return shuffled_teams


def choose_winner_after_draw(
    home_team,
    away_team,
    feature_values,
):
    """
    Resolve a drawn knockout match using Elo ratings.

    The stronger team has a slightly higher probability
    of winning extra time or penalties.
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


def generate_winning_score():
    """
    Generate a realistic scoreline where one team wins.
    """

    winning_scores = [
        (1, 0),
        (2, 0),
        (2, 1),
        (3, 0),
        (3, 1),
        (3, 2),
        (4, 0),
        (4, 1),
        (4, 2),
        (4, 3),
    ]

    score_weights = [
        22,
        13,
        26,
        6,
        13,
        8,
        2,
        4,
        4,
        2,
    ]

    return random.choices(
        population=winning_scores,
        weights=score_weights,
        k=1,
    )[0]


def generate_draw_score():
    """
    Generate a realistic draw scoreline.
    """

    draw_scores = [
        (0, 0),
        (1, 1),
        (2, 2),
        (3, 3),
    ]

    draw_weights = [
        25,
        50,
        20,
        5,
    ]

    return random.choices(
        population=draw_scores,
        weights=draw_weights,
        k=1,
    )[0]


def simulate_match(
    home_team,
    away_team,
    model,
    feature_columns,
    histories,
    elo_ratings,
):
    """
    Simulate one knockout match.

    Tournament matches are treated as neutral-venue matches.
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
        home_score, away_score = generate_winning_score()

        winner = home_team

        print(
            f"Score: {home_team} {home_score}"
            f"–{away_score} {away_team}"
        )

        print(f"Winner: {winner}")

    elif sampled_outcome == 0:
        away_score, home_score = generate_winning_score()

        winner = away_team

        print(
            f"Score: {home_team} {home_score}"
            f"–{away_score} {away_team}"
        )

        print(f"Winner: {winner}")

    else:
        home_score, away_score = generate_draw_score()

        winner = choose_winner_after_draw(
            home_team=home_team,
            away_team=away_team,
            feature_values=feature_values,
        )

        print(
            f"Score after normal time: "
            f"{home_team} {home_score}"
            f"–{away_score} {away_team}"
        )

        print(
            f"{winner} wins after extra time or penalties"
        )

    return winner


def simulate_round(
    teams,
    round_name,
    model,
    feature_columns,
    histories,
    elo_ratings,
):
    """
    Simulate one complete knockout round.
    """

    print("\n===================================")
    print(f"             {round_name.upper()}")
    print("===================================")

    winners = []

    for index in range(0, len(teams), 2):
        home_team = teams[index]
        away_team = teams[index + 1]

        winner = simulate_match(
            home_team=home_team,
            away_team=away_team,
            model=model,
            feature_columns=feature_columns,
            histories=histories,
            elo_ratings=elo_ratings,
        )

        winners.append(winner)

    return winners


def display_teams(teams):
    """
    Display the tournament participants.
    """

    print("\n===================================")
    print("        TOURNAMENT TEAMS")
    print("===================================\n")

    for number, team in enumerate(
        teams,
        start=1,
    ):
        print(f"{number}. {team}")


def display_bracket(teams):
    """
    Display the randomized quarterfinal matchups.
    """

    print("\n===================================")
    print("         RANDOMIZED BRACKET")
    print("===================================\n")

    match_number = 1

    for index in range(0, len(teams), 2):
        home_team = teams[index]
        away_team = teams[index + 1]

        print(
            f"Quarterfinal {match_number}: "
            f"{home_team} vs {away_team}"
        )

        match_number += 1


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

    tournament_teams = ask_for_tournament_teams(
        known_teams
    )

    display_teams(tournament_teams)

    bracket_teams = create_random_bracket(
        tournament_teams
    )

    display_bracket(bracket_teams)

    quarterfinal_winners = simulate_round(
        teams=bracket_teams,
        round_name="Quarterfinals",
        model=model,
        feature_columns=feature_columns,
        histories=histories,
        elo_ratings=elo_ratings,
    )

    semifinal_winners = simulate_round(
        teams=quarterfinal_winners,
        round_name="Semifinals",
        model=model,
        feature_columns=feature_columns,
        histories=histories,
        elo_ratings=elo_ratings,
    )

    final_winners = simulate_round(
        teams=semifinal_winners,
        round_name="Final",
        model=model,
        feature_columns=feature_columns,
        histories=histories,
        elo_ratings=elo_ratings,
    )

    champion = final_winners[0]

    runner_up = (
        semifinal_winners[1]
        if champion == semifinal_winners[0]
        else semifinal_winners[0]
    )

    print("\n===================================")
    print("         TOURNAMENT COMPLETE")
    print("===================================\n")

    print(f"Champion: {champion}")
    print(f"Runner-up: {runner_up}")

    print("\n===================================")


if __name__ == "__main__":
    main()