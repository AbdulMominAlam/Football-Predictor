def create_round_of_32(qualified_teams):
    """
    Create the Round of 32 fixtures.

    Parameters:
        qualified_teams (list): 32 qualified teams

    Returns:
        list: List of (home_team, away_team) tuples
    """

    matches = []

    for i in range(0, len(qualified_teams), 2):
        matches.append(
            (
                qualified_teams[i],
                qualified_teams[i + 1],
            )
        )

    return matches




def create_next_round_matches(winners):
    """
    Pair the winners from one knockout round
    to create the next round's matches.

    Example:
        16 winners -> 8 matches
        8 winners -> 4 matches
        4 winners -> 2 matches
        2 winners -> 1 final
    """

    if len(winners) % 2 != 0:
        raise ValueError(
            "The number of winners must be even."
        )

    matches = []

    for i in range(0, len(winners), 2):
        matches.append(
            (
                winners[i],
                winners[i + 1],
            )
        )

    return matches