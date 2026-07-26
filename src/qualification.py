def get_qualified_teams(group_results):
    """
    Select the 32 teams that qualify from the group stage.

    Qualification:
    - Top two teams from each of the 12 groups = 24 teams
    - Best eight third-place teams = 8 teams

    Each qualifier keeps its group and finishing position so
    the official Round-of-32 bracket can be created later.
    """

    group_winners = []
    runners_up = []
    third_place_candidates = []

    for group_name, ranked_teams in group_results.items():
        # ranked_teams contains:
        # [
        #     (team_name, statistics),
        #     (team_name, statistics),
        #     ...
        # ]

        winner_team, winner_stats = ranked_teams[0]
        runner_up_team, runner_up_stats = ranked_teams[1]
        third_team, third_stats = ranked_teams[2]

        group_winners.append(
            {
                "team": winner_team,
                "group": group_name,
                "position": 1,
                "stats": winner_stats,
            }
        )

        runners_up.append(
            {
                "team": runner_up_team,
                "group": group_name,
                "position": 2,
                "stats": runner_up_stats,
            }
        )

        third_place_candidates.append(
            {
                "team": third_team,
                "group": group_name,
                "position": 3,
                "stats": third_stats,
            }
        )

    # Rank all 12 third-place teams.
    third_place_candidates.sort(
        key=lambda entry: (
            entry["stats"]["points"],
            entry["stats"]["goal_difference"],
            entry["stats"]["goals_for"],
            entry["team"],
        ),
        reverse=True,
    )

    # Only the best eight third-place teams qualify.
    best_third_place = third_place_candidates[:8]

    all_qualified = (
        group_winners
        + runners_up
        + best_third_place
    )

    return {
        "group_winners": group_winners,
        "runners_up": runners_up,
        "best_third_place": best_third_place,
        "all_qualified": all_qualified,
    }