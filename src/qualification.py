def get_qualified_teams(group_results):
    """
    Select the 32 teams that advance from the group stage.

    Qualification:
    - 12 group winners
    - 12 runners-up
    - 8 best third-place teams
    """

    group_winners = []
    runners_up = []
    third_place_candidates = []

    for group_name, ranked_teams in group_results.items():
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

    third_place_candidates.sort(
        key=lambda qualifier: (
            qualifier["stats"]["points"],
            qualifier["stats"]["goal_difference"],
            qualifier["stats"]["goals_for"],
            qualifier["team"],
        ),
        reverse=True,
    )

    best_third_place = third_place_candidates[:8]

    all_qualified = (
        group_winners
        + runners_up
        + best_third_place
    )

    if len(all_qualified) != 32:
        raise ValueError(
            f"Exactly 32 teams must qualify, found {len(all_qualified)}."
        )

    return {
        "group_winners": group_winners,
        "runners_up": runners_up,
        "best_third_place": best_third_place,
        "all_qualified": all_qualified,
    }