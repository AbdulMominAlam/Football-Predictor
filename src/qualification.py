






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