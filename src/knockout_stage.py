GROUP_NAMES = list("ABCDEFGHIJKL")


def _find_group_safe_assignment(left_labels, right_labels):
    """
    Assign every left-side qualifier to one right-side qualifier
    without creating a same-group match.
    """

    assignments = []
    used_right_labels = set()

    def search(index):
        if index == len(left_labels):
            return True

        left_label = left_labels[index]

        for right_label in right_labels:
            if right_label in used_right_labels:
                continue

            if left_label[0] == right_label[0]:
                continue

            used_right_labels.add(right_label)
            assignments.append((left_label, right_label))

            if search(index + 1):
                return True

            assignments.pop()
            used_right_labels.remove(right_label)

        return False

    if not search(0):
        raise ValueError(
            "Could not create valid Round-of-32 pairings."
        )

    return assignments


def create_round_of_32(qualified):
    """
    Create a deterministic custom Round-of-32 bracket.

    Structure:
    - Eight group winners play the eight qualified third-place teams.
    - Four remaining group winners play four runners-up.
    - The eight remaining runners-up play one another.
    - Same-group rematches are avoided.

    This is a project-specific bracket, not FIFA's official
    third-place permutation table.
    """

    qualifier_lookup = {}

    for category in (
        "group_winners",
        "runners_up",
        "best_third_place",
    ):
        for qualifier in qualified[category]:
            label = (
                f"{qualifier['group']}"
                f"{qualifier['position']}"
            )
            qualifier_lookup[label] = qualifier["team"]

    winner_labels = [
        f"{group}1"
        for group in GROUP_NAMES
    ]
    runner_labels = [
        f"{group}2"
        for group in GROUP_NAMES
    ]
    third_labels = sorted(
        f"{qualifier['group']}3"
        for qualifier in qualified["best_third_place"]
    )

    if len(third_labels) != 8:
        raise ValueError(
            "Exactly eight third-place teams must qualify."
        )

    winner_vs_third = _find_group_safe_assignment(
        left_labels=winner_labels[:8],
        right_labels=third_labels,
    )

    winner_vs_runner = _find_group_safe_assignment(
        left_labels=winner_labels[8:],
        right_labels=runner_labels,
    )

    used_runner_labels = {
        runner_label
        for _, runner_label in winner_vs_runner
    }

    remaining_runner_labels = [
        runner_label
        for runner_label in runner_labels
        if runner_label not in used_runner_labels
    ]

    runner_vs_runner = []

    while remaining_runner_labels:
        first_label = remaining_runner_labels.pop(0)

        opponent_index = next(
            (
                index
                for index, candidate_label
                in enumerate(remaining_runner_labels)
                if candidate_label[0] != first_label[0]
            ),
            None,
        )

        if opponent_index is None:
            raise ValueError(
                "Could not safely pair the remaining runners-up."
            )

        second_label = remaining_runner_labels.pop(
            opponent_index
        )

        runner_vs_runner.append(
            (first_label, second_label)
        )

    label_matches = (
        winner_vs_third
        + winner_vs_runner
        + runner_vs_runner
    )

    matches = [
        (
            qualifier_lookup[home_label],
            qualifier_lookup[away_label],
        )
        for home_label, away_label in label_matches
    ]

    if len(matches) != 16:
        raise ValueError(
            f"Round of 32 must contain 16 matches, "
            f"found {len(matches)}."
        )

    teams = [
        team
        for match in matches
        for team in match
    ]

    if len(teams) != 32 or len(set(teams)) != 32:
        raise ValueError(
            "Every qualified team must appear exactly once "
            "in the Round of 32."
        )

    return matches


def create_next_round_matches(winners):
    """
    Pair consecutive winners to create the next knockout round.

    Examples:
    - 16 winners -> 8 Round-of-16 matches
    - 8 winners -> 4 quarterfinals
    - 4 winners -> 2 semifinals
    - 2 winners -> 1 final
    """

    if len(winners) < 2:
        raise ValueError(
            "At least two teams are required to create a round."
        )

    if len(winners) % 2 != 0:
        raise ValueError(
            "The number of winners must be even."
        )

    return [
        (winners[index], winners[index + 1])
        for index in range(0, len(winners), 2)
    ]