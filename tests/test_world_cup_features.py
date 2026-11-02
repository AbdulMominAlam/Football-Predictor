import pandas as pd
import pytest

from features import (
    ELO_K_FACTOR,
    FORM_WINDOW,
    HOME_ADVANTAGE,
    INITIAL_ELO,
    calculate_team_form,
    create_features,
    create_target,
    expected_score,
    update_elo,
)


# ---------------------------------------------------------------------------
# Elo
# ---------------------------------------------------------------------------


def test_expected_score_equal_ratings_is_half():
    assert expected_score(1500, 1500) == pytest.approx(0.5)


def test_expected_score_400_point_gap():
    # A 400-point gap means 10:1 odds.
    assert expected_score(1900, 1500) == pytest.approx(10 / 11)
    assert expected_score(1500, 1900) == pytest.approx(1 / 11)


@pytest.mark.parametrize("a, b", [(1500, 1500), (1720, 1480), (1300, 1850)])
def test_expected_scores_are_complementary(a, b):
    assert expected_score(a, b) + expected_score(b, a) == pytest.approx(1.0)


def test_expected_score_increases_with_rating():
    assert expected_score(1600, 1500) > expected_score(1550, 1500)


def test_update_elo_neutral_win_between_equal_teams():
    home, away = update_elo(1500, 1500, 2, 0, neutral=True)

    assert home == pytest.approx(1500 + ELO_K_FACTOR / 2)
    assert away == pytest.approx(1500 - ELO_K_FACTOR / 2)


def test_update_elo_neutral_draw_between_equal_teams_is_unchanged():
    home, away = update_elo(1500, 1500, 1, 1, neutral=True)

    assert home == pytest.approx(1500)
    assert away == pytest.approx(1500)


def test_update_elo_home_advantage_applied_when_not_neutral():
    home, away = update_elo(1500, 1500, 1, 1, neutral=False)

    expected_home = expected_score(1500 + HOME_ADVANTAGE, 1500)
    change = ELO_K_FACTOR * (0.5 - expected_home)

    # The home side was favoured, so a draw costs it rating.
    assert home == pytest.approx(1500 + change)
    assert away == pytest.approx(1500 - change)
    assert home < 1500 < away


@pytest.mark.parametrize(
    "home_rating, away_rating, home_score, away_score, neutral",
    [
        (1500, 1500, 3, 0, True),
        (1650, 1420, 0, 1, False),
        (1380, 1710, 2, 2, False),
    ],
)
def test_update_elo_is_zero_sum(
    home_rating, away_rating, home_score, away_score, neutral
):
    home, away = update_elo(
        home_rating, away_rating, home_score, away_score, neutral
    )

    assert home + away == pytest.approx(home_rating + away_rating)


def test_update_elo_upset_moves_more_than_expected_win():
    favourite_win, _ = update_elo(1800, 1400, 1, 0, neutral=True)
    underdog_win, _ = update_elo(1400, 1800, 1, 0, neutral=True)

    assert underdog_win - 1400 > favourite_win - 1800 > 0


# ---------------------------------------------------------------------------
# Target and form
# ---------------------------------------------------------------------------


def test_create_target_encodes_results_and_does_not_mutate():
    data = pd.DataFrame(
        {"home_score": [2, 1, 0], "away_score": [0, 1, 3]}
    )

    result = create_target(data)

    assert result["result"].tolist() == [2, 1, 0]
    assert "result" not in data.columns


def test_calculate_team_form():
    history = [
        {"won": 1, "drawn": 0, "points": 3, "goals_scored": 3, "goals_conceded": 1},
        {"won": 0, "drawn": 1, "points": 1, "goals_scored": 1, "goals_conceded": 1},
        {"won": 0, "drawn": 0, "points": 0, "goals_scored": 0, "goals_conceded": 2},
        {"won": 1, "drawn": 0, "points": 3, "goals_scored": 2, "goals_conceded": 0},
    ]

    form = calculate_team_form(history)

    assert form["win_rate"] == pytest.approx(0.5)
    assert form["draw_rate"] == pytest.approx(0.25)
    assert form["points_per_match"] == pytest.approx(7 / 4)
    assert form["average_goals_scored"] == pytest.approx(6 / 4)
    assert form["average_goals_conceded"] == pytest.approx(4 / 4)
    assert form["goal_difference_per_match"] == pytest.approx(2 / 4)


# ---------------------------------------------------------------------------
# create_features
# ---------------------------------------------------------------------------


def _alternating_matches(count, home_score=1, away_score=0, neutral=True):
    """Team A and B alternate home/away; the home side always wins 1-0."""
    rows = []
    dates = pd.date_range("2020-01-01", periods=count, freq="7D")

    for index, date in enumerate(dates):
        home, away = ("A", "B") if index % 2 == 0 else ("B", "A")
        rows.append(
            {
                "date": date,
                "home_team": home,
                "away_team": away,
                "home_score": home_score,
                "away_score": away_score,
                "neutral": neutral,
            }
        )

    return create_target(pd.DataFrame(rows))


def test_create_features_skips_until_both_teams_have_full_history():
    data = _alternating_matches(FORM_WINDOW + 3)

    result = create_features(data)

    # The first FORM_WINDOW matches only build history.
    assert len(result) == 3
    assert result["date"].iloc[0] == data["date"].iloc[FORM_WINDOW]


def test_create_features_uses_only_pre_match_information():
    data = _alternating_matches(FORM_WINDOW + 1)

    # Replay the Elo updates for the first FORM_WINDOW matches by hand.
    ratings = {"A": INITIAL_ELO, "B": INITIAL_ELO}
    for _, match in data.iloc[:FORM_WINDOW].iterrows():
        new_home, new_away = update_elo(
            ratings[match["home_team"]],
            ratings[match["away_team"]],
            match["home_score"],
            match["away_score"],
            neutral=True,
        )
        ratings[match["home_team"]] = new_home
        ratings[match["away_team"]] = new_away

    result = create_features(data)
    row = result.iloc[0]
    last = data.iloc[FORM_WINDOW]

    # Elo features equal the ratings *before* the match is played.
    assert row["home_elo"] == pytest.approx(ratings[last["home_team"]])
    assert row["away_elo"] == pytest.approx(ratings[last["away_team"]])
    assert row["elo_difference"] == pytest.approx(
        row["home_elo"] - row["away_elo"]
    )

    # A played 0,2,4 at home (won) and 1,3 away (lost): 3 wins in 5.
    assert last["home_team"] == "B"
    assert row["away_win_rate_last_5"] == pytest.approx(3 / 5)
    assert row["home_win_rate_last_5"] == pytest.approx(2 / 5)
    assert row["win_rate_difference"] == pytest.approx(-1 / 5)
    assert row["result"] == 2


def test_create_features_neutral_flag_and_no_missing_values():
    data = _alternating_matches(FORM_WINDOW + 2, neutral=False)

    result = create_features(data)

    assert (result["neutral"] == 0).all()
    assert not result.isna().any().any()
