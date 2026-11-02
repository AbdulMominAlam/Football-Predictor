import pandas as pd
import pytest

import premier_league_features as plf
from premier_league_features import (
    ELO_K_FACTOR,
    FEATURE_COLUMNS,
    HOME_ADVANTAGE,
    INITIAL_ELO,
    SEASON_REGRESSION,
    PremierLeagueFeatureBuilder,
)


def make_match(season, date, home, away, home_goals, away_goals):
    return {
        "season": season,
        "date": date,
        "home_team": home,
        "away_team": away,
        "home_goals": home_goals,
        "away_goals": away_goals,
    }


@pytest.fixture
def builder():
    return PremierLeagueFeatureBuilder()


# ---------------------------------------------------------------------------
# Elo
# ---------------------------------------------------------------------------


def test_expected_home_score_includes_home_advantage():
    expected = PremierLeagueFeatureBuilder.expected_home_score(1500, 1500)

    assert expected == pytest.approx(
        1 / (1 + 10 ** (-HOME_ADVANTAGE / 400))
    )
    assert expected > 0.5


def test_expected_home_score_offsets_home_advantage():
    expected = PremierLeagueFeatureBuilder.expected_home_score(
        1500, 1500 + HOME_ADVANTAGE
    )

    assert expected == pytest.approx(0.5)


@pytest.mark.parametrize(
    "home_goals, away_goals, multiplier",
    [
        (1, 0, 1.0),
        (0, 0, 1.0),
        (2, 0, 1.5),
        (3, 0, 1.75),
        (5, 0, 2.0),
        (0, 4, 1.875),
    ],
)
def test_update_elo_goal_difference_multiplier(
    builder, home_goals, away_goals, multiplier
):
    expected_home = builder.expected_home_score(INITIAL_ELO, INITIAL_ELO)
    actual_home, _ = builder.result_values(home_goals, away_goals)

    builder.update_elo("Home", "Away", home_goals, away_goals)

    change = ELO_K_FACTOR * multiplier * (actual_home - expected_home)
    assert builder.elo_ratings["Home"] == pytest.approx(INITIAL_ELO + change)
    assert builder.elo_ratings["Away"] == pytest.approx(INITIAL_ELO - change)


def test_update_elo_is_zero_sum(builder):
    builder.elo_ratings["Home"] = 1580
    builder.elo_ratings["Away"] = 1460

    builder.update_elo("Home", "Away", 0, 3)

    assert (
        builder.elo_ratings["Home"] + builder.elo_ratings["Away"]
        == pytest.approx(1580 + 1460)
    )
    assert builder.elo_ratings["Home"] < 1580


def test_home_draw_between_equal_teams_lowers_home_elo(builder):
    builder.update_elo("Home", "Away", 1, 1)

    assert builder.elo_ratings["Home"] < INITIAL_ELO
    assert builder.elo_ratings["Away"] > INITIAL_ELO


def test_regress_elo_at_new_season(builder):
    builder.elo_ratings["Strong"] = 1700
    builder.elo_ratings["Weak"] = 1300

    builder.regress_elo_at_new_season()

    assert builder.elo_ratings["Strong"] == pytest.approx(
        1700 - (1700 - INITIAL_ELO) * SEASON_REGRESSION
    )
    assert builder.elo_ratings["Weak"] == pytest.approx(
        1300 + (INITIAL_ELO - 1300) * SEASON_REGRESSION
    )


def test_start_season_regresses_and_resets_only_on_change(builder):
    builder.start_season("2024-25")
    builder.elo_ratings["A"] = 1600
    builder.update_season_stats("A", "B", 2, 0)

    builder.start_season("2024-25")
    assert builder.elo_ratings["A"] == 1600
    assert builder.season_stats["A"]["matches"] == 1

    builder.start_season("2025-26")
    assert builder.elo_ratings["A"] == pytest.approx(
        1600 - 100 * SEASON_REGRESSION
    )
    assert builder.season_stats["A"]["matches"] == 0
    assert builder.current_season == "2025-26"


# ---------------------------------------------------------------------------
# Result helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "home_goals, away_goals, values, target, points",
    [
        (2, 1, (1.0, 0.0), "home_win", (3, 0)),
        (0, 2, (0.0, 1.0), "away_win", (0, 3)),
        (1, 1, (0.5, 0.5), "draw", (1, 1)),
    ],
)
def test_result_helpers(home_goals, away_goals, values, target, points):
    cls = PremierLeagueFeatureBuilder

    assert cls.result_values(home_goals, away_goals) == values
    assert cls.match_target(home_goals, away_goals) == target
    assert cls.match_points(home_goals, away_goals) == points


# ---------------------------------------------------------------------------
# Form, season and venue features
# ---------------------------------------------------------------------------


def test_recent_form_defaults_without_history(builder):
    form = builder.get_recent_form("Nobody", 5)

    assert form["win_rate"] == pytest.approx(1 / 3)
    assert form["points_per_match"] == 1.0
    assert form["goal_difference"] == 0.0


def test_recent_form_window(builder):
    # Home team: W 3-0, D 1-1, L 0-2
    builder.update_recent_form("A", "B", 3, 0)
    builder.update_recent_form("A", "C", 1, 1)
    builder.update_recent_form("A", "D", 0, 2)

    full = builder.get_recent_form("A", 5)
    assert full["win_rate"] == pytest.approx(1 / 3)
    assert full["draw_rate"] == pytest.approx(1 / 3)
    assert full["points_per_match"] == pytest.approx(4 / 3)
    assert full["goals_scored"] == pytest.approx(4 / 3)
    assert full["goals_conceded"] == pytest.approx(1.0)
    assert full["goal_difference"] == pytest.approx(1 / 3)

    last_two = builder.get_recent_form("A", 2)
    assert last_two["win_rate"] == 0.0
    assert last_two["points_per_match"] == pytest.approx(0.5)

    # The away side's record is mirrored.
    assert builder.get_recent_form("B", 5)["goals_conceded"] == 3.0


def test_recent_results_capped_at_max_window(builder):
    for _ in range(plf.MAX_FORM_WINDOW + 5):
        builder.update_recent_form("A", "B", 1, 0)

    assert len(builder.recent_results["A"]) == plf.MAX_FORM_WINDOW


def test_season_and_venue_form(builder):
    builder.update_season_stats("A", "B", 2, 0)  # A home win
    builder.update_season_stats("C", "A", 1, 1)  # A away draw

    season = builder.get_season_form("A")
    assert season["matches"] == 2
    assert season["points_per_match"] == pytest.approx(2.0)
    assert season["goal_difference"] == pytest.approx(1.0)

    home = builder.get_venue_form("A", "home")
    away = builder.get_venue_form("A", "away")
    assert home == {"points_per_match": 3.0, "goal_difference": 2.0}
    assert away == {"points_per_match": 1.0, "goal_difference": 0.0}

    # B lost its only (away) match; with no home matches it gets defaults.
    assert builder.get_venue_form("B", "away") == {
        "points_per_match": 0.0,
        "goal_difference": -2.0,
    }
    assert builder.get_venue_form("B", "home") == {
        "points_per_match": 1.0,
        "goal_difference": 0.0,
    }


@pytest.mark.parametrize(
    "days, expected",
    [(3, 3), (0, 0), (30, 30), (90, 30)],
)
def test_rest_days_are_clipped(builder, days, expected):
    previous = pd.Timestamp("2025-01-01")
    builder.last_match_dates["A"] = previous

    assert (
        builder.get_rest_days("A", previous + pd.Timedelta(days=days))
        == expected
    )


def test_rest_days_default_for_first_match(builder):
    assert builder.get_rest_days("A", pd.Timestamp("2025-01-01")) == 7


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def test_pre_match_features_cover_all_feature_columns(builder):
    features = builder.create_pre_match_features(
        "A", "B", pd.Timestamp("2025-08-16")
    )

    assert set(FEATURE_COLUMNS) == set(features)
    assert features["elo_difference"] == HOME_ADVANTAGE


def test_process_match_has_no_leakage(builder):
    row = builder.process_match(
        make_match("2025-26", "2025-08-16", "A", "B", 4, 0)
    )

    # Features describe the state before kick-off.
    assert row["home_elo"] == INITIAL_ELO
    assert row["away_elo"] == INITIAL_ELO
    assert row["home_season_matches"] == 0
    assert row["home_rest_days"] == 7
    assert row["target"] == "home_win"
    assert row["date"] == "2025-08-16"

    # State is updated after the row is produced.
    assert builder.elo_ratings["A"] > INITIAL_ELO
    assert builder.season_stats["A"]["matches"] == 1
    assert builder.last_match_dates["A"] == pd.Timestamp("2025-08-16")


def test_process_match_sequence(builder):
    builder.process_match(make_match("2025-26", "2025-08-16", "A", "B", 2, 0))
    row = builder.process_match(
        make_match("2025-26", "2025-08-23", "B", "A", 1, 1)
    )

    assert row["home_team"] == "B"
    assert row["home_rest_days"] == 7
    assert row["away_rest_days"] == 7
    assert row["home_season_points_per_match"] == 0.0
    assert row["away_season_points_per_match"] == 3.0
    assert row["season_points_difference"] == -3.0
    assert row["away_elo"] > row["home_elo"]
    assert row["target"] == "draw"


def test_process_match_new_season_resets_season_stats(builder):
    builder.process_match(make_match("2024-25", "2025-05-25", "A", "B", 3, 0))
    elo_after_first = builder.elo_ratings["A"]

    row = builder.process_match(
        make_match("2025-26", "2025-08-16", "A", "C", 0, 0)
    )

    assert row["home_season_matches"] == 0
    assert row["home_elo"] == pytest.approx(
        elo_after_first * (1 - SEASON_REGRESSION)
        + INITIAL_ELO * SEASON_REGRESSION
    )
    # Recent form carries across seasons.
    assert row["home_win_rate"] == 1.0


def test_load_completed_matches_filters_and_sorts(tmp_path, monkeypatch):
    csv_file = tmp_path / "matches.csv"
    pd.DataFrame(
        [
            ["2025-26", "2025-08-23", "C", "D", 1, 0, "played"],
            ["2025-26", "2025-08-16", "B", "A", 2, 2, "played"],
            ["2025-26", "2025-08-30", "A", "C", None, None, "scheduled"],
        ],
        columns=[
            "season", "date", "home_team", "away_team",
            "home_goals", "away_goals", "status",
        ],
    ).to_csv(csv_file, index=False)

    monkeypatch.setattr(plf, "INPUT_FILE", csv_file)

    matches = plf.load_completed_matches()

    assert len(matches) == 2
    assert matches["home_team"].tolist() == ["B", "C"]


def test_load_completed_matches_missing_columns(tmp_path, monkeypatch):
    csv_file = tmp_path / "matches.csv"
    pd.DataFrame({"season": ["2025-26"]}).to_csv(csv_file, index=False)

    monkeypatch.setattr(plf, "INPUT_FILE", csv_file)

    with pytest.raises(ValueError, match="Missing columns"):
        plf.load_completed_matches()
