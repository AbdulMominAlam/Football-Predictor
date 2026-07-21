from pathlib import Path

import pandas as pd


# Get the main project folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Input and output file paths
RAW_DATA_FILE = PROJECT_ROOT / "data" / "raw" / "results.csv"
CLEAN_DATA_FILE = PROJECT_ROOT / "data" / "processed" / "clean_results.csv"


def load_data():
    """
    Load the raw football match dataset.
    """

    if not RAW_DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {RAW_DATA_FILE}"
        )

    return pd.read_csv(RAW_DATA_FILE)


def clean_data(data):
    """
    Clean the dataset and remove 2026 FIFA World Cup matches.
    """

    # Remove rows with missing values and create a fresh DataFrame
    data = data.dropna().copy()

    # Convert the date column from text into datetime
    data["date"] = pd.to_datetime(data["date"])

    # Remove matches from the 2026 FIFA World Cup
    is_2026_world_cup = (
        data["tournament"].str.strip().eq("FIFA World Cup")
        & data["date"].dt.year.eq(2026)
    )

    data = data.loc[~is_2026_world_cup].copy()

    # Sort all matches from oldest to newest
    data = data.sort_values("date")

    # Reset the row numbers
    data = data.reset_index(drop=True)

    return data


def save_data(data):
    """
    Save the cleaned dataset inside data/processed.
    """

    CLEAN_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)

    data.to_csv(CLEAN_DATA_FILE, index=False)

    print(f"\nCleaned dataset saved to:\n{CLEAN_DATA_FILE}")


def inspect_data(data):
    """
    Print useful information about the cleaned dataset.
    """

    print("\n========== FIRST 5 ROWS ==========\n")
    print(data.head())

    print("\n========== DATASET SHAPE ==========\n")
    print(data.shape)

    print("\n========== MISSING VALUES ==========\n")
    print(data.isnull().sum())

    print("\n========== DATA TYPES ==========\n")
    print(data.dtypes)

    print("\n========== DATE RANGE ==========\n")
    print(data["date"].min(), "to", data["date"].max())

    remaining_2026_world_cup_matches = data[
        data["tournament"].str.strip().eq("FIFA World Cup")
        & data["date"].dt.year.eq(2026)
    ]

    print("\n========== 2026 WORLD CUP MATCHES REMAINING ==========\n")
    print(len(remaining_2026_world_cup_matches))


def main():
    football_data = load_data()

    football_data = clean_data(football_data)

    inspect_data(football_data)

    save_data(football_data)


if __name__ == "__main__":
    main()