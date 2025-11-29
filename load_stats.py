# load_stats.py
import nflreadpy as nfl
import polars as pl
import numpy as np


def load_nfl_dataset(
    seasons=range(2015, 2025),
    positions=("QB", "RB", "WR", "TE"),
):
    """
    Load player stats and return (X, y) as numpy arrays.

    y = [passing_yards, rushing_yards, receiving_yards, receptions]
    X = simple numeric/context features (you can expand this later).
    """
    seasons = list(seasons)

    # Load polars DataFrame from nflreadpy
    player_stats = nfl.load_player_stats(seasons)


    # Columns we care about right now
    cols = [
        "player_id",
        "player_name",
        "player_display_name",
        "position",
        "team",
        "opponent_team",
        "season",
        "week",
        "passing_yards",
        "rushing_yards",
        "receiving_yards",
        "receptions",
    ]

    player_stats_small = player_stats.select(cols)

    # Filter positions
    player_stats_small = player_stats_small.filter(
        pl.col("position").str.to_uppercase().is_in(list(positions))
    )

    # Fill null,target columns with 0 for now
    target_cols = [
        "passing_yards",
        "rushing_yards",
        "receiving_yards",
        "receptions",
    ]

    player_stats_small = player_stats_small.with_columns(
        [pl.col(c).fill_null(0) for c in target_cols]
    )

    # === Build features X ===
    # Start very simple:
    # - season (normalized-ish)
    # - week
    # - one-hot-ish position (QB/RB/WR/TE)
    # - previous-game stats could be added later

    # Position one-hot
    pos_exprs = []
    pos_names = ["QB", "RB", "WR", "TE"]
    for p in pos_names:
        pos_exprs.append(
            (pl.col("position") == p).cast(pl.Int32).alias(f"is_{p.lower()}")
        )

    df_features = player_stats_small.with_columns(
        *pos_exprs,
        # Normalize-ish season by subtracting 2000 so numbers are smaller
        (pl.col("season") - 2000).alias("season_norm"),
    ).select(
        [
            "season_norm",
            "week",
            "is_qb",
            "is_rb",
            "is_wr",
            "is_te",
        ]
    )

    # Targets
    df_targets = player_stats_small.select(target_cols)

    # Convert to numpy directly (no pandas, no pyarrow)
    X = df_features.to_numpy().astype(np.float32)
    y = df_targets.to_numpy().astype(np.float32)

    return X, y


if __name__ == "__main__":
    X, y = load_nfl_dataset()
    print("X shape:", X.shape)
    print("y shape:", y.shape)
