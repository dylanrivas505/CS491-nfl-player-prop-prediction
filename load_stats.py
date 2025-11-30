# load_stats.py
import nflreadpy as nfl
import polars as pl
import numpy as np


def _rolling_mean(col_name: str, window: int = 3):
    """
    Convenience helper for player-level rolling mean of the previous `window` games.
    Shift by 1 so we do not leak the current game's target into features.
    """
    return (
        pl.col(col_name)
        .shift(1)
        .rolling_mean(window_size=window, min_periods=1)
        .over("player_id")
        .alias(f"{col_name}_prev{window}")
    )


def load_nfl_dataset(
    seasons=range(2015, 2025),
    positions=("QB", "RB", "WR", "TE"),
    normalize=True,
    return_feature_names=False,
    return_norm_stats=False,
    return_target_mask=False,
    return_positions=False,
    return_target_norm_stats=False,
):
    """
    Load player stats and return (X, y) as numpy arrays.

    y = [passing_yards, rushing_yards, receiving_yards, receptions]
    X = richer numeric/context features with position-aware masking:
        - season/week, position one-hots
        - per-game volume/efficiency (attempts, carries, targets, EPA, CPOE)
        - rolling means of prior games (passing/rushing/receiving yards, receptions, targets)
        - target share and air yards share for pass catchers
        - position-specific masking zeros out irrelevant stats (e.g., passing metrics for WR/TE)
    """
    seasons = list(seasons)

    # Load polars DataFrame from nflreadpy
    player_stats = nfl.load_player_stats(seasons)

    target_cols = [
        "passing_yards",
        "rushing_yards",
        "receiving_yards",
        "receptions",
    ]

    # Keep the columns we need for features/targets
    cols = [
        "player_id",
        "position",
        "team",
        "opponent_team",
        "season",
        "week",
        "completions",
        "attempts",
        "passing_yards",
        "passing_tds",
        "passing_interceptions",
        "passing_yards_after_catch",
        "passing_epa",
        "passing_cpoe",
        "carries",
        "rushing_yards",
        "rushing_tds",
        "rushing_epa",
        "receptions",
        "targets",
        "receiving_yards",
        "receiving_tds",
        "receiving_air_yards",
        "receiving_yards_after_catch",
        "receiving_epa",
        "target_share",
        "air_yards_share",
        "wopr",
        "fantasy_points",
    ]

    player_stats_small = player_stats.select(cols)

    # Filter positions of interest
    player_stats_small = player_stats_small.filter(
        pl.col("position").str.to_uppercase().is_in(list(positions))
    )

    # Fill nulls on columns we will use directly
    fill_zero_cols = target_cols + [
        "completions",
        "attempts",
        "passing_tds",
        "passing_interceptions",
        "passing_yards_after_catch",
        "passing_epa",
        "passing_cpoe",
        "carries",
        "rushing_tds",
        "rushing_epa",
        "targets",
        "receiving_tds",
        "receiving_air_yards",
        "receiving_yards_after_catch",
        "receiving_epa",
        "target_share",
        "air_yards_share",
        "wopr",
        "fantasy_points",
    ]
    player_stats_small = player_stats_small.with_columns(
        [pl.col(c).fill_null(0) for c in fill_zero_cols]
    )
    # Track original row order for joins
    player_stats_small = player_stats_small.with_row_count(name="row_id")

    # Opponent defensive “so far” stats (running averages excluding current game)
    opp_group = ["opponent_team", "season"]
    def _opp_allowed_avg(col: str):
        cum = (
            pl.col(col)
            .shift(1)
            .fill_null(0)
            .cumsum()
            .over(opp_group)
        )
        games = pl.cum_count().over(opp_group)
        denom = pl.when(games > 0).then(games).otherwise(1)
        return (cum / denom).alias(f"opp_{col}_allowed_avg")

    opp_sorted = (
        player_stats_small.sort(opp_group + ["week"])
        .with_columns(
            _opp_allowed_avg("passing_yards"),
            _opp_allowed_avg("rushing_yards"),
            _opp_allowed_avg("receiving_yards"),
            _opp_allowed_avg("receptions"),
        )
        .select(
            [
                "row_id",
                "opp_passing_yards_allowed_avg",
                "opp_rushing_yards_allowed_avg",
                "opp_receiving_yards_allowed_avg",
                "opp_receptions_allowed_avg",
            ]
        )
    )
    player_stats_small = player_stats_small.join(opp_sorted, on="row_id", how="left")

    # Position one-hots
    pos_exprs = []
    pos_names = ["QB", "RB", "WR", "TE"]
    for p in pos_names:
        pos_exprs.append(
            (pl.col("position") == p).cast(pl.Int32).alias(f"is_{p.lower()}")
        )

    # Rolling lookback features (previous 3 games per player)
    rolling_exprs = [
        _rolling_mean("passing_yards"),
        _rolling_mean("rushing_yards"),
        _rolling_mean("receiving_yards"),
        _rolling_mean("receptions"),
        _rolling_mean("targets"),
    ]

    # Efficiency features (guard divide-by-zero)
    pass_ypa = pl.when(pl.col("attempts") > 0).then(
        pl.col("passing_yards") / pl.col("attempts")
    ).otherwise(0.0).alias("pass_ypa")
    rush_ypc = pl.when(pl.col("carries") > 0).then(
        pl.col("rushing_yards") / pl.col("carries")
    ).otherwise(0.0).alias("rush_ypc")
    rec_ypt = pl.when(pl.col("targets") > 0).then(
        pl.col("receiving_yards") / pl.col("targets")
    ).otherwise(0.0).alias("rec_ypt")

    player_stats_sorted = (
        player_stats_small.with_columns(
            *pos_exprs,
            *rolling_exprs,
            # Normalize season so values are smaller
            (pl.col("season") - 2000).alias("season_norm"),
            pass_ypa,
            rush_ypc,
            rec_ypt,
        )
        .sort(["player_id", "season", "week"])
    )

    df_features = player_stats_sorted.select(
        [
            # Context
            "season_norm",
            "week",
            "position",
            "is_qb",
            "is_rb",
            "is_wr",
            "is_te",
            # Opponent defensive form
            "opp_passing_yards_allowed_avg",
            "opp_rushing_yards_allowed_avg",
            "opp_receiving_yards_allowed_avg",
            "opp_receptions_allowed_avg",
            # Volume / efficiency
            "completions",
            "attempts",
            "pass_ypa",
            "carries",
            "rush_ypc",
            "targets",
            "rec_ypt",
            "receiving_air_yards",
            "receiving_yards_after_catch",
            "passing_epa",
            "passing_cpoe",
            "rushing_epa",
            "receiving_epa",
            "target_share",
            "air_yards_share",
            "wopr",
            "fantasy_points",
            # Rolling history
            "passing_yards_prev3",
            "rushing_yards_prev3",
            "receiving_yards_prev3",
            "receptions_prev3",
            "targets_prev3",
        ]
    )

    # Position-specific masking to reduce noise (e.g., drop passing stats for WR/TE).
    common_features = [
        "season_norm",
        "week",
        "is_qb",
        "is_rb",
        "is_wr",
        "is_te",
    ]
    pos_feature_groups = {
        "QB": common_features
        + [
            "completions",
            "attempts",
            "pass_ypa",
            "passing_epa",
            "passing_cpoe",
            "carries",
            "rush_ypc",
            "rushing_epa",
            "targets",
            "rec_ypt",
            "receiving_air_yards",
            "receiving_yards_after_catch",
            "receiving_epa",
            "fantasy_points",
            "passing_yards_prev3",
            "rushing_yards_prev3",
            "receptions_prev3",
            "targets_prev3",
        ],
        "RB": common_features
        + [
            "carries",
            "rush_ypc",
            "rushing_epa",
            "targets",
            "rec_ypt",
            "receiving_air_yards",
            "receiving_yards_after_catch",
            "receiving_epa",
            "target_share",
            "air_yards_share",
            "wopr",
            "fantasy_points",
            "rushing_yards_prev3",
            "receiving_yards_prev3",
            "receptions_prev3",
            "targets_prev3",
        ],
        "WR": common_features
        + [
            "targets",
            "rec_ypt",
            "receiving_air_yards",
            "receiving_yards_after_catch",
            "receiving_epa",
            "target_share",
            "air_yards_share",
            "wopr",
            "fantasy_points",
            "receptions_prev3",
            "receiving_yards_prev3",
            "targets_prev3",
        ],
        "TE": common_features
        + [
            "targets",
            "rec_ypt",
            "receiving_air_yards",
            "receiving_yards_after_catch",
            "receiving_epa",
            "target_share",
            "air_yards_share",
            "wopr",
            "fantasy_points",
            "receptions_prev3",
            "receiving_yards_prev3",
            "targets_prev3",
        ],
    }

    all_positions = [p.upper() for p in positions]
    feature_columns = [c for c in df_features.columns if c != "position"]
    feature_allowed_positions = {col: set() for col in feature_columns}
    for col in feature_columns:
        if col in common_features:
            feature_allowed_positions[col] = set(all_positions)

    for pos, cols in pos_feature_groups.items():
        for col in cols:
            feature_allowed_positions.setdefault(col, set()).add(pos)

    mask_exprs = []
    for col in feature_columns:
        allowed = feature_allowed_positions.get(col, set())
        if len(allowed) == len(all_positions):
            continue  # allowed for everyone
        mask_exprs.append(
            pl.when(pl.col("position").is_in(list(allowed)))
            .then(pl.col(col))
            .otherwise(0.0)
            .alias(col)
        )

    if mask_exprs:
        df_features = df_features.with_columns(mask_exprs)
    df_features = df_features.drop("position")

    df_targets = player_stats_sorted.select(target_cols)
    positions_series = player_stats_sorted.select(["position"]).to_series()

    # Replace any lingering nulls with 0 then convert to numpy
    df_features = df_features.with_columns([pl.all().fill_null(0)])

    # Replace any lingering nulls with 0 then convert to numpy
    df_features = df_features.with_columns([pl.all().fill_null(0)])

    X = df_features.to_numpy().astype(np.float32)
    y = df_targets.to_numpy().astype(np.float32)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    feature_names = df_features.columns
    # Build target mask aligned to positions to avoid scoring irrelevant targets
    target_mask = np.ones_like(y, dtype=np.float32)
    pos_to_target_idx = {
        "QB": [0],           # passing yards
        "RB": [1, 2, 3],     # rushing, receiving, receptions
        "WR": [2, 3],        # receiving, receptions
        "TE": [2, 3],        # receiving, receptions
    }
    for i, pos in enumerate(positions_series):
        allowed_idx = pos_to_target_idx.get(pos, [0, 1, 2, 3])
        mask_row = np.zeros(4, dtype=np.float32)
        mask_row[allowed_idx] = 1.0
        target_mask[i] = mask_row

    norm_stats = {}
    if normalize:
        mean = X.mean(axis=0, keepdims=True)
        std = X.std(axis=0, keepdims=True) + 1e-6
        X = (X - mean) / std
        norm_stats = {"mean": mean.squeeze(0), "std": std.squeeze(0)}

    target_norm_stats = {}
    t_mean = y.mean(axis=0, keepdims=True)
    t_std = y.std(axis=0, keepdims=True) + 1e-6
    target_norm_stats = {"mean": t_mean.squeeze(0), "std": t_std.squeeze(0)}

    # Guard against any nan/inf from upstream operations
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    extras = []
    if return_feature_names:
        extras.append(feature_names)
    if return_norm_stats:
        extras.append(norm_stats)
    if return_target_mask:
        extras.append(target_mask)
    if return_positions:
        extras.append(np.array(positions_series))
    if return_target_norm_stats:
        extras.append(target_norm_stats)

    if extras:
        return X, y, *extras

    return X, y


if __name__ == "__main__":
    X, y, names, norm, mask, pos, tnorm = load_nfl_dataset(
        return_feature_names=True,
        return_norm_stats=True,
        return_target_mask=True,
        return_positions=True,
        return_target_norm_stats=True,
    )
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Feature columns:", names)
    print("Mean/std shapes:", norm["mean"].shape, norm["std"].shape)
    print("Mask shape:", mask.shape, "Positions shape:", pos.shape)
    print("Target mean/std:", tnorm)


def train_val_split(X, y, mask=None, positions=None, val_frac=0.2, seed=0):
    """
    Simple random train/val split with optional propagation of mask and positions.
    """
    rng = np.random.default_rng(seed)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    split = int(len(X) * (1 - val_frac))
    train_idx, val_idx = idx[:split], idx[split:]

    def _slice(arr):
        if arr is None:
            return None, None
        return arr[train_idx], arr[val_idx]

    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]
    mask_tr, mask_val = _slice(mask)
    pos_tr, pos_val = _slice(positions)

    train = {"X": X_tr, "y": y_tr, "mask": mask_tr, "positions": pos_tr}
    val = {"X": X_val, "y": y_val, "mask": mask_val, "positions": pos_val}
    return train, val
