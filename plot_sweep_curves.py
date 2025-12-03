"""
Aggregate per-episode reward traces from sweep_experiments.py and plot mean ± std across seeds.
Generates a single plot of the top-K configs ranked by mean final reward.
"""

import argparse
import csv
import glob
import os
from typing import Dict, List, Tuple

import numpy as np


def parse_run_filename(path: str) -> Dict:
    """Parse metadata out of the run filename emitted by sweep_experiments.py."""
    name = os.path.basename(path).replace(".csv", "")
    parts = name.split("_")
    meta = {}
    for part in parts:
        if "=" not in part:
            continue
        key, val = part.split("=", 1)
        meta[key] = val

    meta["fv"] = meta.get("fv", "unknown")
    meta["lr"] = float(meta["lr"])
    meta["rollout"] = int(meta["rollout"])
    meta["hidden"] = tuple(int(x) for x in meta["hidden"].split("x"))
    meta["clip"] = float(meta["clip"])
    meta["temp"] = float(meta["temp"])
    meta["maxep"] = int(meta["maxep"])
    meta["seed"] = int(meta["seed"])
    return meta


def load_rewards(path: str) -> List[float]:
    rewards = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rewards.append(float(row["reward"]))
    return rewards


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values
    cumsum = np.cumsum(np.insert(values, 0, 0))
    smoothed = (cumsum[window:] - cumsum[:-window]) / window
    # Pad to original length for alignment
    pad_len = len(values) - len(smoothed)
    if pad_len > 0:
        smoothed = np.concatenate([np.full(pad_len, smoothed[0]), smoothed])
    return smoothed


def aggregate(groups: Dict, top_k: int, smooth_window: int, group_by: str | None = None, top_per_group: int = 1):
    """
    Return (runs, warnings).
    Each item is (meta, episodes, mean, std, mean_final).
    If group_by is set, returns top_per_group per group value; otherwise returns top_k overall.
    """
    ranked = []
    warnings = []
    cfg_runs = []
    for cfg_key, runs in groups.items():
        reward_lists = runs["rewards"]
        if not reward_lists:
            warnings.append(f"No rewards for config {cfg_key}")
            continue
        # Match sweep_experiments final_reward: mean of last 10 episodes (or all if shorter)
        finals = []
        for r in reward_lists:
            if len(r) == 0:
                continue
            if len(r) >= 10:
                finals.append(float(np.mean(r[-10:])))
            else:
                finals.append(float(np.mean(r)))
        if not finals:
            warnings.append(f"No final rewards for config {cfg_key}")
            continue

        max_len = max(len(r) for r in reward_lists)
        padded = np.full((len(reward_lists), max_len), np.nan, dtype=np.float32)
        for idx, rewards in enumerate(reward_lists):
            padded[idx, : len(rewards)] = rewards

        mean = np.nanmean(padded, axis=0)
        std = np.nanstd(padded, axis=0)
        if smooth_window > 1:
            mean = moving_average(mean, smooth_window)
            std = moving_average(std, smooth_window)

        episodes = np.arange(1, len(mean) + 1)
        cfg_runs.append((runs["meta"], episodes, mean, std, float(np.mean(finals))))

    if group_by is None:
        cfg_runs.sort(key=lambda x: x[4], reverse=True)
        return cfg_runs[:top_k], warnings

    # Bucket by the requested hyperparameter
    buckets: Dict[str, List] = {}
    for meta, episodes, mean, std, mean_final in cfg_runs:
        group_val = meta.get(group_by)
        buckets.setdefault(group_val, []).append((meta, episodes, mean, std, mean_final))

    grouped_runs = []
    for group_val, items in buckets.items():
        items.sort(key=lambda x: x[4], reverse=True)
        grouped_runs.extend(items[:top_per_group])

    return grouped_runs, warnings


def plot_top_runs(runs: List, out_dir: str, label: str) -> str:
    import matplotlib.pyplot as plt  # Local import so the script fails fast if missing

    if not runs:
        raise RuntimeError("No runs to plot.")

    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    for meta, episodes, mean, std, mean_final in runs:
        label_str = (
            f"fv={meta['fv']} lr={meta['lr']} rollout={meta['rollout']} "
            f"hidden={'x'.join(map(str, meta['hidden']))} clip={meta['clip']} "
            f"temp={meta['temp']} maxep={meta['maxep']} "
            f"(final≈{mean_final:.2f})"
        )
        plt.plot(episodes, mean, label=label_str)
        plt.fill_between(episodes, mean - std, mean + std, alpha=0.15)

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(label)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    safe_label = label.lower().replace(" ", "_").replace("/", "_")
    out_path = os.path.join(out_dir, f"reward_curves_{safe_label}.png")
    plt.savefig(out_path, dpi=200)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Plot reward variance across seeds for sweep runs.")
    parser.add_argument("--log-dir", default="sweep_episode_logs", help="Directory with per-run reward CSVs.")
    parser.add_argument("--out-dir", default="outplots", help="Where to write plots.")
    parser.add_argument("--top-k", type=int, default=5, help="How many configs to plot, ranked by mean final reward.")
    parser.add_argument(
        "--smooth",
        type=int,
        default=1,
        help="Optional moving-average window (in episodes) applied to mean/std curves.",
    )
    parser.add_argument(
        "--group-by",
        choices=["lr", "clip", "hidden", "rollout", "fv"],
        help="Group plots by a hyperparameter (plots top-per-group instead of global top-k).",
    )
    parser.add_argument(
        "--top-per-group",
        type=int,
        default=1,
        help="When grouping, how many configs to plot per group (ranked by mean final reward).",
    )
    args = parser.parse_args()

    files = glob.glob(os.path.join(args.log_dir, "*.csv"))
    if not files:
        raise SystemExit(f"No reward logs found in {args.log_dir}. Rerun sweeps to generate them.")

    grouped = {}
    for path in files:
        meta = parse_run_filename(path)
        cfg_key = (
            meta["fv"],
            meta["lr"],
            meta["rollout"],
            meta["hidden"],
            meta["clip"],
            meta["temp"],
            meta["maxep"],
        )
        grouped.setdefault(cfg_key, {"meta": meta, "rewards": []})
        grouped[cfg_key]["rewards"].append(load_rewards(path))

    top_runs, warnings = aggregate(grouped, args.top_k, args.smooth, group_by=args.group_by, top_per_group=args.top_per_group)
    for w in warnings:
        print(f"Warning: {w}")

    if args.group_by:
        title = f"Top {args.top_per_group} per {args.group_by}: mean ± std reward across seeds"
    else:
        title = f"Top {args.top_k} sweep configs: mean ± std reward across seeds"

    out_path = plot_top_runs(top_runs, args.out_dir, title)
    print(f"Wrote plot to {out_path}")


if __name__ == "__main__":
    main()
