"""
Sweep script for PPO hyperparameters and feature variants.
Runs multiple seeds per config, times training, and reports mean/std of masked MSE and rewards.
"""

import itertools
import time
import csv
from collections import defaultdict
from typing import List, Tuple

import numpy as np
import torch

from load_stats import load_nfl_dataset, train_val_split
from envs.nfl_env import NFLPlayerPropEnv
from nfl_ppo import train_ppo

# Position-specific target weights helper
def build_weight_matrix(pos_arr):
    pos_weights = {
        "QB": np.array([1.2, 0.3, 0.1, 0.1], dtype=np.float32),
        "RB": np.array([0.1, 1.0, 0.8, 0.8], dtype=np.float32),
        "WR": np.array([0.05, 0.2, 1.0, 1.0], dtype=np.float32),
        "TE": np.array([0.05, 0.2, 1.0, 1.0], dtype=np.float32),
    }
    default_w = np.ones(4, dtype=np.float32)
    return np.vstack([pos_weights.get(p, default_w) for p in pos_arr])


# Adjust these to keep runtime manageable.
RUNS_CSV = "sweep_runs_top.csv"
RESULTS_CSV = "sweep_results_top.csv"

SWEEP_CONFIG = {
    # Narrowed to better-performing variants
    "feature_variants": ["full", "no_history"],
    "learning_rates": [3e-4, 1e-4],
    "rollout_lens": [1024, 2048],
    "hidden_sizes": [(128, 128), (256, 256)],
    "clip_coefs": [0.2, 0.1],
    "reward_temperatures": [0.5],  # sharper rewards already tried; keep softer setting
    "max_episode_steps": [512],
    "total_timesteps": 50_000,  # give PPO more room to learn
    "num_epochs": 10,
    "seeds": [0, 1, 2],
    "val_frac": 0.2,
}


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def masked_mse(y_true, y_pred, mask):
    denom = np.maximum(mask.sum(axis=1, keepdims=True), 1.0)
    return np.mean(np.sum(((y_pred - y_true) ** 2) * mask, axis=1) / denom[:, 0])


def evaluate_model(model, X, y, mask, device, target_norm_stats=None):
    model.eval()
    with torch.no_grad():
        obs = torch.tensor(X, dtype=torch.float32, device=device)
        mu, _, _ = model.forward(obs)
    preds = mu.detach().cpu().numpy()
    if target_norm_stats is not None:
        preds = preds * target_norm_stats["std"] + target_norm_stats["mean"]
    return masked_mse(y, preds, mask)


def apply_feature_variant(X, feature_names: List[str], variant: str):
    if variant == "full":
        return X, feature_names
    if variant == "no_history":
        keep_idx = [i for i, name in enumerate(feature_names) if "prev3" not in name]
        X_reduced = X[:, keep_idx]
        names_reduced = [feature_names[i] for i in keep_idx]
        return X_reduced, names_reduced
    raise ValueError(f"Unknown feature variant: {variant}")


def run_sweeps():
    device = get_device()
    print(f"Running sweeps on device: {device}")

    # Load once; feature variants are applied downstream
    X, y, feature_names, target_mask, positions, target_norm_stats = load_nfl_dataset(
        seasons=range(2018, 2024),
        return_feature_names=True,
        return_target_mask=True,
        return_positions=True,
        return_target_norm_stats=True,
    )

    results = []

    for feature_variant in SWEEP_CONFIG["feature_variants"]:
        X_variant, feat_names_variant = apply_feature_variant(X, feature_names, feature_variant)

        # Fresh split per variant to keep same shapes
        train, val = train_val_split(
            X_variant,
            y,
            mask=target_mask,
            positions=positions,
            val_frac=SWEEP_CONFIG["val_frac"],
            seed=0,
        )
        train_weights = build_weight_matrix(train["positions"])
        val_weights = build_weight_matrix(val["positions"])

        print(f"\n=== Feature variant: {feature_variant} | num_features={len(feat_names_variant)} ===")
        for lr, rollout_len, hidden_sizes, clip_coef, reward_temp, max_ep_steps in itertools.product(
            SWEEP_CONFIG["learning_rates"],
            SWEEP_CONFIG["rollout_lens"],
            SWEEP_CONFIG["hidden_sizes"],
            SWEEP_CONFIG["clip_coefs"],
            SWEEP_CONFIG["reward_temperatures"],
            SWEEP_CONFIG["max_episode_steps"],
        ):
            print(
                f"\nConfig lr={lr} rollout={rollout_len} hidden={hidden_sizes} "
                f"clip={clip_coef} temp={reward_temp} max_ep={max_ep_steps}"
            )
            for seed in SWEEP_CONFIG["seeds"]:
                env = NFLPlayerPropEnv(
                    train["X"],
                    train["y"],
                    target_mask=train["mask"],
                    shuffle=True,
                    target_mean=target_norm_stats["mean"],
                    target_std=target_norm_stats["std"],
                    max_episode_steps=max_ep_steps,
                    target_weights=np.array([0.7, 1.0, 1.0, 1.0], dtype=np.float32),
                    target_weights_per_sample=train_weights,
                    reward_clip=3.0,
                    reward_positive=True,
                    reward_temperature=reward_temp,
                )

                start = time.perf_counter()
                ep_rewards, model = train_ppo(
                    env,
                    total_timesteps=SWEEP_CONFIG["total_timesteps"],
                    rollout_len=rollout_len,
                    num_epochs=SWEEP_CONFIG["num_epochs"],
                    clip_coef=clip_coef,
                    lr=lr,
                    seed=seed,
                    device=device,
                    return_model=True,
                )
                train_time = time.perf_counter() - start

                val_mse = evaluate_model(model, val["X"], val["y"], val["mask"], device, target_norm_stats)
                if len(ep_rewards) == 0:
                    final_reward = 0.0
                else:
                    final_reward = (
                        float(np.mean(ep_rewards[-10:])) if len(ep_rewards) >= 10 else float(np.mean(ep_rewards))
                    )

                print(
                    f"  Seed {seed}: val_masked_mse={val_mse:.3f} "
                    f"final_reward={final_reward:.3f} time={train_time:.1f}s"
                )

                results.append(
                    {
                        "feature_variant": feature_variant,
                        "lr": lr,
                        "rollout_len": rollout_len,
                        "hidden_sizes": hidden_sizes,
                        "clip_coef": clip_coef,
                        "reward_temperature": reward_temp,
                        "max_episode_steps": max_ep_steps,
                        "seed": seed,
                        "val_mse": val_mse,
                        "final_reward": final_reward,
                        "train_time_sec": train_time,
                    }
                )

    summarize_results(results)


def summarize_results(results):
    grouped = defaultdict(list)
    # Write per-run results
    with open(RUNS_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "feature_variant",
                "lr",
                "rollout_len",
                "hidden_sizes",
                "clip_coef",
                "reward_temperature",
                "max_episode_steps",
                "seed",
                "val_mse",
                "final_reward",
                "train_time_sec",
            ]
        )
        for r in results:
            writer.writerow(
                [
                    r["feature_variant"],
                    r["lr"],
                    r["rollout_len"],
                    r["hidden_sizes"],
                    r["clip_coef"],
                    r["reward_temperature"],
                    r["max_episode_steps"],
                    r["seed"],
                    r["val_mse"],
                    r["final_reward"],
                    r["train_time_sec"],
                ]
            )

    for r in results:
        key = (
            r["feature_variant"],
            r["lr"],
            r["rollout_len"],
            r["hidden_sizes"],
            r["clip_coef"],
            r["reward_temperature"],
            r["max_episode_steps"],
        )
        grouped[key].append(r)

    print("\n=== Aggregated results (mean ± std over seeds) ===")
    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "feature_variant",
                "lr",
                "rollout_len",
                "hidden_sizes",
                "clip_coef",
                "reward_temperature",
                "max_episode_steps",
                "val_mse_mean",
                "val_mse_std",
                "final_reward_mean",
                "final_reward_std",
                "train_time_mean",
                "train_time_std",
            ]
        )

    for key, runs in grouped.items():
        fv, lr, rollout, hidden, clip_coef, reward_temp, max_ep = key
        val_mses = np.array([r["val_mse"] for r in runs])
        rewards = np.array([r["final_reward"] for r in runs])
        times = np.array([r["train_time_sec"] for r in runs])

        val_mean, val_std = val_mses.mean(), val_mses.std()
        rew_mean, rew_std = rewards.mean(), rewards.std()
        time_mean, time_std = times.mean(), times.std()

        print(
            f"[fv={fv} lr={lr} rollout={rollout} hidden={hidden} clip={clip_coef} temp={reward_temp} max_ep={max_ep}] "
            f"val_mse={val_mean:.3f}±{val_std:.3f} "
            f"final_reward={rew_mean:.3f}±{rew_std:.3f} "
            f"time={time_mean:.1f}s±{time_std:.1f}s"
        )

        with open(RESULTS_CSV, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    fv,
                    lr,
                    rollout,
                    hidden,
                    clip_coef,
                    reward_temp,
                    max_ep,
                    val_mean,
                    val_std,
                    rew_mean,
                    rew_std,
                    time_mean,
                    time_std,
                ]
            )


if __name__ == "__main__":
    run_sweeps()
def build_weight_matrix(pos_arr):
    pos_weights = {
        "QB": np.array([1.2, 0.3, 0.1, 0.1], dtype=np.float32),
        "RB": np.array([0.1, 1.0, 0.8, 0.8], dtype=np.float32),
        "WR": np.array([0.05, 0.2, 1.0, 1.0], dtype=np.float32),
        "TE": np.array([0.05, 0.2, 1.0, 1.0], dtype=np.float32),
    }
    default_w = np.ones(4, dtype=np.float32)
    return np.vstack([pos_weights.get(p, default_w) for p in pos_arr])
