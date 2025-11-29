"""
Sweep script for PPO hyperparameters and feature variants.
Runs multiple seeds per config, times training, and reports mean/std of masked MSE and rewards.
"""

import itertools
import time
from collections import defaultdict
from typing import List, Tuple

import numpy as np
import torch

from load_stats import load_nfl_dataset, train_val_split
from envs.nfl_env import NFLPlayerPropEnv
from nfl_ppo import train_ppo


# Adjust these to keep runtime manageable.
SWEEP_CONFIG = {
    "feature_variants": ["full", "no_history"],  # drop rolling history for no_history
    "learning_rates": [3e-4, 1e-4],
    "rollout_lens": [1024, 2048],
    "hidden_sizes": [(128, 128), (256, 256)],
    "total_timesteps": 20_000,
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


def evaluate_model(model, X, y, mask, device):
    model.eval()
    with torch.no_grad():
        obs = torch.tensor(X, dtype=torch.float32, device=device)
        mu, _, _ = model.forward(obs)
    preds = mu.detach().cpu().numpy()
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
    X, y, feature_names, target_mask, positions = load_nfl_dataset(
        return_feature_names=True, return_target_mask=True, return_positions=True
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

        print(f"\n=== Feature variant: {feature_variant} | num_features={len(feat_names_variant)} ===")
        for lr, rollout_len, hidden_sizes in itertools.product(
            SWEEP_CONFIG["learning_rates"],
            SWEEP_CONFIG["rollout_lens"],
            SWEEP_CONFIG["hidden_sizes"],
        ):
            print(f"\nConfig lr={lr} rollout={rollout_len} hidden={hidden_sizes}")
            for seed in SWEEP_CONFIG["seeds"]:
                env = NFLPlayerPropEnv(
                    train["X"],
                    train["y"],
                    target_mask=train["mask"],
                    shuffle=True,
                )

                start = time.perf_counter()
                ep_rewards, model = train_ppo(
                    env,
                    total_timesteps=SWEEP_CONFIG["total_timesteps"],
                    rollout_len=rollout_len,
                    num_epochs=SWEEP_CONFIG["num_epochs"],
                    lr=lr,
                    seed=seed,
                    device=device,
                    return_model=True,
                )
                train_time = time.perf_counter() - start

                val_mse = evaluate_model(model, val["X"], val["y"], val["mask"], device)
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
                        "seed": seed,
                        "val_mse": val_mse,
                        "final_reward": final_reward,
                        "train_time_sec": train_time,
                    }
                )

    summarize_results(results)


def summarize_results(results):
    grouped = defaultdict(list)
    for r in results:
        key = (
            r["feature_variant"],
            r["lr"],
            r["rollout_len"],
            r["hidden_sizes"],
        )
        grouped[key].append(r)

    print("\n=== Aggregated results (mean ± std over seeds) ===")
    for key, runs in grouped.items():
        fv, lr, rollout, hidden = key
        val_mses = np.array([r["val_mse"] for r in runs])
        rewards = np.array([r["final_reward"] for r in runs])
        times = np.array([r["train_time_sec"] for r in runs])

        print(
            f"[fv={fv} lr={lr} rollout={rollout} hidden={hidden}] "
            f"val_mse={val_mses.mean():.3f}±{val_mses.std():.3f} "
            f"final_reward={rewards.mean():.3f}±{rewards.std():.3f} "
            f"time={times.mean():.1f}s±{times.std():.1f}s"
        )


if __name__ == "__main__":
    run_sweeps()
