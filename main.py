# main.py
from envs.nfl_env import NFLPlayerPropEnv
from load_stats import load_nfl_dataset, train_val_split
from nfl_ppo import train_ppo

import numpy as np

if __name__ == "__main__":
    # Load real data with position-aware target masks
    X, y, feature_names, target_mask, positions = load_nfl_dataset(
        return_feature_names=True, return_target_mask=True, return_positions=True
    )
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    # Split train/val for basic evaluation
    train, val = train_val_split(
        X, y, mask=target_mask, positions=positions, val_frac=0.2, seed=0
    )

    print("Loaded X shape:", X.shape, "| y shape:", y.shape)
    print("Num features:", len(feature_names))
    print(f"Train size: {len(train['X'])} | Val size: {len(val['X'])}")

    # Baseline: per-position mean predictor evaluated with masked MSE
    def masked_mse(y_true, y_pred, mask):
        denom = np.maximum(mask.sum(axis=1, keepdims=True), 1.0)
        return np.mean(np.sum(((y_pred - y_true) ** 2) * mask, axis=1) / denom[:, 0])

    def build_position_means(train_dict):
        pos_to_mean = {}
        mask = train_dict["mask"]
        y_train = train_dict["y"]
        positions_arr = train_dict["positions"]

        # Global mean for fallback (masked)
        denom_global = mask.sum(axis=0)
        denom_global[denom_global == 0] = 1
        global_mean = (y_train * mask).sum(axis=0) / denom_global

        for pos in np.unique(positions_arr):
            pos_idx = positions_arr == pos
            y_pos = y_train[pos_idx]
            mask_pos = mask[pos_idx]
            denom = mask_pos.sum(axis=0)
            denom[denom == 0] = 1
            pos_to_mean[pos] = (y_pos * mask_pos).sum(axis=0) / denom
        return pos_to_mean, global_mean

    def predict_with_means(positions_arr, pos_to_mean, global_mean):
        preds = np.zeros((len(positions_arr), y.shape[1]), dtype=np.float32)
        for i, pos in enumerate(positions_arr):
            preds[i] = pos_to_mean.get(pos, global_mean)
        return preds

    pos_means, global_mean = build_position_means(train)
    val_preds = predict_with_means(val["positions"], pos_means, global_mean)
    overall_val_mse = masked_mse(val["y"], val_preds, val["mask"])

    print(f"Baseline masked MSE (per-position mean): {overall_val_mse:.3f}")
    for pos in np.unique(val["positions"]):
        idx = val["positions"] == pos
        mse_pos = masked_mse(val["y"][idx], val_preds[idx], val["mask"][idx])
        print(f"  {pos} val masked MSE: {mse_pos:.3f}")

    # PPO training on train split
    env = NFLPlayerPropEnv(train["X"], train["y"], target_mask=train["mask"], shuffle=True)

    # Sanity check the env
    obs, info = env.reset()
    print("Initial obs shape:", obs.shape)

    for _ in range(5):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        print("Sanity step reward:", reward)
        if done or truncated:
            obs, info = env.reset()

    # Now actually train PPO
    episode_rewards = train_ppo(env, total_timesteps=50_000, seed=0)

    print("Training done. Final ~10 episode rewards mean:",
          np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else episode_rewards)
