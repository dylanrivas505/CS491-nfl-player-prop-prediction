import numpy as np
from load_stats import load_nfl_dataset, train_val_split
from envs.nfl_env import NFLPlayerPropEnv
from nfl_ppo import train_ppo

X, y, feature_names, target_mask, positions, target_norm_stats = load_nfl_dataset(
    seasons=range(2018, 2024),
    return_feature_names=True,
    return_target_mask=True,
    return_positions=True,
    return_target_norm_stats=True,
)

train, val = train_val_split(X, y, mask=target_mask, positions=positions, val_frac=0.2, seed=0)
print('Train size:', len(train['X']), 'Val size:', len(val['X']), 'Features:', len(feature_names))

pos_weights = {
    "QB": np.array([1.2, 0.3, 0.1, 0.1], dtype=np.float32),
    "RB": np.array([0.1, 1.0, 0.8, 0.8], dtype=np.float32),
    "WR": np.array([0.05, 0.2, 1.0, 1.0], dtype=np.float32),
    "TE": np.array([0.05, 0.2, 1.0, 1.0], dtype=np.float32),
}
default_w = np.ones(4, dtype=np.float32)

def build_weight_matrix(pos_arr):
    return np.vstack([pos_weights.get(p, default_w) for p in pos_arr])

train_weights = build_weight_matrix(train["positions"])
val_weights = build_weight_matrix(val["positions"])

env = NFLPlayerPropEnv(
    train['X'],
    train['y'],
    target_mask=train['mask'],
    shuffle=True,
    target_mean=target_norm_stats["mean"],
    target_std=target_norm_stats["std"],
    max_episode_steps=512,
    target_weights=np.array([0.7, 1.0, 1.0, 1.0], dtype=np.float32),
    target_weights_per_sample=train_weights,
    reward_clip=3.0,
    reward_positive=True,
    reward_temperature=0.2,
)

episode_rewards = train_ppo(
    env,
    total_timesteps=12_000,
    rollout_len=256,
    num_epochs=3,
    minibatch_size=64,
    seed=0,
    save_path='ppo_nfl_smoke.pt',
)

print('Train done. Last 5 episode rewards:', episode_rewards[-5:])
