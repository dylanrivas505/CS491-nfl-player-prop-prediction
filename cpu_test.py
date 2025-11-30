import numpy as np
from load_stats import load_nfl_dataset, train_val_split
from envs.nfl_env import NFLPlayerPropEnv
from nfl_ppo import train_ppo

X, y, feature_names, target_mask, positions, target_norm_stats = load_nfl_dataset(
    seasons=range(2023, 2024),
    return_feature_names=True,
    return_target_mask=True,
    return_positions=True,
    return_target_norm_stats=True,
)

train, val = train_val_split(X, y, mask=target_mask, positions=positions, val_frac=0.2, seed=0)
print('Train size:', len(train['X']), 'Val size:', len(val['X']), 'Features:', len(feature_names))

env = NFLPlayerPropEnv(
    train['X'],
    train['y'],
    target_mask=train['mask'],
    shuffle=True,
    target_mean=target_norm_stats["mean"],
    target_std=target_norm_stats["std"],
    max_episode_steps=1024,
)

episode_rewards = train_ppo(
    env,
    total_timesteps=8_000,
    rollout_len=256,
    num_epochs=3,
    minibatch_size=64,
    seed=0,
    save_path='ppo_nfl_smoke.pt',
)

print('Train done. Last 5 episode rewards:', episode_rewards[-5:])
