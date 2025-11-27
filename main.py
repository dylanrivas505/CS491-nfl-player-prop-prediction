# main.py
from envs.nfl_env import NFLPlayerPropEnv
from load_stats import load_nfl_dataset
from nfl_ppo import train_ppo

import numpy as np

if __name__ == "__main__":
    # Load real data
    X, y = load_nfl_dataset()
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    print("Loaded X shape:", X.shape)
    print("Loaded y shape:", y.shape)

    env = NFLPlayerPropEnv(X, y)

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
