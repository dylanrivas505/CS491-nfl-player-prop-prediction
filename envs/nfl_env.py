import gymnasium as gym
from gymnasium import spaces
import numpy as np


class NFLPlayerPropEnv(gym.Env):
    """
    A simple supervised-to-RL environment:
    - Observation: feature vector for a (player, game)
    - Action: predicted [passing_yards, rushing_yards, receiving_yards, receptions]
    - Reward: negative MSE between prediction and actual stats
    """
    
    metadata = {"render_modes": []}

    def __init__(
        self,
        X,
        y,
        target_mask=None,
        shuffle=True,
        target_mean=None,
        target_std=None,
        max_episode_steps=None,
        target_weights=None,
        reward_clip=None,
        reward_positive=True,
        reward_temperature=1.0,
        target_weights_per_sample=None,
    ):
        super().__init__()

        self.X = X.astype(np.float32)   # (num_samples, obs_dim)
        self.y = y.astype(np.float32)   # (num_samples, 4)
        if target_mask is None:
            target_mask = np.ones_like(self.y, dtype=np.float32)
        self.target_mask = target_mask.astype(np.float32)
        self.shuffle = shuffle
        self.use_normalized_targets = target_mean is not None and target_std is not None
        self.target_mean = None if target_mean is None else target_mean.astype(np.float32)
        self.target_std = None if target_std is None else target_std.astype(np.float32)
        self.max_episode_steps = max_episode_steps
        if target_weights is None:
            target_weights = np.ones(self.y.shape[1], dtype=np.float32)
        self.target_weights = target_weights.astype(np.float32)
        self.target_weights_per_sample = None
        if target_weights_per_sample is not None:
            self.target_weights_per_sample = target_weights_per_sample.astype(np.float32)
        self.reward_clip = reward_clip
        self.reward_positive = reward_positive
        self.reward_temperature = reward_temperature
        
        self.num_samples = self.X.shape[0]
        self.obs_dim = self.X.shape[1]
        self.action_dim = self.y.shape[1]
        self.indices = np.arange(self.num_samples)

        # Observation space: continuous vector
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32
        )

        # Action space: normalized if target stats provided, else raw bounds
        if self.use_normalized_targets:
            self.action_space = spaces.Box(
                low=np.ones(self.action_dim, dtype=np.float32) * -5.0,
                high=np.ones(self.action_dim, dtype=np.float32) * 5.0,
                shape=(self.action_dim,),
                dtype=np.float32,
            )
        else:
            self.action_space = spaces.Box(
                low=np.zeros(self.action_dim, dtype=np.float32),
                high=np.ones(self.action_dim, dtype=np.float32) * 2000.0,  # 2000 yards max (won’t be hit)
                shape=(self.action_dim,),
                dtype=np.float32
            )

        self.idx = 0  # position within current episode
        self.steps_in_ep = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = np.random.default_rng(seed)
        if self.shuffle:
            rng.shuffle(self.indices)
        else:
            # Keep deterministic ordering for evaluation
            self.indices = np.arange(self.num_samples)

        self.idx = 0
        self.steps_in_ep = 0
        obs = self.X[self.indices[self.idx]]
        info = {}

        return obs, info

    def step(self, action):
        """
        Compute reward, move to next sample.
        Reward = -MSE(action, truth).
        """
        
        true_vals = self.y[self.indices[self.idx]]
        mask = self.target_mask[self.indices[self.idx]]
        weights = (
            self.target_weights_per_sample[self.indices[self.idx]]
            if self.target_weights_per_sample is not None
            else self.target_weights
        )

        # Normalize targets for reward if stats provided
        if self.target_mean is not None and self.target_std is not None:
            norm_true = (true_vals - self.target_mean) / self.target_std
            norm_action = (action - self.target_mean) / self.target_std
        else:
            norm_true = true_vals
            norm_action = action

        diff = norm_action - norm_true
        if self.reward_clip is not None:
            diff = np.clip(diff, -self.reward_clip, self.reward_clip)

        # Only score relevant targets for this position; avoid divide-by-zero
        weighted_mask = mask * weights
        denom = np.maximum(weighted_mask.sum(), 1.0)
        mse = np.sum((diff ** 2) * weighted_mask) / denom
        if self.reward_positive:
            # Map small errors to rewards near 1, larger errors decay toward 0
            reward = float(np.exp(-mse / max(self.reward_temperature, 1e-8)))
        else:
            reward = -mse

        self.steps_in_ep += 1
        self.idx += 1
        terminated = self.idx >= self.num_samples
        truncated = False
        if self.max_episode_steps is not None and self.steps_in_ep >= self.max_episode_steps:
            truncated = True

        if not terminated:
            obs = self.X[self.indices[self.idx]]
        else:
            obs = np.zeros(self.obs_dim, dtype=np.float32)

        info = {"mse": mse}
        if self.use_normalized_targets:
            # Also report unnormalized MSE with weights applied
            raw_diff = action - true_vals
            raw_mse = np.sum((raw_diff ** 2) * weighted_mask) / denom
            info["mse_raw_scale"] = raw_mse

        return obs, reward, terminated, truncated, info

    def render(self):
        pass
