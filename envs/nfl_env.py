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

    def __init__(self, X, y, target_mask=None, shuffle=True):
        super().__init__()

        self.X = X.astype(np.float32)   # (num_samples, obs_dim)
        self.y = y.astype(np.float32)   # (num_samples, 4)
        if target_mask is None:
            target_mask = np.ones_like(self.y, dtype=np.float32)
        self.target_mask = target_mask.astype(np.float32)
        self.shuffle = shuffle
        
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

        # Action space: continuous predictions, reasonable bounds
        self.action_space = spaces.Box(
            low=np.zeros(self.action_dim, dtype=np.float32),
            high=np.ones(self.action_dim, dtype=np.float32) * 2000.0,  # 2000 yards max (won’t be hit)
            shape=(self.action_dim,),
            dtype=np.float32
        )

        self.idx = 0  # position within current episode

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = np.random.default_rng(seed)
        if self.shuffle:
            rng.shuffle(self.indices)
        else:
            # Keep deterministic ordering for evaluation
            self.indices = np.arange(self.num_samples)

        self.idx = 0
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

        # Only score relevant targets for this position; avoid divide-by-zero
        denom = np.maximum(mask.sum(), 1.0)
        mse = np.sum(((action - true_vals) ** 2) * mask) / denom
        reward = -mse

        self.idx += 1
        terminated = self.idx >= self.num_samples
        truncated = False

        if not terminated:
            obs = self.X[self.indices[self.idx]]
        else:
            obs = np.zeros(self.obs_dim, dtype=np.float32)

        info = {"mse": mse}

        return obs, reward, terminated, truncated, info

    def render(self):
        pass
