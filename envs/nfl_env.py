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

    def __init__(self, X, y):
        super().__init__()

        self.X = X.astype(np.float32)   # (num_samples, obs_dim)
        self.y = y.astype(np.float32)   # (num_samples, 4)
        
        self.num_samples = self.X.shape[0]
        self.obs_dim = self.X.shape[1]
        self.action_dim = self.y.shape[1]

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

        self.idx = 0  # index of the current sample

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.idx = 0
        
        obs = self.X[self.idx]
        info = {}

        return obs, info

    def step(self, action):
        """
        Compute reward, move to next sample.
        Reward = -MSE(action, truth).
        """
        
        true_vals = self.y[self.idx]

        mse = np.mean((action - true_vals) ** 2)
        reward = -mse

        self.idx += 1
        terminated = self.idx >= self.num_samples
        truncated = False

        if not terminated:
            obs = self.X[self.idx]
        else:
            obs = np.zeros(self.obs_dim, dtype=np.float32)

        info = {"mse": mse}

        return obs, reward, terminated, truncated, info

    def render(self):
        pass
