# nfl_ppo.py

import os
import math
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from envs.nfl_env import NFLPlayerPropEnv
from load_stats import load_nfl_dataset  # adjust to your loader if named differently


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=(128, 128)):
        super().__init__()
        layers = []
        last_dim = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ReLU())
            last_dim = h
        self.body = nn.Sequential(*layers)

        # Policy head: mean of Gaussian
        self.mu = nn.Linear(last_dim, act_dim)
        # Log std as a parameter (shared across states)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

        # Value head
        self.v = nn.Linear(last_dim, 1)

    def forward(self, obs):
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        x = self.body(obs)
        mu = self.mu(x)
        value = self.v(x).squeeze(-1)
        return mu, self.log_std, value

    def get_action_and_value(self, obs, action=None):
        mu, log_std, value = self.forward(obs)
        std = log_std.exp()
        dist = torch.distributions.Normal(mu, std)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return action, log_prob, entropy, value


# ==========================
#   PPO Training Loop
# ==========================

def train_ppo(
    env,
    total_timesteps=100_000,
    rollout_len=2048,
    num_epochs=10,
    minibatch_size=64,
    gamma=0.99,
    gae_lambda=0.95,
    clip_coef=0.2,
    vf_coef=0.5,
    ent_coef=0.01,
    max_grad_norm=0.5,
    lr=3e-4,
    seed=0,
    save_path="ppo_nfl.pt",
    device=None,
    return_model=False,
):
    # ---------- Seeding ----------
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# ---------- Device ----------
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")  # Apple Silicon GPU
        else:
            device = torch.device("cpu")
    print(f"Using device: {device}")

    # ---------- Data / Env ----------
    # Load your NFL dataset (X = features, y = targets)
    obs_space = env.observation_space
    act_space = env.action_space

    obs_dim = obs_space.shape[0]
    act_dim = act_space.shape[0]

    # ---------- Model & Optimizer ----------
    model = ActorCritic(obs_dim, act_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # ---------- Storage for rollouts ----------
    # We store rollout_len steps
    obs_buf = torch.zeros((rollout_len, obs_dim), dtype=torch.float32, device=device)
    actions_buf = torch.zeros((rollout_len, act_dim), dtype=torch.float32, device=device)
    logprobs_buf = torch.zeros(rollout_len, dtype=torch.float32, device=device)
    rewards_buf = torch.zeros(rollout_len, dtype=torch.float32, device=device)
    dones_buf = torch.zeros(rollout_len, dtype=torch.float32, device=device)
    values_buf = torch.zeros(rollout_len, dtype=torch.float32, device=device)

    # ---------- Initial reset ----------
    obs, info = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32, device=device)

    global_step = 0
    num_updates = total_timesteps // rollout_len

    episode_rewards = []
    current_ep_reward = 0.0

    for update in range(num_updates):
        # ==========================
        #   Collect Rollout
        # ==========================
        for step in range(rollout_len):
            global_step += 1
            obs_buf[step] = obs

            with torch.no_grad():
                action, log_prob, _, value = model.get_action_and_value(obs)
            # Clip action to env bounds
            action_np = action.cpu().numpy()
            action_np = np.clip(action_np, act_space.low, act_space.high)

            next_obs, reward, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated

            actions_buf[step] = torch.tensor(action_np, dtype=torch.float32, device=device)
            logprobs_buf[step] = log_prob
            rewards_buf[step] = reward
            dones_buf[step] = float(done)
            values_buf[step] = value

            current_ep_reward += reward

            if done:
                episode_rewards.append(current_ep_reward)
                current_ep_reward = 0.0
                next_obs, info = env.reset()

            obs = torch.tensor(next_obs, dtype=torch.float32, device=device)

        # Compute value for the last obs (bootstrap for GAE)
        with torch.no_grad():
            _, _, _, last_value = model.get_action_and_value(obs)

        # ==========================
        #   GAE Advantage Computation
        # ==========================
        advantages = torch.zeros_like(rewards_buf, device=device)
        last_gae_lam = 0
        for t in reversed(range(rollout_len)):
            if t == rollout_len - 1:
                next_non_terminal = 1.0 - dones_buf[t]
                next_values = last_value
            else:
                next_non_terminal = 1.0 - dones_buf[t + 1]
                next_values = values_buf[t + 1]

            delta = rewards_buf[t] + gamma * next_values * next_non_terminal - values_buf[t]
            advantages[t] = last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam

        returns = advantages + values_buf

        # ==========================
        #   PPO Update
        # ==========================
        b_obs = obs_buf
        b_actions = actions_buf
        b_logprobs = logprobs_buf
        b_advantages = advantages
        b_returns = returns
        b_values = values_buf

        # Normalize advantages for stability
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        batch_size = rollout_len
        indices = torch.arange(batch_size, device=device)

        for epoch in range(num_epochs):
            # shuffle indices in-place
            indices = indices[torch.randperm(batch_size, device=device)]
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = indices[start:end]

                mb_obs = b_obs[mb_inds]
                mb_actions = b_actions[mb_inds]
                mb_logprobs_old = b_logprobs[mb_inds]
                mb_adv = b_advantages[mb_inds]
                mb_returns = b_returns[mb_inds]
                mb_values_old = b_values[mb_inds]

                _, new_logprobs, entropy, new_values = model.get_action_and_value(mb_obs, mb_actions)
                new_values = new_values

                # Ratio for policy loss
                ratio = (new_logprobs - mb_logprobs_old).exp()

                # Clipped surrogate objective
                unclipped = ratio * mb_adv
                clipped = torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef) * mb_adv
                policy_loss = -torch.min(unclipped, clipped).mean()

                # Value function loss (clipped)
                value_clipped = mb_values_old + (new_values - mb_values_old).clamp(-clip_coef, clip_coef)
                value_losses = (new_values - mb_returns).pow(2)
                value_losses_clipped = (value_clipped - mb_returns).pow(2)
                value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()

                entropy_loss = entropy.mean()

                loss = policy_loss + vf_coef * value_loss - ent_coef * entropy_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

        # Simple logging
        if len(episode_rewards) > 0:
            avg_reward = np.mean(episode_rewards[-10:])
        else:
            avg_reward = 0.0

        print(
            f"Update {update + 1}/{num_updates} | "
            f"Global step: {global_step} | "
            f"Avg episode reward (last 10): {avg_reward:.3f}"
        )

    # ==========================
    #   Save Trained Model
    # ==========================
    torch.save(model.state_dict(), save_path)
    print(f"Saved PPO model to {save_path}")

    if return_model:
        return episode_rewards, model
    return episode_rewards


if __name__ == "__main__":
    train_ppo()
