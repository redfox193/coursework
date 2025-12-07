from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .config import PPOHyperParams


class ActorCritic(nn.Module):
    """Shared backbone Actor-Critic used by PPO."""

    def __init__(self, obs_size: int, action_size: int) -> None:
        super().__init__()
        self.state_net = nn.Sequential(
            nn.Linear(obs_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.actor = nn.Linear(32, action_size)
        self.critic = nn.Linear(32, 1)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        latent = self.state_net(obs)
        return self.actor(latent), self.critic(latent).squeeze(-1)


@dataclass
class Transition:
    obs: np.ndarray
    action: int
    log_prob: float
    reward: float
    value: float
    done: bool


class RolloutBuffer:
    """Utility buffer to store on-policy data for PPO."""

    def __init__(self) -> None:
        self.data: List[Transition] = []

    def add(self, transition: Transition) -> None:
        self.data.append(transition)

    def clear(self) -> None:
        self.data.clear()

    def __len__(self) -> int:
        return len(self.data)


class PPOAgent:
    """Minimal PPO implementation tailored for RL-MSA."""

    def __init__(self, obs_size: int, action_size: int, config: PPOHyperParams) -> None:
        self.obs_size = obs_size
        self.action_size = action_size
        self.cfg = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = ActorCritic(obs_size, action_size).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=self.cfg.learning_rate
        )
        self.last_update_stats: Dict[str, float] = {}

    @torch.no_grad()
    def act(
        self, obs: np.ndarray, action_mask: np.ndarray, deterministic: bool = False
    ) -> Tuple[int, float, float]:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        logits, value = self.policy(obs_tensor.unsqueeze(0))
        logits = logits.squeeze(0)
        mask_tensor = torch.as_tensor(action_mask, dtype=torch.float32, device=self.device)
        if torch.sum(mask_tensor).item() == 0:
            mask_tensor = torch.ones_like(mask_tensor)
        masked_logits = logits + torch.log(mask_tensor + 1e-8)
        dist = Categorical(logits=masked_logits)
        if deterministic:
            action = torch.argmax(dist.probs).item()
        else:
            action = dist.sample().item()
        log_prob = dist.log_prob(torch.tensor(action, device=self.device)).item()
        return action, log_prob, value.item()

    def update(self, buffer: RolloutBuffer) -> Dict[str, float]:
        obs = torch.as_tensor(
            np.stack([transition.obs for transition in buffer.data]),
            dtype=torch.float32,
            device=self.device,
        )
        actions = torch.as_tensor(
            [transition.action for transition in buffer.data],
            dtype=torch.int64,
            device=self.device,
        )
        old_log_probs = torch.as_tensor(
            [transition.log_prob for transition in buffer.data],
            dtype=torch.float32,
            device=self.device,
        )
        rewards = [transition.reward for transition in buffer.data]
        values = [transition.value for transition in buffer.data]
        dones = [transition.done for transition in buffer.data]

        returns, advantages = self._compute_gae(rewards, values, dones)
        returns = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dataset = torch.utils.data.TensorDataset(obs, actions, old_log_probs, returns, advantages)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.cfg.mini_batch_size,
            shuffle=True,
            drop_last=False,
        )

        policy_losses = []
        value_losses = []
        entropies = []

        for _ in range(self.cfg.epochs):
            for batch_obs, batch_actions, batch_old_log, batch_returns, batch_adv in loader:
                logits, values_pred = self.policy(batch_obs)
                dist = Categorical(logits=logits)
                log_probs = dist.log_prob(batch_actions)
                ratios = torch.exp(log_probs - batch_old_log)
                clipped = torch.clamp(
                    ratios, 1.0 - self.cfg.clip_epsilon, 1.0 + self.cfg.clip_epsilon
                )
                policy_loss = -torch.min(ratios * batch_adv, clipped * batch_adv).mean()

                value_loss = nn.functional.mse_loss(values_pred, batch_returns)
                entropy = dist.entropy().mean()

                loss = (
                    policy_loss
                    + self.cfg.value_coef * value_loss
                    - self.cfg.entropy_coef * entropy
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy.item())

        stats = {
            "policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
            "value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
            "entropy": float(np.mean(entropies)) if entropies else 0.0,
        }
        self.last_update_stats = stats
        return stats

    def _compute_gae(
        self, rewards: List[float], values: List[float], dones: List[bool]
    ) -> Tuple[List[float], List[float]]:
        returns: List[float] = []
        advantages: List[float] = []
        gae = 0.0
        next_value = 0.0
        for step in reversed(range(len(rewards))):
            mask = 1.0 - float(dones[step])
            delta = rewards[step] + self.cfg.gamma * next_value * mask - values[step]
            gae = delta + self.cfg.gamma * self.cfg.gae_lambda * mask * gae
            advantages.insert(0, gae)
            next_value = values[step]
            returns.insert(0, gae + values[step])
        return returns, advantages

