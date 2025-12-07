from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import torch

from .data import load_problem_config
from .environment import RLMSAEnvironment
from .ppo import PPOAgent


def evaluate_policy(
    config_path: str,
    checkpoint_path: str,
    mode: str = "offline",
    episodes: int = 5,
) -> List[Dict[str, float]]:
    """Evaluate a trained policy and collect aggregate metrics."""
    problem = load_problem_config(config_path)
    env = RLMSAEnvironment(problem, mode=mode)
    scenario = problem.online_training if mode == "online" else problem.offline_training
    agent = PPOAgent(env.observation_dim, env.action_dim, scenario.ppo)
    state_dict = torch.load(checkpoint_path, map_location=agent.device)
    agent.policy.load_state_dict(state_dict)

    results: List[Dict[str, float]] = []
    for episode in range(episodes):
        obs = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        while not done:
            action_mask = env.action_mask
            action, _log_prob, _value = agent.act(obs, action_mask, deterministic=True)
            obs, reward, done, _info = env.step(action)
            total_reward += reward
            steps += 1

        buses_used = sum(1 for bus in env.buses.values() if bus.used)
        results.append(
            {
                "episode": episode + 1,
                "steps": steps,
                "reward": total_reward,
                "buses_used": buses_used,
                "total_deadhead_time": env.total_deadhead_time,
                "invalid_actions": env.invalid_actions,
            }
        )
    return results


def save_results(results: List[Dict[str, float]], output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    import json

    with path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

