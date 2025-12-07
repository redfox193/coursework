from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import torch

from .data import load_problem_config
from .environment import RLMSAEnvironment
from .ppo import PPOAgent, RolloutBuffer, Transition


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _train(
    config_path: str,
    mode: str,
    output_dir: str,
    *,
    init_checkpoint: Optional[str] = None,
    deadhead_policy_path: Optional[str] = None,
) -> Dict[str, float]:
    problem = load_problem_config(config_path)
    env = RLMSAEnvironment(problem, mode=mode)
    if mode == "online" and deadhead_policy_path:
        offline_scenario = problem.offline_training
        deadhead_agent = PPOAgent(env.observation_dim, env.action_dim, offline_scenario.ppo)
        offline_state = torch.load(deadhead_policy_path, map_location=deadhead_agent.device)
        deadhead_agent.policy.load_state_dict(offline_state)
        deadhead_agent.policy.eval()
        env.attach_deadhead_policy(deadhead_agent)
    scenario = problem.online_training if mode == "online" else problem.offline_training
    agent = PPOAgent(env.observation_dim, env.action_dim, scenario.ppo)
    if init_checkpoint:
        state_dict = torch.load(init_checkpoint, map_location=agent.device)
        agent.policy.load_state_dict(state_dict)

    buffer = RolloutBuffer()
    out_path = Path(output_dir)
    _ensure_dir(out_path)
    stats = {"episodes": scenario.episodes, "mode": mode}

    for episode in range(scenario.episodes):
        obs = env.reset()
        done = False
        episode_reward = 0.0
        steps = 0
        buffer.clear()
        while not done:
            action_mask = env.action_mask
            action, log_prob, value = agent.act(obs, action_mask)
            next_obs, reward, done, _info = env.step(action)
            transition = Transition(
                obs=obs,
                action=action,
                log_prob=log_prob,
                reward=reward,
                value=value,
                done=done,
            )
            buffer.add(transition)
            obs = next_obs
            episode_reward += reward
            steps += 1

            if len(buffer) >= scenario.ppo.rollout_steps:
                agent.update(buffer)
                buffer.clear()

            if scenario.max_steps_per_episode and steps >= scenario.max_steps_per_episode:
                break

        if len(buffer) > 0:
            agent.update(buffer)
            buffer.clear()

        print(
            f"[{mode}] episode={episode+1}/{scenario.episodes} "
            f"steps={steps} reward={episode_reward:.2f}"
        )

    checkpoint_path = out_path / f"{mode}_policy.pt"
    torch.save(agent.policy.state_dict(), checkpoint_path)
    stats["checkpoint"] = str(checkpoint_path)
    with (out_path / f"{mode}_training_stats.json").open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)
    return stats


def train_offline(config_path: str, output_dir: str) -> Dict[str, float]:
    """Train the offline RL-MSA agent end-to-end."""
    return _train(config_path, "offline", output_dir)


def train_online(
    config_path: str,
    output_dir: str,
    *,
    offline_checkpoint: Optional[str] = None,
) -> Dict[str, float]:
    """Train the online bus-selection agent, optionally bootstrapping from the offline policy."""
    return _train(
        config_path,
        "online",
        output_dir,
        init_checkpoint=offline_checkpoint,
        deadhead_policy_path=offline_checkpoint,
    )

