from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from .config import (
    BusLineConfig,
    ControlPointConfig,
    DepartureDefinition,
    DisturbanceConfig,
    FleetBusConfig,
    FleetConfig,
    MetaConfig,
    PPOHyperParams,
    ProblemConfig,
    RewardWeights,
    TrainingScenario,
)


def _load_reward_weights(raw: Dict[str, Any]) -> RewardWeights:
    return RewardWeights(
        final_buses=float(raw.get("final_buses", 4.0)),
        final_deadhead=float(raw.get("final_deadhead", 0.1)),
        step_unused_penalty=float(raw.get("step_unused_penalty", 4.0)),
        step_deadhead_penalty=float(raw.get("step_deadhead_penalty", 0.1)),
        step_rest_reward=float(raw.get("step_rest_reward", 2.0)),
        step_demand_penalty=float(raw.get("step_demand_penalty", 1.0)),
    )


def _load_ppo(raw: Dict[str, Any]) -> PPOHyperParams:
    return PPOHyperParams(
        learning_rate=float(raw.get("learning_rate", 1e-5)),
        gamma=float(raw.get("gamma", 0.99)),
        clip_epsilon=float(raw.get("clip_epsilon", 0.1)),
        gae_lambda=float(raw.get("gae_lambda", 0.95)),
        epochs=int(raw.get("epochs", 4)),
        mini_batch_size=int(raw.get("mini_batch_size", 64)),
        rollout_steps=int(raw.get("rollout_steps", 1024)),
        entropy_coef=float(raw.get("entropy_coef", 0.01)),
        value_coef=float(raw.get("value_coef", 0.5)),
        max_grad_norm=float(raw.get("max_grad_norm", 0.5)),
    )


def _load_training(raw: Dict[str, Any]) -> TrainingScenario:
    max_steps = raw.get("max_steps_per_episode")
    if max_steps is not None:
        max_steps = int(max_steps)
    return TrainingScenario(
        episodes=int(raw.get("episodes", 1000)),
        max_steps_per_episode=max_steps,
        ppo=_load_ppo(raw.get("ppo", {})),
        reward_weights=_load_reward_weights(raw.get("reward_weights", {})),
    )


def load_problem_config(path: str | Path) -> ProblemConfig:
    """Load a RL-MSA problem instance from disk."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    meta_raw = payload.get("meta", {})
    meta = MetaConfig(
        min_rest_time=float(meta_raw.get("min_rest_time", 10.0)),
        target_bus_set_size=int(meta_raw.get("target_bus_set_size", 8)),
        short_term_horizon=float(meta_raw.get("short_term_horizon", 300.0)),
        time_window_minutes=float(meta_raw.get("time_window_minutes", 60.0)),
        time_normalizer=float(meta_raw.get("time_normalizer", 60.0)),
        shift_duration=float(meta_raw.get("shift_duration", 480.0)),
        min_work_time_before_lunch=float(meta_raw.get("min_work_time_before_lunch", 240.0)),
        lunch_duration=float(meta_raw.get("lunch_duration", 30.0)),
    )

    control_points = []
    for cp in payload.get("control_points", []):
        departures = [
            DepartureDefinition(line_id=dep["line_id"], time_minute=float(dep["time"]))
            for dep in cp.get("departures", [])
        ]
        control_points.append(
            ControlPointConfig(
                cp_id=cp["id"],
                departures=departures,
                initial_buses=int(cp.get("initial_buses", 0)),
            )
        )

    bus_lines = [
        BusLineConfig(
            line_id=line["id"],
            departure_cp=line["departure_cp"],
            arrival_cp=line["arrival_cp"],
            travel_time=float(line["travel_time"]),
        )
        for line in payload.get("bus_lines", [])
    ]

    deadhead_matrix = {
        origin: {dest: float(time) for dest, time in mapping.items()}
        for origin, mapping in payload.get("deadhead_times", {}).items()
    }

    fleet = FleetConfig(
        buses=[
            FleetBusConfig(
                bus_id=bus["id"],
                initial_cp=bus["initial_cp"],
                shift_duration=bus.get("shift_duration"),
                min_work_time_before_lunch=bus.get("min_work_time_before_lunch"),
            )
            for bus in payload.get("fleet", {}).get("buses", [])
        ],
        max_buses=int(payload.get("fleet", {}).get("max_buses", 0)),
    )

    offline_training = _load_training(payload.get("offline_training", {}))
    online_training = _load_training(payload.get("online_training", {}))

    disturbances = [
        DisturbanceConfig(
            kind=entry["kind"],
            line_id=entry.get("line_id"),
            start_minute=float(entry.get("start_minute"))
            if entry.get("start_minute") is not None
            else None,
            end_minute=float(entry.get("end_minute"))
            if entry.get("end_minute") is not None
            else None,
            delay_minutes=float(entry.get("delay_minutes", 0.0)),
        )
        for entry in payload.get("disturbances", [])
    ]

    return ProblemConfig(
        name=payload.get("name", path.stem),
        meta=meta,
        control_points=control_points,
        bus_lines=bus_lines,
        deadhead_matrix=deadhead_matrix,
        fleet=fleet,
        offline_training=offline_training,
        online_training=online_training,
        disturbances=disturbances,
    )

