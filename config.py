from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence


def _minutes(value: float | int) -> float:
    """Utility helper to coerce any numeric duration to float minutes."""
    return float(value)


@dataclass
class RewardWeights:
    """Weights for the final and step-wise reward terms."""

    final_buses: float = 4.0
    final_deadhead: float = 0.1
    step_unused_penalty: float = 4.0
    step_deadhead_penalty: float = 0.1
    step_rest_reward: float = 2.0
    step_demand_penalty: float = 1.0


@dataclass
class PPOHyperParams:
    """Hyper-parameters that control PPO optimisation."""

    learning_rate: float = 1e-5
    gamma: float = 0.99
    clip_epsilon: float = 0.1
    gae_lambda: float = 0.95
    epochs: int = 4
    mini_batch_size: int = 64
    rollout_steps: int = 1024
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5


@dataclass
class TrainingScenario:
    """PPO training schedule for a specific phase (offline / online)."""

    episodes: int = 1000
    max_steps_per_episode: Optional[int] = None
    ppo: PPOHyperParams = field(default_factory=PPOHyperParams)
    reward_weights: RewardWeights = field(default_factory=RewardWeights)


@dataclass
class FleetBusConfig:
    """Initial description of a single bus."""

    bus_id: str
    initial_cp: str
    shift_duration: Optional[float] = None
    min_work_time_before_lunch: Optional[float] = None


@dataclass
class FleetConfig:
    """Initial fleet information."""

    buses: List[FleetBusConfig]
    max_buses: int


@dataclass
class DepartureDefinition:
    """Departure definition belonging to a CP."""

    line_id: str
    time_minute: float


@dataclass
class ControlPointConfig:
    """Definition of a control point (CP)."""

    cp_id: str
    departures: List[DepartureDefinition]
    initial_buses: int = 0


@dataclass
class BusLineConfig:
    """Definition of a bus line."""

    line_id: str
    departure_cp: str
    arrival_cp: str
    travel_time: float


@dataclass
class DisturbanceConfig:
    """Represents exogenous disturbances used in online experiments."""

    kind: str
    line_id: Optional[str] = None
    start_minute: Optional[float] = None
    end_minute: Optional[float] = None
    delay_minutes: float = 0.0


@dataclass
class MetaConfig:
    """Global knobs shared between both phases."""

    min_rest_time: float = 10.0
    target_bus_set_size: int = 8
    short_term_horizon: float = 300.0
    time_window_minutes: float = 60.0
    time_normalizer: float = 60.0
    shift_duration: float = 510.0
    min_work_time_before_lunch: float = 240.0
    lunch_duration: float = 30.0


@dataclass
class ProblemConfig:
    """Aggregated configuration loaded from JSON."""

    name: str
    meta: MetaConfig
    control_points: List[ControlPointConfig]
    bus_lines: List[BusLineConfig]
    deadhead_matrix: Dict[str, Dict[str, float]]
    fleet: FleetConfig
    offline_training: TrainingScenario
    online_training: TrainingScenario
    disturbances: List[DisturbanceConfig] = field(default_factory=list)

    def deadhead_time(self, origin: str, destination: str) -> float:
        """Retrieve deadhead time between two control points."""
        if origin == destination:
            return 0.0
        return _minutes(
            self.deadhead_matrix.get(origin, {}).get(destination, float("inf"))
        )

