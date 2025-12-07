from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import random
import bisect
import math
import numpy as np

from .config import (
    BusLineConfig,
    ControlPointConfig,
    MetaConfig,
    ProblemConfig,
    RewardWeights,
    TrainingScenario,
)


@dataclass
class DepartureEvent:
    """A single decision point corresponding to a timetable departure."""

    line_id: str
    departure_cp: str
    arrival_cp: str
    departure_time: float
    travel_time: float


@dataclass
class BusState:
    """State attached to a physical bus."""

    bus_id: str
    home_cp: str
    current_cp: Optional[str]
    shift_duration: float
    min_work_time_before_lunch: float
    lunch_duration: float
    used: bool = False
    available_time: float = 0.0
    last_service_arrival: float = -math.inf
    busy_reason: Optional[str] = None  # "trip" or "deadhead"
    destination_cp: Optional[str] = None
    last_line_id: Optional[str] = None
    shift_start_time: Optional[float] = None
    lunch_taken: bool = False
    lunch_start_time: Optional[float] = None

    def clone(self) -> "BusState":
        return BusState(
            bus_id=self.bus_id,
            home_cp=self.home_cp,
            current_cp=self.current_cp,
            shift_duration=self.shift_duration,
            used=self.used,
            available_time=self.available_time,
            last_service_arrival=self.last_service_arrival,
            busy_reason=self.busy_reason,
            destination_cp=self.destination_cp,
            last_line_id=self.last_line_id,
            shift_start_time=self.shift_start_time,
        )


@dataclass
class CandidateBus:
    """Container used to build the target bus set at each step."""

    bus: BusState
    requires_deadhead: bool
    deadhead_time: float
    rest_time: float
    priority_reward: float
    source_cp: Optional[str]


class RLMSAEnvironment:
    """Simulator for the RL-MSA offline/online training phases."""

    def __init__(
        self,
        config: ProblemConfig,
        mode: str = "offline",
        *,
        deadhead_policy=None,
        auto_reset: bool = True,
    ) -> None:
        self.config = config
        self.mode = mode
        self.deadhead_policy = deadhead_policy
        self.meta: MetaConfig = config.meta
        self.control_points: Dict[str, ControlPointConfig] = {
            cp.cp_id: cp for cp in config.control_points
        }
        self.cp_departure_times: Dict[str, List[float]] = {
            cp.cp_id: sorted(dep.time_minute for dep in cp.departures)
            for cp in config.control_points
        }
        self.bus_lines: Dict[str, BusLineConfig] = {
            line.line_id: line for line in config.bus_lines
        }
        self.fleet_bus_config = {bus.bus_id: bus for bus in config.fleet.buses}
        self.cp_index = {cp_id: idx for idx, cp_id in enumerate(self.control_points)}
        self.num_cps = len(self.cp_index)
        self.target_action_size = self.meta.target_bus_set_size
        self.short_horizon = self.meta.short_term_horizon
        self.min_rest = self.meta.min_rest_time
        self.time_norm = self.meta.time_normalizer
        self.departure_sequence: List[DepartureEvent] = self._build_departure_sequence()
        self.total_departures = len(self.departure_sequence)
        if self.total_departures == 0:
            raise ValueError(
                "No departures were defined in the configuration. "
                "Please add timetable entries for each control point."
            )
        self.deadhead_planner = None

        # State placeholders initialised during reset
        self.buses: Dict[str, BusState] = {}
        self.current_step: int = 0
        self.current_departure: Optional[DepartureEvent] = None
        self.current_time: float = 0.0
        self.current_candidates: List[Optional[CandidateBus]] = []
        self.current_action_mask: np.ndarray = np.zeros(
            self.target_action_size, dtype=np.float32
        )
        self.reward_weights: RewardWeights = self._select_reward_weights()
        self.training_profile: TrainingScenario = self._select_training_profile()
        self.total_deadhead_time: float = 0.0
        self.invalid_actions: int = 0
        self.uncovered_departures: int = 0
        self.history: List[Dict[str, float]] = []
        self.next_auto_bus_idx = 0

        self._configure_deadhead_planner()

        if auto_reset:
            self.reset()

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #
    def _select_reward_weights(self) -> RewardWeights:
        """Pick the reward configuration matching the current phase."""
        if self.mode == "online":
            return self.config.online_training.reward_weights
        return self.config.offline_training.reward_weights

    def _select_training_profile(self) -> TrainingScenario:
        """Pick the PPO hyper-parameters for the current phase."""
        if self.mode == "online":
            return self.config.online_training
        return self.config.offline_training

    def _build_departure_sequence(self) -> List[DepartureEvent]:
        """Flatten per-CP timetables into a single chronological decision list."""
        sequence: List[DepartureEvent] = []
        for cp in self.config.control_points:
            for dep in cp.departures:
                line = self.bus_lines.get(dep.line_id)
                if line is None:
                    raise KeyError(f"Unknown line_id '{dep.line_id}' in CP '{cp.cp_id}'")
                if line.departure_cp != cp.cp_id:
                    raise ValueError(
                        f"Line {line.line_id} is defined to depart from "
                        f"{line.departure_cp}, but CP {cp.cp_id} lists it."
                    )
                sequence.append(
                    DepartureEvent(
                        line_id=line.line_id,
                        departure_cp=line.departure_cp,
                        arrival_cp=line.arrival_cp,
                        departure_time=float(dep.time_minute),
                        travel_time=float(line.travel_time),
                    )
                )
        # Sort once here so training can simply iterate over the array.
        sequence.sort(key=lambda item: item.departure_time)
        return sequence

    # NOTE: Online-only helper.
    def _configure_deadhead_planner(self) -> None:
        """Attach the time-window planner that reuses the offline policy."""
        if self.mode == "online" and self.deadhead_policy is not None:
            self.deadhead_planner = TimeWindowDeadheadPlanner(
                time_window=self.meta.time_window_minutes,
                min_rest=self.min_rest,
                policy=self.deadhead_policy,
            )
        else:
            self.deadhead_planner = None

    # NOTE: Online-only helper.
    def attach_deadhead_policy(self, policy) -> None:
        """Attach or replace the offline policy used for deadhead planning."""
        self.deadhead_policy = policy
        self._configure_deadhead_planner()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """Reset the simulator to its initial state."""
        del seed  # deterministic env for reproducibility
        self.reward_weights = self._select_reward_weights()
        self.training_profile = self._select_training_profile()
        self.buses = {}
        for bus_cfg in self.config.fleet.buses:
            shift_duration = (
                bus_cfg.shift_duration
                if bus_cfg.shift_duration is not None
                else self.meta.shift_duration
            )
            min_lunch = (
                bus_cfg.min_work_time_before_lunch
                if bus_cfg.min_work_time_before_lunch is not None
                else self.meta.min_work_time_before_lunch
            )
            self.buses[bus_cfg.bus_id] = BusState(
                bus_id=bus_cfg.bus_id,
                home_cp=bus_cfg.initial_cp,
                current_cp=bus_cfg.initial_cp,
                shift_duration=shift_duration,
                min_work_time_before_lunch=min_lunch,
                lunch_duration=self.meta.lunch_duration,
                used=False,
                available_time=0.0,
                last_service_arrival=-math.inf,
                busy_reason=None,
                destination_cp=None,
            )
        self.next_auto_bus_idx = len(self.buses)
        self.current_step = 0
        self.current_departure = self.departure_sequence[0]
        self.current_time = self.current_departure.departure_time
        self.total_deadhead_time = 0.0
        self.invalid_actions = 0
        self.uncovered_departures = 0
        self.history = []
        self.current_candidates = []
        self.current_action_mask = np.zeros(
            self.target_action_size, dtype=np.float32
        )
        return self._build_observation()

    def step(self, action_index: int) -> Tuple[np.ndarray, float, bool, Dict[str, float]]:
        """Run one decision step at the current departure."""
        if self.current_departure is None:
            raise RuntimeError("Environment must be reset before stepping.")

        if (
            action_index < 0
            or action_index >= self.target_action_size
            or self.current_action_mask[action_index] == 0
        ):
            self.invalid_actions += 1
            action_index = self._fallback_action_index()

        candidate = self.current_candidates[action_index]
        if candidate is None:
            candidate = self._auto_assign_candidate()

        reward = self._apply_action(candidate)
        self._maybe_dispatch_deadheads()
        done = False

        self.current_step += 1
        if self.current_step >= self.total_departures:
            done = True
            reward += self._final_reward()
            next_state = np.zeros_like(self._blank_state_template())
            self.current_departure = None
        else:
            self.current_departure = self.departure_sequence[self.current_step]
            self.current_time = self.current_departure.departure_time
            next_state = self._build_observation()

        info = self._build_info(candidate)
        return next_state, reward, done, info

    @property
    def observation_dim(self) -> int:
        return int(self._blank_state_template().shape[0])

    @property
    def action_dim(self) -> int:
        return self.target_action_size

    @property
    def action_mask(self) -> np.ndarray:
        return self.current_action_mask.copy()

    @property
    def next_departure_time(self) -> float:
        if self.current_step + 1 < self.total_departures:
            return self.departure_sequence[self.current_step + 1].departure_time
        return self.current_time

    # ------------------------------------------------------------------ #
    # Observation and action helpers
    # ------------------------------------------------------------------ #
    def _blank_state_template(self) -> np.ndarray:
        cp_feature_size = self.num_cps * 5
        line_feature_size = 3
        bus_feature_size = self.target_action_size * 5
        total = cp_feature_size + line_feature_size + bus_feature_size
        return np.zeros(total, dtype=np.float32)

    def _build_observation(self) -> np.ndarray:
        self._sync_bus_positions()
        cp_features = self._encode_control_points()
        line_features = self._encode_current_line()
        candidate_info = self._build_action_candidates()
        self.current_candidates = candidate_info
        self.current_action_mask = np.array(
            [1.0 if cand is not None else 0.0 for cand in candidate_info],
            dtype=np.float32,
        )
        bus_features = self._encode_candidates(candidate_info)
        observation = np.concatenate([cp_features, line_features, bus_features]).astype(
            np.float32
        )
        return observation

    def _sync_bus_positions(self) -> None:
        """Release buses that have completed their trip/deadhead before current time."""
        for bus in self.buses.values():
            if bus.busy_reason and self.current_time >= bus.available_time:
                bus.current_cp = bus.destination_cp
                if bus.busy_reason == "trip":
                    bus.last_service_arrival = bus.available_time
                bus.busy_reason = None
                bus.destination_cp = None
            if bus.busy_reason is None:
                self._maybe_trigger_lunch(bus)

    def _encode_control_points(self) -> np.ndarray:
        features: List[float] = []
        now = self.current_time
        for cp_id in sorted(self.control_points):
            # Future demand counts (long-term + short-term horizon).
            n_l = self._count_future_departures(cp_id, now, math.inf)
            n_s = self._count_future_departures(cp_id, now, self.short_horizon)
            # Available bus counts (all buses vs buses that already served trips).
            n_a, n_o = self._count_buses_at_cp(cp_id)
            features.extend(
                [
                    self._normalize_cp_index(cp_id),
                    n_l,
                    n_s,
                    n_a,
                    n_o,
                ]
            )
        return np.array(features, dtype=np.float32)

    def _encode_current_line(self) -> np.ndarray:
        dep = self.current_departure
        assert dep is not None
        return np.array(
            [
                self._normalize_cp_index(dep.departure_cp),
                self._normalize_cp_index(dep.arrival_cp),
                dep.travel_time / self.time_norm,
            ],
            dtype=np.float32,
        )

    def _encode_candidates(self, candidates: Sequence[Optional[CandidateBus]]) -> np.ndarray:
        features: List[float] = []
        for cand in candidates:
            if cand is None:
                features.extend([0.0, 0.0, 0.0, 0.0, 0.0])
                continue
            bus = cand.bus
            cp_idx = (
                self._normalize_cp_index(bus.current_cp)
                if bus.current_cp is not None
                else 0.0
            )
            remaining_shift = self._normalize_time(
                self._remaining_shift_time(bus)
            )
            features.extend(
                [
                    1.0 if bus.used else 0.0,
                    self._normalize_time(cand.rest_time),
                    cp_idx,
                    self._normalize_time(cand.deadhead_time),
                    remaining_shift,
                ]
            )
        return np.array(features, dtype=np.float32)

    def _build_action_candidates(self) -> List[Optional[CandidateBus]]:
        dep = self.current_departure
        assert dep is not None
        available = []
        used_buses = []
        travel_time = self._travel_time(dep)
        for bus in self.buses.values():
            self._maybe_trigger_lunch(bus)
            # Skip buses that are still en-route or resting.
            if bus.busy_reason is not None:
                continue
            if bus.available_time > self.current_time:
                continue
            rest = self._rest_time(bus)
            if bus.current_cp == dep.departure_cp and rest >= self.min_rest:
                if not self._shift_can_finish(bus, 0.0, travel_time):
                    continue
                available.append((bus, False, 0.0, rest))
                if bus.used:
                    used_buses.append((bus, False, 0.0, rest))
            elif self.mode != "online":
                # Offline phase can consider deadheading before selection.
                deadhead = self._deadhead_time(bus.current_cp, dep.departure_cp)
                if math.isfinite(deadhead) and rest > deadhead + self.min_rest:
                    if not self._shift_can_finish(bus, deadhead, travel_time):
                        continue
                    available.append((bus, True, deadhead, rest))
                    if bus.used:
                        used_buses.append((bus, True, deadhead, rest))

        # Build sorted groups for priority screening
        used_no_deadhead = sorted(
            [
                CandidateBus(
                    bus=b,
                    requires_deadhead=req,
                    deadhead_time=dh,
                    rest_time=rest,
                    priority_reward=0.0,
                    source_cp=b.current_cp,
                )
                for b, req, dh, rest in available
                if b.used and not req
            ],
            key=lambda c: c.rest_time,
            reverse=True,
        )
        used_with_deadhead = sorted(
            [
                CandidateBus(
                    bus=b,
                    requires_deadhead=req,
                    deadhead_time=dh,
                    rest_time=rest,
                    priority_reward=0.0,
                    source_cp=b.current_cp,
                )
                for b, req, dh, rest in available
                if b.used and req
            ],
            key=lambda c: c.deadhead_time,
        )
        unused_no_deadhead = [
            CandidateBus(
                bus=b,
                requires_deadhead=req,
                deadhead_time=dh,
                rest_time=rest,
                priority_reward=0.0,
                source_cp=b.current_cp,
            )
            for b, req, dh, rest in available
            if not b.used and not req
        ]
        unused_with_deadhead = [
            CandidateBus(
                bus=b,
                requires_deadhead=req,
                deadhead_time=dh,
                rest_time=rest,
                priority_reward=0.0,
                source_cp=b.current_cp,
            )
            for b, req, dh, rest in available
            if not b.used and req
        ]

        ordered_groups: List[List[CandidateBus]] = []
        if self.mode == "offline":
            ordered_groups = [
                used_no_deadhead,
                used_with_deadhead,
                unused_no_deadhead,
                unused_with_deadhead,
            ]
        else:
            ordered_groups = [used_no_deadhead, unused_no_deadhead]

        candidates: List[Optional[CandidateBus]] = []
        for group in ordered_groups:
            for cand in group:
                candidates.append(cand)
                if len(candidates) == self.target_action_size:
                    break
            if len(candidates) == self.target_action_size:
                break
        
        while len(candidates) < self.target_action_size:
            candidates.append(None)

        self._assign_priority_rewards(candidates, used_buses)
        return candidates

    def _assign_priority_rewards(
        self, candidates: Sequence[Optional[CandidateBus]], used_buses: Sequence[Tuple[BusState, bool, float, float]]
    ) -> None:
        used_bus_entries = [entry for entry in used_buses]
        n_used = len(used_bus_entries)
        if n_used == 0:
            return
        # Rank buses by rest time so we can reward dispatching idle assets first.
        used_bus_entries.sort(key=lambda item: item[3], reverse=True)
        ranking: Dict[str, int] = {
            entry[0].bus_id: rank for rank, entry in enumerate(used_bus_entries)
        }
        for cand in candidates:
            if cand is None or not cand.bus.used:
                continue
            rank = ranking.get(cand.bus.bus_id, n_used - 1)
            cand.priority_reward = (n_used - rank) / n_used

    def _fallback_action_index(self) -> int:
        for idx, mask in enumerate(self.current_action_mask):
            if mask == 1:
                return idx
        # If nothing is available, create a bus on the fly
        self.uncovered_departures += 1
        self.current_candidates[0] = self._auto_assign_candidate()
        self.current_action_mask[0] = 1
        return 0

    def _auto_assign_candidate(self) -> CandidateBus:
        if self.current_departure is None:
            raise RuntimeError("Cannot auto assign without an active departure.")
        bus = self._spawn_bus(self.current_departure.departure_cp)
        return CandidateBus(
            bus=bus,
            requires_deadhead=False,
            deadhead_time=0.0,
            rest_time=float("inf"),
            priority_reward=0.0,
            source_cp=bus.current_cp,
        )

    def _spawn_bus(self, cp_id: str) -> BusState:
        if len(self.buses) >= self.config.fleet.max_buses:
            raise RuntimeError(
                "Fleet capacity exceeded. Increase 'fleet.max_buses' in the config."
            )
        bus_id = f"auto-{self.next_auto_bus_idx}"
        self.next_auto_bus_idx += 1
        bus = BusState(
            bus_id=bus_id,
            home_cp=cp_id,
            current_cp=cp_id,
            shift_duration=self.meta.shift_duration,
            min_work_time_before_lunch=self.meta.min_work_time_before_lunch,
            lunch_duration=self.meta.lunch_duration,
            used=False,
            available_time=0.0,
            last_service_arrival=-math.inf,
            busy_reason=None,
            destination_cp=None,
        )
        self.buses[bus_id] = bus
        return bus

    # ------------------------------------------------------------------ #
    # Reward, transitions and metrics
    # ------------------------------------------------------------------ #
    def _apply_action(self, candidate: CandidateBus) -> float:
        dep = self.current_departure
        assert dep is not None
        bus = candidate.bus
        was_used = bus.used
        bus.used = True
        if bus.shift_start_time is None:
            bus.shift_start_time = max(0.0, self.current_time - candidate.deadhead_time)
        bus.last_line_id = dep.line_id
        bus.busy_reason = "trip"
        bus.destination_cp = dep.arrival_cp
        travel_time = self._travel_time(dep)
        arrival_time = dep.departure_time + travel_time
        bus.available_time = arrival_time
        bus.current_cp = None

        self.total_deadhead_time += candidate.deadhead_time

        unused_penalty = 0.0 if was_used else 1.0
        deadhead_penalty = candidate.deadhead_time
        rest_reward = candidate.priority_reward if was_used else 0.0
        demand_penalty = 0.0
        if candidate.requires_deadhead and candidate.source_cp is not None:
            demand_penalty = self._demand_penalty(candidate.source_cp, dep.arrival_cp)

        weights = self.reward_weights
        step_reward = (
            -weights.step_unused_penalty * unused_penalty
            -weights.step_deadhead_penalty * deadhead_penalty
            +weights.step_rest_reward * rest_reward
            -weights.step_demand_penalty * demand_penalty
        )

        self.history.append(
            {
                "event": "trip",
                "departure_time": dep.departure_time,
                "line_id": dep.line_id,
                "bus_id": bus.bus_id,
                "deadhead_time": candidate.deadhead_time,
                "unused_penalty": unused_penalty,
                "rest_reward": rest_reward,
                "duration": travel_time,
            }
        )
        return step_reward

    # NOTE: Online-only helper.
    def _maybe_dispatch_deadheads(self) -> None:
        """Invoke the time-window planner between decisions when in online mode."""
        if self.deadhead_planner is None or self.current_departure is None:
            return
        self.deadhead_planner.plan(self)

    def _final_reward(self) -> float:
        weights = self.reward_weights
        buses_used = sum(1 for bus in self.buses.values() if bus.used)
        return (
            -weights.final_buses * buses_used
            -weights.final_deadhead * self.total_deadhead_time
        )

    def _build_info(self, candidate: CandidateBus) -> Dict[str, float]:
        buses_used = sum(1 for bus in self.buses.values() if bus.used)
        return {
            "current_time": self.current_time,
            "bus_id": candidate.bus.bus_id,
            "requires_deadhead": float(candidate.requires_deadhead),
            "deadhead_time": candidate.deadhead_time,
            "buses_used": float(buses_used),
            "total_deadhead_time": self.total_deadhead_time,
            "invalid_actions": float(self.invalid_actions),
            "uncovered_departures": float(self.uncovered_departures),
        }

    # ------------------------------------------------------------------ #
    # Utility functions
    # ------------------------------------------------------------------ #
    def _shift_can_finish(self, bus: BusState, deadhead_time: float, trip_time: float) -> bool:
        """Check whether the bus can finish the upcoming deadhead+trip before shift expiry."""
        upcoming = deadhead_time + trip_time
        if bus.shift_start_time is None:
            return upcoming <= bus.shift_duration
        elapsed = self.current_time - bus.shift_start_time
        if elapsed >= bus.shift_duration:
            return False
        return (elapsed + upcoming) <= bus.shift_duration

    def _remaining_shift_time(self, bus: BusState) -> float:
        """Return remaining minutes before the bus reaches its shift limit."""
        if bus.shift_start_time is None:
            return bus.shift_duration
        elapsed = self.current_time - bus.shift_start_time
        return max(0.0, bus.shift_duration - elapsed)

    def _needs_lunch(self, bus: BusState) -> bool:
        if bus.lunch_taken or bus.shift_start_time is None:
            return False
        elapsed = self.current_time - bus.shift_start_time
        return elapsed >= bus.min_work_time_before_lunch

    def _maybe_trigger_lunch(self, bus: BusState) -> None:
        if self._needs_lunch(bus) and bus.busy_reason is None:
            self._start_lunch(bus)

    def _start_lunch(self, bus: BusState) -> None:
        bus.busy_reason = "lunch"
        bus.lunch_taken = True
        bus.lunch_start_time = self.current_time
        if bus.shift_start_time is None:
            bus.shift_start_time = self.current_time
        bus.destination_cp = bus.current_cp
        bus.available_time = self.current_time + bus.lunch_duration
        self.history.append(
            {
                "event": "lunch",
                "bus_id": bus.bus_id,
                "start_time": self.current_time,
                "duration": bus.lunch_duration,
            }
        )

    def _normalize_cp_index(self, cp_id: Optional[str]) -> float:
        if cp_id is None:
            return 0.0
        return (self.cp_index[cp_id] + 1) / (self.num_cps + 1)

    def _normalize_time(self, minutes: float) -> float:
        if minutes in (math.inf, -math.inf):
            return 1.0
        return minutes / self.time_norm

    def _count_future_departures(
        self, cp_id: str, start_time: float, horizon: float
    ) -> int:
        times = self.cp_departure_times[cp_id]
        start_idx = bisect.bisect_left(times, start_time)
        if horizon == math.inf:
            return len(times) - start_idx
        end_time = start_time + horizon
        end_idx = bisect.bisect_right(times, end_time)
        return max(0, end_idx - start_idx)

    def _count_buses_at_cp(self, cp_id: str) -> Tuple[int, int]:
        total = 0
        used = 0
        for bus in self.buses.values():
            if bus.busy_reason is not None:
                continue
            if bus.current_cp == cp_id and bus.available_time <= self.current_time:
                total += 1
                if bus.used:
                    used += 1
        return total, used

    def _rest_time(self, bus: BusState) -> float:
        if not bus.used:
            return math.inf
        return max(0.0, self.current_time - bus.last_service_arrival)

    def _deadhead_time(self, origin: Optional[str], destination: str) -> float:
        if origin is None:
            origin = destination
        return self.config.deadhead_time(origin, destination)

    def _travel_time(self, departure: DepartureEvent) -> float:
        travel_time = departure.travel_time
        if self.mode == "online":
            for disturbance in self.config.disturbances:
                if disturbance.line_id and disturbance.line_id != departure.line_id:
                    continue
                if (
                    disturbance.start_minute is not None
                    and departure.departure_time < disturbance.start_minute
                ):
                    continue
                if (
                    disturbance.end_minute is not None
                    and departure.departure_time > disturbance.end_minute
                ):
                    continue
                travel_time += disturbance.delay_minutes
        return travel_time

    def _demand_penalty(self, source_cp: str, target_cp: str) -> float:
        demand_source = self._cp_demand_score(source_cp)
        demand_target = self._cp_demand_score(target_cp)
        return 1.0 if demand_source > demand_target else 0.0

    def _cp_demand_score(self, cp_id: str) -> float:
        n_s = self._count_future_departures(cp_id, self.current_time, self.short_horizon)
        _, n_o = self._count_buses_at_cp(cp_id)
        return n_s / (n_o + 1.0)

    # NOTE: Online-only helper.
    def dispatch_deadhead(
        self, bus_id: str, target_cp: str, deadhead_time: Optional[float] = None
    ) -> bool:
        """Dispatch a bus to perform a deadhead trip immediately."""
        bus = self.buses.get(bus_id)
        if bus is None or bus.busy_reason is not None:
            return False
        if deadhead_time is None:
            deadhead_time = self._deadhead_time(bus.current_cp, target_cp)
        if not math.isfinite(deadhead_time):
            return False
        # Respect shift duration while performing deadhead only movement.
        elapsed = 0.0
        if bus.shift_start_time is not None:
            elapsed = self.current_time - bus.shift_start_time
            if elapsed >= bus.shift_duration:
                return False
        remaining = bus.shift_duration - elapsed
        if remaining <= deadhead_time:
            return False
        if bus.shift_start_time is None:
            bus.shift_start_time = self.current_time
        bus.busy_reason = "deadhead"
        bus.destination_cp = target_cp
        bus.available_time = self.current_time + deadhead_time
        bus.current_cp = None
        self.total_deadhead_time += deadhead_time
        return True

    # NOTE: Online-only helper.
    def clone_for_planning(self) -> "RLMSAEnvironment":
        """Create a lightweight clone used by the time window planner."""
        clone = RLMSAEnvironment(
            self.config,
            mode=self.mode,
            deadhead_policy=None,
            auto_reset=False,
        )
        clone.buses = {bus_id: bus.clone() for bus_id, bus in self.buses.items()}
        clone.current_step = self.current_step
        clone.current_departure = self.current_departure
        clone.current_time = self.current_time
        clone.current_candidates = []
        clone.current_action_mask = np.zeros(self.target_action_size, dtype=np.float32)
        clone.reward_weights = self.reward_weights
        clone.training_profile = self.training_profile
        clone.total_deadhead_time = self.total_deadhead_time
        clone.invalid_actions = self.invalid_actions
        clone.uncovered_departures = self.uncovered_departures
        clone.history = list(self.history)
        clone.next_auto_bus_idx = self.next_auto_bus_idx
        clone.deadhead_planner = None
        return clone


# NOTE: Online-only helper.
class TimeWindowDeadheadPlanner:
    """Implements the time window mechanism powered by the offline policy."""

    def __init__(self, time_window: float, min_rest: float, policy) -> None:
        self.time_window = time_window
        self.min_rest = min_rest
        self.policy = policy

    def plan(self, env: RLMSAEnvironment) -> None:
        if self.policy is None or env.current_departure is None:
            return
        window_end = env.current_time + self.time_window
        planner_env = env.clone_for_planning()
        obs = planner_env._build_observation()

        while (
            planner_env.current_departure is not None
            and planner_env.current_departure.departure_time <= window_end
        ):
            mask = planner_env.action_mask
            action, _, _ = self.policy.act(obs, mask, deterministic=True)
            candidate = planner_env.current_candidates[action]
            if candidate and candidate.requires_deadhead:
                dep = planner_env.current_departure
                latest_depart = dep.departure_time - self.min_rest - candidate.deadhead_time
                if env.current_time <= latest_depart <= env.next_departure_time:
                    env.dispatch_deadhead(
                        candidate.bus.bus_id,
                        dep.departure_cp,
                        candidate.deadhead_time,
                    )
            obs, _, done, _ = planner_env.step(action)
            if done:
                break

