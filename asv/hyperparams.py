"""Hyperparameter management for experiments and training."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any, Dict, Optional

from .config import BoatParams, EnvConfig, SpawnConfig, TurnSessionConfig


@dataclass
class TrainingHyperParameters:
    """Training schedule and optimisation hyperparameters."""

    episodes: int = 1500
    steps_per_episode: int = 3000
    gamma: float = 0.995

    lr: float = 2e-4
    batch_size: int = 256
    replay_size: int = 200_000
    min_replay: int = 10_000
    target_update: int = 4000

    eps_start: float = 1.0
    eps_end: float = 0.10
    eps_decay_episodes: int = 3000
    eps_decay: float = 0.0

    save_every: int = 50
    seed: int = 0
    render: bool = False
    show_hud: bool = True

    log_actions: bool = True
    actions_filename: str = "actions_all.csv"
    print_action_log: bool = False

    n_boats: Optional[int] = None

    turn_deg: float = 15.0
    turn_rate_degps: float = 45.0
    hysteresis_deg: float = 1.5
    allow_cancel: bool = False
    passthrough_throttle: bool = True
    hold_thr_while_turning: bool = False
    substeps: int = 1

    cpa_horizon: float = 60.0
    tcpa_decay: float = 20.0
    dcpa_scale: float = 120.0
    risk_weight: float = 0.10
    colregs_penalty: float = 0.05

    save_best: bool = True
    ema_beta: float = 0.98
    best_filename: str = "q_best.pth"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GlobalHyperParameters:
    """Collect all hyperparameters relevant to an experiment."""

    training: TrainingHyperParameters = field(default_factory=TrainingHyperParameters)
    env: EnvConfig = field(default_factory=EnvConfig)
    boat: BoatParams = field(default_factory=BoatParams)
    turn: TurnSessionConfig = field(default_factory=TurnSessionConfig)
    spawn: SpawnConfig = field(default_factory=SpawnConfig)

    def synchronize(self) -> None:
        """Propagate training-driven defaults into the other configs."""
        hp = self.training
        self.env.render = hp.render
        self.env.show_hud = hp.show_hud
        self.env.seed = hp.seed
        self.env.substeps = hp.substeps
        self.env.max_steps = hp.steps_per_episode
        self.env.cpa_horizon = hp.cpa_horizon
        self.env.tcpa_decay = hp.tcpa_decay
        self.env.dcpa_scale = hp.dcpa_scale
        self.env.risk_weight = hp.risk_weight
        self.env.colregs_penalty = hp.colregs_penalty

        self.turn.turn_deg = hp.turn_deg
        self.turn.turn_rate_degps = hp.turn_rate_degps
        self.turn.hysteresis_deg = hp.hysteresis_deg
        self.turn.allow_cancel = hp.allow_cancel
        self.turn.passthrough_throttle = hp.passthrough_throttle
        self.turn.hold_throttle_while_turning = hp.hold_thr_while_turning

        if hp.n_boats is not None:
            self.spawn.n_boats = hp.n_boats

    def to_nested_dict(self) -> Dict[str, Any]:
        """Return a nested dictionary of all hyperparameters."""
        return {
            "training": asdict(self.training),
            "env": asdict(self.env),
            "boat": asdict(self.boat),
            "turn": asdict(self.turn),
            "spawn": asdict(self.spawn),
        }

    def to_flat_dict(self) -> Dict[str, Any]:
        """Return a flattened dictionary with ``section.key`` entries."""
        result: Dict[str, Any] = {}
        for name in ("training", "env", "boat", "turn", "spawn"):
            section = getattr(self, name)
            if is_dataclass(section):
                for field_ in fields(section):
                    result[f"{name}.{field_.name}"] = getattr(section, field_.name)
        return result
