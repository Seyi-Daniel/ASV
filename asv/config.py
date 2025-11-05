"""Configuration dataclasses for the ASV environment."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class BoatParams:
    """Geometry and kinematic limits for a vessel."""
    length: float = 6.0
    width: float = 2.2

    max_speed: float = 18.0
    min_speed: float = 0.0
    accel_rate: float = 1.6
    decel_rate: float = 1.2


@dataclass
class TurnSessionConfig:
    """Discrete chunked turn behaviour configuration."""
    turn_deg: float = 15.0
    turn_rate_degps: float = 45.0
    allow_cancel: bool = False
    hysteresis_deg: float = 1.5

    passthrough_throttle: bool = True
    hold_throttle_while_turning: bool = False


@dataclass
class SpawnConfig:
    """Initial spawn placement and goal configuration."""
    n_boats: int = 2
    start_speed: float = 0.0

    margin: float = 80.0
    min_sep_factor: float = 2.0

    goal_ahead_distance: float = 450.0
    goal_radius: float = 10.0
    goal_edge_clearance: float = 20.0

    max_spawn_attempts: int = 1000


@dataclass
class EnvConfig:
    """Environment configuration."""
    world_w: float = 100.0
    world_h: float = 100.0
    dt: float = 0.05
    substeps: int = 1
    sensor_range: Optional[float] = None
    seed: Optional[int] = None

    render: bool = False
    pixels_per_meter: float = 10.0
    show_grid: bool = True
    show_sectors: bool = False
    show_trails: bool = True
    show_hud: bool = True

    progress_weight: float = 0.01
    living_penalty: float = -0.001
    goal_bonus: float = 6.0
    collision_penalty: float = -12.0
    oob_penalty: float = -3.0
    max_steps_penalty: float = -0.5

    max_steps: int = 3000

    cpa_horizon: float = 60.0
    tcpa_decay: float = 20.0
    dcpa_scale: float = 120.0
    risk_weight: float = 0.10
    colregs_penalty: float = 0.05
