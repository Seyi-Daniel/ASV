"""Deterministic crossing scenario generator with optional rendering."""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from asv import BoatParams, CrossingScenarioEnv, EnvConfig, TurnSessionConfig

# Ordered bearings, matching the exact positions requested by the user.
STAND_ON_BEARINGS_DEG: Tuple[float, ...] = (
    5.0,
    (5.0 + 112.5) / 2.0,
    112.5,
    5.0 + 0.25 * (112.5 - 5.0),
    5.0 + 0.75 * (112.5 - 5.0),
)


@dataclass(frozen=True)
class VesselState:
    """Description of a vessel participating in the encounter."""

    name: str
    x: float
    y: float
    heading_deg: float
    speed: float

    def bearing_to(self, other: "VesselState") -> float:
        """Return clockwise (starboard) relative bearing to ``other`` in degrees."""

        dx = other.x - self.x
        dy = other.y - self.y
        ch = math.cos(math.radians(self.heading_deg))
        sh = math.sin(math.radians(self.heading_deg))
        x_rel = ch * dx + sh * dy
        y_rel = -sh * dx + ch * dy
        rel_port = math.degrees(math.atan2(y_rel, x_rel))
        rel_port = (rel_port + 360.0) % 360.0
        return (360.0 - rel_port) % 360.0


@dataclass(frozen=True)
class CrossingScenario:
    """Container for the initial geometry of a crossing encounter."""

    agent: VesselState
    stand_on: VesselState
    crossing_point: Tuple[float, float]
    requested_bearing: float

    def describe(self) -> str:
        """Create a human-readable summary of the encounter."""

        bearing = self.agent.bearing_to(self.stand_on)
        return (
            f"Stand-on bearing (requested) : {self.requested_bearing:6.2f}°\n"
            f"Stand-on bearing (realised)  : {bearing:6.2f}°\n"
            f"Agent position               : ({self.agent.x:7.2f}, {self.agent.y:7.2f}) m\n"
            f"Stand-on position            : ({self.stand_on.x:7.2f}, {self.stand_on.y:7.2f}) m\n"
            f"Agent heading                : {self.agent.heading_deg:6.2f}°\n"
            f"Stand-on heading             : {self.stand_on.heading_deg:6.2f}°\n"
            f"Agent speed                  : {self.agent.speed:6.2f} m/s\n"
            f"Stand-on speed               : {self.stand_on.speed:6.2f} m/s"
        )


@dataclass(frozen=True)
class ScenarioRequest:
    """User-controllable parameters for the crossing scenario."""

    crossing_distance: float = 220.0
    agent_speed: float = 7.0
    stand_on_speed: float = 7.0


def compute_crossing_geometry(angle_deg: float, request: ScenarioRequest) -> CrossingScenario:
    """Create the crossing encounter for a single bearing value."""

    beta = math.radians(angle_deg)
    crossing_point = (0.0, 0.0)

    offset_agent = request.crossing_distance * math.cos(beta)
    offset_stand = request.crossing_distance * math.sin(beta)

    agent = VesselState(
        name="give_way",
        x=0.0,
        y=-offset_agent,
        heading_deg=90.0,
        speed=request.agent_speed,
    )

    stand_on = VesselState(
        name="stand_on",
        x=offset_stand,
        y=0.0,
        heading_deg=180.0,
        speed=request.stand_on_speed,
    )

    return CrossingScenario(
        agent=agent,
        stand_on=stand_on,
        crossing_point=crossing_point,
        requested_bearing=angle_deg,
    )


def iter_scenarios(
    angles: Iterable[float], request: ScenarioRequest
) -> Iterable[CrossingScenario]:
    """Yield scenarios for each provided bearing value."""

    for ang in angles:
        yield compute_crossing_geometry(ang, request)


def build_env(args: argparse.Namespace) -> CrossingScenarioEnv:
    cfg = EnvConfig(
        world_w=args.world_width,
        world_h=args.world_height,
        dt=args.dt,
        substeps=args.substeps,
        render=args.render,
        pixels_per_meter=args.pixels_per_meter,
        show_grid=not args.hide_grid,
        show_trails=not args.hide_trails,
        show_hud=not args.hide_hud,
    )
    return CrossingScenarioEnv(cfg=cfg, kin=BoatParams(), tcfg=TurnSessionConfig())


def scenario_states_for_env(env: CrossingScenarioEnv, scenario: CrossingScenario) -> Tuple[list, dict]:
    """Convert the dataclass description into environment-specific state dictionaries."""

    cx = env.world_w / 2.0
    cy = env.world_h / 2.0
    cross_x = cx + scenario.crossing_point[0]
    cross_y = cy + scenario.crossing_point[1]

    def convert(vessel: VesselState) -> dict:
        return {
            "x": cross_x + vessel.x,
            "y": cross_y + vessel.y,
            "heading": math.radians(vessel.heading_deg),
            "speed": vessel.speed,
        }

    states = [convert(scenario.agent), convert(scenario.stand_on)]
    meta = {
        "bearing": scenario.requested_bearing,
        "cross_x": cross_x,
        "cross_y": cross_y,
    }
    return states, meta


def run_render_loop(env: CrossingScenarioEnv, scenario: CrossingScenario, duration: float) -> None:
    states, meta = scenario_states_for_env(env, scenario)
    env.reset_from_states(states, meta=meta)

    steps = max(1, int(round(duration / env.cfg.dt)))
    env.render()
    for _ in range(steps):
        env.step()
        env.render()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cross-distance",
        type=float,
        default=ScenarioRequest.crossing_distance,
        help="Longitudinal distance between the agent and the crossing point in metres.",
    )
    parser.add_argument(
        "--agent-speed",
        type=float,
        default=ScenarioRequest.agent_speed,
        help="Initial speed for the give-way vessel in m/s.",
    )
    parser.add_argument(
        "--stand-on-speed",
        type=float,
        default=ScenarioRequest.stand_on_speed,
        help="Initial speed for the stand-on vessel in m/s.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Enable pygame rendering of each deterministic scenario.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=35.0,
        help="Simulation time per scenario when rendering (seconds).",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=EnvConfig.dt,
        help="Simulation step in seconds.",
    )
    parser.add_argument(
        "--substeps",
        type=int,
        default=EnvConfig.substeps,
        help="Number of integration substeps per frame.",
    )
    parser.add_argument(
        "--world-width",
        type=float,
        default=EnvConfig.world_w,
        help="World width in metres for the renderer.",
    )
    parser.add_argument(
        "--world-height",
        type=float,
        default=EnvConfig.world_h,
        help="World height in metres for the renderer.",
    )
    parser.add_argument(
        "--pixels-per-meter",
        type=float,
        default=EnvConfig.pixels_per_meter,
        help="Display scaling factor for rendering.",
    )
    parser.add_argument(
        "--hide-grid",
        action="store_true",
        help="Disable the background grid overlay.",
    )
    parser.add_argument(
        "--hide-trails",
        action="store_true",
        help="Disable vessel motion trails.",
    )
    parser.add_argument(
        "--hide-hud",
        action="store_true",
        help="Disable the on-screen HUD panel.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    request = ScenarioRequest(
        crossing_distance=args.cross_distance,
        agent_speed=args.agent_speed,
        stand_on_speed=args.stand_on_speed,
    )

    scenarios = list(iter_scenarios(STAND_ON_BEARINGS_DEG, request))

    print("Deterministic crossing scenarios (give-way vs stand-on)")
    print("=" * 56)
    for idx, scenario in enumerate(scenarios, start=1):
        print(f"\nScenario {idx}:")
        print(scenario.describe())

    env = build_env(args)

    if args.render and env._screen is None:
        # Rendering was requested but pygame is unavailable or initialisation failed.
        raise RuntimeError("pygame could not be initialised; rendering is unavailable.")

    try:
        if args.render:
            print("\nRendering scenarios — close the window or press ESC to exit early.")
            for idx, scenario in enumerate(scenarios, start=1):
                print(
                    f"  • Scenario {idx}: bearing {scenario.requested_bearing:6.2f}°"
                )
                run_render_loop(env, scenario, args.duration)
        else:
            print("\nRendering disabled; use --render to open the pygame visualisation.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
