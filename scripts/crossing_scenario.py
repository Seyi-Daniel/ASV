"""Generate a deterministic crossing scenario for the ASV environment.

The utility spawns the default two-boat environment but replaces the random
spawn logic with a deterministic crossing encounter.  The scenario is repeated
five times, positioning the stand-on vessel at the bearings requested by the
user so that each run can be analysed individually.
"""

from __future__ import annotations

import argparse
import math
from typing import Iterable, Tuple

try:  # Allow running the script directly without installing the package.
    from asv.config import BoatParams, EnvConfig, SpawnConfig, TurnSessionConfig
    from asv.env import MultiBoatSectorsEnv
    from asv.scenarios import apply_crossing_layout, iter_crossing_angles
except ModuleNotFoundError:  # pragma: no cover - convenience path fix
    import importlib
    import sys
    import types
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    if "asv" not in sys.modules:
        stub = types.ModuleType("asv")
        stub.__path__ = [str(repo_root / "asv")]
        sys.modules["asv"] = stub

    config_module = importlib.import_module("asv.config")
    BoatParams = config_module.BoatParams  # type: ignore[misc]
    EnvConfig = config_module.EnvConfig  # type: ignore[misc]
    SpawnConfig = config_module.SpawnConfig  # type: ignore[misc]
    TurnSessionConfig = config_module.TurnSessionConfig  # type: ignore[misc]
    MultiBoatSectorsEnv = importlib.import_module("asv.env").MultiBoatSectorsEnv  # type: ignore[misc]
    scenarios = importlib.import_module("asv.scenarios")
    apply_crossing_layout = scenarios.apply_crossing_layout
    iter_crossing_angles = scenarios.iter_crossing_angles


def compute_starboard_bearing(agent, target) -> float:
    """Return the clockwise (starboard) relative bearing from ``agent`` to ``target``."""

    dx = target.x - agent.x
    dy = target.y - agent.y
    ch = math.cos(agent.h)
    sh = math.sin(agent.h)
    x_rel = ch * dx + sh * dy
    y_rel = -sh * dx + ch * dy
    rel_port = math.degrees(math.atan2(y_rel, x_rel))
    rel_port = (rel_port + 360.0) % 360.0
    return (360.0 - rel_port) % 360.0


def run_episode(
    env: MultiBoatSectorsEnv,
    *,
    angle: float,
    steps: int,
    cross_distance: float,
    agent_speed: float,
    stand_on_speed: float,
) -> Tuple[int, str]:
    """Run a single deterministic crossing scenario and return statistics."""

    apply_crossing_layout(
        env,
        angle,
        crossing_distance_agent=cross_distance,
        agent_speed=agent_speed,
        stand_on_speed=stand_on_speed,
    )

    agent = env.ships[0]
    stand_on = env.ships[1]
    bearing = compute_starboard_bearing(agent, stand_on)

    print("\n=== Crossing scenario ===")
    print(f"Stand-on bearing (requested) : {angle:6.2f}°")
    print(f"Stand-on bearing (realised)  : {bearing:6.2f}°")
    print(f"Agent position               : ({agent.x:7.2f}, {agent.y:7.2f}) m")
    print(f"Stand-on position            : ({stand_on.x:7.2f}, {stand_on.y:7.2f}) m")
    print(f"Agent heading                : {math.degrees(agent.h):6.2f}°")
    print(f"Stand-on heading             : {math.degrees(stand_on.h):6.2f}°")
    print(f"Agent speed                  : {agent.u:6.2f} m/s")
    print(f"Stand-on speed               : {stand_on.u:6.2f} m/s")

    reason = ""
    for step in range(steps):
        _, _, done, info = env.step([0, 0])
        if done:
            reason = str(info.get("reason", ""))
            print(f"Episode finished at step {step + 1} with reason: {reason}")
            break
    else:
        reason = "max_steps"
        print(f"Episode stopped after {steps} steps (reason: {reason})")

    return env.step_index, reason


def build_env(args: argparse.Namespace) -> MultiBoatSectorsEnv:
    """Construct the environment according to the CLI arguments."""

    spawn = SpawnConfig(n_boats=2, start_speed=0.0)
    margin = float(spawn.margin)
    max_offset = max(
        abs(args.cross_distance * math.tan(math.radians(angle))) for angle in iter_crossing_angles()
    )
    dim_requirement = max(
        args.world_size,
        2.0 * margin + 2.0 * max_offset + 80.0,
        2.0 * margin + 3.0 * args.cross_distance + 80.0,
    )

    cfg = EnvConfig(
        world_w=dim_requirement,
        world_h=dim_requirement,
        render=args.render,
        show_grid=args.show_grid,
        show_sectors=args.show_sectors,
        show_trails=args.show_trails,
        seed=args.seed,
        max_steps=args.max_steps,
    )
    env = MultiBoatSectorsEnv(cfg=cfg, kin=BoatParams(), tcfg=TurnSessionConfig(), spawn=spawn)
    env.reset(seed=args.seed)
    return env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=600, help="Number of steps to simulate per scenario")
    parser.add_argument("--max-steps", type=int, default=1200, help="Environment hard limit for steps")
    parser.add_argument(
        "--cross-distance",
        type=float,
        default=220.0,
        help="Longitudinal distance between the agent and the crossing point",
    )
    parser.add_argument("--agent-speed", type=float, default=7.0, help="Initial speed for the give-way vessel")
    parser.add_argument(
        "--stand-on-speed",
        type=float,
        default=7.0,
        help="Initial speed for the stand-on vessel",
    )
    parser.add_argument("--seed", type=int, default=13, help="Seed used when resetting the environment")
    parser.add_argument("--world-size", type=float, default=400.0, help="World width/height in metres")
    parser.add_argument("--render", action="store_true", help="Enable pygame rendering (if available)")
    parser.add_argument("--show-grid", action="store_true", help="Display the background grid when rendering")
    parser.add_argument("--show-sectors", action="store_true", help="Display the sensor sectors when rendering")
    parser.add_argument("--show-trails", action="store_true", help="Display ship trails when rendering")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = build_env(args)

    for idx, ang in enumerate(iter_crossing_angles(), start=1):
        env.reset(seed=args.seed + idx)
        step_count, reason = run_episode(
            env,
            angle=ang,
            steps=args.steps,
            cross_distance=args.cross_distance,
            agent_speed=args.agent_speed,
            stand_on_speed=args.stand_on_speed,
        )
        print(f"Summary: scenario {idx} finished after {step_count} steps (reason: {reason}).")


if __name__ == "__main__":
    main()

