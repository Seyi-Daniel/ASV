"""Deterministic crossing encounter generator.

This script produces the five stand-on bearings requested by the user for a
single crossing scenario.  The give-way (agent controlled) vessel travels north
while the stand-on vessel travels west.  Both vessels start the scenario at the
same longitudinal separation from the crossing point; changing that separation
modifies how much time they have before reaching the intersection.

Running the script prints a concise summary of the initial geometry for each of
the five angles.  The output is intentionally textual so that it can be used in
any environment without graphical dependencies.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Iterable, Tuple

# Ordered bearings, matching the exact positions requested by the user.  The
# sequence includes the lower bound, mid-point, upper bound, quarter point, and
# three-quarter point of the interval [5°, 112.5°].
STAND_ON_BEARINGS_DEG: Tuple[float, ...] = (
    5.0,
    (5.0 + 112.5) / 2.0,
    112.5,
    5.0 + 0.25 * (112.5 - 5.0),
    5.0 + 0.75 * (112.5 - 5.0),
)


@dataclass(frozen=True)
class VesselState:
    """Lightweight description of a vessel participating in the encounter."""

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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    request = ScenarioRequest(
        crossing_distance=args.cross_distance,
        agent_speed=args.agent_speed,
        stand_on_speed=args.stand_on_speed,
    )

    print("Deterministic crossing scenarios (give-way vs stand-on)")
    print("=" * 56)
    for idx, scenario in enumerate(iter_scenarios(STAND_ON_BEARINGS_DEG, request), start=1):
        print(f"\nScenario {idx}:")
        print(scenario.describe())


if __name__ == "__main__":
    main()
