"""Predefined scenario layouts for deterministic evaluations.

This module contains helpers to build deterministic situations for the
multi-boat environment so that experiments can be reproduced easily.  The
focus is on providing a *crossing* encounter between two vessels where the
agent-controlled craft acts as the give-way vessel and the second craft is
the stand-on vessel placed at specific relative bearings.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Iterable, Sequence, Tuple

from .env import MultiBoatSectorsEnv
from .utils import clamp

# ---------------------------------------------------------------------------
# Crossing scenario helpers
# ---------------------------------------------------------------------------

# Degree bounds provided for the stand-on vessel relative to the give-way
# (agent controlled) vessel.  The user requested a single scenario to be
# repeated with the stand-on craft positioned at five distinct bearings:
#   1. Exactly at the lower bound of 5 degrees.
#   2. The midpoint between the bounds.
#   3. Exactly at the upper bound of 112.5 degrees.
#   4. The quarter point between the bounds.
#   5. The three-quarter point between the bounds.
#
# The order below follows the wording of the request even though it is not
# monotonic.  This makes it very explicit which encounter corresponds to each
# identifier when iterating over the list.
STAND_ON_BEARINGS_DEG: Tuple[float, ...] = (
    5.0,
    (5.0 + 112.5) / 2.0,
    112.5,
    5.0 + 0.25 * (112.5 - 5.0),
    5.0 + 0.75 * (112.5 - 5.0),
)


def iter_crossing_angles() -> Sequence[float]:
    """Return the ordered bearings for the stand-on vessel.

    The values are supplied as a convenience for scripts that want to iterate
    over all requested encounters.  A tuple is returned (instead of a list) so
    the caller cannot accidentally mutate the canonical ordering.
    """

    return STAND_ON_BEARINGS_DEG


def apply_crossing_layout(
    env: MultiBoatSectorsEnv,
    stand_on_angle_deg: float,
    *,
    crossing_distance_agent: float = 220.0,
    stand_on_speed: float = 7.0,
    agent_speed: float = 7.0,
) -> None:
    """Impose a deterministic crossing encounter on ``env``.

    Parameters
    ----------
    env:
        Environment instance with at least two vessels already created via
        ``reset``.
    stand_on_angle_deg:
        Desired relative bearing (in degrees) of the stand-on vessel measured
        clockwise from the give-way vessel's bow.  The value is expected to be
        within ``[5.0, 112.5]`` as requested by the user.
    crossing_distance_agent:
        Longitudinal distance (in metres) that separates the give-way vessel
        from the nominal crossing point along its track.  Increasing this
        pushes the agent further away from the intersection which provides the
        vessels with more time to react.
    stand_on_speed / agent_speed:
        The initial speeds assigned to the vessels.  The dynamics code keeps
        the speed constant when the throttle command is neutral, so the script
        that uses this helper can simply apply a no-op action each step.

    Notes
    -----
    The layout assumes that boat ``0`` (the agent) travels north (heading
    ``+90°``) whereas boat ``1`` (the stand-on vessel) travels west.  The
    stand-on vessel is positioned so that its relative bearing with respect to
    the agent matches ``stand_on_angle_deg``.  A constant separation is used
    for the agent along-track distance and the transverse offset is derived by
    enforcing the requested bearing.
    """

    if len(env.ships) < 2:
        raise ValueError("Crossing scenario requires at least two vessels")

    # Give-way (agent controlled) boat is always index 0.
    agent = env.ships[0]
    stand_on = env.ships[1]

    # Anchor the scenario around a single crossing point.  Boats are positioned
    # relative to this point to keep them well inside the configured world.
    margin = max(float(env.spawn.margin), 40.0)
    usable_width = max(env.world_w - 2.0 * margin, 1.0)
    crossing_x = margin + 0.5 * usable_width
    crossing_y = clamp(margin + 1.5 * crossing_distance_agent, margin, env.world_h - margin)

    agent_heading = math.pi / 2.0  # 90 degrees, travelling north.
    stand_on_heading = math.pi  # 180 degrees, travelling west.

    # Convert the requested clockwise starboard bearing into offsets for both
    # vessels relative to the shared crossing point.  ``offset_agent`` moves the
    # give-way vessel along its track whereas ``offset_stand`` shifts the
    # stand-on vessel laterally so that the desired bearing is reproduced.
    beta_rad = math.radians(stand_on_angle_deg)
    offset_agent = crossing_distance_agent * math.cos(beta_rad)
    offset_stand = crossing_distance_agent * math.sin(beta_rad)
    max_x = env.world_w - margin
    min_x = margin
    crossing_x = clamp(crossing_x, min_x, max_x)
    if crossing_x + offset_stand > max_x:
        crossing_x = clamp(max_x - offset_stand, min_x, max_x)
    if crossing_x + offset_stand < min_x:
        crossing_x = clamp(min_x - offset_stand, min_x, max_x)

    agent.x = crossing_x
    agent.y = clamp(crossing_y - offset_agent, margin, env.world_h - margin)
    agent.h = agent_heading
    agent.u = agent_speed
    agent.last_thr = 0
    agent.last_helm = 0
    agent.session_active = False
    agent.session_dir = 0

    stand_on.x = clamp(crossing_x + offset_stand, margin, env.world_w - margin)
    stand_on.y = clamp(crossing_y, margin, env.world_h - margin)
    stand_on.h = stand_on_heading
    stand_on.u = stand_on_speed
    stand_on.last_thr = 0
    stand_on.last_helm = 0
    stand_on.session_active = False
    stand_on.session_dir = 0

    env.goals[0] = (
        agent.x,
        clamp(agent.y + 2.5 * crossing_distance_agent, margin, env.world_h - margin),
    )
    env.goals[1] = (
        clamp(stand_on.x - 2.5 * crossing_distance_agent, margin, env.world_w - margin),
        stand_on.y,
    )

    # Reset status tracking to reflect the new deterministic layout.
    env.reached = [False for _ in env.ships]
    env.prev_goal_d = [
        math.hypot(env.goals[i][0] - ship.x, env.goals[i][1] - ship.y)
        for i, ship in enumerate(env.ships)
    ]
    env._traces = [deque(maxlen=600) for _ in env.ships]
    env.time = 0.0
    env.step_index = 0


def configure_crossing_episodes(
    env: MultiBoatSectorsEnv,
    angles: Iterable[float] = STAND_ON_BEARINGS_DEG,
) -> Tuple[Tuple[float, float], ...]:
    """Prepare a batch of crossing encounters.

    The helper iterates over ``angles`` and applies the layout to the
    environment for each value.  The resulting tuples contain the (x, y)
    coordinates of the stand-on vessel for convenience when reporting or
    logging the scenarios.
    """

    positions = []
    for ang in angles:
        env.reset(env.cfg.seed)
        apply_crossing_layout(env, ang)
        stand_on = env.ships[1]
        positions.append((stand_on.x, stand_on.y))
    return tuple(positions)

