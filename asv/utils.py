"""Utility helpers for the ASV project."""
from __future__ import annotations

import math
from typing import Tuple


def wrap_pi(angle: float) -> float:
    """Wrap an angle to the range ``(-pi, pi]``."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def angle_deg(angle: float) -> float:
    """Convert radians to degrees."""
    return angle * 180.0 / math.pi


def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp ``value`` to the inclusive range ``[lo, hi]``."""
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def tcpa_dcpa(
    ax: float,
    ay: float,
    ah: float,
    aspd: float,
    bx: float,
    by: float,
    bh: float,
    bspd: float,
) -> Tuple[float, float]:
    """Compute time and distance to closest approach for two moving points."""
    rel_vx = math.cos(bh) * bspd - math.cos(ah) * aspd
    rel_vy = math.sin(bh) * bspd - math.sin(ah) * aspd
    rx, ry = (bx - ax), (by - ay)
    rv2 = rel_vx * rel_vx + rel_vy * rel_vy
    if rv2 < 1e-9:
        return 0.0, math.hypot(rx, ry)
    tcpa = - (rx * rel_vx + ry * rel_vy) / rv2
    if tcpa < 0.0:
        tcpa = 0.0
    cx = rx + rel_vx * tcpa
    cy = ry + rel_vy * tcpa
    return tcpa, math.hypot(cx, cy)
