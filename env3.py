"""Backward-compatibility wrapper for the refactored environment package."""
from __future__ import annotations

from asv.config import BoatParams, EnvConfig, SpawnConfig, TurnSessionConfig
from asv.env import MultiBoatSectorsEnv

__all__ = [
    "BoatParams",
    "EnvConfig",
    "SpawnConfig",
    "TurnSessionConfig",
    "MultiBoatSectorsEnv",
]
