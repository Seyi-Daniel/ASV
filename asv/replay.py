"""Experience replay buffer utilities."""
from __future__ import annotations

import random
from collections import deque, namedtuple
from typing import Deque

Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))


class Replay:
    """Simple FIFO replay buffer."""

    def __init__(self, capacity: int) -> None:
        self.buf: Deque[Transition] = deque(maxlen=capacity)

    def push(self, *args) -> None:
        self.buf.append(Transition(*args))

    def sample(self, batch_size: int) -> Transition:
        batch = random.sample(self.buf, batch_size)
        return Transition(*zip(*batch))

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.buf)
