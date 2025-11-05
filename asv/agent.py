"""DQN agent logic."""
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn

from .hyperparams import TrainingHyperParameters
from .models import QNet
from .replay import Replay


@dataclass
class ActResult:
    action: int
    was_random: bool
    epsilon: float


class Agent:
    """Double-DQN agent with epsilon-greedy exploration."""

    def __init__(self, in_dim: int, n_actions: int, hp: TrainingHyperParameters, device: torch.device) -> None:
        self.q = QNet(in_dim, n_actions).to(device)
        self.t = QNet(in_dim, n_actions).to(device)
        self.t.load_state_dict(self.q.state_dict())
        self.optim = optim.Adam(self.q.parameters(), lr=hp.lr)

        self.n_actions = n_actions
        self.gamma = hp.gamma
        self.device = device

        self.eps_end = hp.eps_end
        self.eps_value = hp.eps_start
        if hp.eps_decay > 0.0:
            self.eps_decay = hp.eps_decay
        else:
            self.eps_decay = (hp.eps_end / hp.eps_start) ** (1.0 / max(1, hp.eps_decay_episodes))
        self.global_step = 0

    def epsilon(self) -> float:
        return self.eps_value

    def decay_epsilon_episode(self) -> None:
        self.eps_value = max(self.eps_end, self.eps_value * self.eps_decay)

    def act(self, state: np.ndarray) -> ActResult:
        eps = self.epsilon()
        if random.random() < eps:
            return ActResult(random.randrange(self.n_actions), True, eps)
        with torch.no_grad():
            s = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            q_values = self.q(s)[0]
            return ActResult(int(torch.argmax(q_values).item()), False, eps)

    def update(self, replay: Replay, batch_size: int) -> float | None:
        if len(replay) < max(batch_size, 1):
            return None

        batch = replay.sample(batch_size)
        s = torch.from_numpy(np.stack(batch.state)).float().to(self.device)
        a = torch.tensor(batch.action, dtype=torch.long, device=self.device).unsqueeze(1)
        r = torch.tensor(batch.reward, dtype=torch.float32, device=self.device).unsqueeze(1)
        ns = torch.from_numpy(np.stack(batch.next_state)).float().to(self.device)
        d = torch.tensor(batch.done, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_sa = self.q(s).gather(1, a)
        with torch.no_grad():
            next_best = torch.argmax(self.q(ns), dim=1, keepdim=True)
            t_q = self.t(ns).gather(1, next_best)
            target = r + (1.0 - d) * self.gamma * t_q

        loss = F.smooth_l1_loss(q_sa, target)
        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 5.0)
        self.optim.step()
        return float(loss.item())

    def hard_update(self) -> None:
        self.t.load_state_dict(self.q.state_dict())
