from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class ReplayBuffer:
    def __init__(self, capacity: int = 10_000):
        self.capacity = int(capacity)
        self.buffer: list[tuple[np.ndarray, int, float, np.ndarray, float]] = []
        self.pos = 0

    def push(self, state, action, reward, next_state, done) -> None:
        item = (
            np.asarray(state, dtype=np.float32),
            int(action),
            float(reward),
            np.asarray(next_state, dtype=np.float32),
            float(done),
        )
        if len(self.buffer) < self.capacity:
            self.buffer.append(item)
        else:
            self.buffer[self.pos] = item
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int, device: str):
        idx = np.random.choice(len(self.buffer), size=int(batch_size), replace=False)
        batch = [self.buffer[i] for i in idx]
        s, a, r, ns, d = zip(*batch)
        states = torch.tensor(np.stack(s), dtype=torch.float32, device=device)
        actions = torch.tensor(a, dtype=torch.long, device=device)
        rewards = torch.tensor(r, dtype=torch.float32, device=device)
        next_states = torch.tensor(np.stack(ns), dtype=torch.float32, device=device)
        dones = torch.tensor(d, dtype=torch.float32, device=device)
        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self.buffer)


class QNetwork(nn.Module):
    def __init__(self, state_dim: int = 2, n_actions: int = 3, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


@dataclass
class DQNConfig:
    state_dim: int = 2
    n_actions: int = 3
    hidden_dim: int = 128
    lr: float = 1e-3
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    buffer_capacity: int = 10_000
    batch_size: int = 64
    target_update_freq: int = 10


class DQNAgent:
    def __init__(self, cfg: DQNConfig, device: str = "cpu"):
        self.cfg = cfg
        self.device = device

        self.policy_net = QNetwork(cfg.state_dim, cfg.n_actions, cfg.hidden_dim).to(device)
        self.target_net = QNetwork(cfg.state_dim, cfg.n_actions, cfg.hidden_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.lr)
        self.memory = ReplayBuffer(cfg.buffer_capacity)

        self.epsilon = cfg.epsilon_start
        self.training_losses: list[float] = []
        self._episode_counter = 0

    def act(self, state: np.ndarray, training: bool = True) -> int:
        if training and random.random() < self.epsilon:
            return random.randrange(self.cfg.n_actions)
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.policy_net(s)
            return int(q.argmax(dim=1).item())

    def store(self, s, a, r, ns, done) -> None:
        self.memory.push(s, a, r, ns, done)

    def learn_step(self) -> float | None:
        if len(self.memory) < self.cfg.batch_size:
            return None
        s, a, r, ns, d = self.memory.sample(self.cfg.batch_size, self.device)

        q = self.policy_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            nq = self.target_net(ns).max(dim=1)[0]
            target = r + self.cfg.gamma * nq * (1.0 - d)

        loss = nn.MSELoss()(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        loss_value = float(loss.item())
        self.training_losses.append(loss_value)
        return loss_value

    def end_episode(self) -> None:
        self._episode_counter += 1
        self.epsilon = max(self.cfg.epsilon_end, self.epsilon * self.cfg.epsilon_decay)
        if (self._episode_counter % self.cfg.target_update_freq) == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def state_payload(self) -> dict[str, Any]:
        return {
            "policy_state_dict": self.policy_net.state_dict(),
            "target_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "training_losses": list(self.training_losses),
            "hparams": vars(self.cfg),
        }

    def load_payload(self, payload: dict[str, Any]) -> None:
        self.policy_net.load_state_dict(payload["policy_state_dict"])
        self.target_net.load_state_dict(payload.get("target_state_dict", payload["policy_state_dict"]))
        if "optimizer_state_dict" in payload:
            self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        self.epsilon = float(payload.get("epsilon", self.cfg.epsilon_end))
        self.training_losses = list(payload.get("training_losses", []))


def clone_agent(agent: DQNAgent) -> DQNAgent:
    cloned = make_agent_from_hparams(vars(agent.cfg), device=agent.device)
    cloned.load_payload(agent.state_payload())
    return cloned


def make_agent_from_hparams(hparams: dict | None, device: str = "cpu") -> DQNAgent:
    if hparams is None:
        cfg = DQNConfig()
    else:
        filtered = {k: v for k, v in hparams.items() if k in DQNConfig.__annotations__}
        cfg = DQNConfig(**filtered)
    return DQNAgent(cfg=cfg, device=device)
