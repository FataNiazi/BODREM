import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agents.network import QNetwork
from agents.replay_buffer import ReplayBuffer

BATCH_SIZE      = 64
REPLAY_CAPACITY = 10_000
GAMMA           = 0.99
LR              = 1e-3
EPS_START       = 1.0
EPS_END         = 0.05
EPS_DECAY       = 0.995


class DQNAgent:
    def __init__(self, state_dim: int, action_dim: int):
        self.action_dim = action_dim
        self.epsilon    = EPS_START

        self.q_net      = QNetwork(state_dim, action_dim)
        self.target_net = QNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=LR)
        self.buffer    = ReplayBuffer(REPLAY_CAPACITY)

    def select_action(self, state: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            return int(self.q_net(s).argmax(dim=1).item())

    def learn(self):
        if len(self.buffer) < BATCH_SIZE:
            return

        states, actions, rewards, next_states, dones = self.buffer.sample(BATCH_SIZE)

        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1).values
            targets    = rewards + GAMMA * max_next_q * (1 - dones)

        loss = nn.functional.mse_loss(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def decay_epsilon(self):
        self.epsilon = max(EPS_END, self.epsilon * EPS_DECAY)

    def sync_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())
