"""
DQN training for MountainCar-v0
State:  [position, velocity]  (2 continuous values)
Actions: 0=push left, 1=no push, 2=push right
Goal:   reach position >= 0.5
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import gymnasium as gym
from agents.dqn_agent import DQNAgent
from domain_randomization import BayesianDomainRandomizer

EPISODES      = 500
MAX_STEPS     = 200
TARGET_UPDATE = 10
SAVE_PATH     = "models/dqn_mountaincar.pt"


def _apply_params(env: gym.Env, params: dict) -> None:
    """Push sampled BDR parameters onto the unwrapped MountainCar env."""
    uw = env.unwrapped
    if "gravity"   in params: uw.gravity   = params["gravity"]
    if "force"     in params: uw.force     = params["force"]
    if "max_speed" in params: uw.max_speed = params["max_speed"]


def train():
    env   = gym.make("MountainCar-v0")
    agent = DQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
    )
    bdr = BayesianDomainRandomizer()

    best_reward = -float("inf")

    for ep in range(1, EPISODES + 1):
        params = bdr.sample()
        _apply_params(env, params)

        state, _ = env.reset()
        total_reward = 0.0

        for _ in range(MAX_STEPS):
            action                                       = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done                                         = terminated or truncated

            agent.buffer.push(state, action, reward, next_state, float(done))
            agent.learn()

            state        = next_state
            total_reward += reward

            if done:
                break

        bdr.update(params, total_reward)
        agent.decay_epsilon()

        if ep % TARGET_UPDATE == 0:
            agent.sync_target()

        if total_reward > best_reward:
            best_reward = total_reward
            torch.save(agent.q_net.state_dict(), SAVE_PATH)

        if ep % 50 == 0:
            dist = bdr.get_distribution()
            dist_str = "  ".join(f"{k}=({v[0]:.5f}, ±{v[1]:.5f})" for k, v in dist.items())
            print(f"Episode {ep:4d} | reward {total_reward:7.1f} | "
                  f"best {best_reward:7.1f} | eps {agent.epsilon:.3f} | {dist_str}")

    env.close()
    print(f"\nTraining complete. Best reward: {best_reward:.1f}")
    print(f"Model saved to {SAVE_PATH}")


if __name__ == "__main__":
    train()
