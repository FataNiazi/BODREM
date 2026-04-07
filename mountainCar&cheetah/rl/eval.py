from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .envs import DEFAULT_TARGET_PARAMS, make_gym_env, make_tinysim_env, state_to_obs


@dataclass(frozen=True)
class MeanReturnResult:
    mean_return: float
    returns: list[float]
    episodes: int


def run_train_episode(
    agent: Any,
    force: float,
    gravity: float,
    tau_steps: int,
    seed: int,
) -> dict[str, float | bool]:
    env = make_gym_env(force=force, gravity=gravity, seed=seed)
    state, _ = env.reset(seed=seed)

    losses: list[float] = []
    solved = False
    steps = 0

    for _ in range(int(tau_steps)):
        action = agent.act(state, training=True)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = bool(terminated or truncated)

        agent.store(state, action, reward, next_state, float(done))
        loss = agent.learn_step()
        if loss is not None:
            losses.append(float(loss))

        state = next_state
        steps += 1

        if terminated:
            solved = True
            break
        if truncated:
            break

    env.close()
    agent.end_episode()

    return {
        "solved": bool(solved),
        "steps": float(steps),
        "loss_mean": float(np.mean(losses)) if losses else float("nan"),
    }


def evaluate_mean_return(
    policy: Any,
    env_factory: Callable[..., Any],
    n_episodes: int,
    seed: int | None = None,
    deterministic: bool = True,
    max_steps: int | None = None,
) -> MeanReturnResult:
    returns: list[float] = []

    for episode_idx in range(int(n_episodes)):
        episode_seed = None if seed is None else int(seed) + int(episode_idx)
        env = _build_env(env_factory=env_factory, episode_seed=episode_seed)
        try:
            obs, _ = env.reset(seed=episode_seed)
            done = False
            truncated = False
            steps = 0
            total_reward = 0.0

            while not done and not truncated:
                action = _predict_action(policy=policy, observation=obs, deterministic=deterministic)
                obs, reward, done, truncated, _ = env.step(action)
                total_reward += float(reward)
                steps += 1
                if max_steps is not None and steps >= int(max_steps):
                    break

            returns.append(float(total_reward))
        finally:
            env.close()

    mean_return = float(np.mean(returns)) if returns else float("nan")
    return MeanReturnResult(
        mean_return=mean_return,
        returns=returns,
        episodes=int(n_episodes),
    )


def target_eval_tinysim_detailed(
    agent: Any,
    n_eval_target: int,
    max_steps: int,
    target_params: dict[str, float] | None = None,
    seed: int | None = None,
) -> tuple[float, int, int]:
    del seed
    params = target_params or DEFAULT_TARGET_PARAMS
    successes = 0

    for _ in range(int(n_eval_target)):
        sim_env = make_tinysim_env(force=params["force"], gravity=params["gravity"])
        state = sim_env.reset()
        obs = state_to_obs(state)

        solved = False
        for _ in range(int(max_steps)):
            action = agent.act(obs, training=False)
            state = sim_env.step(action)
            obs = state_to_obs(state)
            if bool(state["done"]):
                solved = True
                break

        successes += int(solved)

    rate = float(successes / max(1, int(n_eval_target)))
    return rate, int(successes), int(n_eval_target)


def target_eval_tinysim(
    agent: Any,
    n_eval_target: int,
    max_steps: int,
    target_params: dict[str, float] | None = None,
    seed: int | None = None,
) -> float:
    rate, _, _ = target_eval_tinysim_detailed(
        agent=agent,
        n_eval_target=n_eval_target,
        max_steps=max_steps,
        target_params=target_params,
        seed=seed,
    )
    return float(rate)


def _build_env(env_factory: Callable[..., Any], episode_seed: int | None):
    try:
        return env_factory(seed=episode_seed)
    except TypeError:
        return env_factory()


def _predict_action(policy: Any, observation: Any, deterministic: bool) -> Any:
    if hasattr(policy, "predict"):
        action = policy.predict(observation, deterministic=deterministic)
        if isinstance(action, tuple):
            return action[0]
        return action
    if hasattr(policy, "act"):
        return policy.act(observation, training=False)
    raise TypeError(
        "Policy must define either '.predict(observation, deterministic=...)' "
        "or '.act(observation, training=False)'."
    )


__all__ = [
    "MeanReturnResult",
    "evaluate_mean_return",
    "run_train_episode",
    "target_eval_tinysim",
    "target_eval_tinysim_detailed",
]
