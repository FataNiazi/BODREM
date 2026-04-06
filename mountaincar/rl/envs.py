from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from .halfcheetah_env import (
    HALFCHEETAH_DEFAULT_ENV_ID,
    HAS_MUJOCO,
    HalfCheetahParamApplier,
    make_halfcheetah_env,
    set_env_params as set_halfcheetah_env_params,
)
from .halfcheetah_params import (
    DEFAULT_HALFCHEETAH_BOUNDS,
    HALFCHEETAH_PARAM_NAMES,
    HalfCheetahBounds,
    HalfCheetahParams,
    clip_params as clip_halfcheetah_params,
    coerce_params as coerce_halfcheetah_params,
    sample_params as sample_halfcheetah_params,
)

try:
    from tinysim.mountain_car import MountainCarEnv
    HAS_TINYSIM = True
except Exception:  # pragma: no cover
    MountainCarEnv = Any  # type: ignore[assignment]
    HAS_TINYSIM = False

DEFAULT_FORCE = 0.001
DEFAULT_GRAVITY = 0.0025
DEFAULT_TARGET_PARAMS = {"force": 0.0010, "gravity": 0.0050}


@dataclass(frozen=True)
class PhysicsParams:
    force: float
    gravity: float


@dataclass(frozen=True)
class EvalResult:
    success_rate: float
    successes: int
    episodes: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(prefer_gpu: bool = True) -> str:
    if prefer_gpu and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def make_gym_env(force: float, gravity: float, seed: int | None = None):
    env = gym.make("MountainCar-v0")
    env.reset(seed=seed)
    env.unwrapped.force = float(force)
    env.unwrapped.gravity = float(gravity)
    return env


def make_tinysim_env(force: float, gravity: float):
    if not HAS_TINYSIM:
        raise RuntimeError("tinysim is required for TinySim evaluation.")
    sim_env = MountainCarEnv()
    sim_env.force = float(force)
    sim_env.gravity = float(gravity)
    return sim_env


def state_to_obs(state: dict[str, Any]) -> np.ndarray:
    return np.array([state["position"], state["velocity"]], dtype=np.float32)


def reset_tinysim_state(
    sim_env: MountainCarEnv,
    random_start: bool = False,
    start_position_range: tuple[float, float] = (-0.6, -0.4),
) -> dict[str, Any]:
    state = sim_env.reset()
    if random_start:
        sim_env.position = float(
            np.random.uniform(low=start_position_range[0], high=start_position_range[1])
        )
        sim_env.velocity = 0.0
        state = {
            "position": sim_env.position,
            "velocity": sim_env.velocity,
            "done": bool(sim_env.position >= sim_env.goal_position),
        }
    return state


def sample_uniform_params(
    bounds: dict[str, tuple[float, float]],
    rng: np.random.Generator,
) -> tuple[float, float]:
    force = float(rng.uniform(bounds["force"][0], bounds["force"][1]))
    gravity = float(rng.uniform(bounds["gravity"][0], bounds["gravity"][1]))
    return force, gravity


def clip_params(
    force: float,
    gravity: float,
    bounds: dict[str, tuple[float, float]],
) -> tuple[float, float]:
    f = float(np.clip(force, bounds["force"][0], bounds["force"][1]))
    g = float(np.clip(gravity, bounds["gravity"][0], bounds["gravity"][1]))
    return f, g


__all__ = [
    "DEFAULT_FORCE",
    "DEFAULT_GRAVITY",
    "DEFAULT_TARGET_PARAMS",
    "DEFAULT_HALFCHEETAH_BOUNDS",
    "EvalResult",
    "HALFCHEETAH_DEFAULT_ENV_ID",
    "HALFCHEETAH_PARAM_NAMES",
    "HAS_MUJOCO",
    "HAS_TINYSIM",
    "HalfCheetahBounds",
    "HalfCheetahParamApplier",
    "HalfCheetahParams",
    "PhysicsParams",
    "clip_halfcheetah_params",
    "clip_params",
    "coerce_halfcheetah_params",
    "make_gym_env",
    "make_halfcheetah_env",
    "make_tinysim_env",
    "pick_device",
    "reset_tinysim_state",
    "sample_halfcheetah_params",
    "sample_uniform_params",
    "set_halfcheetah_env_params",
    "set_seed",
    "state_to_obs",
]
