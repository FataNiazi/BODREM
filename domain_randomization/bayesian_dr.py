"""
Bayesian Domain Randomization for MountainCar-v0.

Maintains a posterior distribution over environment parameters and updates
it via a conjugate Gaussian-Gaussian model after each training episode.

Parameters randomized:
  gravity   — default 0.0025  (controls how fast the car rolls back down)
  force     — default 0.001   (engine thrust applied per action)
  max_speed — default 0.07    (velocity clamp)

THe posterion is updated such that the parameter
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass
class _ParamConfig:
    """Prior / posterior state for a single scalar parameter."""
    mean: float          
    std: float
    noise: float
    low: float
    high: float


# Default prior configuration — centred on MountainCar-v0 defaults
_DEFAULTS: Dict[str, _ParamConfig] = {
    "gravity":   _ParamConfig(mean=0.0025,  std=0.0005,  noise=0.001,  low=0.001,  high=0.006),
    "force":     _ParamConfig(mean=0.001,   std=0.0002,  noise=0.0005, low=0.0002, high=0.003),
    "max_speed": _ParamConfig(mean=0.07,    std=0.01,    noise=0.02,   low=0.03,   high=0.15),
}


class BayesianDomainRandomizer:
    """
    Maintains and updates a Bayesian posterior over MountainCar-v0 physics
    parameters, and samples new parameter sets from that posterior.

    Usage
    -----
    bdr = BayesianDomainRandomizer()

    # At the start of each episode:
    params = bdr.sample()           # dict[str, float]
    apply_params(env, params)

    # After each episode:
    bdr.update(params, episode_reward)

    # Inspect the current posterior:
    dist = bdr.get_distribution()   # dict[str, (mean, std)]
    """

    def __init__(self, config: Dict[str, _ParamConfig] | None = None) -> None:
        # Deep-copy defaults so instances are independent
        cfg = config or _DEFAULTS
        self._params: Dict[str, _ParamConfig] = {
            k: _ParamConfig(**v.__dict__) for k, v in cfg.items()
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sample(self) -> Dict[str, float]:
        """
        Draw one parameter set from the current posterior.

        Returns a dict mapping parameter name → sampled scalar value,
        clipped to each parameter's hard [low, high] bounds.
        """
        sampled: Dict[str, float] = {}
        for name, cfg in self._params.items():
            value = np.random.normal(cfg.mean, cfg.std)
            value = float(np.clip(value, cfg.low, cfg.high))
            sampled[name] = value
        return sampled

    def update(self, params: Dict[str, float], episode_reward: float) -> None:
        """
        Bayesian update of the posterior given the observed episode reward.

        The likelihood model treats *low reward* as evidence that the sampled
        parameters are *difficult* (informative), pulling the posterior mean
        toward those parameter values.  High reward means the parameters were
        easy, so the update has little effect.

        Conjugate Gaussian update (known noise variance):
            μ_new = (μ_prior/σ_prior² + x/σ_obs²) / (1/σ_prior² + 1/σ_obs²)
            σ_new = 1 / sqrt(1/σ_prior² + 1/σ_obs²)

        The likelihood weight is scaled by difficulty = exp(-reward / scale),
        so episodes with very negative reward contribute a strong pull.
        """
        scale = 200.0  # expected range of episode rewards (MountainCar ~[-200, 0])
        difficulty = float(np.exp(-episode_reward / scale))  # ∈ (0, 1]

        for name, cfg in self._params.items():
            if name not in params:
                continue
            x = params[name]

            # Effective observation noise: shrink with difficulty (hard episodes
            # are more informative — tighter likelihood)
            effective_noise = cfg.noise / (difficulty + 1e-6)

            # Conjugate Gaussian posterior update
            prior_precision = 1.0 / (cfg.std ** 2)
            likelihood_precision = 1.0 / (effective_noise ** 2)

            posterior_precision = prior_precision + likelihood_precision
            posterior_mean = (
                prior_precision * cfg.mean + likelihood_precision * x
            ) / posterior_precision
            posterior_std = 1.0 / np.sqrt(posterior_precision)

            cfg.mean = float(np.clip(posterior_mean, cfg.low, cfg.high))
            cfg.std  = float(np.clip(posterior_std,  1e-6,   (cfg.high - cfg.low) / 2))

    def get_distribution(self) -> Dict[str, Tuple[float, float]]:
        """
        Return the current posterior as a dict of {param: (mean, std)}.

        Useful for logging or plotting how the distribution evolves.
        """
        return {name: (cfg.mean, cfg.std) for name, cfg in self._params.items()}

    def reset_posterior(self) -> None:
        """Reset all parameters to their original prior."""
        self._params = {
            k: _ParamConfig(**v.__dict__) for k, v in _DEFAULTS.items()
        }
