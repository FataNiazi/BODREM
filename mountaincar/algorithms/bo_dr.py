from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..rl.envs import make_gym_env
from .base import Algorithm
from .doraemon_math import project_mu_to_bounds


def _bounds_keys(bounds: dict[str, tuple[float, float]]) -> list[str]:
    return [str(key) for key in bounds.keys()]


def _skopt_dimensions(bounds: dict[str, tuple[float, float]]):
    from skopt.space import Real

    return [Real(low, high, name=f"{name}_center") for name, (low, high) in bounds.items()]


def sample_dr_params(
    mu: np.ndarray,
    sigma: np.ndarray,
    bounds: dict[str, tuple[float, float]],
    rng: np.random.Generator,
) -> np.ndarray:
    context = np.asarray(mu, dtype=np.float64) + rng.normal(0.0, 1.0, size=np.asarray(mu).shape) * np.asarray(sigma, dtype=np.float64)
    return project_mu_to_bounds(context, bounds)


def build_stress_eval_params(
    mu: np.ndarray,
    sigma: np.ndarray,
    bounds: dict[str, tuple[float, float]],
    n_eval: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    k_hi = 1.5
    k_mid = 1.0
    mu_arr = project_mu_to_bounds(mu, bounds)
    sigma_arr = np.asarray(sigma, dtype=np.float64).reshape(-1)
    if sigma_arr.size == 1 and mu_arr.size > 1:
        sigma_arr = np.repeat(sigma_arr, mu_arr.size)
    if sigma_arr.size != mu_arr.size:
        raise ValueError(f"Sigma dimensionality {sigma_arr.size} does not match context dimensionality {mu_arr.size}")

    pts: list[np.ndarray] = [project_mu_to_bounds(mu_arr, bounds)]
    for idx in range(mu_arr.size):
        basis = np.zeros_like(mu_arr)
        basis[idx] = 1.0
        pts.append(project_mu_to_bounds(mu_arr - k_hi * sigma_arr * basis, bounds))
        pts.append(project_mu_to_bounds(mu_arr - k_mid * sigma_arr * basis, bounds))
        pts.append(project_mu_to_bounds(mu_arr + k_mid * sigma_arr * basis, bounds))
        pts.append(project_mu_to_bounds(mu_arr + k_hi * sigma_arr * basis, bounds))

    if n_eval <= len(pts):
        return pts[:n_eval]
    while len(pts) < n_eval:
        pts.append(sample_dr_params(mu_arr, sigma_arr, bounds, rng))
    return pts


def surrogate_success_rate(
    agent,
    mu: np.ndarray,
    sigma: np.ndarray,
    n_eval: int,
    tau_steps: int,
    bounds: dict[str, tuple[float, float]],
    seed: int,
    mode: str = "stress",
    eval_runner: Any | None = None,
) -> float:
    rng = np.random.default_rng(seed)
    successes = 0

    if mode == "stress":
        eval_params = build_stress_eval_params(mu, sigma, bounds, n_eval=n_eval, rng=rng)
    else:
        eval_params = [sample_dr_params(mu, sigma, bounds, rng) for _ in range(n_eval)]

    for i, context in enumerate(eval_params):
        if callable(eval_runner):
            score, _, _ = eval_runner(
                agent=agent,
                context=np.asarray(context, dtype=np.float64),
                seed=seed + i,
                tau_steps=int(tau_steps),
            )
            successes += int(score > 0.0)
            continue

        context_arr = np.asarray(context, dtype=np.float64)
        if context_arr.size != 2:
            raise RuntimeError(
                "MountainCar fallback in surrogate_success_rate only supports 2D contexts. "
                "Provide eval_runner for non-MountainCar tasks."
            )
        force, gravity = context_arr[:2]
        env = make_gym_env(force=float(force), gravity=float(gravity), seed=seed + i)
        s, _ = env.reset(seed=seed + i)
        solved = False

        for _ in range(int(tau_steps)):
            a = agent.act(s, training=False)
            ns, _, terminated, truncated, _ = env.step(a)
            s = ns
            if terminated:
                solved = True
                break
            if truncated:
                break

        env.close()
        successes += int(solved)

    return float(successes / max(1, n_eval))


def train_under_dr(
    agent,
    mu: np.ndarray,
    sigma_init: np.ndarray,
    K: int,
    bounds: dict[str, tuple[float, float]],
    tau_steps: int,
    alpha_low: float,
    alpha_high: float,
    sigma_min: np.ndarray,
    sigma_max: np.ndarray,
    surrogate_eval_episodes: int,
    sigma_expand_factor: float,
    sigma_shrink_factor: float,
    sigma_update_interval: int,
    surrogate_mode: str,
    seed: int,
    train_episode_fn: Any | None = None,
    eval_runner: Any | None = None,
):
    rng = np.random.default_rng(seed)
    sigma = sigma_init.copy().astype(np.float32)
    g_history: list[float] = []
    loss_history: list[float] = []

    for k in range(int(K)):
        context_k = sample_dr_params(mu, sigma, bounds, rng)
        if callable(train_episode_fn):
            train_info = train_episode_fn(agent=agent, context=context_k, seed=seed + 1000 + k, tau_steps=int(tau_steps))
            ep_losses = [float(train_info.get("loss_mean", math.nan))]
        else:
            context_arr = np.asarray(context_k, dtype=np.float64)
            if context_arr.size != 2:
                raise RuntimeError(
                    "MountainCar fallback in train_under_dr only supports 2D contexts. "
                    "Provide train_episode_fn for non-MountainCar tasks."
                )
            force_k, gravity_k = context_arr[:2]
            env = make_gym_env(force=float(force_k), gravity=float(gravity_k), seed=seed + 1000 + k)
            s, _ = env.reset(seed=seed + 1000 + k)
            ep_losses = []

            for _ in range(int(tau_steps)):
                a = agent.act(s, training=True)
                ns, r, terminated, truncated, _ = env.step(a)
                done = bool(terminated or truncated)
                agent.store(s, a, r, ns, float(done))
                loss = agent.learn_step()
                if loss is not None:
                    ep_losses.append(float(loss))
                s = ns
                if done:
                    break

            env.close()
            agent.end_episode()

        g_t = surrogate_success_rate(
            agent=agent,
            mu=mu,
            sigma=sigma,
            n_eval=int(surrogate_eval_episodes),
            tau_steps=int(tau_steps),
            bounds=bounds,
            seed=seed + 5000 + k,
            mode=surrogate_mode,
            eval_runner=eval_runner,
        )

        g_history.append(float(g_t))
        loss_history.append(float(np.mean(ep_losses)) if ep_losses else math.nan)

        if ((k + 1) % max(1, int(sigma_update_interval))) == 0:
            g_window = g_history[-max(1, int(sigma_update_interval)) :]
            g_bar = float(np.mean(g_window))
            if g_bar >= float(alpha_high):
                sigma = np.minimum(sigma * float(sigma_expand_factor), sigma_max)
            elif g_bar <= float(alpha_low):
                sigma = np.maximum(sigma * float(sigma_shrink_factor), sigma_min)

    return agent, g_history, loss_history, sigma


class BODRAlgorithm(Algorithm):
    def execute(self) -> None:
        try:
            from skopt import Optimizer
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("BODRAlgorithm requires scikit-optimize.") from exc

        T = int(self.budget.get("T", 0))
        K = int(self.budget.get("K", 1))
        B_r = int(self.budget.get("B_r", 10))
        B_r_extra = int(self.budget.get("B_r_extra", 0))
        surrogate_eval_episodes = int(self.budget.get("surrogate_eval_episodes", 6))

        alpha_low = float(self.budget.get("alpha_low", 0.52))
        alpha_high = float(self.budget.get("alpha_high", 0.78))
        sigma_expand = float(self.budget.get("sigma_expand", 1.08))
        sigma_shrink = float(self.budget.get("sigma_shrink", 0.92))
        sigma_update_interval = int(self.budget.get("sigma_update_interval", 4))
        surrogate_mode = str(self.budget.get("surrogate_mode", "stress"))
        bo_epsilon_reset = float(self.budget.get("bo_epsilon_reset", 0.12))
        contender_margin = float(self.budget.get("contender_margin", 0.15))

        context_dim = int(len(self.edge_bounds))
        sigma0 = self._coerce_parameter_vector(
            self.budget.get("sigma0", [0.00018]),
            size=context_dim,
            name="sigma0",
        ).astype(np.float32)
        sigma_min = self._coerce_parameter_vector(
            self.budget.get("sigma_min", [0.00003]),
            size=context_dim,
            name="sigma_min",
        ).astype(np.float32)
        sigma_max = self._coerce_parameter_vector(
            self.budget.get("sigma_max", [0.00060]),
            size=context_dim,
            name="sigma_max",
        ).astype(np.float32)

        bounds = dict(self.dr_bounds if self.cfg.profile in {"smoke", "prototype"} else self.edge_bounds)
        mu0 = self._project_to_bounds(self.mu0, bounds)

        optimizer = Optimizer(
            dimensions=_skopt_dimensions(bounds),
            base_estimator="GP",
            acq_func="EI",
            random_state=int(self.state.seed),
        )

        agent_prev = self.agent
        sigma_prev = sigma0.copy()

        f0, succ0, eps0 = self._evaluate_target(n_eval_target=B_r, seed=int(self.state.seed + 7))
        optimizer.tell(mu0.tolist(), -float(f0))

        is_best0 = self._save_best_if_needed(f0, step=0, extra={"center": mu0.tolist()})
        self._append_history(
            {
                "method": self.method.value,
                "step": 0,
                **self._context_row("mu", mu0, bounds),
                **self._context_row("sigma_end", sigma_prev, bounds, project=False),
                "g_mean": math.nan,
                "g_last": math.nan,
                self.primary_metric: float(f0),
                "target_solve_rate": float(f0),
                "target_successes": int(succ0),
                "target_episodes": int(eps0),
                "is_best": bool(is_best0),
            }
        )

        for t in range(1, T + 1):
            mu_t = self._project_to_bounds(np.asarray(optimizer.ask(), dtype=np.float64), bounds)

            agent_t = self._clone_agent(agent_prev)
            sigma_start = sigma_prev.copy()
            if hasattr(agent_t, "epsilon"):
                agent_t.epsilon = float(max(float(agent_t.epsilon), bo_epsilon_reset))

            agent_t, g_hist, _, sigma_end = train_under_dr(
                agent=agent_t,
                mu=mu_t,
                sigma_init=sigma_start,
                K=K,
                bounds=bounds,
                tau_steps=int(self.tau_steps),
                alpha_low=alpha_low,
                alpha_high=alpha_high,
                sigma_min=sigma_min,
                sigma_max=sigma_max,
                surrogate_eval_episodes=surrogate_eval_episodes,
                sigma_expand_factor=sigma_expand,
                sigma_shrink_factor=sigma_shrink,
                sigma_update_interval=sigma_update_interval,
                surrogate_mode=surrogate_mode,
                seed=int(self.state.seed + 10_000 * t),
                train_episode_fn=self._train_episode,
                eval_runner=self._evaluate_context,
            )

            self.agent = agent_t
            f_base, succ_base, eps_base = self._evaluate_target(
                n_eval_target=B_r,
                seed=int(self.state.seed + 333 + t),
            )

            total_succ = int(succ_base)
            total_eps = int(max(1, eps_base))
            weighted_score_sum = float(f_base) * float(total_eps)
            if B_r_extra > 0 and float(f_base) >= max(0.0, float(self.state.best_score) - contender_margin):
                f_extra, succ_extra, eps_extra = self._evaluate_target(
                    n_eval_target=B_r_extra,
                    seed=int(self.state.seed + 777 + t),
                )
                total_succ += int(succ_extra)
                extra_eps = int(max(1, eps_extra))
                total_eps += extra_eps
                weighted_score_sum += float(f_extra) * float(extra_eps)

            f_t = float(weighted_score_sum / max(1, total_eps))
            optimizer.tell(mu_t.tolist(), -float(f_t))

            is_best = self._save_best_if_needed(
                score=f_t,
                step=t,
                extra={"center": mu_t.tolist(), "sigma_end": sigma_end.tolist()},
            )

            self._append_history(
                {
                    "method": self.method.value,
                    "step": int(t),
                    **self._context_row("mu", mu_t, bounds),
                    **self._context_row("sigma_start", sigma_start, bounds, project=False),
                    **self._context_row("sigma_end", sigma_end, bounds, project=False),
                    "g_mean": float(np.mean(g_hist)) if g_hist else math.nan,
                    "g_last": float(g_hist[-1]) if g_hist else math.nan,
                    self.primary_metric: float(f_t),
                    "target_solve_rate": float(f_t),
                    "target_successes": int(total_succ),
                    "target_episodes": int(total_eps),
                    "is_best": bool(is_best),
                }
            )

            agent_prev = agent_t
            sigma_prev = sigma_end.copy()
