from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..rl.envs import make_gym_env
from .base import Algorithm
from .doraemon_math import (
    doraemon_update_beta,
    make_beta_dist_at_mu,
    mean_conc_from_beta,
    sample_beta_params,
)


def train_under_doraemon_beta(
    agent,
    mu: np.ndarray,
    conc_init: np.ndarray,
    blocks: int,
    episodes_per_block: int,
    bounds: dict[str, tuple[float, float]],
    tau_steps: int,
    alpha_target: float,
    kl_max: float,
    n_candidates: int,
    mean_std: float,
    logc_std: float,
    c_min: np.ndarray,
    c_max: np.ndarray,
    min_log_w: float,
    max_log_w: float,
    seed: int,
    freeze_mean: bool,
    train_episode_fn: Any | None = None,
):
    rng = np.random.default_rng(seed)
    dist = make_beta_dist_at_mu(mu, conc_init, bounds)

    g_history: list[float] = []
    conc_history: list[np.ndarray] = []
    block_history: list[dict[str, float | int | str]] = []

    for b in range(int(blocks)):
        z_batch = []
        s_batch = []
        loss_batch = []

        for e in range(int(episodes_per_block)):
            context_e, z_e = sample_beta_params(dist, rng)
            if callable(train_episode_fn):
                train_info = train_episode_fn(
                    agent=agent,
                    context=np.asarray(context_e, dtype=np.float64),
                    seed=seed + 1000 * (b + 1) + e,
                    tau_steps=int(tau_steps),
                )
                losses = [float(train_info.get("loss_mean", math.nan))]
                solved = bool(train_info.get("solved", train_info.get("success", False)))
            else:
                context_arr = np.asarray(context_e, dtype=np.float64)
                if context_arr.size != 2:
                    raise RuntimeError(
                        "MountainCar fallback in train_under_doraemon_beta only supports 2D contexts. "
                        "Provide a train_episode_fn for non-MountainCar tasks."
                    )
                force_e, gravity_e = context_arr[:2]
                env = make_gym_env(force=force_e, gravity=gravity_e, seed=seed + 1000 * (b + 1) + e)
                s, _ = env.reset(seed=seed + 1000 * (b + 1) + e)
                losses = []
                solved = False

                for _ in range(int(tau_steps)):
                    a = agent.act(s, training=True)
                    ns, r, terminated, truncated, _ = env.step(a)
                    done = bool(terminated or truncated)
                    agent.store(s, a, r, ns, float(done))
                    loss = agent.learn_step()
                    if loss is not None:
                        losses.append(float(loss))
                    s = ns
                    if terminated:
                        solved = True
                        break
                    if truncated:
                        break

                env.close()
                agent.end_episode()

            z_batch.append(z_e)
            s_batch.append(1.0 if solved else 0.0)
            if losses:
                loss_batch.append(float(np.mean(losses)))

        z_batch_np = np.asarray(z_batch, dtype=np.float64)
        s_batch_np = np.asarray(s_batch, dtype=np.float64)

        dist, upd = doraemon_update_beta(
            old_dist=dist,
            z_samples=z_batch_np,
            success_samples=s_batch_np,
            alpha_target=alpha_target,
            kl_max=kl_max,
            rng=rng,
            n_candidates=n_candidates,
            mean_std=mean_std,
            logc_std=logc_std,
            c_min=np.asarray(c_min, dtype=np.float64),
            c_max=np.asarray(c_max, dtype=np.float64),
            min_log_w=min_log_w,
            max_log_w=max_log_w,
            freeze_mean=freeze_mean,
        )

        _, conc_end = mean_conc_from_beta(dist["alpha"], dist["beta"])
        g_history.append(float(upd["g_hat"]))
        conc_history.append(conc_end.copy())

        block_history.append(
            {
                "block": int(b + 1),
                "mode": str(upd["mode"]),
                "emp_success": float(np.mean(s_batch_np)) if len(s_batch_np) else 0.0,
                "g_hat": float(upd["g_hat"]),
                "entropy": float(upd["entropy"]),
                "kl": float(upd["kl"]),
                **{f"conc_end_{key}": float(value) for key, value in zip(bounds.keys(), conc_end)},
                "loss_mean": float(np.mean(loss_batch)) if loss_batch else math.nan,
            }
        )

    return agent, g_history, conc_history, dist, block_history


class DoraemonAlgorithm(Algorithm):
    def execute(self) -> None:
        adapt_iters = int(self.budget.get("adapt_iters", 0))
        blocks = int(self.budget.get("blocks", 1))
        episodes_per_block = int(self.budget.get("episodes_per_block", 1))
        B_r = int(self.budget.get("B_r", 10))

        alpha_target = float(self.budget.get("alpha_target", 0.60))
        kl_max = float(self.budget.get("kl_max", 0.04))
        n_candidates = int(self.budget.get("n_candidates", 300))
        mean_std = float(self.budget.get("mean_std", 0.06))
        logc_std = float(self.budget.get("logc_std", 0.20))
        context_dim = int(len(self.edge_bounds))
        c_init = self._coerce_parameter_vector(
            self.budget.get("c_init", [18.0]),
            size=context_dim,
            name="c_init",
        ).astype(np.float32)
        c_min = self._coerce_parameter_vector(
            self.budget.get("c_min", [2.2]),
            size=context_dim,
            name="c_min",
        ).astype(np.float32)
        c_max = self._coerce_parameter_vector(
            self.budget.get("c_max", [220.0]),
            size=context_dim,
            name="c_max",
        ).astype(np.float32)
        min_log_w = float(self.budget.get("min_log_w", -20.0))
        max_log_w = float(self.budget.get("max_log_w", 20.0))

        bounds = dict(self.edge_bounds)
        center = self._project_to_bounds(self.mu0, bounds)
        conc_prev = np.asarray(c_init, dtype=np.float32).copy()

        f0, succ0, eps0 = self._evaluate_target(n_eval_target=B_r, seed=int(self.state.seed + 7))
        is_best0 = self._save_best_if_needed(
            score=f0,
            step=0,
            extra={"center": center.tolist(), "conc": conc_prev.tolist()},
        )

        self._append_history(
            {
                "method": self.method.value,
                "step": 0,
                **self._context_row("center", center, bounds),
                **self._context_row("conc_end", conc_prev, bounds, project=False),
                "g_mean": math.nan,
                "g_last": math.nan,
                self.primary_metric: float(f0),
                "target_solve_rate": float(f0),
                "target_successes": int(succ0),
                "target_episodes": int(eps0),
                "is_best": bool(is_best0),
            }
        )

        for t in range(1, adapt_iters + 1):
            self._set_epsilon_floor(float(self.budget.get("epsilon_floor", 0.10)))

            self.agent, g_hist, conc_hist, _, block_hist = train_under_doraemon_beta(
                agent=self.agent,
                mu=center,
                conc_init=conc_prev,
                blocks=blocks,
                episodes_per_block=episodes_per_block,
                bounds=bounds,
                tau_steps=int(self.tau_steps),
                alpha_target=alpha_target,
                kl_max=kl_max,
                n_candidates=n_candidates,
                mean_std=mean_std,
                logc_std=logc_std,
                c_min=c_min,
                c_max=c_max,
                min_log_w=min_log_w,
                max_log_w=max_log_w,
                seed=int(self.state.seed + 10_000 * t),
                freeze_mean=False,
                train_episode_fn=self._train_episode,
            )

            f_t, succ_t, eps_t = self._evaluate_target(n_eval_target=B_r, seed=int(self.state.seed + 333 + t))
            conc_end = conc_hist[-1] if conc_hist else conc_prev

            is_best = self._save_best_if_needed(
                score=f_t,
                step=t,
                extra={"center": center.tolist(), "conc": np.asarray(conc_end).tolist()},
            )

            self._append_history(
                {
                    "method": self.method.value,
                    "step": int(t),
                    **self._context_row("center", center, bounds),
                    **self._context_row("conc_end", conc_end, bounds, project=False),
                    "g_mean": float(np.mean(g_hist)) if g_hist else math.nan,
                    "g_last": float(g_hist[-1]) if g_hist else math.nan,
                    "entropy_last": float(block_hist[-1]["entropy"]) if block_hist else math.nan,
                    "kl_last": float(block_hist[-1]["kl"]) if block_hist else math.nan,
                    self.primary_metric: float(f_t),
                    "target_solve_rate": float(f_t),
                    "target_successes": int(succ_t),
                    "target_episodes": int(eps_t),
                    "is_best": bool(is_best),
                }
            )

            conc_prev = np.asarray(conc_end, dtype=np.float32).copy()
