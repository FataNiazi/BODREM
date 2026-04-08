from __future__ import annotations

import math

import numpy as np

from .base import Algorithm


def _skopt_dimensions(bounds: dict[str, tuple[float, float]]):
    from skopt.space import Real

    return [Real(low, high, name=f"{name}_center") for name, (low, high) in bounds.items()]


class BOAlgorithm(Algorithm):
    def execute(self) -> None:
        try:
            from skopt import Optimizer
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("BOAlgorithm requires scikit-optimize.") from exc

        bo_iters = int(self.budget.get("bo_iters", 0))
        train_per_iter = int(self.budget.get("bo_train_episodes_per_iter", 1))
        target_eval_episodes = int(self.budget.get("target_eval_episodes", 10))
        epsilon_floor = float(self.budget.get("bo_epsilon_reset", 0.10))

        bounds = dict(self.task_config.get("context_bounds") or self.edge_bounds)
        mu0 = self._project_to_bounds(self.mu0, bounds)

        opt = Optimizer(
            dimensions=_skopt_dimensions(bounds),
            base_estimator="GP",
            acq_func="EI",
            random_state=int(self.state.seed if self.state else 0),
        )

        score0, succ0, eps0 = self._evaluate_target(
            n_eval_target=target_eval_episodes,
            seed=int(self.state.seed + 7 if self.state else 7),
        )
        opt.tell(mu0.tolist(), -float(score0))

        is_best0 = self._save_best_if_needed(
            score=score0,
            step=0,
            extra={"center": mu0.tolist(), "phase": "baseline"},
        )

        row0 = {
            "method": self.method.value,
            "step": 0,
            "train_episodes": 0,
            "loss_mean": math.nan,
            self.primary_metric: float(score0),
            "target_solve_rate": float(score0),
            "target_successes": int(succ0),
            "target_episodes": int(eps0),
            "is_best": bool(is_best0),
        }
        row0.update(self._context_row("mu", mu0, bounds))
        self._append_history(row0)

        cumulative = 0
        for t in range(1, bo_iters + 1):
            mu = self._project_to_bounds(np.asarray(opt.ask(), dtype=np.float64), bounds)
            self._set_epsilon_floor(epsilon_floor)

            losses: list[float] = []
            for e in range(train_per_iter):
                out = self._train_episode(
                    context=mu,
                    seed=int(self.state.seed + 10_000 * t + e),
                )
                loss_mean = float(out.get("loss_mean", math.nan))
                if not math.isnan(loss_mean):
                    losses.append(loss_mean)

            cumulative += train_per_iter

            score_t, succ_t, eps_t = self._evaluate_target(
                n_eval_target=target_eval_episodes,
                seed=int(self.state.seed + 100 + t),
            )
            opt.tell(mu.tolist(), -float(score_t))

            is_best = self._save_best_if_needed(
                score=score_t,
                step=t,
                extra={"center": mu.tolist(), "phase": "bo"},
            )

            row = {
                "method": self.method.value,
                "step": int(t),
                "train_episodes": int(cumulative),
                "loss_mean": float(np.mean(losses)) if losses else math.nan,
                self.primary_metric: float(score_t),
                "target_solve_rate": float(score_t),
                "target_successes": int(succ_t),
                "target_episodes": int(eps_t),
                "is_best": bool(is_best),
            }
            row.update(self._context_row("mu", mu, bounds))
            self._append_history(row)
