from __future__ import annotations

import math

from .base import Algorithm


class DRAlgorithm(Algorithm):
    def execute(self) -> None:
        total_episodes = int(self.budget.get("dr_total_episodes", 0))
        eval_every = int(self.budget.get("dr_eval_every", 1))
        target_eval_episodes = int(self.budget.get("target_eval_episodes", 10))

        score0, succ0, eps0 = self._evaluate_target(
            n_eval_target=target_eval_episodes,
            seed=int(self.state.seed + 11 if self.state else 11),
        )
        is_best0 = self._save_best_if_needed(score=score0, step=0, extra={"phase": "baseline"})

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
        row0.update(self._context_row("center", self.mu0, self.edge_bounds))
        self._append_history(row0)

        for ep in range(1, total_episodes + 1):
            context = self._sample_uniform_context(self.edge_bounds)
            train_info = self._train_episode(
                context=context,
                seed=int(self.state.seed + ep),
            )

            if (ep % max(1, eval_every) != 0) and ep != total_episodes:
                continue

            score_t, succ_t, eps_t = self._evaluate_target(
                n_eval_target=target_eval_episodes,
                seed=int(self.state.seed + 200 + ep),
            )
            is_best = self._save_best_if_needed(
                score=score_t,
                step=ep,
                extra={"phase": "dr", "context": context.tolist()},
            )

            row = {
                "method": self.method.value,
                "step": int(ep),
                "train_episodes": int(ep),
                "loss_mean": float(train_info.get("loss_mean", math.nan)),
                self.primary_metric: float(score_t),
                "target_solve_rate": float(score_t),
                "target_successes": int(succ_t),
                "target_episodes": int(eps_t),
                "is_best": bool(is_best),
            }
            row.update(self._context_row("center", context, self.edge_bounds))
            self._append_history(row)
