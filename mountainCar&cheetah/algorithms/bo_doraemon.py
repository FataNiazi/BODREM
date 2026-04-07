from __future__ import annotations

import math

import numpy as np

from .base import Algorithm
from .doraemon import train_under_doraemon_beta
from .doraemon_math import denormalize_context, normalize_context, project_mu_to_bounds


def choose_conc_start(conc_init_mode: str, conc0: np.ndarray, conc_prev: np.ndarray, conc_blend: float) -> np.ndarray:
    mode = str(conc_init_mode).strip().lower()
    if mode == "carry":
        return np.asarray(conc_prev, dtype=np.float32).copy()
    if mode == "blend":
        w = float(np.clip(conc_blend, 0.0, 1.0))
        return (1.0 - w) * np.asarray(conc0, dtype=np.float32) + w * np.asarray(conc_prev, dtype=np.float32)
    return np.asarray(conc0, dtype=np.float32).copy()


def center_shift_norm(
    mu_new: np.ndarray,
    mu_old: np.ndarray | None,
    bounds: dict[str, tuple[float, float]],
) -> float:
    if mu_old is None:
        return 0.0
    span = np.asarray([max(1e-12, float(high - low)) for low, high in bounds.values()], dtype=np.float64)
    delta = (project_mu_to_bounds(mu_new, bounds) - project_mu_to_bounds(mu_old, bounds)) / np.clip(span, 1e-8, None)
    return float(np.linalg.norm(delta, ord=2))


def scheduled_epsilon_floor(
    bo_iter: int,
    total_bo_iters: int,
    shift_norm: float,
    floor_start: float,
    floor_end: float,
    distance_scale: float,
) -> float:
    if total_bo_iters <= 1:
        base = float(floor_end)
    else:
        frac = 1.0 - (float(bo_iter - 1) / float(total_bo_iters - 1))
        frac = float(np.clip(frac, 0.0, 1.0))
        base = float(floor_end + (floor_start - floor_end) * frac)
    return float(np.clip(base + distance_scale * max(0.0, shift_norm), floor_end, 1.0))


def sample_random_centers_near_mu(
    mu: np.ndarray,
    bounds: dict[str, tuple[float, float]],
    n_points: int,
    radius_frac: float,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    count = max(0, int(n_points))
    if count == 0:
        return []

    z0 = normalize_context(mu, bounds)
    centers: list[np.ndarray] = []
    for _ in range(count):
        z = np.clip(z0 + rng.normal(0.0, float(radius_frac), size=z0.size), 1e-4, 1.0 - 1e-4)
        centers.append(denormalize_context(z, bounds))
    return centers


def _skopt_dimensions(bounds: dict[str, tuple[float, float]]):
    from skopt.space import Real

    return [Real(low, high, name=f"{name}_center") for name, (low, high) in bounds.items()]


class BODoraemonAlgorithm(Algorithm):
    def execute(self) -> None:
        try:
            from skopt import Optimizer
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("BODoraemonAlgorithm requires scikit-optimize.") from exc

        T = int(self.budget.get("T", 0))
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

        freeze_doraemon_mean = bool(self.budget.get("freeze_doraemon_mean", True))
        conc_init_mode = str(self.budget.get("conc_init_mode", "blend"))
        conc_blend = float(self.budget.get("conc_blend", 0.5))

        stabilization_blocks = int(self.budget.get("stabilization_blocks", 2))
        bo_init_random_points = int(self.budget.get("bo_init_random_points", 3))
        bo_init_blocks = int(self.budget.get("bo_init_blocks", 2))
        bo_init_radius_frac = float(self.budget.get("bo_init_radius_frac", 0.08))
        competence_min_g_hat = float(self.budget.get("competence_min_g_hat", 0.45))
        competence_extra_rounds = int(self.budget.get("competence_extra_rounds", 1))
        competence_extra_blocks = int(self.budget.get("competence_extra_blocks", 2))
        center_hold_rounds = int(self.budget.get("center_hold_rounds", 2))
        center_hold_blocks = int(self.budget.get("center_hold_blocks", max(1, blocks // 2)))

        epsilon_floor_start = float(self.budget.get("epsilon_floor_start", 0.08))
        epsilon_floor_end = float(self.budget.get("epsilon_floor_end", 0.02))
        epsilon_distance_scale = float(self.budget.get("epsilon_distance_scale", 0.03))

        early_eval_replication_points = int(self.budget.get("early_eval_replication_points", 6))
        early_eval_replicates = int(self.budget.get("early_eval_replicates", 3))

        bounds = dict(self.edge_bounds)
        mu0 = self._project_to_bounds(self.mu0, bounds)

        optimizer = Optimizer(
            dimensions=_skopt_dimensions(bounds),
            base_estimator="GP",
            acq_func="EI",
            random_state=int(self.state.seed),
        )

        conc_prev = np.asarray(c_init, dtype=np.float32).copy()
        agent_prev = self.agent
        seed_cursor = int(self.state.seed) * 1000 + 17
        eval_counter = 0

        def next_seed() -> int:
            nonlocal seed_cursor
            out = int(seed_cursor)
            seed_cursor += 1009
            return out

        def eval_target_with_replicates() -> tuple[float, int, int, int, list[float]]:
            nonlocal eval_counter
            n_reps = early_eval_replicates if eval_counter < early_eval_replication_points else 1
            rates: list[float] = []
            total_succ = 0
            total_eps = 0
            for r in range(max(1, int(n_reps))):
                rate_r, succ_r, eps_r = self._evaluate_target(n_eval_target=B_r, seed=next_seed() + 7919 * r)
                rates.append(float(rate_r))
                total_succ += int(succ_r)
                total_eps += int(eps_r)
            eval_counter += 1
            mean_rate = float(np.mean(rates)) if rates else 0.0
            return mean_rate, int(total_succ), int(total_eps), int(n_reps), rates

        def adapt_once(
            agent_in,
            mu_center: np.ndarray,
            conc_start: np.ndarray,
            n_blocks: int,
            bo_iter: int,
            shift_norm: float,
        ):
            eps_floor = scheduled_epsilon_floor(
                bo_iter=bo_iter,
                total_bo_iters=max(1, int(T)),
                shift_norm=shift_norm,
                floor_start=epsilon_floor_start,
                floor_end=epsilon_floor_end,
                distance_scale=epsilon_distance_scale,
            )
            if hasattr(agent_in, "epsilon"):
                agent_in.epsilon = float(max(float(agent_in.epsilon), float(eps_floor)))

            out_agent, g_hist, conc_hist, _, block_hist = train_under_doraemon_beta(
                agent=agent_in,
                mu=mu_center,
                conc_init=np.asarray(conc_start, dtype=np.float32),
                blocks=int(n_blocks),
                episodes_per_block=int(episodes_per_block),
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
                seed=next_seed(),
                freeze_mean=freeze_doraemon_mean,
                train_episode_fn=self._train_episode,
            )
            conc_end = np.asarray(conc_hist[-1] if conc_hist else conc_start, dtype=np.float32)
            return out_agent, conc_end, list(g_hist), list(block_hist), float(eps_floor)

        if stabilization_blocks > 0:
            agent_prev, conc_prev, _, _, _ = adapt_once(
                agent_in=agent_prev,
                mu_center=mu0,
                conc_start=conc_prev,
                n_blocks=stabilization_blocks,
                bo_iter=1,
                shift_norm=0.0,
            )

        self.agent = agent_prev
        f0, succ0, eps0, reps0, rates0 = eval_target_with_replicates()
        optimizer.tell(mu0.tolist(), -float(f0))

        is_best0 = self._save_best_if_needed(
            score=f0,
            step=0,
            extra={"center": mu0.tolist(), "phase": "init_mu0", "rates": rates0},
        )

        self._append_history(
            {
                "method": self.method.value,
                "step": 0,
                **self._context_row("mu", mu0, bounds),
                **self._context_row("conc_end", conc_prev, bounds, project=False),
                "g_mean": math.nan,
                "g_last": math.nan,
                self.primary_metric: float(f0),
                "target_solve_rate": float(f0),
                "target_successes": int(succ0),
                "target_episodes": int(eps0),
                "target_eval_reps": int(reps0),
                "target_eval_replica_rates": [float(x) for x in rates0],
                "is_best": bool(is_best0),
            }
        )

        continuation_agent = self._clone_agent(agent_prev)
        continuation_conc = np.asarray(conc_prev, dtype=np.float32).copy()
        continuation_center = mu0.copy()
        continuation_score = float(f0)

        init_rng = np.random.default_rng(int(self.state.seed) + 2026)
        init_centers = sample_random_centers_near_mu(
            mu=mu0,
            bounds=bounds,
            n_points=bo_init_random_points,
            radius_frac=bo_init_radius_frac,
            rng=init_rng,
        )

        for i, mu_i in enumerate(init_centers, start=1):
            shift_i = center_shift_norm(mu_i, mu0, bounds)
            agent_i = self._new_start_agent(seed=int(self.state.seed + 50_000 + i))
            conc_i = np.asarray(c_init, dtype=np.float32).copy()

            agent_i, conc_i, g_hist_i, _, _ = adapt_once(
                agent_in=agent_i,
                mu_center=mu_i,
                conc_start=conc_i,
                n_blocks=bo_init_blocks,
                bo_iter=1,
                shift_norm=shift_i,
            )

            rounds_used = 0
            g_last_i = float(g_hist_i[-1]) if g_hist_i else math.nan
            while (not np.isfinite(g_last_i) or g_last_i < competence_min_g_hat) and rounds_used < competence_extra_rounds:
                rounds_used += 1
                agent_i, conc_i, g_more_i, _, _ = adapt_once(
                    agent_in=agent_i,
                    mu_center=mu_i,
                    conc_start=conc_i,
                    n_blocks=competence_extra_blocks,
                    bo_iter=1,
                    shift_norm=0.0,
                )
                g_hist_i.extend(g_more_i)
                g_last_i = float(g_hist_i[-1]) if g_hist_i else math.nan

            self.agent = agent_i
            f_i, _, _, _, _ = eval_target_with_replicates()
            tell_ok = bool(np.isfinite(g_last_i) and (g_last_i >= competence_min_g_hat))
            if tell_ok:
                optimizer.tell(mu_i.tolist(), -float(f_i))

            if float(f_i) > float(self.state.best_score):
                self._save_best_if_needed(
                    score=float(f_i),
                    step=0,
                    extra={"center": mu_i.tolist(), "phase": "init_random", "index": int(i)},
                )

            if tell_ok and float(f_i) >= continuation_score:
                continuation_score = float(f_i)
                continuation_center = mu_i.copy()
                continuation_conc = np.asarray(conc_i, dtype=np.float32).copy()
                continuation_agent = self._clone_agent(agent_i)

        agent_prev = continuation_agent
        conc_prev = continuation_conc
        prev_center = continuation_center.copy()

        for t in range(1, T + 1):
            mu_t = self._project_to_bounds(np.asarray(optimizer.ask(), dtype=np.float64), bounds)
            agent_t = self._clone_agent(agent_prev)
            shift_t = center_shift_norm(mu_t, prev_center, bounds)
            conc_start = choose_conc_start(
                conc_init_mode=conc_init_mode,
                conc0=np.asarray(c_init, dtype=np.float32),
                conc_prev=np.asarray(conc_prev, dtype=np.float32),
                conc_blend=conc_blend,
            )

            g_hist: list[float] = []
            conc_end = np.asarray(conc_start, dtype=np.float32).copy()
            eps_floor_applied = math.nan

            hold_schedule = [blocks] + [center_hold_blocks] * max(0, center_hold_rounds - 1)
            for hold_idx, round_blocks in enumerate(hold_schedule, start=1):
                agent_t, conc_end, g_round, _, eps_floor_applied = adapt_once(
                    agent_in=agent_t,
                    mu_center=mu_t,
                    conc_start=conc_end,
                    n_blocks=round_blocks,
                    bo_iter=t,
                    shift_norm=shift_t if hold_idx == 1 else 0.0,
                )
                g_hist.extend(g_round)

            competence_rounds = 0
            g_last = float(g_hist[-1]) if g_hist else math.nan
            while (not np.isfinite(g_last) or g_last < competence_min_g_hat) and competence_rounds < competence_extra_rounds:
                competence_rounds += 1
                agent_t, conc_end, g_more, _, eps_floor_applied = adapt_once(
                    agent_in=agent_t,
                    mu_center=mu_t,
                    conc_start=conc_end,
                    n_blocks=competence_extra_blocks,
                    bo_iter=t,
                    shift_norm=0.0,
                )
                g_hist.extend(g_more)
                g_last = float(g_hist[-1]) if g_hist else math.nan

            self.agent = agent_t
            f_t, succ_t, eps_t, reps_t, rates_t = eval_target_with_replicates()
            tell_ok = bool(np.isfinite(g_last) and (g_last >= competence_min_g_hat))
            if tell_ok:
                optimizer.tell(mu_t.tolist(), -float(f_t))

            is_best = self._save_best_if_needed(
                score=float(f_t),
                step=t,
                extra={"center": mu_t.tolist(), "conc": conc_end.tolist()},
            )

            self._append_history(
                {
                    "method": self.method.value,
                    "step": int(t),
                    **self._context_row("mu", mu_t, bounds),
                    "center_shift_norm": float(shift_t),
                    **self._context_row("conc_end", conc_end, bounds, project=False),
                    "g_mean": float(np.mean(g_hist)) if g_hist else math.nan,
                    "g_last": float(g_last) if np.isfinite(g_last) else math.nan,
                    self.primary_metric: float(f_t),
                    "target_solve_rate": float(f_t),
                    "target_successes": int(succ_t),
                    "target_episodes": int(eps_t),
                    "target_eval_reps": int(reps_t),
                    "target_eval_replica_rates": [float(x) for x in rates_t],
                    "competence_extra_rounds_used": int(competence_rounds),
                    "bo_tell_accepted": bool(tell_ok),
                    "epsilon_floor_applied": float(eps_floor_applied),
                    "is_best": bool(is_best),
                }
            )

            agent_prev = agent_t
            conc_prev = np.asarray(conc_end, dtype=np.float32).copy()
            prev_center = mu_t.copy()
