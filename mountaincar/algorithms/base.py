from __future__ import annotations

import hashlib
import math
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..config import get_profile_budget
from ..rl.checkpoints import (
    clone_agent as clone_dqn_agent,
    load_warmstart_agent,
    make_agent_from_hparams,
    require_sb3,
    save_agent_checkpoint,
    save_checkpoint_metadata_sidecar,
)
from ..rl.envs import make_gym_env, set_seed
from ..rl.eval import evaluate_mean_return, run_train_episode, target_eval_tinysim_detailed
from ..rl.halfcheetah_env import make_halfcheetah_env
from ..rl.halfcheetah_params import DEFAULT_HALFCHEETAH_BOUNDS
from ..rl.target_context import HiddenTargetManager, make_halfcheetah_target_manager
from ..types import ArtifactPaths, MethodName, MethodResult, MethodRunConfig, RLBackend, TaskName
from ..utils.io import ensure_run_dir, slugify, write_csv_rows, write_json


@dataclass
class RunState:
    seed: int
    run_dir: Path
    used_warmstart: bool
    history: list[dict[str, Any]]
    best_score: float
    best_checkpoint: Path
    last_checkpoint: Path


class Algorithm(ABC):
    method: MethodName

    def __init__(self, cfg: MethodRunConfig):
        self.cfg = cfg
        self.method = cfg.method
        self.task = cfg.task
        self.backend = cfg.backend
        self.primary_metric = cfg.primary_metric
        self.device = cfg.device
        self.task_config = dict(cfg.task_config)

        mc_dr_bounds = cfg.dr_bounds.as_dict()
        mc_edge_bounds = cfg.edge_bounds.as_dict()
        hc_bounds = DEFAULT_HALFCHEETAH_BOUNDS.as_dict()

        default_dr = hc_bounds if self.task == TaskName.HALFCHEETAH else mc_dr_bounds
        default_edge = hc_bounds if self.task == TaskName.HALFCHEETAH else mc_edge_bounds

        self.dr_bounds = self._resolve_context_bounds(self.task_config.get("dr_bounds"), fallback=default_dr)
        self.edge_bounds = self._resolve_context_bounds(self.task_config.get("edge_bounds"), fallback=default_edge)
        self.context_bounds = self._resolve_context_bounds(
            self.task_config.get("context_bounds"),
            fallback=self.edge_bounds,
        )
        self.context_names = self._resolve_context_names()

        raw_mu0 = self.task_config.get("mu0")
        if raw_mu0 is None:
            if self.task == TaskName.HALFCHEETAH:
                raw_mu0 = [(low + high) * 0.5 for low, high in self.context_bounds.values()]
            else:
                raw_mu0 = cfg.mu0
        self.mu0 = self._coerce_context(raw_mu0, bounds=self.context_bounds)

        self.tau_steps = int(cfg.episode_steps)
        self.budget = get_profile_budget(cfg.method, cfg.profile)
        self.budget.update(dict(cfg.method_overrides))

        self.target_manager = self.task_config.get("target_manager")
        self.target_public_metadata = self._resolve_target_public_metadata()
        self.target_params = self._resolve_target_params()

        self.agent: Any = None
        self.state: RunState | None = None
        self.rng: np.random.Generator | None = None

    def run(self, seed: int) -> MethodResult:
        set_seed(int(seed))
        self.rng = np.random.default_rng(int(seed))

        prefix = f"{slugify(self.method.value)}_seed{int(seed)}"
        run_dir = ensure_run_dir(self.cfg.output_root, prefix)

        checkpoint_ext = ".zip" if self.backend == RLBackend.SB3_SAC else ".pt"
        best_checkpoint = run_dir / f"best_policy{checkpoint_ext}"
        last_checkpoint = run_dir / f"last_policy{checkpoint_ext}"

        self.state = RunState(
            seed=int(seed),
            run_dir=run_dir,
            used_warmstart=False,
            history=[],
            best_score=-math.inf,
            best_checkpoint=best_checkpoint,
            last_checkpoint=last_checkpoint,
        )

        self._prepare_target(seed=int(seed))
        self.agent, used_warmstart = self._init_agent(int(seed))
        self.state.used_warmstart = bool(used_warmstart)

        self.execute()
        self._save_last_checkpoint(extra={"seed": int(seed), "best_score": float(self.state.best_score)})
        self._flush_history()

        public_target_metadata = self._public_target_metadata()
        manifest = {
            "method": self.method.value,
            "seed": int(seed),
            "task": self.task.value,
            "env_id": self.cfg.env_id,
            "backend": self.backend.value,
            "primary_metric": self.primary_metric,
            "eval_episodes": int(self.cfg.eval_episodes),
            "profile": self.cfg.profile,
            "init_mode": self.cfg.init_mode,
            "used_warmstart": bool(self.state.used_warmstart),
            "context_names": list(self.context_names),
            "context_dim": int(self.context_dim),
            "target_params": self._manifest_target_params(),
            "target_public_metadata": dict(public_target_metadata),
            "dr_bounds": dict(self.dr_bounds),
            "edge_bounds": dict(self.edge_bounds),
            "episode_steps": int(self.tau_steps),
            "budget": dict(self.budget),
            "run_dir": str(run_dir),
            "best_score": float(self.state.best_score),
            "best_checkpoint": str(self.state.best_checkpoint),
            "last_checkpoint": str(self.state.last_checkpoint),
            "history_rows": int(len(self.state.history)),
            **dict(public_target_metadata),
        }
        write_json(run_dir / "manifest.json", manifest)

        artifacts = ArtifactPaths(
            run_dir=str(run_dir),
            history_json=str(run_dir / "history.json"),
            history_csv=str(run_dir / "history.csv"),
            best_checkpoint=str(self.state.best_checkpoint),
            last_checkpoint=str(self.state.last_checkpoint),
        )

        return MethodResult(
            method=self.method,
            seed=int(seed),
            profile=self.cfg.profile,
            artifacts=artifacts,
            history=list(self.state.history),
            best_score=float(self.state.best_score),
            metadata=manifest,
        )

    @abstractmethod
    def execute(self) -> None:
        pass

    def _prepare_target(self, seed: int) -> None:
        if self.task != TaskName.HALFCHEETAH:
            self.target_params = dict(self.cfg.target_params.as_dict())
            return

        if isinstance(self.target_manager, HiddenTargetManager):
            target_obj = self.target_manager.sample_once()
            self.target_public_metadata = self.target_manager.public_metadata_dict()
            self.target_params = self._normalize_target_params(target_obj)
            self.task_config.setdefault("target_manager", self.target_manager)
            self.task_config.setdefault("target_params", dict(self.target_params))
            return

        raw_target = self.task_config.get("target_params")
        if isinstance(raw_target, Mapping):
            self.target_params = {str(k): float(v) for k, v in raw_target.items()}
            if not self.target_public_metadata:
                self.target_public_metadata = self._hash_target_metadata(self.target_params)
            return

        self.target_manager = make_halfcheetah_target_manager(
            base_seed=int(seed),
            bounds=self.context_bounds,
            env_id=self.cfg.env_id,
        )
        target_obj = self.target_manager.sample_once()
        self.target_params = self._normalize_target_params(target_obj)
        self.target_public_metadata = self.target_manager.public_metadata_dict()
        self.task_config["target_manager"] = self.target_manager
        self.task_config["target_params"] = dict(self.target_params)

    def _init_agent(self, seed: int) -> tuple[Any, bool]:
        del seed
        checkpoint = self.cfg.checkpoint_path
        mode = str(self.cfg.init_mode).strip().lower()
        checkpoint_exists = checkpoint is not None and Path(checkpoint).exists()

        if mode == "scratch":
            return self._make_fresh_agent(), False

        if mode == "warmstart":
            if not checkpoint_exists:
                raise ValueError(
                    "init_mode='warmstart' requires an existing checkpoint_path. "
                    f"Got: {checkpoint!r}"
                )
            return self._load_agent_from_checkpoint(checkpoint), True

        if mode == "auto":
            if checkpoint_exists:
                return self._load_agent_from_checkpoint(checkpoint), True
            return self._make_fresh_agent(), False

        raise ValueError(f"Unknown init_mode '{self.cfg.init_mode}'. Expected auto|scratch|warmstart")

    def _new_start_agent(self, seed: int) -> Any:
        agent, _ = self._init_agent(seed)
        return agent

    def _clone_agent(self, agent: Any) -> Any:
        if hasattr(agent, "state_payload"):
            return clone_dqn_agent(agent)

        if self.backend == RLBackend.SB3_SAC and hasattr(agent, "save"):
            try:
                from stable_baselines3 import SAC
                from stable_baselines3.common.base_class import BaseAlgorithm as SB3BaseAlgorithm

                if isinstance(agent, SB3BaseAlgorithm):
                    state = self._ensure_state()
                    temp_path = state.run_dir / (
                        f".tmp_clone_{slugify(self.method.value)}_{state.seed}_{np.random.randint(1_000_000)}.zip"
                    )
                    agent.save(str(temp_path))
                    clone = SAC.load(str(temp_path), device=self.device)
                    try:
                        temp_path.unlink(missing_ok=True)
                    except Exception:
                        pass
                    metadata_sidecar = temp_path.with_name(f"{temp_path.name}.metadata.json")
                    try:
                        metadata_sidecar.unlink(missing_ok=True)
                    except Exception:
                        pass
                    return clone
            except Exception:
                pass

        return deepcopy(agent)

    @property
    def context_dim(self) -> int:
        return int(len(self.context_names))

    def _resolve_context_names(self) -> list[str]:
        names = self.task_config.get("context_names")
        if isinstance(names, Sequence) and not isinstance(names, (str, bytes)):
            resolved = [str(name) for name in names if str(name).strip()]
            if resolved:
                return resolved
        return [str(key) for key in self.context_bounds.keys()]

    def _resolve_context_bounds(
        self,
        raw_bounds: Any = None,
        fallback: Mapping[str, tuple[float, float]] | None = None,
    ) -> dict[str, tuple[float, float]]:
        if raw_bounds is None:
            if fallback is not None:
                return {str(k): (float(v[0]), float(v[1])) for k, v in fallback.items()}
            if self.task == TaskName.HALFCHEETAH:
                return DEFAULT_HALFCHEETAH_BOUNDS.as_dict()
            return {"force": (0.0, 1.0), "gravity": (0.0, 1.0)}

        if isinstance(raw_bounds, Mapping):
            return {str(k): (float(v[0]), float(v[1])) for k, v in raw_bounds.items()}

        if isinstance(raw_bounds, Sequence) and not isinstance(raw_bounds, (str, bytes)):
            seq = list(raw_bounds)
            names = list(getattr(self, "context_names", []))
            if len(names) < len(seq):
                names.extend(f"context_{i}" for i in range(len(names), len(seq)))
            return {
                str(name): (float(bounds[0]), float(bounds[1]))
                for name, bounds in zip(names, seq)
            }

        raise ValueError(f"Could not normalize context bounds from: {raw_bounds!r}")

    def _coerce_context(
        self,
        raw_context: Any,
        bounds: Mapping[str, tuple[float, float]],
    ) -> np.ndarray:
        keys = [str(k) for k in bounds.keys()]
        if isinstance(raw_context, Mapping):
            arr = np.asarray([float(raw_context[k]) for k in keys], dtype=np.float64).reshape(-1)
        else:
            arr = np.asarray(raw_context, dtype=np.float64).reshape(-1)

        if arr.size == 1 and len(keys) > 1:
            arr = np.repeat(arr, len(keys))

        if arr.size != len(keys):
            raise ValueError(
                f"Expected context dimensionality {len(keys)}, got {arr.size}. "
                "Provide matching task_config['mu0'] / context_bounds."
            )
        return arr.astype(np.float64, copy=True)

    def _bounds_keys(self, bounds: Mapping[str, tuple[float, float]]) -> list[str]:
        return [str(k) for k in bounds.keys()]

    def _project_to_bounds(
        self,
        mu: np.ndarray,
        bounds: Mapping[str, tuple[float, float]],
    ) -> np.ndarray:
        keys = self._bounds_keys(bounds)
        arr = np.asarray(mu, dtype=np.float64).reshape(-1)
        if arr.size != len(keys):
            raise ValueError(f"Expected context with {len(keys)} parameters, got {arr.size}")
        clipped = [float(np.clip(value, bounds[k][0], bounds[k][1])) for value, k in zip(arr, keys)]
        return np.asarray(clipped, dtype=np.float64)

    def _sample_uniform_context(
        self,
        bounds: Mapping[str, tuple[float, float]],
    ) -> np.ndarray:
        if self.rng is None:
            raise RuntimeError("RNG is not initialized.")
        keys = self._bounds_keys(bounds)
        return np.asarray(
            [float(self.rng.uniform(bounds[k][0], bounds[k][1])) for k in keys],
            dtype=np.float64,
        )

    def _coerce_parameter_vector(
        self,
        value: Any,
        *,
        size: int,
        name: str,
        allow_legacy_pair_for_nd: bool = True,
    ) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
        if arr.size == int(size):
            return arr.astype(np.float64, copy=True)
        if arr.size == 1:
            return np.repeat(arr.astype(np.float64, copy=True), int(size))
        if allow_legacy_pair_for_nd and arr.size == 2 and int(size) > 2:
            # Legacy MountainCar-style pair ([force, gravity]) projected to isotropic ND default.
            return np.repeat(np.asarray([float(np.mean(arr))], dtype=np.float64), int(size))
        raise ValueError(
            f"Parameter '{name}' expects size {int(size)} (or scalar). Got size {arr.size}."
        )

    def _context_to_kwargs(
        self,
        context: np.ndarray,
        bounds: Mapping[str, tuple[float, float]] | None = None,
    ) -> dict[str, float]:
        resolved = bounds or self.context_bounds
        values = self._project_to_bounds(context, resolved)
        return {k: float(v) for k, v in zip(self._bounds_keys(resolved), values)}

    def _context_row(
        self,
        prefix: str,
        context: np.ndarray,
        bounds: Mapping[str, tuple[float, float]] | None = None,
        *,
        project: bool = True,
    ) -> dict[str, float]:
        resolved = bounds or self.context_bounds
        keys = self._bounds_keys(resolved)
        values = np.asarray(context, dtype=np.float64).reshape(-1)
        if values.size != len(keys):
            raise ValueError(f"Expected {len(keys)} values for '{prefix}', got {values.size}")
        if project:
            values = self._project_to_bounds(values, resolved)

        row = {f"{prefix}_{k}": float(v) for k, v in zip(keys, values)}
        if keys == ["force", "gravity"] and values.size == 2:
            row[f"{prefix}_force"] = float(values[0])
            row[f"{prefix}_gravity"] = float(values[1])
        return row

    def _call_env_factory(self, factory: Any, *, seed: int | None, context: np.ndarray | Mapping[str, float] | None):
        for kwargs in (
            {"seed": seed, "context": context},
            {"seed": seed},
            {},
        ):
            try:
                return factory(**kwargs)
            except TypeError:
                continue
        return factory()

    def _default_halfcheetah_env_factory(self, *, seed: int | None, context: np.ndarray | Mapping[str, float] | None):
        if context is None:
            params = self._context_to_kwargs(self.mu0, self.context_bounds)
        elif isinstance(context, Mapping):
            params = {str(k): float(v) for k, v in context.items()}
        else:
            params = self._context_to_kwargs(np.asarray(context, dtype=np.float64), self.context_bounds)
        return make_halfcheetah_env(params=params, seed=seed, env_id=self.cfg.env_id)

    def _make_fresh_agent(self) -> Any:
        factory = self.task_config.get("agent_factory") or self.task_config.get("make_agent_fn")
        if callable(factory):
            return factory(cfg=self.cfg, seed=self.state.seed if self.state else None, device=self.device)

        if self.backend == RLBackend.SB3_SAC:
            require_sb3()
            from stable_baselines3 import SAC

            env_factory = self.task_config.get("train_env_factory") or self.task_config.get("env_factory")
            if not callable(env_factory):
                env_factory = lambda **kwargs: self._default_halfcheetah_env_factory(
                    seed=kwargs.get("seed"),
                    context=kwargs.get("context"),
                )

            env = self._call_env_factory(
                env_factory,
                seed=self.state.seed if self.state else None,
                context=self.mu0,
            )
            try:
                sb3_kwargs = dict(self.task_config.get("sb3_kwargs", {}))
                policy_kwargs = dict(self.task_config.get("policy_kwargs", {}))
                return SAC(
                    "MlpPolicy",
                    env,
                    device=self.device,
                    seed=self.state.seed if self.state else None,
                    policy_kwargs=policy_kwargs,
                    **sb3_kwargs,
                )
            finally:
                try:
                    env.close()
                except Exception:
                    pass

        return make_agent_from_hparams(None, device=self.device)

    def _load_agent_from_checkpoint(self, checkpoint_path: Path | None) -> Any:
        if checkpoint_path is None:
            raise ValueError("checkpoint_path is required for warmstart loading.")

        if self.backend == RLBackend.SB3_SAC:
            require_sb3()
            from stable_baselines3 import SAC

            return SAC.load(str(checkpoint_path), device=self.device)

        return load_warmstart_agent(checkpoint_path=checkpoint_path, device=self.device)

    def _train_episode(
        self,
        context: np.ndarray,
        seed: int,
        tau_steps: int | None = None,
        **kwargs: Any,
    ) -> dict[str, float | bool]:
        steps = int(tau_steps if tau_steps is not None else self.tau_steps)
        agent = kwargs.get("agent", self.agent)

        if self.task == TaskName.HALFCHEETAH:
            runner = self.task_config.get("train_episode_fn")
            if callable(runner):
                return runner(
                    agent=agent,
                    context=np.asarray(context, dtype=np.float64),
                    seed=int(seed),
                    tau_steps=steps,
                    config=self.cfg,
                    task_config=dict(self.task_config),
                )

            if self.backend == RLBackend.SB3_SAC and hasattr(agent, "set_env") and hasattr(agent, "learn"):
                env_factory = self.task_config.get("train_env_factory") or self.task_config.get("env_factory")
                if not callable(env_factory):
                    env_factory = lambda **factory_kwargs: self._default_halfcheetah_env_factory(
                        seed=factory_kwargs.get("seed"),
                        context=factory_kwargs.get("context"),
                    )
                env = self._call_env_factory(
                    env_factory,
                    seed=int(seed),
                    context=np.asarray(context, dtype=np.float64),
                )
                try:
                    agent.set_env(env)
                    agent.learn(total_timesteps=steps, reset_num_timesteps=False, progress_bar=False)
                finally:
                    try:
                        env.close()
                    except Exception:
                        pass
                eval_result = evaluate_mean_return(
                    policy=agent,
                    env_factory=lambda **factory_kwargs: self._call_env_factory(
                        env_factory,
                        seed=factory_kwargs.get("seed"),
                        context=np.asarray(context, dtype=np.float64),
                    ),
                    n_episodes=1,
                    seed=int(seed) + 100_000,
                    deterministic=bool(self.task_config.get("deterministic_eval", True)),
                    max_steps=steps,
                )
                solve_threshold = float(self.task_config.get("surrogate_success_threshold", 0.0))
                solved = bool(float(eval_result.mean_return) >= solve_threshold)
                return {
                    "solved": solved,
                    "steps": float(steps),
                    "loss_mean": math.nan,
                    "episode_return": float(eval_result.mean_return),
                }

            raise RuntimeError(
                "HalfCheetah training requires either task_config['train_episode_fn'] "
                "or backend='sb3_sac' with a Stable-Baselines3 model."
            )

        context_kwargs = self._context_to_kwargs(context, self.edge_bounds)
        return run_train_episode(
            agent=agent,
            force=float(context_kwargs["force"]),
            gravity=float(context_kwargs["gravity"]),
            tau_steps=steps,
            seed=int(seed),
        )

    def _evaluate_context(
        self,
        *,
        agent: Any,
        context: np.ndarray,
        seed: int,
        tau_steps: int,
        n_eval: int = 1,
    ) -> tuple[float, int, int]:
        if self.task == TaskName.HALFCHEETAH:
            eval_factory = self.task_config.get("eval_env_factory")
            if not callable(eval_factory):
                eval_factory = lambda **kwargs: self._default_halfcheetah_env_factory(
                    seed=kwargs.get("seed"),
                    context=kwargs.get("context"),
                )

            result = evaluate_mean_return(
                policy=agent,
                env_factory=lambda **kwargs: self._call_env_factory(
                    eval_factory,
                    seed=kwargs.get("seed"),
                    context=np.asarray(context, dtype=np.float64),
                ),
                n_episodes=max(1, int(n_eval)),
                seed=int(seed),
                deterministic=bool(self.task_config.get("deterministic_eval", True)),
                max_steps=int(tau_steps),
            )
            episodes = int(result.episodes)
            return float(result.mean_return), episodes, episodes

        context_kwargs = self._context_to_kwargs(context, self.edge_bounds)
        successes = 0
        for episode_idx in range(max(1, int(n_eval))):
            episode_seed = int(seed) + int(episode_idx)
            env = make_gym_env(
                force=float(context_kwargs["force"]),
                gravity=float(context_kwargs["gravity"]),
                seed=episode_seed,
            )
            try:
                obs, _ = env.reset(seed=episode_seed)
                solved = False
                for _ in range(int(tau_steps)):
                    action = agent.act(obs, training=False)
                    obs, _, terminated, truncated, _ = env.step(action)
                    if terminated:
                        solved = True
                        break
                    if truncated:
                        break
                successes += int(solved)
            finally:
                env.close()

        episodes = max(1, int(n_eval))
        return float(successes / episodes), int(successes), int(episodes)

    def _evaluate_target(
        self,
        n_eval_target: int,
        seed: int,
        context: np.ndarray | None = None,
        tau_steps: int | None = None,
        **kwargs: Any,
    ) -> tuple[float, int, int]:
        del context
        agent = kwargs.get("agent", self.agent)
        steps = int(tau_steps if tau_steps is not None else self.tau_steps)

        if self.task == TaskName.HALFCHEETAH:
            target_manager = self.task_config.get("target_manager")
            target_params: Mapping[str, Any] | None = None
            if isinstance(target_manager, HiddenTargetManager):
                target_obj = target_manager.sample_once()
                self.target_public_metadata = target_manager.public_metadata_dict()
                self.target_params = self._normalize_target_params(target_obj)
                target_params = self.target_params
            elif isinstance(self.target_params, Mapping) and self.target_params:
                target_params = self.target_params

            if target_params is None:
                raise RuntimeError(
                    "HalfCheetah evaluation requires a hidden target manager or target_params."
                )

            target_env_factory = self.task_config.get("target_env_factory")
            if not callable(target_env_factory):
                target_env_factory = lambda **factory_kwargs: self._default_halfcheetah_env_factory(
                    seed=factory_kwargs.get("seed"),
                    context=target_params,
                )

            episodes = int(n_eval_target if n_eval_target else self.cfg.eval_episodes)
            result = evaluate_mean_return(
                policy=agent,
                env_factory=lambda **factory_kwargs: self._call_env_factory(
                    target_env_factory,
                    seed=factory_kwargs.get("seed"),
                    context=target_params,
                ),
                n_episodes=max(1, episodes),
                seed=int(seed),
                deterministic=bool(self.task_config.get("deterministic_eval", True)),
                max_steps=steps,
            )
            count = int(result.episodes)
            return float(result.mean_return), count, count

        return target_eval_tinysim_detailed(
            agent=agent,
            n_eval_target=int(n_eval_target),
            max_steps=steps,
            target_params=dict(self.target_params),
            seed=int(seed),
        )

    def _ensure_state(self) -> RunState:
        if self.state is None:
            raise RuntimeError("Algorithm state not initialized.")
        return self.state

    def _append_history(self, row: dict[str, Any]) -> None:
        state = self._ensure_state()
        state.history.append(row)
        self._flush_history()

    def _flush_history(self) -> None:
        state = self._ensure_state()
        write_json(state.run_dir / "history.json", state.history)
        write_csv_rows(state.run_dir / "history.csv", state.history)

    def _save_best_if_needed(self, score: float, step: int, extra: dict[str, Any] | None = None) -> bool:
        state = self._ensure_state()
        if float(score) <= float(state.best_score):
            return False
        state.best_score = float(score)
        payload = {"step": int(step), "score": float(score), **(extra or {})}
        self._save_agent_checkpoint(state.best_checkpoint, extra=payload)
        return True

    def _save_last_checkpoint(self, extra: dict[str, Any] | None = None) -> None:
        state = self._ensure_state()
        self._save_agent_checkpoint(state.last_checkpoint, extra=(extra or {}))

    def _set_epsilon_floor(self, floor: float) -> None:
        if hasattr(self.agent, "epsilon"):
            self.agent.epsilon = float(max(float(self.agent.epsilon), float(floor)))

    def _save_agent_checkpoint(self, path: Path, extra: dict[str, Any] | None = None) -> None:
        payload = dict(extra or {})
        if hasattr(self.agent, "state_payload"):
            save_agent_checkpoint(
                self.agent,
                path,
                extra=payload,
                backend=self.backend.value,
                env_id=self.cfg.env_id,
                task=self.task.value,
            )
            return

        if hasattr(self.agent, "save"):
            save_path = path if path.suffix == ".zip" else path.with_suffix(".zip")
            save_path.parent.mkdir(parents=True, exist_ok=True)
            self.agent.save(str(save_path))
            metadata = self._checkpoint_metadata(extra=payload)
            save_checkpoint_metadata_sidecar(save_path, metadata)
            state = self.state
            if state is not None:
                if path == state.best_checkpoint:
                    state.best_checkpoint = save_path
                if path == state.last_checkpoint:
                    state.last_checkpoint = save_path
            return

        raise TypeError(f"Unsupported agent type for checkpoint save: {type(self.agent)!r}")

    def _checkpoint_metadata(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        metadata = {
            "task": self.task.value,
            "env_id": self.cfg.env_id,
            "backend": self.backend.value,
            "primary_metric": self.primary_metric,
            "context_names": list(self.context_names),
            "context_dim": int(self.context_dim),
        }
        metadata.update(self._public_target_metadata())
        if extra:
            metadata.update(self._json_safe(extra))
        return metadata

    def _public_target_metadata(self) -> dict[str, Any]:
        if isinstance(self.target_public_metadata, Mapping) and self.target_public_metadata:
            return self._json_safe(dict(self.target_public_metadata))

        if self.task == TaskName.HALFCHEETAH:
            if self.target_params:
                return self._hash_target_metadata(self.target_params)
            return {
                "task": self.task.value,
                "env_id": self.cfg.env_id,
                "target_family": "halfcheetah_hidden_target_v1",
                "target_hash": "",
                "hidden_target": True,
            }

        return {"hidden_target": False}

    def _resolve_target_params(self) -> dict[str, Any]:
        if self.task == TaskName.HALFCHEETAH:
            raw_target = self.task_config.get("target_params")
            if isinstance(raw_target, Mapping):
                return {str(k): float(v) for k, v in raw_target.items()}
            return {}
        return dict(self.cfg.target_params.as_dict())

    def _resolve_target_public_metadata(self) -> dict[str, Any]:
        meta = self.task_config.get("target_public_metadata")
        if isinstance(meta, Mapping):
            return self._json_safe(dict(meta))

        target = self.task_config.get("target_manager")
        if hasattr(target, "public_metadata_dict"):
            try:
                return self._json_safe(dict(target.public_metadata_dict()))
            except Exception:
                return {}
        return {}

    def _manifest_target_params(self) -> dict[str, Any]:
        if self.task == TaskName.HALFCHEETAH:
            return {}
        return dict(self.target_params)

    def _normalize_target_params(self, raw: Any) -> dict[str, float]:
        if isinstance(raw, Mapping):
            return {str(k): float(v) for k, v in raw.items()}
        if hasattr(raw, "as_dict"):
            maybe = raw.as_dict()
            if isinstance(maybe, Mapping):
                return {str(k): float(v) for k, v in maybe.items()}
        arr = np.asarray(raw, dtype=np.float64).reshape(-1)
        keys = list(self.context_bounds.keys())
        if arr.size != len(keys):
            raise ValueError(f"Expected target vector size {len(keys)}, got {arr.size}")
        return {str(k): float(v) for k, v in zip(keys, arr)}

    def _hash_target_metadata(self, target: Mapping[str, Any]) -> dict[str, Any]:
        canonical = self._json_safe({str(k): target[k] for k in sorted(target.keys())})
        digest = hashlib.sha256(str(canonical).encode("utf-8")).hexdigest()
        return {
            "task": self.task.value,
            "env_id": self.cfg.env_id,
            "target_family": "halfcheetah_hidden_target_v1",
            "target_hash": digest,
            "hidden_target": True,
        }

    def _json_safe(self, value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(k): self._json_safe(v) for k, v in value.items()}
        if isinstance(value, np.ndarray):
            return [self._json_safe(v) for v in value.tolist()]
        if isinstance(value, (list, tuple)):
            return [self._json_safe(v) for v in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if hasattr(value, "as_dict"):
            return self._json_safe(value.as_dict())
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                pass
        return str(value)
