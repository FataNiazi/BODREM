from __future__ import annotations

from pathlib import Path
from typing import Any

from .types import (
    InitMode,
    MethodName,
    MethodRunConfig,
    PhysicsBounds,
    PhysicsPoint,
    PrimaryMetricName,
    ProfileName,
    RLBackend,
    SuiteConfig,
    TaskName,
)

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


PROJECT_DIR = Path(__file__).resolve().parent
RUNS_ROOT = PROJECT_DIR / "runs"
RUNS_ROOT.mkdir(parents=True, exist_ok=True)

DEFAULT_TARGET_PARAMS = PhysicsPoint(force=0.0010, gravity=0.0050)
DEFAULT_DR_BOUNDS = PhysicsBounds(force=(0.0003, 0.0030), gravity=(0.0010, 0.0062))
DEFAULT_EDGE_BOUNDS = PhysicsBounds(force=(0.0007, 0.0018), gravity=(0.0030, 0.0059))
DEFAULT_MU0 = (0.0010, 0.0025)
DEFAULT_EPISODE_STEPS = 400
DEFAULT_EVAL_EPISODES = 5
DEFAULT_WARMSTART_CHECKPOINT = PROJECT_DIR / "mountain_car_saved" / "mountain_car_dqn.pt"
DEFAULT_TASK = TaskName.MOUNTAINCAR
DEFAULT_MOUNTAINCAR_ENV_ID = "MountainCar-v0"
DEFAULT_HALFCHEETAH_ENV_ID = "HalfCheetah-v4"
DEFAULT_MOUNTAINCAR_BACKEND = RLBackend.NATIVE_DQN
DEFAULT_HALFCHEETAH_BACKEND = RLBackend.SB3_SAC
DEFAULT_MOUNTAINCAR_PRIMARY_METRIC: PrimaryMetricName = "target_solve_rate"
DEFAULT_HALFCHEETAH_PRIMARY_METRIC: PrimaryMetricName = "mean_episodic_return"

DEFAULT_METHOD_ORDER = [
    MethodName.BO,
    MethodName.DR,
    MethodName.BO_DR,
    MethodName.DORAEMON,
    MethodName.BO_DORAEMON,
]

PROFILE_PRESETS: dict[MethodName, dict[ProfileName, dict[str, Any]]] = {
    MethodName.BO: {
        "smoke": dict(bo_iters=3, bo_train_episodes_per_iter=20, target_eval_episodes=5),
        "prototype": dict(bo_iters=12, bo_train_episodes_per_iter=120, target_eval_episodes=12),
        "edge": dict(bo_iters=20, bo_train_episodes_per_iter=180, target_eval_episodes=20),
    },
    MethodName.DR: {
        "smoke": dict(dr_total_episodes=60, dr_eval_every=20, target_eval_episodes=5),
        "prototype": dict(dr_total_episodes=1440, dr_eval_every=120, target_eval_episodes=12),
        "edge": dict(dr_total_episodes=3600, dr_eval_every=180, target_eval_episodes=20),
    },
    MethodName.BO_DR: {
        "smoke": dict(T=3, K=20, B_r=5, B_r_extra=0, surrogate_eval_episodes=6),
        "prototype": dict(T=12, K=120, B_r=12, B_r_extra=0, surrogate_eval_episodes=10),
        "edge": dict(T=20, K=180, B_r=20, B_r_extra=0, surrogate_eval_episodes=16),
    },
    MethodName.DORAEMON: {
        "smoke": dict(adapt_iters=3, blocks=5, episodes_per_block=4, B_r=5),
        "prototype": dict(adapt_iters=12, blocks=20, episodes_per_block=6, B_r=12),
        "edge": dict(adapt_iters=20, blocks=30, episodes_per_block=6, B_r=20),
    },
    MethodName.BO_DORAEMON: {
        "smoke": dict(T=3, blocks=5, episodes_per_block=4, B_r=5),
        "prototype": dict(T=12, blocks=20, episodes_per_block=6, B_r=12),
        "edge": dict(T=20, blocks=30, episodes_per_block=6, B_r=20),
    },
}


def normalize_profile(profile: str) -> ProfileName:
    p = str(profile).strip().lower()
    if p not in {"smoke", "prototype", "edge"}:
        raise ValueError(f"Unknown profile '{profile}'. Expected one of smoke|prototype|edge")
    return p  # type: ignore[return-value]


def normalize_init_mode(init_mode: str) -> InitMode:
    mode = str(init_mode).strip().lower()
    if mode not in {"auto", "scratch", "warmstart"}:
        raise ValueError(f"Unknown init_mode '{init_mode}'. Expected one of auto|scratch|warmstart")
    return mode  # type: ignore[return-value]


def normalize_task(task: str | TaskName) -> TaskName:
    return TaskName.from_any(task)


def default_env_id(task: TaskName) -> str:
    if task == TaskName.HALFCHEETAH:
        return DEFAULT_HALFCHEETAH_ENV_ID
    return DEFAULT_MOUNTAINCAR_ENV_ID


def normalize_backend(backend: str | RLBackend, task: TaskName | None = None) -> RLBackend:
    backend_name = RLBackend.from_any(backend)
    if task == TaskName.MOUNTAINCAR and backend_name != RLBackend.NATIVE_DQN:
        raise ValueError("MountainCar currently supports backend='native_dqn' only.")
    return backend_name


def default_backend(task: TaskName) -> RLBackend:
    if task == TaskName.HALFCHEETAH:
        return DEFAULT_HALFCHEETAH_BACKEND
    return DEFAULT_MOUNTAINCAR_BACKEND


def normalize_primary_metric(metric: str, task: TaskName | None = None) -> PrimaryMetricName:
    token = str(metric).strip().lower()
    if token not in {"target_solve_rate", "mean_episodic_return"}:
        raise ValueError(
            "Unknown primary_metric "
            f"'{metric}'. Expected one of target_solve_rate|mean_episodic_return"
        )
    metric_name = token  # type: ignore[assignment]
    if task == TaskName.MOUNTAINCAR and metric_name != "target_solve_rate":
        raise ValueError("MountainCar currently uses primary_metric='target_solve_rate'.")
    return metric_name


def default_primary_metric(task: TaskName) -> PrimaryMetricName:
    if task == TaskName.HALFCHEETAH:
        return DEFAULT_HALFCHEETAH_PRIMARY_METRIC
    return DEFAULT_MOUNTAINCAR_PRIMARY_METRIC


def parse_methods(methods: list[str] | None) -> list[MethodName]:
    if not methods:
        return list(DEFAULT_METHOD_ORDER)
    return [MethodName.from_any(m) for m in methods]


def resolve_checkpoint_path(checkpoint_path: str | Path | None) -> Path | None:
    if checkpoint_path is None:
        return None
    token = str(checkpoint_path).strip()
    if not token:
        return None
    p = Path(token)
    if not p.is_absolute():
        p = (PROJECT_DIR / p).resolve()
    return p


def default_device() -> str:
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_profile_budget(method: MethodName, profile: ProfileName) -> dict[str, Any]:
    return dict(PROFILE_PRESETS[method][profile])


def build_suite_config(
    *,
    seeds: list[int],
    methods: list[str] | list[MethodName] | None = None,
    task: str | TaskName = DEFAULT_TASK,
    env_id: str | None = None,
    backend: str | RLBackend | None = None,
    profile: str = "smoke",
    init_mode: str = "auto",
    primary_metric: str | None = None,
    eval_episodes: int = DEFAULT_EVAL_EPISODES,
    task_config: dict[str, Any] | None = None,
    checkpoint_path: str | Path | None = None,
    device: str | None = None,
    output_root: str | Path | None = None,
    episode_steps: int = DEFAULT_EPISODE_STEPS,
    target_params: PhysicsPoint = DEFAULT_TARGET_PARAMS,
    dr_bounds: PhysicsBounds = DEFAULT_DR_BOUNDS,
    edge_bounds: PhysicsBounds = DEFAULT_EDGE_BOUNDS,
    mu0: tuple[float, float] = DEFAULT_MU0,
    method_overrides: dict[MethodName, dict[str, Any]] | None = None,
) -> SuiteConfig:
    task_name = normalize_task(task)
    method_names = parse_methods([m.value if isinstance(m, MethodName) else str(m) for m in methods] if methods else None)
    root = Path(output_root) if output_root is not None else RUNS_ROOT
    root.mkdir(parents=True, exist_ok=True)

    if isinstance(target_params, dict):
        target_params = PhysicsPoint(
            force=float(target_params["force"]),
            gravity=float(target_params["gravity"]),
        )
    if isinstance(dr_bounds, dict):
        dr_bounds = PhysicsBounds(
            force=(float(dr_bounds["force"][0]), float(dr_bounds["force"][1])),
            gravity=(float(dr_bounds["gravity"][0]), float(dr_bounds["gravity"][1])),
        )
    if isinstance(edge_bounds, dict):
        edge_bounds = PhysicsBounds(
            force=(float(edge_bounds["force"][0]), float(edge_bounds["force"][1])),
            gravity=(float(edge_bounds["gravity"][0]), float(edge_bounds["gravity"][1])),
        )

    normalized_overrides: dict[MethodName, dict[str, Any]] = {}
    for raw_key, override in (method_overrides or {}).items():
        method_key = raw_key if isinstance(raw_key, MethodName) else MethodName.from_any(str(raw_key))
        normalized_overrides[method_key] = dict(override)

    resolved_env_id = (env_id or default_env_id(task_name)).strip()
    resolved_backend = normalize_backend(backend or default_backend(task_name), task=task_name)
    resolved_metric = normalize_primary_metric(primary_metric or default_primary_metric(task_name), task=task_name)

    return SuiteConfig(
        seeds=[int(s) for s in seeds],
        methods=method_names,
        task=task_name,
        env_id=resolved_env_id,
        backend=resolved_backend,
        profile=normalize_profile(profile),
        init_mode=normalize_init_mode(init_mode),
        primary_metric=resolved_metric,
        eval_episodes=int(eval_episodes),
        target_params=target_params,
        dr_bounds=dr_bounds,
        edge_bounds=edge_bounds,
        episode_steps=int(episode_steps),
        mu0=(float(mu0[0]), float(mu0[1])),
        checkpoint_path=resolve_checkpoint_path(checkpoint_path),
        device=(device or default_device()).strip().lower(),
        output_root=root,
        task_config=dict(task_config or {}),
        method_overrides=normalized_overrides,
    )


def build_method_run_config(suite_cfg: SuiteConfig, method: MethodName) -> MethodRunConfig:
    return MethodRunConfig(
        method=method,
        task=suite_cfg.task,
        env_id=suite_cfg.env_id,
        backend=suite_cfg.backend,
        profile=suite_cfg.profile,
        init_mode=suite_cfg.init_mode,
        primary_metric=suite_cfg.primary_metric,
        eval_episodes=suite_cfg.eval_episodes,
        target_params=suite_cfg.target_params,
        dr_bounds=suite_cfg.dr_bounds,
        edge_bounds=suite_cfg.edge_bounds,
        episode_steps=suite_cfg.episode_steps,
        mu0=suite_cfg.mu0,
        checkpoint_path=suite_cfg.checkpoint_path,
        device=suite_cfg.device,
        output_root=suite_cfg.output_root,
        task_config=dict(suite_cfg.task_config),
        method_overrides=dict(suite_cfg.method_overrides.get(method, {})),
    )
