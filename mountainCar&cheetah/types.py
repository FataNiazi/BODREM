from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Literal


ProfileName = Literal["smoke", "prototype", "edge"]
InitMode = Literal["auto", "scratch", "warmstart"]
PrimaryMetricName = Literal["target_solve_rate", "mean_episodic_return"]


class TaskName(str, Enum):
    MOUNTAINCAR = "mountaincar"
    HALFCHEETAH = "halfcheetah"

    @classmethod
    def from_any(cls, raw: str | "TaskName") -> "TaskName":
        if isinstance(raw, TaskName):
            return raw
        token = str(raw).strip().lower()
        aliases = {
            "mountaincar": cls.MOUNTAINCAR,
            "mountain_car": cls.MOUNTAINCAR,
            "mountain-car": cls.MOUNTAINCAR,
            "halfcheetah": cls.HALFCHEETAH,
            "half_cheetah": cls.HALFCHEETAH,
            "half-cheetah": cls.HALFCHEETAH,
        }
        if token not in aliases:
            valid = ", ".join([item.value for item in cls])
            raise ValueError(f"Unknown task '{raw}'. Expected one of: {valid}")
        return aliases[token]


class RLBackend(str, Enum):
    NATIVE_DQN = "native_dqn"
    SB3_SAC = "sb3_sac"

    @classmethod
    def from_any(cls, raw: str | "RLBackend") -> "RLBackend":
        if isinstance(raw, RLBackend):
            return raw
        token = str(raw).strip().lower()
        aliases = {
            "native_dqn": cls.NATIVE_DQN,
            "native-dqn": cls.NATIVE_DQN,
            "dqn": cls.NATIVE_DQN,
            "sb3_sac": cls.SB3_SAC,
            "sb3-sac": cls.SB3_SAC,
            "sac": cls.SB3_SAC,
        }
        if token not in aliases:
            valid = ", ".join([item.value for item in cls])
            raise ValueError(f"Unknown backend '{raw}'. Expected one of: {valid}")
        return aliases[token]


class MethodName(str, Enum):
    BO = "BO"
    DR = "DR"
    BO_DR = "BO+DR"
    DORAEMON = "DORAEMON"
    BO_DORAEMON = "BO+DORAEMON"

    @classmethod
    def from_any(cls, raw: str | MethodName) -> MethodName:
        if isinstance(raw, MethodName):
            return raw
        token = str(raw).strip().lower()
        aliases = {
            "bo": cls.BO,
            "dr": cls.DR,
            "bo+dr": cls.BO_DR,
            "bo_dr": cls.BO_DR,
            "bodr": cls.BO_DR,
            "doraemon": cls.DORAEMON,
            "bo+doraemon": cls.BO_DORAEMON,
            "bo_doraemon": cls.BO_DORAEMON,
            "bodoraemon": cls.BO_DORAEMON,
        }
        if token not in aliases:
            valid = ", ".join([m.value for m in MethodName])
            raise ValueError(f"Unknown method '{raw}'. Expected one of: {valid}")
        return aliases[token]


@dataclass(frozen=True)
class PhysicsPoint:
    force: float
    gravity: float

    def as_dict(self) -> dict[str, float]:
        return {"force": float(self.force), "gravity": float(self.gravity)}


@dataclass(frozen=True)
class PhysicsBounds:
    force: tuple[float, float]
    gravity: tuple[float, float]

    def as_dict(self) -> dict[str, tuple[float, float]]:
        return {
            "force": (float(self.force[0]), float(self.force[1])),
            "gravity": (float(self.gravity[0]), float(self.gravity[1])),
        }


@dataclass(frozen=True)
class MethodRunConfig:
    method: MethodName
    task: TaskName
    env_id: str
    backend: RLBackend
    profile: ProfileName
    init_mode: InitMode
    primary_metric: PrimaryMetricName
    eval_episodes: int
    target_params: PhysicsPoint
    dr_bounds: PhysicsBounds
    edge_bounds: PhysicsBounds
    episode_steps: int
    mu0: tuple[float, float]
    checkpoint_path: Path | None
    device: str
    output_root: Path
    task_config: dict[str, Any] = field(default_factory=dict)
    method_overrides: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SuiteConfig:
    seeds: list[int]
    methods: list[MethodName]
    task: TaskName
    env_id: str
    backend: RLBackend
    profile: ProfileName
    init_mode: InitMode
    primary_metric: PrimaryMetricName
    eval_episodes: int
    target_params: PhysicsPoint
    dr_bounds: PhysicsBounds
    edge_bounds: PhysicsBounds
    episode_steps: int
    mu0: tuple[float, float]
    checkpoint_path: Path | None
    device: str
    output_root: Path
    task_config: dict[str, Any] = field(default_factory=dict)
    method_overrides: dict[MethodName, dict[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class ArtifactPaths:
    run_dir: str
    history_json: str
    history_csv: str
    best_checkpoint: str
    last_checkpoint: str


@dataclass
class MethodResult:
    method: MethodName
    seed: int
    profile: ProfileName
    artifacts: ArtifactPaths
    history: list[dict[str, Any]]
    best_score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SeedResult:
    seed: int
    methods: dict[MethodName, MethodResult]


@dataclass
class SuiteResult:
    run_dir: str
    profile: ProfileName
    seeds: list[int]
    methods: list[MethodName]
    master_csv: str
    master_json: str
    aggregate_csv: str
    manifest_json: str
    by_seed: dict[int, SeedResult]
    rows: list[dict[str, Any]]
    aggregate_rows: list[dict[str, Any]]


@dataclass(frozen=True)
class BatchTask:
    method: MethodName
    seed: int
    seed_index: int
    repetition: int
    task_id: str


@dataclass
class BatchResult:
    run_dir: str
    profile: ProfileName
    methods: list[MethodName]
    seeds: list[int]
    repetitions: int
    attempts_csv: str
    attempts_json: str
    aggregate_by_method_csv: str
    aggregate_by_method_seed_csv: str
    manifest_json: str
    rows: list[dict[str, Any]]
    aggregate_method_rows: list[dict[str, Any]]
    aggregate_method_seed_rows: list[dict[str, Any]]
    total_tasks: int
    succeeded: int
    failed: int
