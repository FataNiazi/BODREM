from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, Mapping, TypeVar

import numpy as np

from .halfcheetah_env import HALFCHEETAH_DEFAULT_ENV_ID
from .halfcheetah_params import (
    DEFAULT_HALFCHEETAH_BOUNDS,
    HalfCheetahBounds,
    HalfCheetahParams,
    sample_params,
)


TargetT = TypeVar("TargetT")


@dataclass(frozen=True)
class HiddenTargetPublicMetadata:
    task: str
    env_id: str
    target_family: str
    target_hash: str
    hidden_target: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "task": str(self.task),
            "env_id": str(self.env_id),
            "target_family": str(self.target_family),
            "target_hash": str(self.target_hash),
            "hidden_target": bool(self.hidden_target),
        }


@dataclass
class HiddenTargetManager(Generic[TargetT]):
    base_seed: int
    task: str
    env_id: str
    sampler: Callable[[np.random.Generator], TargetT]
    serializer: Callable[[TargetT], Mapping[str, Any] | list[float] | tuple[float, ...]]
    target_family: str = "hidden_target"
    _target: TargetT | None = field(default=None, init=False, repr=False)

    def sample_once(self) -> TargetT:
        if self._target is None:
            rng = np.random.default_rng(self.target_seed())
            self._target = self.sampler(rng)
        return self._target

    def get_target(self) -> TargetT:
        return self.sample_once()

    def target_seed(self) -> int:
        return self._seed_for("target", 0)

    def train_seed(self, index: int = 0) -> int:
        return self._seed_for("train", index)

    def eval_seed(self, index: int = 0) -> int:
        return self._seed_for("eval", index)

    def seed_schedule(self, purpose: str, count: int) -> list[int]:
        return [self._seed_for(purpose, idx) for idx in range(int(count))]

    def public_metadata(self) -> HiddenTargetPublicMetadata:
        target = self.sample_once()
        digest = hashlib.sha256(self._canonical_json(target).encode("utf-8")).hexdigest()
        return HiddenTargetPublicMetadata(
            task=str(self.task),
            env_id=str(self.env_id),
            target_family=str(self.target_family),
            target_hash=str(digest),
        )

    def public_metadata_dict(self) -> dict[str, Any]:
        return self.public_metadata().as_dict()

    def _seed_for(self, purpose: str, index: int) -> int:
        namespace = self._namespace_token(purpose)
        sequence = np.random.SeedSequence([int(self.base_seed), namespace, int(index)])
        return int(sequence.generate_state(1, dtype=np.uint32)[0])

    def _canonical_json(self, target: TargetT) -> str:
        payload = self.serializer(target)
        normalized = self._normalize_json_value(payload)
        return json.dumps(normalized, sort_keys=True, separators=(",", ":"))

    @staticmethod
    def _normalize_json_value(value: Any) -> Any:
        if hasattr(value, "as_dict"):
            return HiddenTargetManager._normalize_json_value(value.as_dict())
        if isinstance(value, Mapping):
            return {
                str(key): HiddenTargetManager._normalize_json_value(item)
                for key, item in sorted(value.items(), key=lambda kv: str(kv[0]))
            }
        if isinstance(value, np.ndarray):
            return [HiddenTargetManager._normalize_json_value(item) for item in value.tolist()]
        if isinstance(value, (list, tuple)):
            return [HiddenTargetManager._normalize_json_value(item) for item in value]
        if isinstance(value, (np.floating, float)):
            return float(value)
        if isinstance(value, (np.integer, int)):
            return int(value)
        if isinstance(value, (np.bool_, bool)):
            return bool(value)
        return value

    @staticmethod
    def _namespace_token(namespace: str) -> int:
        digest = hashlib.blake2b(str(namespace).encode("utf-8"), digest_size=4).digest()
        return int.from_bytes(digest, byteorder="big", signed=False)


@dataclass
class HalfCheetahTargetManager(HiddenTargetManager[HalfCheetahParams]):
    target_family: str = "halfcheetah_hidden_target_v1"


def make_halfcheetah_target_manager(
    base_seed: int,
    bounds: HalfCheetahBounds = DEFAULT_HALFCHEETAH_BOUNDS,
    env_id: str = HALFCHEETAH_DEFAULT_ENV_ID,
) -> HalfCheetahTargetManager:
    return HalfCheetahTargetManager(
        base_seed=int(base_seed),
        task="halfcheetah",
        env_id=str(env_id),
        sampler=lambda rng: sample_params(rng=rng, bounds=bounds),
        serializer=lambda params: params.as_dict(),
    )


__all__ = [
    "HalfCheetahTargetManager",
    "HiddenTargetManager",
    "HiddenTargetPublicMetadata",
    "make_halfcheetah_target_manager",
]
