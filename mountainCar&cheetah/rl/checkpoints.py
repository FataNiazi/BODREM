from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from .agent import (
    DQNAgent,
    clone_agent as _clone_agent,
    make_agent_from_hparams as _make_agent_from_hparams,
)

try:
    import stable_baselines3  # noqa: F401
    HAS_SB3 = True
except Exception as exc:  # pragma: no cover
    stable_baselines3 = None  # type: ignore[assignment]
    HAS_SB3 = False
    _SB3_IMPORT_ERROR = exc
else:  # pragma: no cover
    _SB3_IMPORT_ERROR = None

DEFAULT_CHECKPOINT_BACKEND = "native_dqn"
DEFAULT_CHECKPOINT_FORMAT = "torch_payload_v2"


def make_agent_from_hparams(hparams: dict | None, device: str = "cpu") -> DQNAgent:
    return _make_agent_from_hparams(hparams, device=device)


def require_sb3() -> None:
    if HAS_SB3:
        return
    message = (
        "stable-baselines3 is required for backend='sb3_sac'. "
        "Install via 'pip install stable-baselines3'."
    )
    if _SB3_IMPORT_ERROR is not None:
        raise RuntimeError(f"{message} Original import error: {_SB3_IMPORT_ERROR}")
    raise RuntimeError(message)


def load_warmstart_agent(checkpoint_path: str | Path, device: str = "cpu") -> DQNAgent:
    path = Path(checkpoint_path)
    if path.suffix == ".zip":
        raise ValueError(
            "Received a '.zip' checkpoint, which is typically an SB3 archive. "
            "load_warmstart_agent only supports native DQN torch checkpoints."
        )

    try:
        payload = torch.load(str(path), map_location=device)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load checkpoint '{path}'. If this is an SB3 checkpoint, use backend-specific "
            "loading instead of load_warmstart_agent."
        ) from exc

    metadata = payload.get("checkpoint_metadata", {}) if isinstance(payload, dict) else {}
    backend = str(metadata.get("backend", DEFAULT_CHECKPOINT_BACKEND)).strip().lower()
    if backend != DEFAULT_CHECKPOINT_BACKEND:
        raise ValueError(
            f"Checkpoint backend mismatch for '{path}': expected '{DEFAULT_CHECKPOINT_BACKEND}', "
            f"found '{backend}'."
        )

    hparams = payload.get("hparams")
    agent = _make_agent_from_hparams(hparams, device=device)
    agent.load_payload(payload)
    return agent


def save_agent_checkpoint(
    agent: DQNAgent,
    path: str | Path,
    extra: dict | None = None,
    *,
    backend: str = DEFAULT_CHECKPOINT_BACKEND,
    env_id: str | None = None,
    task: str | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = agent.state_payload()
    payload["extra"] = extra or {}
    payload["checkpoint_metadata"] = build_checkpoint_metadata(
        backend=backend,
        env_id=env_id,
        task=task,
        path=path,
    )
    torch.save(payload, str(path))


def clone_agent(agent: DQNAgent) -> DQNAgent:
    return _clone_agent(agent)


def build_checkpoint_metadata(
    *,
    backend: str = DEFAULT_CHECKPOINT_BACKEND,
    env_id: str | None = None,
    task: str | None = None,
    path: str | Path | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = {
        "backend": str(backend),
        "env_id": None if env_id is None else str(env_id),
        "task": None if task is None else str(task),
        "format": DEFAULT_CHECKPOINT_FORMAT,
    }
    if path is not None:
        metadata["checkpoint_path"] = str(Path(path))
    if extra:
        metadata.update(dict(extra))
    return metadata


def checkpoint_metadata_sidecar_path(path: str | Path) -> Path:
    checkpoint_path = Path(path)
    return checkpoint_path.with_name(f"{checkpoint_path.name}.metadata.json")


def save_checkpoint_metadata_sidecar(path: str | Path, metadata: dict[str, Any]) -> Path:
    sidecar = checkpoint_metadata_sidecar_path(path)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return sidecar


def load_checkpoint_metadata(path: str | Path, device: str = "cpu") -> dict[str, Any]:
    checkpoint_path = Path(path)
    sidecar = checkpoint_metadata_sidecar_path(checkpoint_path)
    if sidecar.exists():
        return json.loads(sidecar.read_text(encoding="utf-8"))

    if checkpoint_path.suffix == ".zip":
        return build_checkpoint_metadata(
            backend="sb3_sac",
            path=checkpoint_path,
            extra={"format": "sb3_zip_archive"},
        )

    payload = torch.load(str(checkpoint_path), map_location=device)
    if isinstance(payload, dict) and "checkpoint_metadata" in payload:
        return dict(payload["checkpoint_metadata"])
    return build_checkpoint_metadata(path=checkpoint_path)


__all__ = [
    "DEFAULT_CHECKPOINT_BACKEND",
    "HAS_SB3",
    "build_checkpoint_metadata",
    "checkpoint_metadata_sidecar_path",
    "clone_agent",
    "load_checkpoint_metadata",
    "load_warmstart_agent",
    "make_agent_from_hparams",
    "require_sb3",
    "save_agent_checkpoint",
    "save_checkpoint_metadata_sidecar",
]
