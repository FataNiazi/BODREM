from __future__ import annotations

import json
import math
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from .config import build_method_run_config, build_suite_config
from .registry import list_methods as registry_list_methods
from .registry import load_algorithm_class, normalize_method_name
from .types import (
    BatchResult,
    BatchTask,
    MethodName,
    MethodResult,
    MethodRunConfig,
    SeedResult,
    SuiteConfig,
    SuiteResult,
)
from .utils.io import ensure_run_dir, slugify, write_csv_rows, write_json


PUBLIC_METADATA_KEYS = {
    "task",
    "env_id",
    "backend",
    "primary_metric",
    "eval_episodes",
    "target_family",
    "target_hash",
    "hidden_target",
    "used_warmstart",
}


def list_methods() -> list[MethodName]:
    return registry_list_methods()


def _coerce_suite_config(config: SuiteConfig | dict[str, Any]) -> SuiteConfig:
    if isinstance(config, SuiteConfig):
        return config
    if isinstance(config, dict):
        task_config = config.get("task_config")
        if task_config is None and config.get("task_config_json") is not None:
            task_config = json.loads(str(config["task_config_json"]))
        if task_config is not None and not isinstance(task_config, dict):
            raise ValueError("task_config must be a mapping if provided.")
        kwargs: dict[str, Any] = {
            "seeds": [int(s) for s in config.get("seeds", [])],
            "methods": config.get("methods"),
            "task": config.get("task", "mountaincar"),
            "env_id": config.get("env_id"),
            "backend": config.get("backend"),
            "profile": str(config.get("profile", "smoke")),
            "init_mode": str(config.get("init_mode", "auto")),
            "primary_metric": config.get("primary_metric"),
            "eval_episodes": int(config.get("eval_episodes", 5)),
            "task_config": task_config,
            "checkpoint_path": config.get("checkpoint_path"),
            "device": config.get("device"),
            "output_root": config.get("output_root"),
            "episode_steps": int(config.get("episode_steps", 400)),
            "mu0": tuple(config.get("mu0", (0.0010, 0.0025))),
            "method_overrides": config.get("method_overrides"),
        }
        if "target_params" in config and config.get("target_params") is not None:
            kwargs["target_params"] = config["target_params"]
        if "dr_bounds" in config and config.get("dr_bounds") is not None:
            kwargs["dr_bounds"] = config["dr_bounds"]
        if "edge_bounds" in config and config.get("edge_bounds") is not None:
            kwargs["edge_bounds"] = config["edge_bounds"]
        return build_suite_config(**kwargs)
    raise TypeError(f"Unsupported suite config type: {type(config)!r}")


def run_method(
    method: str | MethodName,
    seed: int,
    config: MethodRunConfig | SuiteConfig | dict[str, Any],
) -> MethodResult:
    method_name = normalize_method_name(method)
    if isinstance(config, MethodRunConfig):
        if config.method == method_name:
            method_cfg = config
        else:
            method_cfg = replace(config, method=method_name, method_overrides={})
    else:
        suite_cfg = _coerce_suite_config(config)
        method_cfg = build_method_run_config(suite_cfg, method_name)

    algorithm_cls = load_algorithm_class(method_name)
    algorithm = algorithm_cls(method_cfg)
    result = algorithm.run(seed=int(seed))
    if not isinstance(result, MethodResult):
        raise TypeError(f"Algorithm '{method_name.value}' returned unsupported type: {type(result)!r}")
    return result


def _safe_public_metadata(
    result: MethodResult,
    *,
    suite_cfg: SuiteConfig,
) -> dict[str, Any]:
    metadata = dict(result.metadata or {})
    public: dict[str, Any] = {}
    for key in PUBLIC_METADATA_KEYS:
        if key in metadata:
            public[key] = metadata[key]

    public.setdefault("task", suite_cfg.task.value)
    public.setdefault("env_id", suite_cfg.env_id)
    public.setdefault("backend", suite_cfg.backend.value)
    public.setdefault("primary_metric", suite_cfg.primary_metric)
    public.setdefault("eval_episodes", int(suite_cfg.eval_episodes))
    return public


def _metric_value_from_history(history: list[dict[str, Any]], metric_name: str) -> float:
    if not history:
        return math.nan

    last_row = history[-1]
    candidates = [str(metric_name), "final_primary_metric", "target_solve_rate", "score", "return", "mean_return"]
    for key in candidates:
        if key in last_row:
            return _coerce_float(last_row.get(key))
    return math.nan


def _final_metric_name(suite_cfg: SuiteConfig) -> str:
    return str(suite_cfg.primary_metric)


def _final_metric_value(result: MethodResult, suite_cfg: SuiteConfig) -> float:
    metric_name = _final_metric_name(suite_cfg)
    value = _metric_value_from_history(result.history, metric_name)
    if math.isnan(value) and metric_name != "target_solve_rate":
        value = _metric_value_from_history(result.history, "target_solve_rate")
    if math.isnan(value):
        value = _coerce_float(result.best_score)
    return float(value)


def run_seed(seed: int, config: SuiteConfig | dict[str, Any]) -> SeedResult:
    suite_cfg = _coerce_suite_config(config)
    methods = list(suite_cfg.methods)

    method_results: dict[MethodName, MethodResult] = {}
    for method in methods:
        method_results[method] = run_method(method=method, seed=int(seed), config=suite_cfg)

    return SeedResult(seed=int(seed), methods=method_results)


def _row_from_method_result(result: MethodResult, *, suite_cfg: SuiteConfig) -> dict[str, Any]:
    final_primary_metric = _final_metric_value(result, suite_cfg)
    public_metadata = _safe_public_metadata(result, suite_cfg=suite_cfg)
    return {
        "method": result.method.value,
        "seed": int(result.seed),
        "profile": result.profile,
        "task": suite_cfg.task.value,
        "env_id": suite_cfg.env_id,
        "backend": suite_cfg.backend.value,
        "primary_metric": suite_cfg.primary_metric,
        "best_score": float(result.best_score),
        "final_primary_metric": float(final_primary_metric),
        "final_target_solve_rate": float(final_primary_metric),
        "run_dir": result.artifacts.run_dir,
        "best_checkpoint": result.artifacts.best_checkpoint,
        "last_checkpoint": result.artifacts.last_checkpoint,
        "history_rows": int(len(result.history)),
        "public_metadata_json": json.dumps(public_metadata, sort_keys=True),
    }


def _coerce_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return math.nan


def _row_final_primary_metric(row: dict[str, Any]) -> float:
    value = _coerce_float(row.get("final_primary_metric"))
    if math.isnan(value):
        value = _coerce_float(row.get("final_target_solve_rate"))
    return value


def _aggregate_rows(
    rows: list[dict[str, Any]],
    methods: list[MethodName],
    metric_name: str,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for method in methods:
        part = [r for r in rows if r["method"] == method.value]
        if not part:
            continue
        best_scores = [_coerce_float(r.get("best_score")) for r in part]
        best_scores = [v for v in best_scores if not math.isnan(v)]
        final_scores = [_row_final_primary_metric(r) for r in part]
        final_scores = [v for v in final_scores if not math.isnan(v)]
        out.append(
            {
                "method": method.value,
                "primary_metric": metric_name,
                "seeds_count": int(len(part)),
                "best_score_mean": float(sum(best_scores) / len(best_scores)) if best_scores else math.nan,
                "best_score_min": float(min(best_scores)) if best_scores else math.nan,
                "best_score_max": float(max(best_scores)) if best_scores else math.nan,
                "final_primary_mean": float(sum(final_scores) / len(final_scores)) if final_scores else math.nan,
                "final_primary_min": float(min(final_scores)) if final_scores else math.nan,
                "final_primary_max": float(max(final_scores)) if final_scores else math.nan,
                "final_target_mean": float(sum(final_scores) / len(final_scores)) if final_scores else math.nan,
                "final_target_min": float(min(final_scores)) if final_scores else math.nan,
                "final_target_max": float(max(final_scores)) if final_scores else math.nan,
            }
        )
    return out


def run_suite(config: SuiteConfig | dict[str, Any]) -> SuiteResult:
    suite_cfg = _coerce_suite_config(config)
    suite_dir = ensure_run_dir(
        Path(suite_cfg.output_root),
        f"all_methods_{suite_cfg.task.value}_{slugify(suite_cfg.env_id)}_{suite_cfg.backend.value}_{suite_cfg.primary_metric}",
    )
    suite_cfg_local = replace(suite_cfg, output_root=suite_dir)

    by_seed: dict[int, SeedResult] = {}
    rows: list[dict[str, Any]] = []
    public_metadata_rows: list[dict[str, Any]] = []

    for seed in suite_cfg_local.seeds:
        seed_result = run_seed(seed=int(seed), config=suite_cfg_local)
        by_seed[int(seed)] = seed_result
        for method_result in seed_result.methods.values():
            rows.append(_row_from_method_result(method_result, suite_cfg=suite_cfg_local))
            public_metadata_rows.append(
                {
                    "method": method_result.method.value,
                    "seed": int(method_result.seed),
                    **_safe_public_metadata(method_result, suite_cfg=suite_cfg_local),
                }
            )

    aggregate_rows = _aggregate_rows(rows, suite_cfg_local.methods, _final_metric_name(suite_cfg_local))

    master_csv = suite_dir / "master_results.csv"
    master_json = suite_dir / "master_results.json"
    aggregate_csv = suite_dir / "aggregate_by_method.csv"
    manifest_json = suite_dir / "manifest.json"

    write_csv_rows(master_csv, rows)
    write_json(master_json, rows)
    write_csv_rows(aggregate_csv, aggregate_rows)

    manifest = {
        "run_dir": str(suite_dir),
        "task": suite_cfg_local.task.value,
        "env_id": suite_cfg_local.env_id,
        "backend": suite_cfg_local.backend.value,
        "primary_metric": suite_cfg_local.primary_metric,
        "eval_episodes": int(suite_cfg_local.eval_episodes),
        "profile": suite_cfg_local.profile,
        "init_mode": suite_cfg_local.init_mode,
        "methods": [m.value for m in suite_cfg_local.methods],
        "seeds": [int(s) for s in suite_cfg_local.seeds],
        "task_config": dict(suite_cfg_local.task_config),
        "master_csv": str(master_csv),
        "master_json": str(master_json),
        "aggregate_csv": str(aggregate_csv),
        "method_public_metadata": public_metadata_rows,
    }
    write_json(manifest_json, manifest)

    return SuiteResult(
        run_dir=str(suite_dir),
        profile=suite_cfg_local.profile,
        seeds=[int(s) for s in suite_cfg_local.seeds],
        methods=list(suite_cfg_local.methods),
        master_csv=str(master_csv),
        master_json=str(master_json),
        aggregate_csv=str(aggregate_csv),
        manifest_json=str(manifest_json),
        by_seed=by_seed,
        rows=rows,
        aggregate_rows=aggregate_rows,
    )


def _dedupe_preserve_order(values: list[int]) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for value in values:
        item = int(value)
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _build_batch_tasks(
    methods: list[MethodName],
    seeds: list[int],
    repetitions: int,
    dedupe_seeds: bool,
) -> tuple[list[int], list[BatchTask]]:
    effective_seeds = _dedupe_preserve_order(seeds) if dedupe_seeds else [int(s) for s in seeds]
    tasks: list[BatchTask] = []

    for repetition in range(int(repetitions)):
        for seed_index, seed in enumerate(effective_seeds):
            for method in methods:
                task_id = f"{method.value}|seed={int(seed)}|idx={int(seed_index)}|rep={int(repetition)}"
                tasks.append(
                    BatchTask(
                        method=method,
                        seed=int(seed),
                        seed_index=int(seed_index),
                        repetition=int(repetition),
                        task_id=task_id,
                    )
                )

    return effective_seeds, tasks


def _batch_attempt_dir(batch_dir: Path, task: BatchTask) -> Path:
    name = (
        f"{slugify(task.method.value)}"
        f"_seed{int(task.seed)}"
        f"_idx{int(task.seed_index)}"
        f"_rep{int(task.repetition)}"
    )
    return batch_dir / "attempts" / name


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_attempt_ok_row(
    task: BatchTask,
    suite_cfg: SuiteConfig,
    result: MethodResult,
    started_at: str,
    ended_at: str,
    duration_sec: float,
) -> dict[str, Any]:
    row = _row_from_method_result(result, suite_cfg=suite_cfg)
    return {
        "task_id": task.task_id,
        "method": task.method.value,
        "seed": int(task.seed),
        "seed_index": int(task.seed_index),
        "repetition": int(task.repetition),
        "profile": suite_cfg.profile,
        "task": suite_cfg.task.value,
        "env_id": suite_cfg.env_id,
        "backend": suite_cfg.backend.value,
        "primary_metric": suite_cfg.primary_metric,
        "status": "ok",
        "best_score": float(row["best_score"]),
        "final_primary_metric": float(row["final_primary_metric"]),
        "final_target_solve_rate": float(row["final_target_solve_rate"]),
        "history_rows": int(row["history_rows"]),
        "run_dir": str(row["run_dir"]),
        "best_checkpoint": str(row["best_checkpoint"]),
        "last_checkpoint": str(row["last_checkpoint"]),
        "error_type": "",
        "error_message": "",
        "started_at": started_at,
        "ended_at": ended_at,
        "duration_sec": float(duration_sec),
        "public_metadata_json": str(row.get("public_metadata_json", "{}")),
    }


def _build_attempt_failed_row(
    task: BatchTask,
    suite_cfg: SuiteConfig,
    started_at: str,
    ended_at: str,
    duration_sec: float,
    error: Exception,
) -> dict[str, Any]:
    return {
        "task_id": task.task_id,
        "method": task.method.value,
        "seed": int(task.seed),
        "seed_index": int(task.seed_index),
        "repetition": int(task.repetition),
        "profile": suite_cfg.profile,
        "task": suite_cfg.task.value,
        "env_id": suite_cfg.env_id,
        "backend": suite_cfg.backend.value,
        "primary_metric": suite_cfg.primary_metric,
        "status": "failed",
        "best_score": math.nan,
        "final_primary_metric": math.nan,
        "final_target_solve_rate": math.nan,
        "history_rows": 0,
        "run_dir": "",
        "best_checkpoint": "",
        "last_checkpoint": "",
        "error_type": type(error).__name__,
        "error_message": str(error),
        "started_at": started_at,
        "ended_at": ended_at,
        "duration_sec": float(duration_sec),
        "public_metadata_json": "{}",
    }


def _aggregate_batch_by_method(
    rows: list[dict[str, Any]],
    methods: list[MethodName],
    metric_name: str,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for method in methods:
        all_rows = [r for r in rows if r.get("method") == method.value]
        if not all_rows:
            continue

        ok_rows = [r for r in all_rows if str(r.get("status", "")).lower() == "ok"]
        best_scores = [_coerce_float(r.get("best_score")) for r in ok_rows]
        best_scores = [v for v in best_scores if not math.isnan(v)]
        final_scores = [_row_final_primary_metric(r) for r in ok_rows]
        final_scores = [v for v in final_scores if not math.isnan(v)]

        out.append(
            {
                "method": method.value,
                "primary_metric": metric_name,
                "attempts_total": int(len(all_rows)),
                "attempts_succeeded": int(len(ok_rows)),
                "attempts_failed": int(len(all_rows) - len(ok_rows)),
                "best_score_mean": float(sum(best_scores) / len(best_scores)) if best_scores else math.nan,
                "best_score_min": float(min(best_scores)) if best_scores else math.nan,
                "best_score_max": float(max(best_scores)) if best_scores else math.nan,
                "final_primary_mean": float(sum(final_scores) / len(final_scores)) if final_scores else math.nan,
                "final_primary_min": float(min(final_scores)) if final_scores else math.nan,
                "final_primary_max": float(max(final_scores)) if final_scores else math.nan,
                "final_target_mean": float(sum(final_scores) / len(final_scores)) if final_scores else math.nan,
                "final_target_min": float(min(final_scores)) if final_scores else math.nan,
                "final_target_max": float(max(final_scores)) if final_scores else math.nan,
            }
        )
    return out


def _aggregate_batch_by_method_seed(
    rows: list[dict[str, Any]],
    methods: list[MethodName],
    metric_name: str,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    for method in methods:
        method_rows = [r for r in rows if r.get("method") == method.value]
        if not method_rows:
            continue

        seeds = sorted({int(r.get("seed", 0)) for r in method_rows})
        for seed in seeds:
            all_rows = [r for r in method_rows if int(r.get("seed", 0)) == int(seed)]
            ok_rows = [r for r in all_rows if str(r.get("status", "")).lower() == "ok"]

            best_scores = [_coerce_float(r.get("best_score")) for r in ok_rows]
            best_scores = [v for v in best_scores if not math.isnan(v)]
            final_scores = [_row_final_primary_metric(r) for r in ok_rows]
            final_scores = [v for v in final_scores if not math.isnan(v)]
            seed_slots = {int(r.get("seed_index", 0)) for r in all_rows}

            out.append(
                {
                    "method": method.value,
                    "primary_metric": metric_name,
                    "seed": int(seed),
                    "seed_slots": int(len(seed_slots)),
                    "attempts_total": int(len(all_rows)),
                    "attempts_succeeded": int(len(ok_rows)),
                    "attempts_failed": int(len(all_rows) - len(ok_rows)),
                    "best_score_mean": float(sum(best_scores) / len(best_scores)) if best_scores else math.nan,
                    "best_score_min": float(min(best_scores)) if best_scores else math.nan,
                    "best_score_max": float(max(best_scores)) if best_scores else math.nan,
                    "final_primary_mean": float(sum(final_scores) / len(final_scores)) if final_scores else math.nan,
                    "final_primary_min": float(min(final_scores)) if final_scores else math.nan,
                    "final_primary_max": float(max(final_scores)) if final_scores else math.nan,
                    "final_target_mean": float(sum(final_scores) / len(final_scores)) if final_scores else math.nan,
                    "final_target_min": float(min(final_scores)) if final_scores else math.nan,
                    "final_target_max": float(max(final_scores)) if final_scores else math.nan,
                }
            )

    return out


def _read_json_list(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array in '{path}'.")
    out: list[dict[str, Any]] = []
    for item in data:
        if isinstance(item, dict):
            out.append(dict(item))
    return out


def _read_json_dict(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in '{path}'.")
    return dict(data)


def _validate_resume_manifest(
    manifest: dict[str, Any],
    suite_cfg: SuiteConfig,
    repetitions: int,
    dedupe_seeds: bool,
) -> None:
    if not manifest:
        return

    expected_methods = [m.value for m in suite_cfg.methods]
    expected_seeds = [int(s) for s in suite_cfg.seeds]

    checks: list[tuple[str, Any, Any]] = [
        ("task", manifest.get("task"), suite_cfg.task.value),
        ("env_id", manifest.get("env_id"), suite_cfg.env_id),
        ("backend", manifest.get("backend"), suite_cfg.backend.value),
        ("primary_metric", manifest.get("primary_metric"), suite_cfg.primary_metric),
        ("eval_episodes", manifest.get("eval_episodes"), int(suite_cfg.eval_episodes)),
        ("task_config", manifest.get("task_config"), dict(suite_cfg.task_config)),
        ("profile", manifest.get("profile"), suite_cfg.profile),
        ("init_mode", manifest.get("init_mode"), suite_cfg.init_mode),
        ("methods", manifest.get("methods"), expected_methods),
        ("seeds", manifest.get("seeds"), expected_seeds),
        ("repetitions", manifest.get("repetitions"), int(repetitions)),
        ("dedupe_seeds", manifest.get("dedupe_seeds"), bool(dedupe_seeds)),
    ]

    mismatches: list[str] = []
    for key, existing, expected in checks:
        if existing is None:
            continue
        if existing != expected:
            mismatches.append(f"{key}: existing={existing!r} expected={expected!r}")

    if mismatches:
        raise ValueError("Resume configuration mismatch: " + "; ".join(mismatches))


def _task_status_index(rows: list[dict[str, Any]]) -> dict[str, str]:
    index: dict[str, str] = {}
    for row in rows:
        task_id = str(row.get("task_id", "")).strip()
        if not task_id:
            continue
        index[task_id] = str(row.get("status", "")).strip().lower()
    return index


def _write_batch_artifacts(
    *,
    batch_dir: Path,
    suite_cfg: SuiteConfig,
    repetitions: int,
    dedupe_seeds: bool,
    on_error: str,
    tasks: list[BatchTask],
    rows: list[dict[str, Any]],
) -> tuple[Path, Path, Path, Path]:
    attempts_csv = batch_dir / "attempts.csv"
    attempts_json = batch_dir / "attempts.json"
    aggregate_by_method_csv = batch_dir / "aggregate_by_method.csv"
    aggregate_by_method_seed_csv = batch_dir / "aggregate_by_method_seed.csv"
    manifest_json = batch_dir / "manifest.json"

    aggregate_method_rows = _aggregate_batch_by_method(rows, suite_cfg.methods, suite_cfg.primary_metric)
    aggregate_method_seed_rows = _aggregate_batch_by_method_seed(rows, suite_cfg.methods, suite_cfg.primary_metric)
    public_metadata_rows: list[dict[str, Any]] = []
    for row in rows:
        raw = row.get("public_metadata_json", "{}")
        try:
            parsed = json.loads(str(raw))
        except Exception:
            parsed = {}
        if isinstance(parsed, dict):
            public_metadata_rows.append(
                {
                    "task_id": row.get("task_id", ""),
                    "method": row.get("method", ""),
                    "seed": row.get("seed", ""),
                    **parsed,
                }
            )

    succeeded = int(sum(1 for r in rows if str(r.get("status", "")).lower() == "ok"))
    failed = int(sum(1 for r in rows if str(r.get("status", "")).lower() == "failed"))

    write_csv_rows(attempts_csv, rows)
    write_json(attempts_json, rows)
    write_csv_rows(aggregate_by_method_csv, aggregate_method_rows)
    write_csv_rows(aggregate_by_method_seed_csv, aggregate_method_seed_rows)

    manifest = {
        "run_dir": str(batch_dir),
        "task": suite_cfg.task.value,
        "env_id": suite_cfg.env_id,
        "backend": suite_cfg.backend.value,
        "primary_metric": suite_cfg.primary_metric,
        "eval_episodes": int(suite_cfg.eval_episodes),
        "profile": suite_cfg.profile,
        "init_mode": suite_cfg.init_mode,
        "methods": [m.value for m in suite_cfg.methods],
        "seeds": [int(s) for s in suite_cfg.seeds],
        "repetitions": int(repetitions),
        "dedupe_seeds": bool(dedupe_seeds),
        "on_error": str(on_error),
        "task_config": dict(suite_cfg.task_config),
        "attempts_csv": str(attempts_csv),
        "attempts_json": str(attempts_json),
        "aggregate_by_method_csv": str(aggregate_by_method_csv),
        "aggregate_by_method_seed_csv": str(aggregate_by_method_seed_csv),
        "manifest_json": str(manifest_json),
        "attempt_public_metadata": public_metadata_rows,
        "total_tasks": int(len(tasks)),
        "succeeded": int(succeeded),
        "failed": int(failed),
    }
    write_json(manifest_json, manifest)

    return attempts_csv, attempts_json, aggregate_by_method_csv, aggregate_by_method_seed_csv


def run_batch(
    config: SuiteConfig | dict[str, Any],
    *,
    repetitions: int,
    on_error: Literal["continue", "fail-fast"] = "continue",
    dedupe_seeds: bool = False,
    resume_from: str | Path | None = None,
    rerun_failed: bool = False,
) -> BatchResult:
    if int(repetitions) < 1:
        raise ValueError("repetitions must be >= 1")

    if on_error not in {"continue", "fail-fast"}:
        raise ValueError("on_error must be one of: continue|fail-fast")

    suite_cfg = _coerce_suite_config(config)
    _, tasks = _build_batch_tasks(
        methods=list(suite_cfg.methods),
        seeds=[int(s) for s in suite_cfg.seeds],
        repetitions=int(repetitions),
        dedupe_seeds=bool(dedupe_seeds),
    )

    if resume_from is not None:
        resume_path = Path(resume_from)
        batch_dir = resume_path.parent if resume_path.is_file() else resume_path
        batch_dir = batch_dir.resolve()
        if not batch_dir.exists():
            raise FileNotFoundError(f"Resume path does not exist: {batch_dir}")
    else:
        batch_dir = ensure_run_dir(
            Path(suite_cfg.output_root),
            f"batch_run_{suite_cfg.task.value}_{slugify(suite_cfg.env_id)}_{suite_cfg.backend.value}_{suite_cfg.primary_metric}",
        )

    attempts_json_path = batch_dir / "attempts.json"
    manifest_path = batch_dir / "manifest.json"
    existing_rows = _read_json_list(attempts_json_path)

    if resume_from is not None:
        existing_manifest = _read_json_dict(manifest_path)
        _validate_resume_manifest(
            existing_manifest,
            suite_cfg=suite_cfg,
            repetitions=int(repetitions),
            dedupe_seeds=bool(dedupe_seeds),
        )

    rows: list[dict[str, Any]] = list(existing_rows)
    status_index = _task_status_index(rows)

    for task in tasks:
        prior = status_index.get(task.task_id)
        if prior == "ok":
            continue
        if prior == "failed" and not rerun_failed:
            continue

        started_at = _utc_now_iso()
        t0 = time.perf_counter()

        try:
            method_cfg = build_method_run_config(suite_cfg, task.method)
            attempt_root = _batch_attempt_dir(batch_dir, task)
            attempt_cfg = replace(method_cfg, output_root=attempt_root)
            result = run_method(method=task.method, seed=int(task.seed), config=attempt_cfg)

            ended_at = _utc_now_iso()
            duration_sec = float(time.perf_counter() - t0)
            row = _build_attempt_ok_row(
                task=task,
                suite_cfg=suite_cfg,
                result=result,
                started_at=started_at,
                ended_at=ended_at,
                duration_sec=duration_sec,
            )
            rows.append(row)
            status_index[task.task_id] = "ok"
        except Exception as exc:
            ended_at = _utc_now_iso()
            duration_sec = float(time.perf_counter() - t0)
            row = _build_attempt_failed_row(
                task=task,
                suite_cfg=suite_cfg,
                started_at=started_at,
                ended_at=ended_at,
                duration_sec=duration_sec,
                error=exc,
            )
            rows.append(row)
            status_index[task.task_id] = "failed"

            _write_batch_artifacts(
                batch_dir=batch_dir,
                suite_cfg=suite_cfg,
                repetitions=int(repetitions),
                dedupe_seeds=bool(dedupe_seeds),
                on_error=str(on_error),
                tasks=tasks,
                rows=rows,
            )
            if on_error == "fail-fast":
                break

        _write_batch_artifacts(
            batch_dir=batch_dir,
            suite_cfg=suite_cfg,
            repetitions=int(repetitions),
            dedupe_seeds=bool(dedupe_seeds),
            on_error=str(on_error),
            tasks=tasks,
            rows=rows,
        )

    attempts_csv, attempts_json, aggregate_by_method_csv, aggregate_by_method_seed_csv = _write_batch_artifacts(
        batch_dir=batch_dir,
        suite_cfg=suite_cfg,
        repetitions=int(repetitions),
        dedupe_seeds=bool(dedupe_seeds),
        on_error=str(on_error),
        tasks=tasks,
        rows=rows,
    )

    aggregate_method_rows = _aggregate_batch_by_method(rows, suite_cfg.methods, suite_cfg.primary_metric)
    aggregate_method_seed_rows = _aggregate_batch_by_method_seed(rows, suite_cfg.methods, suite_cfg.primary_metric)
    succeeded = int(sum(1 for r in rows if str(r.get("status", "")).lower() == "ok"))
    failed = int(sum(1 for r in rows if str(r.get("status", "")).lower() == "failed"))

    return BatchResult(
        run_dir=str(batch_dir),
        profile=suite_cfg.profile,
        methods=list(suite_cfg.methods),
        seeds=[int(s) for s in suite_cfg.seeds],
        repetitions=int(repetitions),
        attempts_csv=str(attempts_csv),
        attempts_json=str(attempts_json),
        aggregate_by_method_csv=str(aggregate_by_method_csv),
        aggregate_by_method_seed_csv=str(aggregate_by_method_seed_csv),
        manifest_json=str(batch_dir / "manifest.json"),
        rows=rows,
        aggregate_method_rows=aggregate_method_rows,
        aggregate_method_seed_rows=aggregate_method_seed_rows,
        total_tasks=int(len(tasks)),
        succeeded=int(succeeded),
        failed=int(failed),
    )
