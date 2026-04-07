from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
import itertools
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.ticker import MaxNLocator


METHOD_ALIASES: dict[str, str] = {
    "BO": "BO",
    "DR": "DR",
    "BO+DR": "BO+DR",
    "BO_DR": "BO+DR",
    "BOPLUSDR": "BO+DR",
    "DORAEMON": "DORAEMON",
    "BO+DORAEMON": "BO+DORAEMON",
    "BO_DORAEMON": "BO+DORAEMON",
    "BOPLUSDORAEMON": "BO+DORAEMON",
}

METHOD_SLUGS: dict[str, str] = {
    "BO": "bo",
    "DR": "dr",
    "BO+DR": "bo_dr",
    "DORAEMON": "doraemon",
    "BO+DORAEMON": "bo_doraemon",
}

METHOD_COLORS: dict[str, str] = {
    "BO": "#0072B2",
    "DR": "#E69F00",
    "BO+DR": "#009E73",
    "DORAEMON": "#CC79A7",
    "BO+DORAEMON": "#D55E00",
}

METHOD_ORDER: list[str] = ["BO", "DR", "BO+DR", "DORAEMON", "BO+DORAEMON"]
TASK_TOTAL_EPISODES: dict[str, float] = {
    "mountaincar": 3600.0,
    "halfcheetah": 1800.0,
}
TASK_MAX_REWARD: dict[str, float] = {
    "mountaincar": 1.0,
    "halfcheetah": 5000.0,
}


@dataclass
class RunSeries:
    method: str
    method_slug: str
    variant: str
    task: str
    env_id: str
    seed: int
    run_dir: str
    primary_metric_name: str
    step: np.ndarray
    primary_metric: np.ndarray
    g_last: np.ndarray
    center_names: list[str]
    center: np.ndarray
    spread_kind: str | None
    spread_names: list[str]
    spread: np.ndarray
    episode_x: np.ndarray
    budget: float
    target_map: dict[str, float] = field(default_factory=dict)
    target_hidden: bool = False
    target_hash: str | None = None
    bounds_map: dict[str, tuple[float, float]] = field(default_factory=dict)


@dataclass
class MethodSummary:
    method: str
    variant: str
    task: str
    env_id: str
    metric_name: str
    metric_steps: np.ndarray
    metric_x: np.ndarray
    metric_mean: np.ndarray
    metric_std: np.ndarray
    metric_n: np.ndarray
    g_steps: np.ndarray
    g_mean: np.ndarray
    g_std: np.ndarray
    g_n: np.ndarray
    center_names: list[str]
    center_steps: np.ndarray
    center_mean: np.ndarray
    center_std: np.ndarray
    center_n: np.ndarray
    spread_kind: str | None
    spread_names: list[str]
    spread_steps: np.ndarray
    spread_mean: np.ndarray
    spread_std: np.ndarray
    spread_n: np.ndarray
    target_center: np.ndarray
    target_hidden: bool
    target_hashes: list[str]
    seeds: list[int]
    run_count: int


def normalize_method_name(name: str) -> str:
    key = str(name).strip().upper().replace(" ", "").replace("-", "_")
    key = key.replace("PLUS", "+")
    key = key.replace("_", "") if key in {"BOPLUSDR", "BOPLUSDORAEMON"} else key
    if key in METHOD_ALIASES:
        return METHOD_ALIASES[key]
    if str(name).strip().upper() in METHOD_ALIASES:
        return METHOD_ALIASES[str(name).strip().upper()]
    if str(name).strip() in METHOD_ALIASES:
        return METHOD_ALIASES[str(name).strip()]
    # Support common fallback spellings.
    fallback = str(name).strip().upper().replace("_", "+")
    fallback = fallback.replace(" ", "")
    if fallback in METHOD_ALIASES:
        return METHOD_ALIASES[fallback]
    return str(name).strip().upper()


def method_slug(method: str) -> str:
    return METHOD_SLUGS.get(method, method.lower().replace("+", "_").replace(" ", "_"))


def _slugify(text: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "_", str(text).strip().lower())
    return value.strip("_") or "default"


def _infer_variant(run_dir: Path, manifest: dict[str, Any]) -> str:
    if isinstance(manifest.get("variant"), str) and str(manifest.get("variant")).strip():
        raw = str(manifest.get("variant")).strip()
        return raw
    # Accept common run-dir forms like "v1_profile", "bo_v1_seed3", or nested ".../v2_x/...".
    pattern = re.compile(r"(?:^|[_\-])(v\d+)(?:$|[_\-])")
    for part in reversed(run_dir.parts):
        m = pattern.search(part)
        if m:
            return m.group(1)
    profile = manifest.get("profile")
    if isinstance(profile, str) and profile.strip():
        return f"profile_{profile.strip()}"
    return "default"


def to_float(value: Any) -> float:
    if value is None:
        return math.nan
    if isinstance(value, (float, int)):
        return float(value)
    text = str(value).strip()
    if not text:
        return math.nan
    lower = text.lower()
    if lower in {"nan", "none", "null"}:
        return math.nan
    try:
        return float(text)
    except ValueError:
        return math.nan


def to_int(value: Any, default: int = 0) -> int:
    val = to_float(value)
    if math.isnan(val):
        return int(default)
    return int(val)


def parse_seed_from_run_dir(run_dir: Path) -> int:
    match = re.search(r"_seed(\d+)", run_dir.name)
    if match:
        return int(match.group(1))
    return 0


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_path(path_like: str | None, *, base: Path) -> Path | None:
    if not path_like:
        return None
    candidate = Path(path_like)
    if not candidate.is_absolute():
        candidate = (base / candidate).resolve()
    return candidate


def _run_dirs_from_master(master_path: Path) -> list[Path]:
    if not master_path.exists():
        return []
    data = _read_json(master_path)
    run_dirs: list[Path] = []
    if isinstance(data, list):
        for row in data:
            if not isinstance(row, dict):
                continue
            run_dir = row.get("run_dir")
            if not run_dir:
                continue
            run_dirs.append(Path(str(run_dir)).resolve())
    return run_dirs


def _run_dirs_from_attempts(attempts_path: Path) -> list[Path]:
    if not attempts_path.exists():
        return []
    data = _read_json(attempts_path)
    run_dirs: list[Path] = []
    if isinstance(data, list):
        for row in data:
            if not isinstance(row, dict):
                continue
            run_dir = row.get("run_dir")
            status = str(row.get("status", "")).lower()
            if run_dir and status in {"ok", "success", "succeeded", ""}:
                run_dirs.append(Path(str(run_dir)).resolve())
    return run_dirs


def discover_run_directories(input_path: str | Path) -> list[Path]:
    path = Path(input_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Input path not found: {path}")

    candidates: list[Path] = []

    def add_run_dir(run_dir: Path) -> None:
        run_dir = run_dir.resolve()
        if (run_dir / "history.csv").exists():
            candidates.append(run_dir)

    def parse_manifest(manifest_path: Path) -> None:
        manifest = _read_json(manifest_path)
        base = manifest_path.parent

        if isinstance(manifest, dict):
            run_dir = manifest.get("run_dir")
            if run_dir:
                add_run_dir(Path(str(run_dir)).resolve())

            for key in ("master_json", "attempts_json"):
                child = _resolve_path(manifest.get(key), base=base)
                if child and child.exists():
                    if key == "master_json":
                        for rd in _run_dirs_from_master(child):
                            add_run_dir(rd)
                    else:
                        for rd in _run_dirs_from_attempts(child):
                            add_run_dir(rd)

            if not candidates:
                maybe_master = base / "master_results.json"
                maybe_attempts = base / "attempts.json"
                for rd in _run_dirs_from_master(maybe_master):
                    add_run_dir(rd)
                for rd in _run_dirs_from_attempts(maybe_attempts):
                    add_run_dir(rd)

    if path.is_file():
        if path.name == "manifest.json":
            parse_manifest(path)
        elif path.name in {"master_results.json", "attempts.json"}:
            if path.name == "master_results.json":
                for rd in _run_dirs_from_master(path):
                    add_run_dir(rd)
            else:
                for rd in _run_dirs_from_attempts(path):
                    add_run_dir(rd)
        elif path.name == "history.csv":
            add_run_dir(path.parent)
        else:
            raise ValueError(f"Unsupported input file: {path.name}")
    else:
        if (path / "history.csv").exists():
            add_run_dir(path)
        if (path / "manifest.json").exists():
            parse_manifest(path / "manifest.json")
        if (path / "master_results.json").exists():
            for rd in _run_dirs_from_master(path / "master_results.json"):
                add_run_dir(rd)
        if (path / "attempts.json").exists():
            for rd in _run_dirs_from_attempts(path / "attempts.json"):
                add_run_dir(rd)

        if not candidates:
            for hist in path.rglob("history.csv"):
                add_run_dir(hist.parent)

    unique: list[Path] = []
    seen: set[str] = set()
    for run_dir in candidates:
        key = str(run_dir)
        if key in seen:
            continue
        seen.add(key)
        unique.append(run_dir)

    if not unique:
        raise ValueError(f"No runnable histories found under input: {path}")
    return sorted(unique)


def _extract_context_names(row: dict[str, str], fallback_task: str) -> list[str]:
    keys = list(row.keys())
    mu = [k[len("mu_") :] for k in keys if k.startswith("mu_")]
    center = [k[len("center_") :] for k in keys if k.startswith("center_")]
    if mu:
        return mu
    if center:
        return center
    if fallback_task == "mountaincar" and all(k in row for k in ("force", "gravity")):
        return ["force", "gravity"]
    return []


def _extract_center_for_names(row: dict[str, str], names: list[str], fallback_task: str) -> list[float]:
    values: list[float] = []
    for name in names:
        val = math.nan
        if f"mu_{name}" in row:
            val = to_float(row.get(f"mu_{name}"))
        elif f"center_{name}" in row:
            val = to_float(row.get(f"center_{name}"))
        elif fallback_task == "mountaincar" and name in {"force", "gravity"}:
            val = to_float(row.get(name))
        values.append(val)
    return values


def _extract_spread_names(row: dict[str, str]) -> tuple[str | None, list[str]]:
    keys = list(row.keys())
    conc = [k[len("conc_end_") :] for k in keys if k.startswith("conc_end_")]
    if conc:
        return "concentration", conc
    sigma = [k[len("sigma_end_") :] for k in keys if k.startswith("sigma_end_")]
    if sigma:
        return "sigma", sigma
    return None, []


def _extract_spread_for_names(row: dict[str, str], kind: str | None, names: list[str]) -> list[float]:
    if not kind or not names:
        return []
    prefix = "conc_end_" if kind == "concentration" else "sigma_end_"
    return [to_float(row.get(f"{prefix}{name}")) for name in names]


def _extract_target_map(manifest: dict[str, Any], context_names: list[str]) -> tuple[dict[str, float], bool, str | None]:
    raw = manifest.get("target_params")
    target_hidden = False
    target_hash = None

    public = manifest.get("target_public_metadata")
    if isinstance(public, dict):
        target_hidden = bool(public.get("hidden_target", False))
        if public.get("target_hash") is not None:
            target_hash = str(public.get("target_hash"))

    if manifest.get("target_hash") is not None:
        target_hash = str(manifest.get("target_hash"))

    if not isinstance(raw, dict):
        return {}, target_hidden, target_hash

    out: dict[str, float] = {}
    for name in context_names:
        if name not in raw:
            continue
        val = to_float(raw.get(name))
        if math.isnan(val):
            continue
        out[name] = float(val)
    return out, target_hidden, target_hash


def _extract_bounds_map(manifest: dict[str, Any], context_names: list[str]) -> dict[str, tuple[float, float]]:
    raw = manifest.get("dr_bounds")
    if not isinstance(raw, dict):
        raw = manifest.get("edge_bounds")
    if not isinstance(raw, dict):
        return {}

    out: dict[str, tuple[float, float]] = {}
    for name in context_names:
        pair = raw.get(name)
        if not isinstance(pair, (list, tuple)) or len(pair) < 2:
            continue
        lo = to_float(pair[0])
        hi = to_float(pair[1])
        if not (np.isfinite(lo) and np.isfinite(hi)):
            continue
        if hi < lo:
            lo, hi = hi, lo
        out[name] = (float(lo), float(hi))
    return out


def _compute_episode_x(
    steps: list[int],
    *,
    task: str,
) -> np.ndarray:
    step_arr = np.asarray(steps, dtype=float)
    if step_arr.size == 0:
        return np.zeros(0, dtype=float)
    n_iters = max(1.0, float(np.max(step_arr) - np.min(step_arr)))
    budget = TASK_TOTAL_EPISODES.get(str(task).strip().lower(), float(n_iters))
    per_iter = float(budget) / n_iters
    return (step_arr - float(np.min(step_arr))) * per_iter


def _resolve_metric_name(
    manifest: dict[str, Any],
    first_row: dict[str, str],
) -> str:
    desired = str(manifest.get("primary_metric") or "").strip()
    if desired and desired in first_row:
        return desired
    if "mean_episodic_return" in first_row:
        return "mean_episodic_return"
    return "target_solve_rate"


def _read_history_rows(history_path: Path) -> list[dict[str, str]]:
    with history_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def load_run_series(
    input_path: str | Path,
    *,
    methods: set[str] | None = None,
    task_filter: str | None = None,
    variants: set[str] | None = None,
) -> list[RunSeries]:
    run_dirs = discover_run_directories(input_path)
    rows: list[RunSeries] = []
    task_filter_norm = task_filter.lower() if task_filter and task_filter.lower() != "auto" else None

    for run_dir in run_dirs:
        history_path = run_dir / "history.csv"
        manifest_path = run_dir / "manifest.json"
        history_rows = _read_history_rows(history_path)
        if not history_rows:
            continue
        manifest: dict[str, Any] = {}
        if manifest_path.exists():
            data = _read_json(manifest_path)
            if isinstance(data, dict):
                manifest = data

        first = history_rows[0]
        raw_method = str(first.get("method") or manifest.get("method") or "").strip()
        method = normalize_method_name(raw_method)
        if methods and method not in methods:
            continue

        task = str(manifest.get("task") or "mountaincar").strip().lower()
        if task_filter_norm and task != task_filter_norm:
            continue

        env_id = str(manifest.get("env_id") or "")
        variant = _infer_variant(run_dir, manifest)
        if variants and variant not in variants:
            continue
        budget = TASK_TOTAL_EPISODES.get(task, math.nan)
        seed = to_int(manifest.get("seed"), default=parse_seed_from_run_dir(run_dir))
        metric_name = _resolve_metric_name(manifest, first)

        context_names = _extract_context_names(first, task)
        if not context_names and isinstance(manifest.get("context_names"), list):
            context_names = [str(x) for x in manifest.get("context_names") if str(x).strip()]
        spread_kind, spread_names = _extract_spread_names(first)
        target_map, target_hidden, target_hash = _extract_target_map(manifest, context_names)
        bounds_map = _extract_bounds_map(manifest, context_names)

        step_vals: list[int] = []
        metric_vals: list[float] = []
        g_vals: list[float] = []
        center_vals: list[list[float]] = []
        spread_vals: list[list[float]] = []

        for row in history_rows:
            step_vals.append(to_int(row.get("step"), default=len(step_vals)))
            metric_vals.append(to_float(row.get(metric_name)))
            g_vals.append(to_float(row.get("g_last")))
            center_vals.append(_extract_center_for_names(row, context_names, task))
            spread_vals.append(_extract_spread_for_names(row, spread_kind, spread_names))
        episode_x = _compute_episode_x(step_vals, task=task)

        # Keep DR without synthetic spread visuals; only plot spread when emitted by method artifacts.

        center_arr = np.asarray(center_vals, dtype=float)
        if center_arr.ndim == 1:
            center_arr = center_arr[:, None]
        spread_arr = np.asarray(spread_vals, dtype=float)
        if spread_names:
            if spread_arr.ndim == 1:
                spread_arr = spread_arr[:, None]
        else:
            spread_arr = np.zeros((len(step_vals), 0), dtype=float)

        rows.append(
            RunSeries(
                method=method,
                method_slug=method_slug(method),
                variant=variant,
                task=task,
                env_id=env_id,
                seed=seed,
                run_dir=str(run_dir),
                primary_metric_name=metric_name,
                step=np.asarray(step_vals, dtype=int),
                primary_metric=np.asarray(metric_vals, dtype=float),
                g_last=np.asarray(g_vals, dtype=float),
                center_names=list(context_names),
                center=center_arr,
                spread_kind=spread_kind,
                spread_names=list(spread_names),
                spread=spread_arr,
                episode_x=episode_x,
                budget=float(budget) if np.isfinite(budget) else math.nan,
                target_map=target_map,
                target_hidden=target_hidden,
                target_hash=target_hash,
                bounds_map=bounds_map,
            )
        )

    if not rows:
        raise ValueError("No runs matched filters")
    return rows


def _aggregate_scalar(series_list: list[RunSeries], *, value_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    step_to_vals: dict[int, list[float]] = {}
    for run in series_list:
        values = getattr(run, value_name)
        for step, val in zip(run.step, values, strict=False):
            if math.isnan(float(val)):
                continue
            step_to_vals.setdefault(int(step), []).append(float(val))

    if not step_to_vals:
        return (
            np.zeros(0, dtype=int),
            np.zeros(0, dtype=float),
            np.zeros(0, dtype=float),
            np.zeros(0, dtype=int),
        )

    steps = np.asarray(sorted(step_to_vals.keys()), dtype=int)
    means: list[float] = []
    stds: list[float] = []
    counts: list[int] = []
    for step in steps:
        vals = np.asarray(step_to_vals[int(step)], dtype=float)
        means.append(float(np.nanmean(vals)))
        stds.append(float(np.nanstd(vals)))
        counts.append(int(np.count_nonzero(~np.isnan(vals))))
    return steps, np.asarray(means), np.asarray(stds), np.asarray(counts, dtype=int)


def _aggregate_step_x(series_list: list[RunSeries]) -> tuple[np.ndarray, np.ndarray]:
    step_to_x: dict[int, list[float]] = {}
    for run in series_list:
        for step, x in zip(run.step, run.episode_x, strict=False):
            if not np.isfinite(float(x)):
                continue
            step_to_x.setdefault(int(step), []).append(float(x))
    if not step_to_x:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=float)
    steps = np.asarray(sorted(step_to_x.keys()), dtype=int)
    x_mean = np.asarray([float(np.mean(step_to_x[int(step)])) for step in steps], dtype=float)
    return steps, x_mean


def _aggregate_vector(
    series_list: list[RunSeries],
    *,
    value_name: str,
    dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    step_to_vals: dict[int, list[np.ndarray]] = {}
    for run in series_list:
        values = getattr(run, value_name)
        if values.shape[1] != dim:
            continue
        for i, step in enumerate(run.step):
            row = values[i, :]
            step_to_vals.setdefault(int(step), []).append(row.astype(float))

    if not step_to_vals:
        return (
            np.zeros(0, dtype=int),
            np.zeros((0, dim), dtype=float),
            np.zeros((0, dim), dtype=float),
            np.zeros((0, dim), dtype=int),
        )

    steps = np.asarray(sorted(step_to_vals.keys()), dtype=int)
    mean = np.zeros((len(steps), dim), dtype=float)
    std = np.zeros((len(steps), dim), dtype=float)
    n = np.zeros((len(steps), dim), dtype=int)

    for idx, step in enumerate(steps):
        mat = np.vstack(step_to_vals[int(step)])
        valid = ~np.isnan(mat)
        n[idx, :] = np.sum(valid, axis=0)
        for col in range(dim):
            if n[idx, col] == 0:
                mean[idx, col] = math.nan
                std[idx, col] = math.nan
                continue
            vals = mat[valid[:, col], col]
            mean[idx, col] = float(np.mean(vals))
            std[idx, col] = float(np.std(vals))
    return steps, mean, std, n


def summarize_by_method(series: list[RunSeries]) -> dict[str, MethodSummary]:
    by_method: dict[tuple[str, str, str, str], list[RunSeries]] = {}
    for run in series:
        by_method.setdefault((run.method, run.variant, run.task, run.primary_metric_name), []).append(run)

    summaries: dict[str, MethodSummary] = {}
    for (method, variant, task, metric_name), runs in by_method.items():
        first = runs[0]
        metric_steps, metric_mean, metric_std, metric_n = _aggregate_scalar(runs, value_name="primary_metric")
        x_steps, metric_x = _aggregate_step_x(runs)
        if len(metric_steps) and len(x_steps) and not np.array_equal(metric_steps, x_steps):
            # fallback alignment by step labels
            x_map = {int(s): float(x) for s, x in zip(x_steps, metric_x, strict=False)}
            metric_x = np.asarray([x_map.get(int(s), float(s)) for s in metric_steps], dtype=float)
        elif not len(metric_x):
            metric_x = metric_steps.astype(float)
        g_steps, g_mean, g_std, g_n = _aggregate_scalar(runs, value_name="g_last")
        if first.center_names:
            center_steps, center_mean, center_std, center_n = _aggregate_vector(
                runs,
                value_name="center",
                dim=len(first.center_names),
            )
        else:
            center_steps = np.zeros(0, dtype=int)
            center_mean = np.zeros((0, 0), dtype=float)
            center_std = np.zeros((0, 0), dtype=float)
            center_n = np.zeros((0, 0), dtype=int)

        if first.spread_names:
            spread_steps, spread_mean, spread_std, spread_n = _aggregate_vector(
                runs,
                value_name="spread",
                dim=len(first.spread_names),
            )
        else:
            spread_steps = np.zeros(0, dtype=int)
            spread_mean = np.zeros((0, 0), dtype=float)
            spread_std = np.zeros((0, 0), dtype=float)
            spread_n = np.zeros((0, 0), dtype=int)

        target_center = np.full(len(first.center_names), np.nan, dtype=float)
        for dim, name in enumerate(first.center_names):
            vals: list[float] = []
            for run in runs:
                if name not in run.target_map:
                    continue
                v = to_float(run.target_map.get(name))
                if math.isnan(v):
                    continue
                vals.append(float(v))
            if vals:
                target_center[dim] = float(np.mean(vals))

        target_hidden = any(bool(run.target_hidden) for run in runs)
        target_hashes = sorted({str(run.target_hash) for run in runs if run.target_hash})

        key = f"{method}__{_slugify(task)}__{_slugify(variant)}__{_slugify(metric_name)}"
        summaries[key] = MethodSummary(
            method=method,
            variant=variant,
            task=task,
            env_id=first.env_id,
            metric_name=metric_name,
            metric_steps=metric_steps,
            metric_x=metric_x,
            metric_mean=metric_mean,
            metric_std=metric_std,
            metric_n=metric_n,
            g_steps=g_steps,
            g_mean=g_mean,
            g_std=g_std,
            g_n=g_n,
            center_names=list(first.center_names),
            center_steps=center_steps,
            center_mean=center_mean,
            center_std=center_std,
            center_n=center_n,
            spread_kind=first.spread_kind,
            spread_names=list(first.spread_names),
            spread_steps=spread_steps,
            spread_mean=spread_mean,
            spread_std=spread_std,
            spread_n=spread_n,
            target_center=target_center,
            target_hidden=target_hidden,
            target_hashes=target_hashes,
            seeds=sorted({int(r.seed) for r in runs}),
            run_count=len(runs),
        )

    return summaries


def generate_param_pairs(names: Iterable[str]) -> list[tuple[str, str]]:
    vals = list(names)
    return list(itertools.combinations(vals, 2))


def _align_series_to_steps(steps: np.ndarray, run_steps: np.ndarray, values: np.ndarray) -> np.ndarray:
    out = np.full(len(steps), np.nan, dtype=float)
    index = {int(step): i for i, step in enumerate(run_steps)}
    for i, step in enumerate(steps):
        j = index.get(int(step))
        if j is None:
            continue
        out[i] = float(values[j])
    return out


def _map_steps_to_x(ref_steps: np.ndarray, ref_x: np.ndarray, steps: np.ndarray) -> np.ndarray:
    if steps.size == 0:
        return np.zeros(0, dtype=float)
    if ref_steps.size == 0 or ref_x.size == 0:
        return steps.astype(float)
    step_to_x = {int(s): float(x) for s, x in zip(ref_steps, ref_x, strict=False)}
    out = np.asarray([step_to_x.get(int(s), math.nan) for s in steps], dtype=float)
    if np.any(~np.isfinite(out)):
        interp = np.interp(
            steps.astype(float),
            ref_steps.astype(float),
            ref_x.astype(float),
            left=float(ref_x[0]),
            right=float(ref_x[-1]),
        )
        out = np.where(np.isfinite(out), out, interp)
    return out


def _method_color(method: str) -> str:
    return METHOD_COLORS.get(method, "#1f77b4")


def _pretty_metric_name(metric_name: str) -> str:
    if metric_name == "target_solve_rate":
        return "Target solve-rate"
    if metric_name == "mean_episodic_return":
        return "Mean episodic return"
    return metric_name.replace("_", " ")


def _configure_plot_style() -> None:
    # Paper-like neutral style close to the notebook examples.
    try:
        plt.style.use("seaborn-v0_8-darkgrid")
    except Exception:
        pass
    plt.rcParams.update(
        {
            "figure.facecolor": "#F4F4F6",
            "axes.facecolor": "#EAEAF2",
            "axes.grid": True,
            "grid.alpha": 0.35,
            "grid.linestyle": "-",
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.frameon": True,
            "legend.framealpha": 0.85,
            "font.size": 10,
        }
    )


def _save_figure(fig: plt.Figure, out_base: Path) -> list[str]:
    files: list[str] = []
    for ext in ("png", "pdf"):
        path = out_base.with_suffix(f".{ext}")
        fig.savefig(path, bbox_inches="tight", dpi=180)
        files.append(str(path))
    plt.close(fig)
    return files


def _set_square_limits(ax: plt.Axes, x: np.ndarray, y: np.ndarray, *, pad_ratio: float = 0.06) -> None:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size == 0 or y.size == 0:
        return
    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(y)), float(np.max(y))
    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)
    half = 0.5 * max(x_max - x_min, y_max - y_min)
    if not np.isfinite(half) or half <= 0:
        half = 1.0
    half *= 1.0 + pad_ratio
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_aspect("equal", adjustable="box")


def _set_metric_ylim(
    ax: plt.Axes,
    values: np.ndarray,
    *,
    task: str | None = None,
) -> None:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return
    vmin = float(np.min(vals))
    vmax = float(np.max(vals))
    task_ceiling: float | None = None
    if task:
        task_norm = str(task).strip().lower()
        if task_norm in TASK_MAX_REWARD:
            task_ceiling = float(TASK_MAX_REWARD[task_norm])
    if vmin == vmax:
        pad = max(abs(vmax) * 0.25, 1.0)
        floor = vmin - 0.25 * pad
        ceiling = vmax + pad
        if task_ceiling is not None:
            ceiling = float(task_ceiling)
            if ceiling <= floor:
                floor = ceiling - max(abs(ceiling) * 0.25, 1.0)
        ax.set_ylim(floor, ceiling)
        return
    ceiling = vmax + 0.25 * abs(vmax)
    floor = vmin - 0.05 * (vmax - vmin)
    if ceiling <= floor:
        ceiling = floor + max(abs(floor) * 0.25, 1.0)
    if task_ceiling is not None:
        ceiling = float(task_ceiling)
        if ceiling <= floor:
            floor = ceiling - max(abs(ceiling) * 0.25, 1.0)
    ax.set_ylim(floor, ceiling)


def _summary_target_xy(summary: MethodSummary, x_name: str, y_name: str) -> tuple[float, float] | None:
    if summary.target_center.size == 0:
        return None
    idx = {name: i for i, name in enumerate(summary.center_names)}
    if x_name not in idx or y_name not in idx:
        return None
    x = float(summary.target_center[idx[x_name]])
    y = float(summary.target_center[idx[y_name]])
    if not (np.isfinite(x) and np.isfinite(y)):
        return None
    return x, y


def _method_has_spread_envelope(method: str) -> bool:
    return method in {"BO+DR", "DORAEMON", "BO+DORAEMON"}


def _spread_radius(
    kind: str | None,
    spread_value: float,
    *,
    axis_span: float,
) -> float:
    if not np.isfinite(spread_value):
        return math.nan
    if kind == "sigma":
        return max(float(spread_value), 0.0)
    if kind == "concentration":
        base = max(axis_span, 1e-6) * 0.35
        return base / math.sqrt(max(float(spread_value), 1e-8))
    if kind == "bounds":
        return max(float(spread_value), 0.0)
    return math.nan


def _draw_spread_envelopes(
    ax: plt.Axes,
    summary: MethodSummary,
    *,
    x_name: str,
    y_name: str,
    color: str,
) -> None:
    if not _method_has_spread_envelope(summary.method):
        return
    if summary.spread_mean.size == 0 or not summary.spread_names:
        return
    if summary.center_mean.size == 0:
        return

    spread_idx = {name: i for i, name in enumerate(summary.spread_names)}
    center_idx = {name: i for i, name in enumerate(summary.center_names)}
    if x_name not in spread_idx or y_name not in spread_idx:
        return
    if x_name not in center_idx or y_name not in center_idx:
        return

    ix_s = spread_idx[x_name]
    iy_s = spread_idx[y_name]
    ix_c = center_idx[x_name]
    iy_c = center_idx[y_name]

    x_vals = summary.center_mean[:, ix_c]
    y_vals = summary.center_mean[:, iy_c]
    x_span = float(np.nanmax(x_vals) - np.nanmin(x_vals)) if np.any(np.isfinite(x_vals)) else 1.0
    y_span = float(np.nanmax(y_vals) - np.nanmin(y_vals)) if np.any(np.isfinite(y_vals)) else 1.0

    spread_step_to_row = {int(step): idx for idx, step in enumerate(summary.spread_steps)}
    for i in range(len(summary.center_steps)):
        step = int(summary.center_steps[i]) if i < len(summary.center_steps) else i
        spread_i = spread_step_to_row.get(step)
        if spread_i is None:
            continue
        cx = float(summary.center_mean[i, ix_c])
        cy = float(summary.center_mean[i, iy_c])
        sx = float(summary.spread_mean[spread_i, ix_s]) if spread_i < summary.spread_mean.shape[0] else math.nan
        sy = float(summary.spread_mean[spread_i, iy_s]) if spread_i < summary.spread_mean.shape[0] else math.nan
        if not (np.isfinite(cx) and np.isfinite(cy)):
            continue
        rx = _spread_radius(summary.spread_kind, sx, axis_span=x_span)
        ry = _spread_radius(summary.spread_kind, sy, axis_span=y_span)
        if not (np.isfinite(rx) and np.isfinite(ry)):
            continue
        if rx <= 0 or ry <= 0:
            continue
        ell = Ellipse(
            (cx, cy),
            width=2.0 * rx,
            height=2.0 * ry,
            facecolor=color,
            edgecolor=color,
            alpha=0.08,
            linewidth=0.8,
            zorder=1,
        )
        ax.add_patch(ell)


def plot_average_comparison(
    summaries: dict[str, MethodSummary],
    *,
    out_dir: Path,
) -> list[str]:
    if not summaries:
        return []

    grouped: dict[tuple[str, str, str], list[MethodSummary]] = {}
    for summary in summaries.values():
        grouped.setdefault((summary.task, summary.metric_name, summary.variant), []).append(summary)

    outputs: list[str] = []
    for (task, metric_name, variant), items in grouped.items():
        by_method = {item.method: item for item in items}
        all_x = sorted({float(x) for item in items for x in item.metric_x if np.isfinite(x)})
        if not all_x:
            continue
        step_axis = np.asarray(all_x, dtype=float)

        fig, ax = plt.subplots(figsize=(9.6, 5.4))
        ax.set_facecolor("#EAEAF2")
        plotted_vals: list[np.ndarray] = []
        missing: list[str] = []
        for method in METHOD_ORDER:
            item = by_method.get(method)
            if item is None:
                missing.append(method)
                continue
            color = _method_color(method)
            x = item.metric_x
            y = item.metric_mean
            sd = item.metric_std
            plotted_vals.append(y)
            ax.plot(x, y, label=method, color=color, linewidth=2.2, marker="o", markersize=3)
            ax.fill_between(x, y - sd, y + sd, color=color, alpha=0.12)

        ax.set_title(f"Average performance comparison ({task}, {variant})")
        ax.set_xlabel("approx training episodes")
        ax.set_ylabel(_pretty_metric_name(metric_name))
        ax.set_xlim(float(np.min(step_axis)), float(np.max(step_axis)))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
        if plotted_vals:
            _set_metric_ylim(ax, np.concatenate(plotted_vals), task=task)
        ax.grid(alpha=0.25)
        ax.legend(loc="best")
        if missing:
            ax.text(0.01, 0.02, f"missing: {', '.join(missing)}", transform=ax.transAxes, fontsize=8, alpha=0.8)
        outputs.extend(_save_figure(fig, out_dir / f"compare_average_{task}_{_slugify(variant)}_{metric_name}"))

    return outputs


def plot_method_summary(
    method: str,
    runs: list[RunSeries],
    summary: MethodSummary,
    *,
    out_dir: Path,
    alpha_target: float,
) -> list[str]:
    color = _method_color(method)
    show_center_panel = summary.task == "mountaincar" and len(summary.center_names) == 2 and summary.center_mean.size > 0
    ncols = 4 if show_center_panel else 3
    fig, axes = plt.subplots(1, ncols, figsize=(24 if show_center_panel else 18, 4.8))
    if isinstance(axes, np.ndarray):
        axes_list = list(axes)
    else:
        axes_list = [axes]
    for ax in axes_list:
        ax.set_facecolor("#EAEAF2")
    ax_metric = axes_list[0]
    ax_g = axes_list[1]
    ax_spread = axes_list[2]
    ax_center = axes_list[3] if show_center_panel else None

    x_metric = summary.metric_x
    x_g = _map_steps_to_x(summary.metric_steps, summary.metric_x, summary.g_steps)
    x_spread = _map_steps_to_x(summary.metric_steps, summary.metric_x, summary.spread_steps)
    x_center = _map_steps_to_x(summary.metric_steps, summary.metric_x, summary.center_steps)
    x_pool: list[np.ndarray] = [x_metric, x_g, x_spread, x_center]
    non_empty_x = [arr for arr in x_pool if arr.size > 0]
    global_x_min = float(min(np.nanmin(arr) for arr in non_empty_x)) if non_empty_x else 0.0
    global_x_max = float(max(np.nanmax(arr) for arr in non_empty_x)) if non_empty_x else 1.0

    steps = summary.metric_steps
    for run in runs:
        aligned = _align_series_to_steps(steps, run.step, run.primary_metric)
        ax_metric.plot(x_metric, aligned, color=color, alpha=0.22, linewidth=1, marker="o", markersize=2)
    if len(summary.metric_mean):
        ax_metric.plot(x_metric, summary.metric_mean, color=color, linewidth=2.2, marker="o", markersize=4, label=f"{method} mean")
        ax_metric.fill_between(
            x_metric,
            summary.metric_mean - summary.metric_std,
            summary.metric_mean + summary.metric_std,
            color=color,
            alpha=0.18,
            label="mean ± 1 SD",
        )
    ax_metric.set_title(f"{_pretty_metric_name(summary.metric_name)} vs episodes")
    ax_metric.set_xlabel("episodes")
    ax_metric.set_ylabel(_pretty_metric_name(summary.metric_name))
    ax_metric.set_xlim(global_x_min, global_x_max)
    ax_metric.xaxis.set_major_locator(MaxNLocator(nbins=8))
    _set_metric_ylim(ax_metric, summary.metric_mean, task=summary.task)
    ax_metric.grid(alpha=0.25)
    ax_metric.legend(loc="best")

    has_g = any(np.any(np.isfinite(run.g_last)) for run in runs)
    if has_g and len(summary.g_mean):
        g_steps = summary.g_steps
        for run in runs:
            aligned = _align_series_to_steps(g_steps, run.step, run.g_last)
            ax_g.plot(x_g, aligned, color=color, alpha=0.2, linewidth=1, marker="o", markersize=2)
        ax_g.plot(x_g, summary.g_mean, color=color, linewidth=2.2, marker="o", markersize=4, label="g (IS estimate)")
        ax_g.fill_between(
            x_g,
            summary.g_mean - summary.g_std,
            summary.g_mean + summary.g_std,
            color=color,
            alpha=0.18,
            label="g ± 1 SD",
        )
        ax_g.axhline(alpha_target, linestyle="--", color="#d55e00", linewidth=1.6, label=f"alpha-target ({alpha_target:g})")
        ax_g.legend(loc="best")
    else:
        ax_g.text(0.5, 0.5, "g_last unavailable", ha="center", va="center", transform=ax_g.transAxes)
    ax_g.set_title("DORAEMON Success Constraint")
    ax_g.set_xlabel("episodes")
    ax_g.set_ylabel("g estimate")
    ax_g.set_xlim(global_x_min, global_x_max)
    ax_g.xaxis.set_major_locator(MaxNLocator(nbins=8))
    ax_g.grid(alpha=0.25)

    if summary.spread_names and summary.spread_mean.size:
        spread_steps = summary.spread_steps
        for i, name in enumerate(summary.spread_names):
            y = summary.spread_mean[:, i]
            sd = summary.spread_std[:, i]
            c = plt.cm.tab10(i % 10)
            ax_spread.plot(x_spread, y, color=c, linewidth=1.8, marker="o", markersize=3, label=name)
            ax_spread.fill_between(x_spread, y - sd, y + sd, color=c, alpha=0.14)
        legend_cols = 2 if len(summary.spread_names) > 4 else 1
        ax_spread.legend(loc="best", ncol=legend_cols, fontsize=8)
        ylabel = "Beta concentration" if summary.spread_kind == "concentration" else ("Spread radius (bounds)" if summary.spread_kind == "bounds" else "Sigma")
        ax_spread.set_ylabel(ylabel)
        if summary.spread_kind == "concentration":
            title = "Beta concentration"
        elif summary.spread_kind == "bounds":
            title = "Bounds-derived spread"
        else:
            title = "Sigma spread"
    else:
        ax_spread.text(0.5, 0.5, "spread unavailable", ha="center", va="center", transform=ax_spread.transAxes)
        ax_spread.set_ylabel("spread")
        title = "Spread"
    ax_spread.set_title(title)
    ax_spread.set_xlabel("episodes")
    ax_spread.set_xlim(global_x_min, global_x_max)
    ax_spread.xaxis.set_major_locator(MaxNLocator(nbins=8))
    ax_spread.tick_params(axis="x", labelsize=8)
    if summary.spread_mean.size:
        _set_metric_ylim(ax_spread, summary.spread_mean.ravel())
    ax_spread.grid(alpha=0.25)

    if ax_center is not None:
        x_name, y_name = summary.center_names
        for run in runs:
            if run.center.shape[1] != 2:
                continue
            ax_center.plot(run.center[:, 0], run.center[:, 1], color=color, alpha=0.22, linewidth=1)
        ax_center.plot(
            summary.center_mean[:, 0],
            summary.center_mean[:, 1],
            color=color,
            linewidth=2.3,
            alpha=0.9,
        )
        _draw_spread_envelopes(ax_center, summary, x_name=x_name, y_name=y_name, color=color)
        target_xy = _summary_target_xy(summary, x_name, y_name)
        if target_xy is not None:
            tx, ty = target_xy
            ax_center.scatter([tx], [ty], marker="*", s=140, color="#111111", edgecolors="white", linewidths=0.8, zorder=5, label="target")
        scatter = ax_center.scatter(
            summary.center_mean[:, 0],
            summary.center_mean[:, 1],
            c=x_center,
            cmap="viridis",
            s=36,
            zorder=3,
        )
        cbar = fig.colorbar(scatter, ax=ax_center, fraction=0.048, pad=0.03)
        cbar.set_label("episodes")
        ax_center.set_title(f"Centre trajectory in ({x_name}, {y_name})")
        ax_center.set_xlabel(f"{x_name} centre")
        ax_center.set_ylabel(f"{y_name} centre")
        all_x = summary.center_mean[:, 0]
        all_y = summary.center_mean[:, 1]
        if target_xy is not None:
            all_x = np.append(all_x, target_xy[0])
            all_y = np.append(all_y, target_xy[1])
        _set_square_limits(ax_center, all_x, all_y)
        ax_center.grid(alpha=0.25)
        if summary.target_hidden and target_xy is None:
            tag = summary.target_hashes[0][:8] if summary.target_hashes else "redacted"
            ax_center.text(0.02, 0.02, f"target hidden ({tag})", transform=ax_center.transAxes, fontsize=9, alpha=0.8)
        if target_xy is not None:
            ax_center.legend(loc="best")

    fig.suptitle(f"{method} [{summary.variant}] ({summary.task}, n={summary.run_count} runs)", fontsize=13)
    return _save_figure(fig, out_dir / f"{method_slug(method)}_summary")


def _plot_center_mountaincar(
    method: str,
    runs: list[RunSeries],
    summary: MethodSummary,
    *,
    out_dir: Path,
) -> list[str]:
    if len(summary.center_names) != 2 or summary.center_mean.size == 0:
        return []

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.set_facecolor("#EAEAF2")
    color = _method_color(method)
    for run in runs:
        if run.center.shape[1] != 2:
            continue
        ax.plot(run.center[:, 0], run.center[:, 1], color=color, alpha=0.25, linewidth=1)
    ax.plot(summary.center_mean[:, 0], summary.center_mean[:, 1], color=color, linewidth=2.5, label="mean trajectory")
    _draw_spread_envelopes(ax, summary, x_name=summary.center_names[0], y_name=summary.center_names[1], color=color)
    target_xy = _summary_target_xy(summary, summary.center_names[0], summary.center_names[1])
    if target_xy is not None:
        tx, ty = target_xy
        ax.scatter([tx], [ty], marker="*", s=140, color="#111111", edgecolors="white", linewidths=0.8, zorder=6, label="target")
    center_x = _map_steps_to_x(summary.metric_steps, summary.metric_x, summary.center_steps)
    scatter = ax.scatter(summary.center_mean[:, 0], summary.center_mean[:, 1], c=center_x, cmap="viridis", s=32)
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.048, pad=0.03)
    cbar.set_label("episodes")
    ax.set_title(f"{method} centre trajectory")
    ax.set_xlabel(summary.center_names[0])
    ax.set_ylabel(summary.center_names[1])
    all_x = summary.center_mean[:, 0]
    all_y = summary.center_mean[:, 1]
    if target_xy is not None:
        all_x = np.append(all_x, target_xy[0])
        all_y = np.append(all_y, target_xy[1])
    _set_square_limits(ax, all_x, all_y)
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    if summary.target_hidden and target_xy is None:
        tag = summary.target_hashes[0][:8] if summary.target_hashes else "redacted"
        ax.text(0.02, 0.02, f"target hidden ({tag})", transform=ax.transAxes, fontsize=9, alpha=0.8)
    return _save_figure(fig, out_dir / f"{method_slug(method)}_center_2d")


def _plot_center_halfcheetah_pairs(
    method: str,
    runs: list[RunSeries],
    summary: MethodSummary,
    *,
    out_dir: Path,
    per_page: int = 6,
) -> list[str]:
    names = summary.center_names
    if len(names) < 2 or summary.center_mean.size == 0:
        return []

    name_to_idx = {name: i for i, name in enumerate(names)}
    pairs = generate_param_pairs(names)
    outputs: list[str] = []

    pages = [pairs[i : i + per_page] for i in range(0, len(pairs), per_page)]
    for page_idx, page in enumerate(pages, start=1):
        n = len(page)
        ncols = 3
        nrows = int(math.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.2 * nrows))
        if isinstance(axes, np.ndarray):
            axes_flat = list(axes.reshape(-1))
        else:
            axes_flat = [axes]

        color = _method_color(method)
        for ax_idx, (a, b) in enumerate(page):
            ax = axes_flat[ax_idx]
            ax.set_facecolor("#EAEAF2")
            ia = name_to_idx[a]
            ib = name_to_idx[b]

            for run in runs:
                if run.center.shape[1] <= max(ia, ib):
                    continue
                ax.plot(run.center[:, ia], run.center[:, ib], color=color, alpha=0.2, linewidth=0.9)
            ax.plot(
                summary.center_mean[:, ia],
                summary.center_mean[:, ib],
                color=color,
                linewidth=2.0,
                label="mean",
            )
            _draw_spread_envelopes(ax, summary, x_name=a, y_name=b, color=color)
            target_xy = _summary_target_xy(summary, a, b)
            if target_xy is not None:
                tx, ty = target_xy
                ax.scatter([tx], [ty], marker="*", s=90, color="#111111", edgecolors="white", linewidths=0.7, zorder=6, label="target")
            ax.scatter(
                summary.center_mean[:, ia],
                summary.center_mean[:, ib],
                c=_map_steps_to_x(summary.metric_steps, summary.metric_x, summary.center_steps),
                cmap="viridis",
                s=16,
            )
            ax.set_title(f"{a} vs {b}")
            ax.set_xlabel(a)
            ax.set_ylabel(b)
            all_x = summary.center_mean[:, ia]
            all_y = summary.center_mean[:, ib]
            if target_xy is not None:
                all_x = np.append(all_x, target_xy[0])
                all_y = np.append(all_y, target_xy[1])
            _set_square_limits(ax, all_x, all_y)
            ax.grid(alpha=0.2)

        for extra in axes_flat[n:]:
            extra.axis("off")

        hidden_note = ""
        if summary.target_hidden and not np.any(np.isfinite(summary.target_center)):
            tag = summary.target_hashes[0][:8] if summary.target_hashes else "redacted"
            hidden_note = f" | target hidden ({tag})"
        fig.suptitle(f"{method} centre trajectory pairs (page {page_idx}/{len(pages)}){hidden_note}")
        outputs.extend(_save_figure(fig, out_dir / f"{method_slug(method)}_center_pairs_page_{page_idx:02d}"))

    return outputs


def plot_center_trajectories(
    method: str,
    runs: list[RunSeries],
    summary: MethodSummary,
    *,
    out_dir: Path,
) -> list[str]:
    if summary.task == "halfcheetah":
        return _plot_center_halfcheetah_pairs(method, runs, summary, out_dir=out_dir)
    return _plot_center_mountaincar(method, runs, summary, out_dir=out_dir)


def generate_plots(
    *,
    input_path: str | Path,
    output_dir: str | Path,
    methods: list[str] | None = None,
    variants: list[str] | None = None,
    task: str = "auto",
    alpha_target: float = 0.5,
) -> dict[str, Any]:
    _configure_plot_style()
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    method_filter: set[str] | None = None
    if methods:
        method_filter = {normalize_method_name(m) for m in methods}
    variant_filter: set[str] | None = None
    if variants:
        variant_filter = {str(v).strip() for v in variants if str(v).strip()}

    series = load_run_series(
        input_path,
        methods=method_filter,
        task_filter=task,
        variants=variant_filter,
    )
    by_method: dict[str, list[RunSeries]] = {}
    for run in series:
        key = f"{run.method}__{_slugify(run.task)}__{_slugify(run.variant)}__{_slugify(run.primary_metric_name)}"
        by_method.setdefault(key, []).append(run)

    summaries = summarize_by_method(series)

    generated_files: list[str] = []
    method_rows: list[dict[str, Any]] = []
    ordered_keys = sorted(
        by_method.keys(),
        key=lambda k: (METHOD_ORDER.index(k.split("__", 1)[0]) if k.split("__", 1)[0] in METHOD_ORDER else 999, k),
    )
    for key in ordered_keys:
        runs = by_method[key]
        summary = summaries[key]
        files = []
        variant_dir = out_dir / _slugify(summary.variant)
        variant_dir.mkdir(parents=True, exist_ok=True)
        files.extend(plot_method_summary(summary.method, runs, summary, out_dir=variant_dir, alpha_target=alpha_target))
        files.extend(plot_center_trajectories(summary.method, runs, summary, out_dir=variant_dir))
        generated_files.extend(files)
        method_rows.append(
            {
                "method": summary.method,
                "variant": summary.variant,
                "task": summary.task,
                "env_id": summary.env_id,
                "metric_name": summary.metric_name,
                "seed_count": int(len(summary.seeds)),
                "run_count": int(summary.run_count),
                "center_dims": int(len(summary.center_names)),
                "spread_dims": int(len(summary.spread_names)),
                "target_hidden": bool(summary.target_hidden),
                "generated_files": files,
            }
        )

    comparison_files = plot_average_comparison(summaries, out_dir=out_dir)
    generated_files.extend(comparison_files)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_path": str(Path(input_path).expanduser().resolve()),
        "output_dir": str(out_dir),
        "variant_filter": sorted(variant_filter) if variant_filter else None,
        "methods": sorted({s.method for s in summaries.values()}),
        "variants": sorted({s.variant for s in summaries.values()}),
        "run_count": int(len(series)),
        "generated_file_count": int(len(generated_files)),
        "comparison_files": comparison_files,
        "method_summaries": method_rows,
        "generated_files": generated_files,
    }
    with (out_dir / "plot_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reconstruct paper-style plots from MountainCar/HalfCheetah run artifacts")
    parser.add_argument("--input", required=True, help="Input path: manifest.json, run dir, suite dir, or artifact root")
    parser.add_argument("--output", required=True, help="Directory where plots and plot_manifest.json are written")
    parser.add_argument(
        "--methods",
        default=None,
        help="Comma-separated method filter (e.g. BO,DR,BO+DR,DORAEMON,BO+DORAEMON)",
    )
    parser.add_argument(
        "--variant",
        default=None,
        help="Comma-separated variant filter (e.g. v1,v2,v3,profile_edge)",
    )
    parser.add_argument(
        "--task",
        default="auto",
        choices=["auto", "mountaincar", "halfcheetah"],
        help="Optional task filter",
    )
    parser.add_argument(
        "--alpha-target",
        type=float,
        default=0.5,
        help="Reference line for g plot (default: 0.5)",
    )
    return parser.parse_args(argv)


def _parse_csv_arg(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    items = [x.strip() for x in raw.split(",") if x.strip()]
    return items or None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    methods = _parse_csv_arg(args.methods)
    variants = _parse_csv_arg(args.variant)
    manifest = generate_plots(
        input_path=args.input,
        output_dir=args.output,
        methods=methods,
        variants=variants,
        task=args.task,
        alpha_target=float(args.alpha_target),
    )
    print(json.dumps({"generated_file_count": manifest["generated_file_count"], "output_dir": manifest["output_dir"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
