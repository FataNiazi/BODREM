from __future__ import annotations

import importlib.util
import inspect
import json
import math
import random
import contextlib
import io
import sys
import time
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


PROJECT_DIR = Path(__file__).resolve().parent
RUNS_ROOT = PROJECT_DIR / "runs"
RUNS_ROOT.mkdir(parents=True, exist_ok=True)

RUNTIME_DIR = PROJECT_DIR / "method_runtimes"

METHOD_ORDER = ["BO", "DR", "BO+DR", "DORAEMON", "BO+DORAEMON"]
EPISODE_STEPS = 400

# Centralized shared physics target and domain bounds.
TARGET_PARAMS = {
    "force": 0.0010,
    "gravity": 0.0050,
}

DR_BOUNDS = {
    "force": (0.0003, 0.0030),
    "gravity": (0.0010, 0.0062),
}

EDGE_BOUNDS = {
    "force": (0.0003, 0.003),
    "gravity": (0.001, 0.006),
}

MU0 = (0.0010, 0.0025)
DEFAULT_WARMSTART_CHECKPOINT = PROJECT_DIR / "mountain_car_saved" / "mountain_car_dqn.pt"

PROFILE_PRESETS: dict[str, dict[str, dict[str, Any]]] = {
    "BO": {
        "smoke": dict(bo_iters=3, bo_train_episodes_per_iter=20, target_eval_episodes=5),
        "prototype": dict(bo_iters=12, bo_train_episodes_per_iter=120, target_eval_episodes=12),
        "edge": dict(bo_iters=20, bo_train_episodes_per_iter=180, target_eval_episodes=20),
    },
    "DR": {
        "smoke": dict(dr_total_episodes=60, dr_eval_every=20, target_eval_episodes=5),
        "prototype": dict(dr_total_episodes=1440, dr_eval_every=120, target_eval_episodes=12),
        "edge": dict(dr_total_episodes=3600, dr_eval_every=180, target_eval_episodes=20),
    },
    "BO+DR": {
        "smoke": dict(T=3, K=20, B_r=5, B_r_extra=0, surrogate_eval_episodes=6),
        "prototype": dict(T=12, K=120, B_r=12, B_r_extra=0, surrogate_eval_episodes=10),
        "edge": dict(T=20, K=180, B_r=20, B_r_extra=0, surrogate_eval_episodes=16),
    },
    "DORAEMON": {
        "smoke": dict(adapt_iters=3, blocks=5, episodes_per_block=4, B_r=5),
        "prototype": dict(adapt_iters=12, blocks=20, episodes_per_block=6, B_r=12),
        "edge": dict(adapt_iters=20, blocks=30, episodes_per_block=6, B_r=20),
    },
    "BO+DORAEMON": {
        "smoke": dict(T=3, blocks=5, episodes_per_block=4, B_r=5),
        "prototype": dict(T=12, blocks=20, episodes_per_block=6, B_r=12),
        "edge": dict(T=20, blocks=30, episodes_per_block=6, B_r=20),
    },
}

METHOD_SOURCE_KEY = {
    "BO": "BO_VS_DR",
    "DR": "BO_VS_DR",
    "BO+DR": "BO_DR",
    "DORAEMON": "DORAEMON_ONLY",
    "BO+DORAEMON": "BO_DORAEMON_PAPER",
}

RUNTIME_FILE_BY_SOURCE = {
    "BO_VS_DR": RUNTIME_DIR / "bo_vs_dr_runtime.py",
    "BO_DR": RUNTIME_DIR / "bo_dr_runtime.py",
    "DORAEMON_ONLY": RUNTIME_DIR / "doraemon_only_runtime.py",
    "BO_DORAEMON_PAPER": RUNTIME_DIR / "bo_doraemon_paper_runtime.py",
}


def _normalize_profile(profile: str) -> str:
    p = str(profile).strip().lower()
    if p not in {"smoke", "prototype", "edge"}:
        raise ValueError(f"Unknown profile '{profile}'. Expected one of smoke|prototype|edge")
    return p


def _resolve_checkpoint_path(checkpoint_path: str | Path | None) -> Path | None:
    if checkpoint_path is None:
        return None
    cp = str(checkpoint_path).strip()
    if not cp:
        return None
    p = Path(cp)
    if not p.is_absolute():
        p = (PROJECT_DIR / p).resolve()
    return p


def _default_device() -> str:
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _set_global_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)


def _copy_bounds(bounds: dict[str, tuple[float, float]]) -> dict[str, tuple[float, float]]:
    return {
        "force": (float(bounds["force"][0]), float(bounds["force"][1])),
        "gravity": (float(bounds["gravity"][0]), float(bounds["gravity"][1])),
    }


def _clip_mu_to_bounds(mu: np.ndarray, bounds: dict[str, tuple[float, float]]) -> np.ndarray:
    return np.array(
        [
            float(np.clip(mu[0], bounds["force"][0], bounds["force"][1])),
            float(np.clip(mu[1], bounds["gravity"][0], bounds["gravity"][1])),
        ],
        dtype=np.float32,
    )


def _load_runtime_namespace(source_key: str) -> dict[str, Any]:
    runtime_path = RUNTIME_FILE_BY_SOURCE.get(source_key)
    if runtime_path is None:
        raise KeyError(f"Unknown runtime source key: {source_key}")
    if not runtime_path.exists():
        raise FileNotFoundError(f"Missing runtime source file: {runtime_path}")

    module_name = f"tinysim_runtime_{source_key}_{int(time.time() * 1_000_000)}"
    spec = importlib.util.spec_from_file_location(module_name, runtime_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not create module spec for runtime file: {runtime_path}")

    runtime_mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = runtime_mod

    try:
        # Runtime modules are extracted from notebooks and may contain
        # top-level print statements from config cells. Suppress those
        # import-time prints to keep benchmark logs clean and avoid
        # confusion about centralized overrides.
        with contextlib.redirect_stdout(io.StringIO()):
            spec.loader.exec_module(runtime_mod)
    except ModuleNotFoundError as exc:
        missing = getattr(exc, "name", "unknown")
        raise RuntimeError(
            f"Missing dependency '{missing}' while loading runtime '{runtime_path.name}'. "
            f"Install required packages before running benchmarks."
        ) from exc

    return runtime_mod.__dict__


def _apply_common_overrides(ns: dict[str, Any]) -> None:
    ns["PROJECT_DIR"] = PROJECT_DIR
    ns["RUNS_ROOT"] = RUNS_ROOT
    ns["WARMSTART_CHECKPOINT"] = DEFAULT_WARMSTART_CHECKPOINT
    ns["TARGET_PARAMS"] = dict(TARGET_PARAMS)


def _apply_method_overrides(method: str, ns: dict[str, Any]) -> None:
    mu0 = np.array([float(MU0[0]), float(MU0[1])], dtype=np.float32)
    edge_bounds = _copy_bounds(EDGE_BOUNDS)
    dr_bounds = _copy_bounds(DR_BOUNDS)

    if method in {"BO", "DR"}:
        ns["DR_BOUNDS"] = dr_bounds
        ns["EDGE_BOUNDS"] = edge_bounds
        ns["TRAIN_BOUNDS"] = edge_bounds
        ns["BO_BOUNDS"] = edge_bounds
        ns["MU0"] = mu0
        ns["MU0_EDGE"] = _clip_mu_to_bounds(mu0, edge_bounds)
        ns["TAU_STEPS"] = int(EPISODE_STEPS)

    elif method == "BO+DR":
        ns["DR_BOUNDS"] = dr_bounds
        ns["EDGE_DR_BOUNDS"] = edge_bounds
        ns["MU0"] = mu0
        ns["MU0_EDGE"] = _clip_mu_to_bounds(mu0, edge_bounds)
        ns["tau_steps"] = int(EPISODE_STEPS)

    elif method == "DORAEMON":
        ns["DR_BOUNDS"] = dr_bounds
        ns["EDGE_DR_BOUNDS"] = edge_bounds
        ns["MU0"] = mu0
        ns["MU0_EDGE"] = _clip_mu_to_bounds(mu0, edge_bounds)
        ns["ADAPT_CENTER"] = ns["MU0_EDGE"].copy()
        ns["TAU_STEPS"] = int(EPISODE_STEPS)

    elif method == "BO+DORAEMON":
        ns["DR_BOUNDS"] = dr_bounds
        ns["EDGE_DR_BOUNDS"] = edge_bounds
        ns["MU0"] = mu0
        ns["MU0_EDGE"] = _clip_mu_to_bounds(mu0, edge_bounds)
        ns["TAU_STEPS"] = int(EPISODE_STEPS)


def init_agent(
    checkpoint_path: str | Path | None,
    seed: int,
    device: str,
    *,
    make_agent_from_hparams: Callable[..., Any],
    load_warmstart_agent: Callable[..., Any],
) -> tuple[Any, bool]:
    """Initialize DQN agent with optional warm-start checkpoint."""
    _set_global_seed(seed)
    resolved = _resolve_checkpoint_path(checkpoint_path)
    if resolved is not None and resolved.exists():
        return load_warmstart_agent(checkpoint_path=resolved, device=device), True
    return make_agent_from_hparams(None, device=device), False


def _patch_loader_for_optional_warmstart(
    ns: dict[str, Any],
    checkpoint_path: str | Path | None,
    seed: int,
    device: str,
) -> bool:
    make_agent_from_hparams = ns["make_agent_from_hparams"]
    original_loader = ns["load_warmstart_agent"]

    def _patched_loader(
        checkpoint_path: str | Path | None = None,
        device: str = device,
        **_: Any,
    ) -> Any:
        agent, _ = init_agent(
            checkpoint_path=checkpoint_path,
            seed=seed,
            device=device,
            make_agent_from_hparams=make_agent_from_hparams,
            load_warmstart_agent=original_loader,
        )
        return agent

    ns["load_warmstart_agent"] = _patched_loader

    resolved = _resolve_checkpoint_path(checkpoint_path)
    return bool(resolved is not None and resolved.exists())


def _call_with_supported_kwargs(fn: Callable[..., Any], kwargs: dict[str, Any]) -> Any:
    sig = inspect.signature(fn)
    accepted = set(sig.parameters.keys())
    filtered = {k: v for k, v in kwargs.items() if k in accepted}
    return fn(**filtered)


def _safe_float(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return math.nan
    if math.isinf(v):
        return math.nan
    return v


def _last_history_value(history: list[dict[str, Any]], key: str) -> float:
    if not history:
        return math.nan
    return _safe_float(history[-1].get(key))


def _extract_scores(method: str, result: dict[str, Any]) -> tuple[float, float]:
    if method == "DORAEMON":
        best_target = _safe_float(result.get("best_target_score"))
        final_target = _safe_float(result.get("posthoc_target_rate"))
        if math.isnan(best_target):
            best_target = final_target
        return best_target, final_target

    best_target = _safe_float(result.get("best_score"))
    final_target = _last_history_value(result.get("history", []), "target_solve_rate")
    return best_target, final_target


def _prepare_runtime(
    method: str,
    seed: int,
    checkpoint_path: str | Path | None,
    device: str | None,
) -> tuple[dict[str, Any], bool]:
    source_key = METHOD_SOURCE_KEY[method]
    ns = _load_runtime_namespace(source_key)

    _apply_common_overrides(ns)
    _apply_method_overrides(method, ns)

    selected_device = (device or ns.get("DEVICE") or _default_device()).strip().lower()
    ns["DEVICE"] = selected_device
    _set_global_seed(seed)

    used_warmstart = _patch_loader_for_optional_warmstart(
        ns=ns,
        checkpoint_path=checkpoint_path,
        seed=seed,
        device=selected_device,
    )
    return ns, used_warmstart


def _run_bo(
    seed: int,
    profile: str,
    checkpoint_path: str | Path | None,
    device: str | None,
) -> tuple[dict[str, Any], bool]:
    ns, used_warmstart = _prepare_runtime("BO", seed, checkpoint_path, device)
    cfg = PROFILE_PRESETS["BO"][_normalize_profile(profile)]

    result = ns["run_pure_bo"](
        checkpoint_path=checkpoint_path,
        mu0=ns["MU0_EDGE"],
        bounds=ns["BO_BOUNDS"],
        bo_iters=cfg["bo_iters"],
        train_episodes_per_iter=cfg["bo_train_episodes_per_iter"],
        target_eval_episodes=cfg["target_eval_episodes"],
        tau_steps=int(EPISODE_STEPS),
        target_params=ns["TARGET_PARAMS"],
        seed=seed,
    )
    return result, used_warmstart


def _run_dr(
    seed: int,
    profile: str,
    checkpoint_path: str | Path | None,
    device: str | None,
) -> tuple[dict[str, Any], bool]:
    ns, used_warmstart = _prepare_runtime("DR", seed, checkpoint_path, device)
    cfg = PROFILE_PRESETS["DR"][_normalize_profile(profile)]

    result = ns["run_pure_dr"](
        checkpoint_path=checkpoint_path,
        total_episodes=cfg["dr_total_episodes"],
        eval_every=cfg["dr_eval_every"],
        target_eval_episodes=cfg["target_eval_episodes"],
        tau_steps=int(EPISODE_STEPS),
        bounds=ns["TRAIN_BOUNDS"],
        target_params=ns["TARGET_PARAMS"],
        seed=seed,
    )
    return result, used_warmstart


def _run_bo_dr(
    seed: int,
    profile: str,
    checkpoint_path: str | Path | None,
    device: str | None,
) -> tuple[dict[str, Any], bool]:
    ns, used_warmstart = _prepare_runtime("BO+DR", seed, checkpoint_path, device)
    cfg = PROFILE_PRESETS["BO+DR"][_normalize_profile(profile)]
    tau_steps = int(EPISODE_STEPS)

    bounds = ns["DR_BOUNDS"] if _normalize_profile(profile) in {"smoke", "prototype"} else ns["EDGE_DR_BOUNDS"]

    result = ns["run_bo_doraemon"](
        checkpoint_path=checkpoint_path,
        mu0=ns["MU0"],
        sigma0=ns["SIGMA0"],
        T=cfg["T"],
        K=cfg["K"],
        B_r=cfg["B_r"],
        B_r_extra=cfg["B_r_extra"],
        bounds=bounds,
        tau_steps=tau_steps,
        alpha_low=ns["alpha_low"],
        alpha_high=ns["alpha_high"],
        sigma_min=ns["SIGMA_MIN"],
        sigma_max=ns["SIGMA_MAX"],
        surrogate_eval_episodes=cfg["surrogate_eval_episodes"],
        sigma_expand_factor=ns["sigma_expand"],
        sigma_shrink_factor=ns["sigma_shrink"],
        sigma_update_interval=int(ns["sigma_update_interval"]),
        surrogate_mode=ns["surrogate_mode"],
        bo_epsilon_reset=float(ns["bo_epsilon_reset"]),
        contender_margin=float(ns["contender_margin"]),
        early_stop_patience=int(ns["early_stop_patience"]),
        target_params=ns["TARGET_PARAMS"],
        seed=seed,
    )
    return result, used_warmstart


def _run_doraemon_only(
    seed: int,
    profile: str,
    checkpoint_path: str | Path | None,
    device: str | None,
) -> tuple[dict[str, Any], bool]:
    ns, used_warmstart = _prepare_runtime("DORAEMON", seed, checkpoint_path, device)
    cfg = PROFILE_PRESETS["DORAEMON"][_normalize_profile(profile)]

    kwargs = {
        "checkpoint_path": checkpoint_path,
        "center": ns["ADAPT_CENTER"],
        "conc0": ns["C_INIT"],
        "adapt_iters": cfg["adapt_iters"],
        "blocks": cfg["blocks"],
        "episodes_per_block": cfg["episodes_per_block"],
        "B_r": cfg["B_r"],
        "bounds": ns["EDGE_DR_BOUNDS"],
        "tau_steps": int(EPISODE_STEPS),
        "alpha_target": float(ns["ALPHA_TARGET"]),
        "kl_max": float(ns["KL_MAX"]),
        "n_candidates": int(ns["CANDIDATES_PER_UPDATE"]),
        "mean_std": float(ns["MEAN_PERTURB_STD"]),
        "logc_std": float(ns["LOGC_PERTURB_STD"]),
        "c_min": ns["C_MIN"],
        "c_max": ns["C_MAX"],
        "min_log_w": float(ns["MIN_WEIGHT_LOG"]),
        "max_log_w": float(ns["MAX_WEIGHT_LOG"]),
        "target_params": ns["TARGET_PARAMS"],
        "seed": seed,
        "eval_target_during_train": True,
        "select_best_by_target": False,
    }
    result = _call_with_supported_kwargs(ns["run_doraemon_only"], kwargs)
    return result, used_warmstart


def _run_bo_doraemon_paperlike(
    seed: int,
    profile: str,
    checkpoint_path: str | Path | None,
    device: str | None,
) -> tuple[dict[str, Any], bool]:
    ns, used_warmstart = _prepare_runtime("BO+DORAEMON", seed, checkpoint_path, device)
    cfg = PROFILE_PRESETS["BO+DORAEMON"][_normalize_profile(profile)]

    result = ns["run_bo_doraemon_paperlike"](
        checkpoint_path=checkpoint_path,
        mu0=ns["MU0_EDGE"],
        conc0=ns["C_INIT"],
        T=cfg["T"],
        blocks=cfg["blocks"],
        episodes_per_block=cfg["episodes_per_block"],
        B_r=cfg["B_r"],
        bounds=ns["EDGE_DR_BOUNDS"],
        tau_steps=int(EPISODE_STEPS),
        alpha_target=float(ns["ALPHA_TARGET"]),
        kl_max=float(ns["KL_MAX"]),
        n_candidates=int(ns["CANDIDATES_PER_UPDATE"]),
        mean_std=float(ns["MEAN_PERTURB_STD"]),
        logc_std=float(ns["LOGC_PERTURB_STD"]),
        c_min=ns["C_MIN"],
        c_max=ns["C_MAX"],
        min_log_w=float(ns["MIN_WEIGHT_LOG"]),
        max_log_w=float(ns["MAX_WEIGHT_LOG"]),
        target_params=ns["TARGET_PARAMS"],
        seed=seed,
        freeze_doraemon_mean=bool(ns["FREEZE_DORAEMON_MEAN"]),
        conc_init_mode=str(ns["CONC_INIT_MODE"]),
        conc_blend=float(ns["CONC_BLEND"]),
    )
    return result, used_warmstart


RUNNER_BY_METHOD: dict[str, Callable[[int, str, str | Path | None, str | None], tuple[dict[str, Any], bool]]] = {
    "BO": _run_bo,
    "DR": _run_dr,
    "BO+DR": _run_bo_dr,
    "DORAEMON": _run_doraemon_only,
    "BO+DORAEMON": _run_bo_doraemon_paperlike,
}


def run_all_methods_for_seed(
    seed: int,
    methods: list[str] | None = None,
    profile: str = "smoke",
    checkpoint_path: str | Path | None = None,
    device: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    profile = _normalize_profile(profile)
    selected_methods = methods or METHOD_ORDER

    rows: list[dict[str, Any]] = []
    raw_results: dict[str, dict[str, Any]] = {}

    for method in selected_methods:
        if method not in RUNNER_BY_METHOD:
            raise ValueError(f"Unknown method '{method}'. Valid methods: {METHOD_ORDER}")

        print(f"\\n=== Seed {seed} | Method {method} | Profile {profile} ===")
        t0 = time.perf_counter()
        result, used_warmstart = RUNNER_BY_METHOD[method](
            seed=seed,
            profile=profile,
            checkpoint_path=checkpoint_path,
            device=device,
        )
        elapsed = float(time.perf_counter() - t0)

        best_target, final_target = _extract_scores(method, result)

        row = {
            "method": method,
            "seed": int(seed),
            "profile": profile,
            "used_warmstart": bool(used_warmstart),
            "best_target_solve_rate": best_target,
            "final_target_solve_rate": final_target,
            "run_dir": str(result.get("run_dir", "")),
            "best_checkpoint": str(result.get("best_checkpoint", "")),
            "elapsed_sec": elapsed,
        }
        rows.append(row)
        raw_results[method] = result

    return rows, raw_results


def _save_plots(master_df: pd.DataFrame, run_dir: Path) -> list[str]:
    plot_paths: list[str] = []
    if master_df.empty:
        return plot_paths

    method_order = [m for m in METHOD_ORDER if m in set(master_df["method"].tolist())]

    best_group = (
        master_df.groupby("method", as_index=False)["best_target_solve_rate"]
        .agg(["mean", "std"])
        .reindex(method_order)
        .reset_index()
    )

    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.bar(
        best_group["method"],
        best_group["mean"],
        yerr=best_group["std"].fillna(0.0),
        capsize=4,
    )
    ax1.set_ylim(0.0, 1.05)
    ax1.set_ylabel("Best Target Solve Rate")
    ax1.set_title("Best Solve Rate by Method (mean ± std over seeds)")
    ax1.grid(axis="y", alpha=0.3)
    fig1.tight_layout()
    path1 = run_dir / "best_target_mean_std.png"
    fig1.savefig(path1, dpi=150)
    plt.close(fig1)
    plot_paths.append(str(path1))

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    for method in method_order:
        part = master_df[master_df["method"] == method]
        ax2.plot(
            part["seed"],
            part["final_target_solve_rate"],
            marker="o",
            linestyle="-",
            label=method,
        )
    ax2.set_ylim(0.0, 1.05)
    ax2.set_xlabel("Seed")
    ax2.set_ylabel("Final Target Solve Rate")
    ax2.set_title("Per-Seed Final Solve Rate by Method")
    ax2.grid(alpha=0.3)
    ax2.legend(loc="best")
    fig2.tight_layout()
    path2 = run_dir / "final_target_per_seed.png"
    fig2.savefig(path2, dpi=150)
    plt.close(fig2)
    plot_paths.append(str(path2))

    return plot_paths


def run_all_methods(
    seeds: list[int],
    methods: list[str] | None = None,
    profile: str = "smoke",
    checkpoint_path: str | Path | None = None,
    device: str | None = None,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    profile = _normalize_profile(profile)
    selected_methods = methods or METHOD_ORDER

    output_base = Path(output_root) if output_root is not None else RUNS_ROOT
    output_base.mkdir(parents=True, exist_ok=True)
    run_dir = output_base / time.strftime("all_methods_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    raw_by_seed: dict[int, dict[str, dict[str, Any]]] = {}

    print("Centralized settings:")
    print("  TARGET_PARAMS:", TARGET_PARAMS)
    print("  DR_BOUNDS:", DR_BOUNDS)
    print("  EDGE_BOUNDS:", EDGE_BOUNDS)
    print("  EPISODE_STEPS:", EPISODE_STEPS)

    for seed in seeds:
        rows, raw = run_all_methods_for_seed(
            seed=int(seed),
            methods=selected_methods,
            profile=profile,
            checkpoint_path=checkpoint_path,
            device=device,
        )
        all_rows.extend(rows)
        raw_by_seed[int(seed)] = raw

    master_df = pd.DataFrame(all_rows)

    master_csv = run_dir / "master_results.csv"
    master_json = run_dir / "master_results.json"
    aggregate_csv = run_dir / "aggregate_by_method.csv"

    master_df.to_csv(master_csv, index=False)
    with open(master_json, "w", encoding="utf-8") as f:
        json.dump(all_rows, f, indent=2)

    agg_df = (
        master_df.groupby("method", as_index=False)
        .agg(
            seeds_count=("seed", "count"),
            best_target_mean=("best_target_solve_rate", "mean"),
            best_target_std=("best_target_solve_rate", "std"),
            best_target_min=("best_target_solve_rate", "min"),
            best_target_max=("best_target_solve_rate", "max"),
            final_target_mean=("final_target_solve_rate", "mean"),
            final_target_std=("final_target_solve_rate", "std"),
            final_target_min=("final_target_solve_rate", "min"),
            final_target_max=("final_target_solve_rate", "max"),
            elapsed_sec_mean=("elapsed_sec", "mean"),
        )
    )
    agg_df.to_csv(aggregate_csv, index=False)

    plot_paths = _save_plots(master_df, run_dir)

    resolved_checkpoint = _resolve_checkpoint_path(checkpoint_path)
    manifest = {
        "run_dir": str(run_dir),
        "master_csv": str(master_csv),
        "master_json": str(master_json),
        "aggregate_csv": str(aggregate_csv),
        "plot_paths": plot_paths,
        "methods": selected_methods,
        "profile": profile,
        "seeds": [int(s) for s in seeds],
        "checkpoint_path": str(resolved_checkpoint) if resolved_checkpoint else None,
        "target_params": dict(TARGET_PARAMS),
        "dr_bounds": _copy_bounds(DR_BOUNDS),
        "edge_bounds": _copy_bounds(EDGE_BOUNDS),
        "episode_steps": int(EPISODE_STEPS),
    }

    with open(run_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return {
        **manifest,
        "master_df": master_df,
        "aggregate_df": agg_df,
        "raw_results_by_seed": raw_by_seed,
    }
