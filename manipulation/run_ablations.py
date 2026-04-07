"""Back-to-back ablation runner for the manipulation task.

Edit `EXPERIMENTS` below to define which runs to launch. Each entry has:
    method:  "doraemon" | "bo" | "bodrem"
    params:  full hyperparameter dict for the chosen method

Run:
    python run_ablations.py
"""

from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

import numpy as np

from pipeline import (
    DEVICE,
    pretrain_manipulation_agent,
    run_bo_doraemon_paperlike,
    run_bo_only,
    run_doraemon_only,
    set_global_seed,
)


# ════════════════════════════════════════════════════════════════════════════
# Paths and shared constants
# ════════════════════════════════════════════════════════════════════════════

PROJECT_DIR          = Path(__file__).resolve().parent
RUNS_ROOT            = PROJECT_DIR / "runs_manipulation"
WARMSTART_CHECKPOINT = PROJECT_DIR / "manipulation_saved" / "manipulation_sac.pt"

GLOBAL_SEED          = 42
PRETRAIN_EPISODES    = 500          # used only if WARMSTART_CHECKPOINT is missing
MAX_EPISODE_STEPS    = 128

# Default initial center (default env physics — friction=1.0, mass=0.1)
MU0 = np.array([1.0, 0.1], dtype=np.float32)

# DORAEMON Beta concentration bounds (shared default)
C_INIT = [18.0, 18.0]
C_MIN  = [2.2, 2.2]
C_MAX  = [220.0, 220.0]


# ════════════════════════════════════════════════════════════════════════════
# DIFFICULTY PRESETS
# ════════════════════════════════════════════════════════════════════════════

DIFFICULTY = {
    "easy": {
        "bounds":        {"friction": (0.6, 1.4), "mass": (0.06, 0.20)},
        "target_params": {"friction": 0.7, "mass": 0.18},
    },
    "medium": {
        "bounds":        {"friction": (0.2, 1.5), "mass": (0.05, 0.40)},
        "target_params": {"friction": 0.4, "mass": 0.30},
    },
    "hard": {
        "bounds":        {"friction": (0.1, 2.5), "mass": (0.02, 0.50)},
        "target_params": {"friction": 0.15, "mass": 0.45},
    },
}


# ════════════════════════════════════════════════════════════════════════════
# BEST-TUNED METHOD CONFIGS
# ════════════════════════════════════════════════════════════════════════════

def doraemon_best(difficulty: dict, seed: int = GLOBAL_SEED) -> dict:
    """DORAEMON-only: needs unfrozen mean, looser KL, more eps/block, lower
    alpha_target so the success constraint is achievable on harder bounds."""
    return {
        "checkpoint_path": WARMSTART_CHECKPOINT,
        "mu0":             MU0.tolist(),
        "conc0":           C_INIT,
        "blocks":               70,        # many adaptation steps
        "episodes_per_block":   100,       # 25 * 150 = 3750 eps
        "B_r":                  20,
        "bounds":               difficulty["bounds"],
        "max_steps":            MAX_EPISODE_STEPS,
        "alpha_target":         0.50,      # ↓ from 0.60 — easier to satisfy
        "kl_max":               0.06,      # ↑ from 0.04 — faster trust-region
        "n_candidates":         400,       # ↑ more candidates per update
        "mean_std":             0.10,      # ↑ from 0.06 — bigger mean drift
        "logc_std":             0.25,
        "c_min":                C_MIN,
        "c_max":                C_MAX,
        "min_log_w":            -20.0,
        "max_log_w":             20.0,
        "target_params":        difficulty["target_params"],
        "seed":                 seed,
        "freeze_doraemon_mean": True,     # ← critical: BO is absent
        "eval_every_blocks":    1,
    }


def bo_best(difficulty: dict, seed: int = GLOBAL_SEED) -> dict:
    """BO-only: needs many SAC episodes per BO iter so the policy can actually
    fine-tune at each proposed center, plus enough T for GP exploration."""
    return {
        "checkpoint_path":   WARMSTART_CHECKPOINT,
        "mu0":               MU0.tolist(),
        "T":                 20,           # ↑ more BO iters
        "episodes_per_iter": 350,          # ↑ from 200 — SAC needs more
        "B_r":               20,
        "bounds":            difficulty["bounds"],
        "max_steps":         MAX_EPISODE_STEPS,
        "target_params":     difficulty["target_params"],
        "seed":              seed,
    }


def bodrem_best(difficulty: dict, seed: int = GLOBAL_SEED) -> dict:
    """BODREM: BO drives the center, DORAEMON's inner loop refines the spread.
    Mean is frozen because BO already moves the center — DORAEMON just needs
    enough blocks per BO iter for the IS estimate to be reliable."""
    return {
        "checkpoint_path": WARMSTART_CHECKPOINT,
        "mu0":             MU0.tolist(),
        "conc0":           C_INIT,
        "T":                    20,        # 20 BO iters
        "blocks":               5,         # 5 DORAEMON blocks per BO iter
        "episodes_per_block":   70,        # 20 * 5 * 70 = 7000 eps
        "B_r":                  20,
        "bounds":               difficulty["bounds"],
        "max_steps":            MAX_EPISODE_STEPS,
        "alpha_target":         0.60,
        "kl_max":               0.05,
        "n_candidates":         300,
        "mean_std":             0.06,
        "logc_std":             0.20,
        "c_min":                C_MIN,
        "c_max":                C_MAX,
        "min_log_w":            -20.0,
        "max_log_w":             20.0,
        "target_params":        difficulty["target_params"],
        "seed":                 seed,
        "freeze_doraemon_mean": False,     # BO handles the center (Set it to False)
        "conc_init_mode":       "reset",
        "conc_blend":           0.5,
    }


METHOD_BUILDERS = {
    "doraemon": doraemon_best,
    "bo":       bo_best,
    "bodrem":   bodrem_best,
}


def make_experiment(method: str, difficulty_name: str) -> dict:
    if method not in METHOD_BUILDERS:
        raise ValueError(f"unknown method '{method}'")
    if difficulty_name not in DIFFICULTY:
        raise ValueError(f"unknown difficulty '{difficulty_name}'")
    return {
        "method": method,
        "params": METHOD_BUILDERS[method](DIFFICULTY[difficulty_name]),
        "difficulty": difficulty_name,
    }


# ════════════════════════════════════════════════════════════════════════════
# EXPERIMENTS
#
# This is the main ablation set for the paper. The format for each experiment
# is:
#     <experiment-name>: make_experiment(<method>, <difficulty>)
#     where:
#     - <experiment-name> is an arbitrary string identifier for the run (will
#         be used in the run name and summary)
#     - <method> is one of "doraemon", "bo", "bodrem"
#     - <difficulty> is one of "easy", "medium", "hard" (see DIFFICULTY above)

# The parameters for each model can be tuned in the METHOD_BUILDERS section 
# above. The difficulty presets can be tuned in the DIFFICULTY section above.
# The budget (number of episodes) for each run is determined by the 
# parameters and printed in the summary table before execution.
# ════════════════════════════════════════════════════════════════════════════

EXPERIMENTS: dict[str, dict] = {
    "doraemon_medium": make_experiment("doraemon", "medium"),
    "bo_medium":       make_experiment("bo",       "medium"),
    "bodrem_medium":   make_experiment("bodrem",   "medium"),

    "doraemon_easy": make_experiment("doraemon", "easy"),
    "bo_easy":       make_experiment("bo",       "easy"),
    "bodrem_easy":   make_experiment("bodrem",   "easy"),

    "doraemon_hard": make_experiment("doraemon", "hard"),
    "bo_hard":       make_experiment("bo",       "hard"),
    "bodrem_hard":   make_experiment("bodrem",   "hard"),
}


# budget reporter
def _episode_budget(spec: dict) -> int:
    p = spec["params"]
    if spec["method"] == "doraemon":
        return int(p["blocks"]) * int(p["episodes_per_block"])
    if spec["method"] == "bo":
        return int(p["T"]) * int(p["episodes_per_iter"])
    if spec["method"] == "bodrem":
        return int(p["T"]) * int(p["blocks"]) * int(p["episodes_per_block"])
    raise ValueError(spec["method"])


# ════════════════════════════════════════════════════════════════════════════
# Dispatch
# ════════════════════════════════════════════════════════════════════════════

METHOD_DISPATCH = {
    "doraemon": run_doraemon_only,
    "bo":       run_bo_only,
    "bodrem":   run_bo_doraemon_paperlike,
}


def ensure_warmstart() -> None:
    if WARMSTART_CHECKPOINT.exists():
        print(f"[warmstart] using existing checkpoint: {WARMSTART_CHECKPOINT}")
        return
    print(f"[warmstart] no checkpoint found at {WARMSTART_CHECKPOINT}, pre-training...")
    WARMSTART_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    pretrain_manipulation_agent(
        n_episodes=PRETRAIN_EPISODES,
        friction=1.0, mass=0.1,
        max_steps=MAX_EPISODE_STEPS,
        seed=GLOBAL_SEED,
        save_path=WARMSTART_CHECKPOINT,
    )


def run_one(name: str, spec: dict) -> dict:
    method = spec["method"]
    if method not in METHOD_DISPATCH:
        raise ValueError(f"Unknown method '{method}' for experiment '{name}'. "
                         f"Choices: {list(METHOD_DISPATCH)}")
    runner = METHOD_DISPATCH[method]

    difficulty = spec.get("difficulty", "?")
    budget     = _episode_budget(spec)
    run_name = f"{name}_{time.strftime('%Y%m%d_%H%M%S')}"
    print("\n" + "═" * 78)
    print(f"▶ EXPERIMENT '{name}'  method={method}  difficulty={difficulty}  "
          f"budget={budget} eps  run_name={run_name}")
    print("═" * 78)

    set_global_seed(int(spec["params"].get("seed", GLOBAL_SEED)))
    return runner(runs_root=RUNS_ROOT, run_name=run_name, **spec["params"])


def _print_budget_table() -> None:
    print("\n" + "─" * 78)
    print(f"  {'name':28s}  {'method':10s}  {'difficulty':10s}  {'budget (eps)':>12s}")
    print("─" * 78)
    total = 0
    for name, spec in EXPERIMENTS.items():
        b = _episode_budget(spec)
        total += b
        print(f"  {name:28s}  {spec['method']:10s}  "
              f"{spec.get('difficulty','?'):10s}  {b:>12d}")
    print("─" * 78)
    print(f"  {'TOTAL':28s}  {'':10s}  {'':10s}  {total:>12d}")
    print("─" * 78 + "\n")


def main() -> None:
    print(f"DEVICE: {DEVICE}")
    print(f"RUNS_ROOT: {RUNS_ROOT}")
    print(f"Experiments queued ({len(EXPERIMENTS)}): {list(EXPERIMENTS)}")
    _print_budget_table()

    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    ensure_warmstart()

    summary = {}
    t_total_start = time.time()

    for name, spec in EXPERIMENTS.items():
        t0 = time.time()
        try:
            result = run_one(name, spec)
            elapsed = time.time() - t0
            summary[name] = {
                "status":      "ok",
                "method":      spec["method"],
                "difficulty":  spec.get("difficulty", "?"),
                "budget_eps":  _episode_budget(spec),
                "elapsed_s":   round(elapsed, 1),
                "best_score":  result.get("best_score"),
                "run_dir":     result.get("run_dir"),
            }
            print(f"\n✓ '{name}' done in {elapsed:.1f}s | "
                  f"best={result.get('best_score'):.3f} | dir={result.get('run_dir')}")
        except Exception as exc:
            elapsed = time.time() - t0
            summary[name] = {
                "status":     "failed",
                "method":     spec["method"],
                "difficulty": spec.get("difficulty", "?"),
                "budget_eps": _episode_budget(spec),
                "elapsed_s":  round(elapsed, 1),
                "error":      f"{type(exc).__name__}: {exc}",
            }
            print(f"\n✗ '{name}' FAILED after {elapsed:.1f}s: {exc}")
            traceback.print_exc()

    total = time.time() - t_total_start
    summary_path = RUNS_ROOT / f"ablation_summary_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_path, "w") as f:
        json.dump({"total_elapsed_s": round(total, 1), "results": summary}, f, indent=2, default=str)

    print("\n" + "═" * 78)
    print(f"ALL DONE in {total:.1f}s")
    print(f"Summary written to {summary_path}")
    print("═" * 78)
    for name, info in summary.items():
        tag = "OK " if info["status"] == "ok" else "ERR"
        score = info.get("best_score")
        score_s = f"{score:.3f}" if isinstance(score, (int, float)) else "—"
        print(f"  [{tag}] {name:28s} method={info['method']:8s} "
              f"diff={info.get('difficulty','?'):6s} "
              f"budget={info.get('budget_eps','?'):>5}eps "
              f"best={score_s} elapsed={info['elapsed_s']}s")


if __name__ == "__main__":
    main()
