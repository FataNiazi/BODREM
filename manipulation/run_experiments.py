"""Multi-machine experiment runner for easy-vs-hard comparison.

Target: 3 seeds × 2 difficulties (easy, hard) × 3 methods = 18 runs total.

Designed to run on two machines with different capacities. Use the --seeds
flag to split the work:

    # On H200 machine (faster — takes 2 seeds = 12 runs):
    python run_experiments.py --seeds 42,123 --parallel 4

    # On RTX 5090 machine (takes 1 seed = 6 runs):
    python run_experiments.py --seeds 456 --parallel 2

Or run a specific method only:
    python run_experiments.py --seeds 42 --filter bodrem --parallel 2

Flags:
    --seeds      Comma-separated seeds to run (e.g. "42,123")
    --parallel   Number of parallel workers (default: 1)
    --filter     Only run one method: bo, doraemon, or bodrem
    --resume     Skip experiments whose run directory already exists
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from pipeline import (
    DEVICE,
    run_bo_doraemon_paperlike,
    run_bo_only,
    run_doraemon_only,
    set_global_seed,
)


# ════════════════════════════════════════════════════════════════════════════
# PATHS
# ════════════════════════════════════════════════════════════════════════════

PROJECT_DIR          = Path(__file__).resolve().parent
RUNS_ROOT            = PROJECT_DIR / "runs_manipulation"
WARMSTART_CHECKPOINT = PROJECT_DIR / "manipulation_saved" / "manipulation_sac.pt"

MAX_EPISODE_STEPS = 128
MU0    = [1.0, 0.1]
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
    "hard": {
        "bounds":        {"friction": (0.1, 2.5), "mass": (0.02, 0.50)},
        "target_params": {"friction": 0.15, "mass": 0.45},
    },
}


# ════════════════════════════════════════════════════════════════════════════
# CANONICAL CONFIGS
#
# These are the "best" configs for each method. DO NOT change them between
# machines — both must use identical configs for the comparison to be valid.
# ════════════════════════════════════════════════════════════════════════════

def bo_config(difficulty: dict, seed: int) -> dict:
    return {
        "checkpoint_path":   WARMSTART_CHECKPOINT,
        "mu0":               list(MU0),
        "T":                 20,
        "episodes_per_iter": 350,
        "B_r":               20,
        "bounds":            difficulty["bounds"],
        "max_steps":         MAX_EPISODE_STEPS,
        "target_params":     difficulty["target_params"],
        "seed":              seed,
    }


def doraemon_config(difficulty: dict, seed: int) -> dict:
    return {
        "checkpoint_path":      WARMSTART_CHECKPOINT,
        "mu0":                  list(MU0),
        "conc0":                C_INIT,
        "blocks":               20,
        "episodes_per_block":   350,
        "B_r":                  20,
        "bounds":               difficulty["bounds"],
        "max_steps":            MAX_EPISODE_STEPS,
        "alpha_target":         0.50,
        "kl_max":               0.06,
        "n_candidates":         400,
        "mean_std":             0.10,
        "logc_std":             0.25,
        "c_min":                C_MIN,
        "c_max":                C_MAX,
        "min_log_w":            -20.0,
        "max_log_w":             20.0,
        "target_params":        difficulty["target_params"],
        "seed":                 seed,
        "freeze_doraemon_mean": True,
        "eval_every_blocks":    1,
    }


def bodrem_config(difficulty: dict, seed: int) -> dict:
    return {
        "checkpoint_path":      WARMSTART_CHECKPOINT,
        "mu0":                  list(MU0),
        "conc0":                C_INIT,
        "T":                    20,
        "blocks":               5,
        "episodes_per_block":   70,
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
        "freeze_doraemon_mean": True,
        "conc_init_mode":       "reset",
        "conc_blend":           0.5,
    }


METHOD_RUNNERS = {
    "bo":       (run_bo_only,               bo_config),
    "doraemon": (run_doraemon_only,         doraemon_config),
    "bodrem":   (run_bo_doraemon_paperlike, bodrem_config),
}

METHODS = ["bo", "doraemon", "bodrem"]
DIFFICULTIES = ["easy", "hard"]


# ════════════════════════════════════════════════════════════════════════════
# EXECUTION
# ════════════════════════════════════════════════════════════════════════════

def run_name_for(method: str, difficulty: str, seed: int) -> str:
    """Deterministic run name — no timestamp so --resume can find it."""
    return f"{method}_{difficulty}_seed{seed}"


def run_exists(method: str, difficulty: str, seed: int) -> bool:
    """Check if a run directory already exists and looks complete."""
    run_dir = RUNS_ROOT / run_name_for(method, difficulty, seed)
    return run_dir.exists() and (run_dir / "last_policy.pt").exists()


def run_one(method: str, difficulty: str, seed: int) -> dict:
    runner, config_fn = METHOD_RUNNERS[method]
    params = config_fn(DIFFICULTY[difficulty], seed=seed)
    run_name = run_name_for(method, difficulty, seed)

    set_global_seed(seed)
    return runner(runs_root=RUNS_ROOT, run_name=run_name, **params)


def run_one_safe(method: str, difficulty: str, seed: int) -> dict:
    try:
        start = time.time()
        result = run_one(method, difficulty, seed)
        return {
            "method": method,
            "difficulty": difficulty,
            "seed": seed,
            "status": "OK",
            "elapsed": time.time() - start,
        }
    except Exception as e:
        return {
            "method": method,
            "difficulty": difficulty,
            "seed": seed,
            "status": "ERR",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=str, required=True,
                        help="Comma-separated seeds (e.g. '42,123')")
    parser.add_argument("--parallel", type=int, default=1,
                        help="Number of parallel workers (default: 1)")
    parser.add_argument("--filter", type=str, default=None,
                        choices=METHODS,
                        help="Only run one method")
    parser.add_argument("--resume", action="store_true",
                        help="Skip experiments whose output directory already exists")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    methods = [args.filter] if args.filter else METHODS

    experiments = [
        (method, difficulty, seed)
        for seed in seeds
        for method in methods
        for difficulty in DIFFICULTIES
    ]

    if args.resume:
        original = len(experiments)
        experiments = [e for e in experiments if not run_exists(*e)]
        skipped = original - len(experiments)
        if skipped:
            print(f"↻ Skipping {skipped} already-completed experiments")

    print("=" * 80)
    print("EXPERIMENT RUNNER")
    print("=" * 80)
    print(f"Device:     {DEVICE}")
    print(f"Seeds:      {seeds}")
    print(f"Methods:    {methods}")
    print(f"Parallel:   {args.parallel}")
    print(f"Total runs: {len(experiments)}")
    print(f"Output:     {RUNS_ROOT}")
    print("\nExperiments:")
    for i, (m, d, s) in enumerate(experiments, 1):
        print(f"  {i:2d}. {m:10} {d:6} seed={s}")
    print("=" * 80 + "\n")

    if not experiments:
        print("Nothing to do.")
        return

    RUNS_ROOT.mkdir(parents=True, exist_ok=True)

    start_all = time.time()
    results = []

    if args.parallel == 1:
        for m, d, s in experiments:
            print(f"\n▶ {m} / {d} / seed={s}")
            r = run_one_safe(m, d, s)
            results.append(r)
            if r["status"] == "OK":
                print(f"  ✓ Done in {r['elapsed']:.1f}s")
            else:
                print(f"  ✗ ERROR: {r['error']}")
    else:
        with ProcessPoolExecutor(max_workers=args.parallel) as ex:
            futures = {
                ex.submit(run_one_safe, m, d, s): (m, d, s)
                for m, d, s in experiments
            }
            for fut in as_completed(futures):
                m, d, s = futures[fut]
                r = fut.result()
                results.append(r)
                if r["status"] == "OK":
                    print(f"✓ {m} / {d} / seed={s} in {r['elapsed']:.1f}s")
                else:
                    print(f"✗ {m} / {d} / seed={s} FAILED: {r['error']}")

    total_elapsed = time.time() - start_all

    print("\n" + "=" * 80)
    print(f"ALL DONE in {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print("=" * 80)
    n_ok = sum(1 for r in results if r["status"] == "OK")
    n_err = sum(1 for r in results if r["status"] == "ERR")
    print(f"Successful: {n_ok}/{len(results)}")
    print(f"Failed:     {n_err}/{len(results)}")

    if n_err:
        print("\nFailed experiments:")
        for r in results:
            if r["status"] == "ERR":
                print(f"  - {r['method']}/{r['difficulty']}/seed={r['seed']}: {r['error']}")

    summary_file = RUNS_ROOT / f"summary_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSummary: {summary_file}")


if __name__ == "__main__":
    main()
