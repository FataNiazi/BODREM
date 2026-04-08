# MountainCar + HalfCheetah

Unified experiment runner for:

- MountainCar (`task=mountaincar`, native DQN)
- HalfCheetah (`task=halfcheetah`, SB3 SAC)

Supported methods:

- `BO`
- `DR`
- `BO+DR`
- `DORAEMON`
- `BO+DORAEMON`

## Working Directory

Resolve and use repo root before running commands:

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"
```

Compatibility note: `mountaincar` is an alias to `mountaincar&cheetah`, so commands use `python -m mountaincar...`.
Do not run `python -m mountaincar...` from inside `mountaincar&cheetah/`, otherwise local `types.py` can shadow the stdlib `types` module.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```


List methods:

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
(cd "$REPO_ROOT" && "$REPO_ROOT/.venv/bin/python" -m mountaincar.cli list-methods)
```

## Smoke Example (Short, 2 Algos x 2 Seeds)

One short HalfCheetah smoke run (2 methods, 2 seeds) to show run structure:

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
RUN_DIR=$(cd "$REPO_ROOT" && "$REPO_ROOT/.venv/bin/python" -m mountaincar.cli run \
  --task halfcheetah \
  --backend sb3_sac \
  --primary-metric mean_episodic_return \
  --eval-episodes 2 \
  --methods DR,DORAEMON \
  --seeds 1,2 \
  --profile smoke \
  --init-mode scratch \
  --task-config-json '{"surrogate_success_threshold": 0.0}')

echo "$RUN_DIR"
ls "$RUN_DIR"
```

Generate plots for this smoke example:

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
(cd "$REPO_ROOT" && "$REPO_ROOT/.venv/bin/python" -m mountaincar.plot_results \
  --input "$RUN_DIR" \
  --output './mountaincar&cheetah/runs/plots/halfcheetah_smoke_2algo_2seed' \
  --task halfcheetah)
```

## Edge Variants (Full Runs)

All full-run examples below use `profile="edge"`.

### MountainCar edge sweep (v1/v2/v3, seeds 1-10)

Method-first order (all seeds for one method-variant, then next):

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"
.venv/bin/python - <<'PY'
from pathlib import Path
from mountaincar.config import build_suite_config
from mountaincar.runner import run_suite
from mountaincar.types import MethodName

seeds = [1,2,3,4,5,6,7,8,9,10]
base_out = Path("mountaincar&cheetah/runs/mountaincar_edge_v123")

sweep = {
    MethodName.BO: [
        ("v1_baseline",    {"bo_iters": 20, "bo_train_episodes_per_iter": 180}),  # 3600
        ("v2_more_outer",  {"bo_iters": 30, "bo_train_episodes_per_iter": 120}),  # 3600
        ("v3_fewer_outer", {"bo_iters": 10, "bo_train_episodes_per_iter": 360}),  # 3600
    ],
    MethodName.DR: [
        ("v1_baseline",    {"dr_total_episodes": 3600, "dr_eval_every": 180}),
        ("v2_more_evals",  {"dr_total_episodes": 3600, "dr_eval_every": 120}),
        ("v3_fewer_evals", {"dr_total_episodes": 3600, "dr_eval_every": 360}),
    ],
    MethodName.BO_DR: [
        ("v1_baseline",    {"T": 20, "K": 180}),  # 3600
        ("v2_more_outer",  {"T": 30, "K": 120}),  # 3600
        ("v3_fewer_outer", {"T": 10, "K": 360}),  # 3600
    ],
    MethodName.DORAEMON: [
        ("v1_baseline",    {"adapt_iters": 20, "blocks": 30, "episodes_per_block": 6}),   # 3600
        ("v2_more_outer",  {"adapt_iters": 20, "blocks": 15, "episodes_per_block": 12}),  # 3600
        ("v3_fewer_outer", {"adapt_iters": 10, "blocks": 15, "episodes_per_block": 24}),  # 3600
    ],
    MethodName.BO_DORAEMON: [
        ("v1_baseline", {
            "T": 20, "blocks": 30, "episodes_per_block": 6,
            "stabilization_blocks": 0, "bo_init_random_points": 0,
            "center_hold_rounds": 1, "competence_extra_rounds": 0,
        }),  # 3600
        ("v2_more_outer", {
            "T": 20, "blocks": 15, "episodes_per_block": 12,
            "stabilization_blocks": 0, "bo_init_random_points": 0,
            "center_hold_rounds": 1, "competence_extra_rounds": 0,
        }),  # 3600
        ("v3_fewer_outer", {
            "T": 10, "blocks": 15, "episodes_per_block": 24,
            "stabilization_blocks": 0, "bo_init_random_points": 0,
            "center_hold_rounds": 1, "competence_extra_rounds": 0,
        }),  # 3600
    ],
}

for method, variants in sweep.items():
    for variant_name, override in variants:
        cfg = build_suite_config(
            seeds=seeds,
            methods=[method],
            task="mountaincar",
            profile="edge",
            init_mode="scratch",
            output_root=base_out / method.value.replace("+","plus").lower() / variant_name,
            method_overrides={method: override},
        )
        res = run_suite(cfg)
        print(method.value, variant_name, res.run_dir)
PY
```

Plot all run outputs from this MountainCar sweep:

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
BASE="$REPO_ROOT/mountaincar&cheetah/runs/mountaincar_edge_v123"
OUT_BASE="$REPO_ROOT/mountaincar&cheetah/runs/plots/mountaincar_edge_v123"

find "$BASE" -type f -name master_results.csv | while read -r csv; do
  RUN_DIR="$(dirname "$csv")"
  REL="${RUN_DIR#$BASE/}"
  OUT_DIR="$OUT_BASE/$REL"
  mkdir -p "$OUT_DIR"
  (cd "$REPO_ROOT" && "$REPO_ROOT/.venv/bin/python" -m mountaincar.plot_results \
    --input "$RUN_DIR" \
    --output "$OUT_DIR" \
    --task mountaincar)
done
```

### HalfCheetah edge sweep (fixed 1800 override, seeds 1-5)

Method-first order (all seeds for one algo, then next):

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"
.venv/bin/python - <<'PY'
from pathlib import Path
from mountaincar.config import build_suite_config
from mountaincar.runner import run_suite
from mountaincar.types import MethodName

seeds = [1,2,3,4,5]
methods = [MethodName.BO, MethodName.DR, MethodName.BO_DR, MethodName.DORAEMON, MethodName.BO_DORAEMON]
base_out = Path("mountaincar&cheetah/runs/halfcheetah_edge_1800")

overrides = {
    MethodName.BO: {"bo_iters": 10, "bo_train_episodes_per_iter": 180},   # 1800
    MethodName.DR: {"dr_total_episodes": 1800, "dr_eval_every": 180},      # 1800
    MethodName.BO_DR: {"T": 10, "K": 180},                                  # 1800
    MethodName.DORAEMON: {"adapt_iters": 10, "blocks": 15, "episodes_per_block": 12},  # 1800
    MethodName.BO_DORAEMON: {
        "T": 10, "blocks": 15, "episodes_per_block": 12,                    # 1800
        "stabilization_blocks": 0, "bo_init_random_points": 0,
        "center_hold_rounds": 1, "competence_extra_rounds": 0,
    },
}

for method in methods:
    cfg = build_suite_config(
        seeds=seeds,
        methods=[method],
        task="halfcheetah",
        backend="sb3_sac",
        primary_metric="mean_episodic_return",
        eval_episodes=5,
        profile="edge",
        init_mode="scratch",
        task_config={"surrogate_success_threshold": 0.0},
        output_root=base_out / method.value.replace("+","plus").lower(),
        method_overrides={method: overrides[method]},
    )
    res = run_suite(cfg)
    print(method.value, res.run_dir)
PY
```

Plot all run outputs from this HalfCheetah sweep:

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
BASE="$REPO_ROOT/mountaincar&cheetah/runs/halfcheetah_edge_1800"
OUT_BASE="$REPO_ROOT/mountaincar&cheetah/runs/plots/halfcheetah_edge_1800"

find "$BASE" -type f -name master_results.csv | while read -r csv; do
  RUN_DIR="$(dirname "$csv")"
  REL="${RUN_DIR#$BASE/}"
  OUT_DIR="$OUT_BASE/$REL"
  mkdir -p "$OUT_DIR"
  (cd "$REPO_ROOT" && "$REPO_ROOT/.venv/bin/python" -m mountaincar.plot_results \
    --input "$RUN_DIR" \
    --output "$OUT_DIR" \
    --task halfcheetah)
done
```

## Useful Variants

MountainCar smoke (all methods, one seed):

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
RUN_DIR=$(cd "$REPO_ROOT" && "$REPO_ROOT/.venv/bin/python" -m mountaincar.cli run \
  --task mountaincar \
  --methods BO,DR,BO+DR,DORAEMON,BO+DORAEMON \
  --seeds 1 \
  --profile smoke \
  --init-mode scratch)

(cd "$REPO_ROOT" && "$REPO_ROOT/.venv/bin/python" -m mountaincar.plot_results \
  --input "$RUN_DIR" \
  --output './mountaincar&cheetah/runs/plots/mountaincar_smoke_all_methods_seed1' \
  --task mountaincar)
```

HalfCheetah batch mode:

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
RUN_DIR=$(cd "$REPO_ROOT" && "$REPO_ROOT/.venv/bin/python" -m mountaincar.cli batch-run \
  --task halfcheetah \
  --backend sb3_sac \
  --primary-metric mean_episodic_return \
  --eval-episodes 5 \
  --methods BO,DR \
  --seeds 1,2 \
  --repetitions 2 \
  --profile smoke \
  --init-mode scratch \
  --on-error continue)

(cd "$REPO_ROOT" && "$REPO_ROOT/.venv/bin/python" -m mountaincar.plot_results \
  --input "$RUN_DIR" \
  --output './mountaincar&cheetah/runs/plots/halfcheetah_batch_smoke_bo_dr' \
  --task halfcheetah)
```

## Outputs

Suite runs write:

- `master_results.csv`
- `master_results.json`
- `aggregate_by_method.csv`
- `manifest.json`

Each method run writes:

- `history.json`
- `history.csv`
- `best_policy.pt` (MountainCar) or `best_policy.zip` (HalfCheetah)
- `last_policy.pt` (MountainCar) or `last_policy.zip` (HalfCheetah)
- `manifest.json`

Plot outputs include:

- `plot_manifest.json`
- `summary_statistics.json`
- per-method PNG/PDF plots and comparison figures

## Notes

- HalfCheetah requires `stable-baselines3` and MuJoCo support (`mujoco` / Gymnasium MuJoCo stack).
- `BO`, `BO+DR`, and `BO+DORAEMON` require `scikit-optimize`.
- DORAEMON variants require `scipy`.
