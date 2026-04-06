# MountainCar (Modular Rewrite)

Clean modular MountainCar research package with a shared `Algorithm` superclass and five concrete methods:

- `BO`
- `DR`
- `BO+DR`
- `DORAEMON`
- `BO+DORAEMON`

The same runner can also operate in a task-aware HalfCheetah mode using the same public API.

## Project Layout

- `algorithms/`: method implementations and shared algorithm base class
- `rl/`: shared DQN, env helpers, eval helpers, checkpoint I/O
- `config.py`: profiles, bounds, defaults, suite config builders
- `types.py`: typed result/config dataclasses
- `registry.py`: method-to-algorithm resolution
- `runner.py`: programmatic execution entrypoints
- `cli.py`: command-line interface
- `tests/`: lightweight smoke/integration tests

## Setup

From `/Users/kevaanbuch/Desktop/Uni/CSC415/mountaincar`:

```bash
./scripts/setup_env.sh
```

Install dev dependencies too:

```bash
./scripts/setup_env.sh --dev
```

Install TinySim from a local repo path (recommended when `pip install tinysim` is unavailable):

```bash
./scripts/setup_env.sh --tinysim-path ../TinySim
```

Activate environment:

```bash
source .venv/bin/activate
```

## CLI Usage

List supported methods:

```bash
python -m mountaincar.cli list-methods
```

Run a suite:

```bash
python -m mountaincar.cli run \
  --methods BO,DR,BO+DR,DORAEMON,BO+DORAEMON \
  --seeds 1,2,3 \
  --profile smoke \
  --init-mode auto
```

Optional runtime flags:

```bash
python -m mountaincar.cli run \
  --methods BO+DORAEMON \
  --seeds 1 \
  --profile prototype \
  --init-mode warmstart \
  --checkpoint-path ./mountain_car_saved/mountain_car_dqn.pt \
  --device cpu \
  --output-root ./runs
```

HalfCheetah usage:

```bash
python -m mountaincar.cli run \
  --task halfcheetah \
  --backend sb3_sac \
  --primary-metric mean_episodic_return \
  --eval-episodes 5 \
  --methods BO,DR,BO+DR,DORAEMON,BO+DORAEMON \
  --seeds 7 \
  --profile smoke \
  --init-mode auto \
  --task-config-json '{"surrogate_success_threshold": 0.0}'
```

Run repeated batch attempts:

```bash
python -m mountaincar.cli batch-run \
  --methods BO,DR,BO+DR \
  --seeds 1,1,2 \
  --repetitions 3 \
  --profile smoke \
  --init-mode scratch \
  --on-error continue
```

HalfCheetah batch example:

```bash
python -m mountaincar.cli batch-run \
  --task halfcheetah \
  --backend sb3_sac \
  --primary-metric mean_episodic_return \
  --eval-episodes 5 \
  --methods BO,DR \
  --seeds 7,8 \
  --repetitions 2 \
  --profile smoke \
  --init-mode auto \
  --on-error continue
```

## Python API Usage

```python
from mountaincar.config import build_suite_config
from mountaincar.runner import run_suite, run_seed, run_method, run_batch, list_methods

cfg = build_suite_config(
    seeds=[1, 2],
    methods=["BO", "DR", "BO+DR", "DORAEMON", "BO+DORAEMON"],
    profile="smoke",
    init_mode="auto",
)

suite_result = run_suite(cfg)
print(suite_result.run_dir)

seed_result = run_seed(1, cfg)
method_result = run_method("BO", 1, cfg)
batch_result = run_batch(cfg, repetitions=2)
print(method_result.best_score)
```

## Profiles

Available profiles:

- `smoke`: minimal quick runs
- `prototype`: medium budget
- `edge`: largest default budget

Profile budgets are defined in `config.py` and can be overridden per method via `method_overrides` in `build_suite_config(...)`.

## Outputs

Each suite run writes a timestamped run directory containing:

- `master_results.csv`
- `master_results.json`
- `aggregate_by_method.csv`
- `manifest.json`

Each batch run writes:

- `attempts.csv`
- `attempts.json`
- `aggregate_by_method.csv`
- `aggregate_by_method_seed.csv`
- `manifest.json`

Each method run writes:

- `history.json`
- `history.csv`
- `best_policy.pt` for MountainCar or `best_policy.zip` for HalfCheetah/SB3
- `last_policy.pt` for MountainCar or `last_policy.zip` for HalfCheetah/SB3
- `manifest.json`

## Tests

If you installed dev dependencies:

```bash
pytest -q tests
```

If `pytest` is unavailable, you can still run smoke checks through Python imports in the existing test files.

## Notes

- Real algorithm runs require runtime dependencies (`numpy`, `torch`, `gymnasium`, `scipy`, `scikit-optimize`, `tinysim`).
- `BO`, `BO+DR`, and `BO+DORAEMON` require `scikit-optimize`.
- DORAEMON variants require `scipy`.
- HalfCheetah runs additionally require `stable-baselines3` and MuJoCo support via `mujoco` or `gymnasium[mujoco]`.
- MountainCar keeps the native DQN path and does not require the HalfCheetah-specific dependencies.
