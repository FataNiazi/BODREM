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

Run all commands from repo root:

```bash
cd <repo-root>
```

Compatibility note: `mountaincar` is an alias to `mountaincar&cheetah`, so commands use `python -m mountaincar...`.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

If TinySim must come from a local repo:

```bash
source .venv/bin/activate
pip install -e ../TinySim
```

List methods:

```bash
python -m mountaincar.cli list-methods
```

## Smoke Example (Single Run)

One minimal HalfCheetah smoke run (single method, single seed) to show run structure:

```bash
RUN_DIR=$(python -m mountaincar.cli run \
  --task halfcheetah \
  --backend sb3_sac \
  --primary-metric mean_episodic_return \
  --eval-episodes 5 \
  --methods DR \
  --seeds 1 \
  --profile smoke \
  --init-mode scratch \
  --task-config-json '{"surrogate_success_threshold": 0.0}')

echo "$RUN_DIR"
ls "$RUN_DIR"
```

Generate plots for this smoke example:

```bash
python -m mountaincar.plot_results \
  --input "$RUN_DIR" \
  --output './mountaincar&cheetah/runs/plots/halfcheetah_smoke_example' \
  --task halfcheetah
```

## Edge Variants (Full Runs)

Use the following three execution variants for each task. All commands use `--profile edge`.

### MountainCar (seeds 1-10)

Variant A: Combined edge suite (all methods and seeds in one command)

```bash
python -m mountaincar.cli run \
  --task mountaincar \
  --methods BO,DR,BO+DR,DORAEMON,BO+DORAEMON \
  --seeds 1,2,3,4,5,6,7,8,9,10 \
  --profile edge \
  --init-mode scratch
```

Variant B: Sequential by algorithm (all seeds for one algo, then next algo)

```bash
for method in BO DR BO+DR DORAEMON BO+DORAEMON; do
  python -m mountaincar.cli run \
    --task mountaincar \
    --methods "$method" \
    --seeds 1,2,3,4,5,6,7,8,9,10 \
    --profile edge \
    --init-mode scratch
done
```

Variant C: Sequential by seed (all algorithms per seed, then next seed)

```bash
for seed in 1 2 3 4 5 6 7 8 9 10; do
  python -m mountaincar.cli run \
    --task mountaincar \
    --methods BO,DR,BO+DR,DORAEMON,BO+DORAEMON \
    --seeds "$seed" \
    --profile edge \
    --init-mode scratch
done
```

### HalfCheetah (seeds 1-5)

Variant A: Combined edge suite (all methods and seeds in one command)

```bash
python -m mountaincar.cli run \
  --task halfcheetah \
  --backend sb3_sac \
  --primary-metric mean_episodic_return \
  --eval-episodes 5 \
  --methods BO,DR,BO+DR,DORAEMON,BO+DORAEMON \
  --seeds 1,2,3,4,5 \
  --profile edge \
  --init-mode scratch \
  --task-config-json '{"surrogate_success_threshold": 0.0}'
```

Variant B: Sequential by algorithm (all seeds for one algo, then next algo)

```bash
for method in BO DR BO+DR DORAEMON BO+DORAEMON; do
  python -m mountaincar.cli run \
    --task halfcheetah \
    --backend sb3_sac \
    --primary-metric mean_episodic_return \
    --eval-episodes 5 \
    --methods "$method" \
    --seeds 1,2,3,4,5 \
    --profile edge \
    --init-mode scratch \
    --task-config-json '{"surrogate_success_threshold": 0.0}'
done
```

Variant C: Sequential by seed (all algorithms per seed, then next seed)

```bash
for seed in 1 2 3 4 5; do
  python -m mountaincar.cli run \
    --task halfcheetah \
    --backend sb3_sac \
    --primary-metric mean_episodic_return \
    --eval-episodes 5 \
    --methods BO,DR,BO+DR,DORAEMON,BO+DORAEMON \
    --seeds "$seed" \
    --profile edge \
    --init-mode scratch \
    --task-config-json '{"surrogate_success_threshold": 0.0}'
done
```

## Useful Variants

MountainCar smoke (all methods, one seed):

```bash
python -m mountaincar.cli run \
  --task mountaincar \
  --methods BO,DR,BO+DR,DORAEMON,BO+DORAEMON \
  --seeds 1 \
  --profile smoke \
  --init-mode scratch
```

HalfCheetah batch mode:

```bash
python -m mountaincar.cli batch-run \
  --task halfcheetah \
  --backend sb3_sac \
  --primary-metric mean_episodic_return \
  --eval-episodes 5 \
  --methods BO,DR \
  --seeds 1,2 \
  --repetitions 2 \
  --profile smoke \
  --init-mode scratch \
  --on-error continue
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
