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

## Paths and Working Directory

Repository structure is now:

- `root/manipulation`
- `root/mountaincar&cheetah`

Run commands from repo root:

```bash
cd /Users/kevaanbuch/Desktop/Uni/CSC415/BODREM
```

Compatibility note: `mountaincar` is an alias to `mountaincar&cheetah`, so module commands use `python -m mountaincar...`.

## Setup

Create env and install dependencies:

```bash
./"mountaincar&cheetah"/scripts/setup_env.sh
source ./"mountaincar&cheetah"/.venv/bin/activate
```

If TinySim must come from local source:

```bash
./"mountaincar&cheetah"/scripts/setup_env.sh --tinysim-path ../TinySim
source ./"mountaincar&cheetah"/.venv/bin/activate
```

List available methods:

```bash
python -m mountaincar.cli list-methods
```

## Smoke Test + Plots (2 Algos, 2 Seeds)

Run README smoke validation for HalfCheetah (`DR` + `DORAEMON`, seeds `1,2`):

```bash
RUN_DIR=$(python -m mountaincar.cli run \
  --task halfcheetah \
  --backend sb3_sac \
  --primary-metric mean_episodic_return \
  --eval-episodes 5 \
  --methods DR,DORAEMON \
  --seeds 1,2 \
  --profile smoke \
  --init-mode scratch \
  --task-config-json '{"surrogate_success_threshold": 0.0}')

echo "$RUN_DIR"
```

Generate plots from that smoke run:

```bash
python -m mountaincar.plot_results \
  --input "$RUN_DIR" \
  --output './mountaincar&cheetah/runs/plots/halfcheetah_smoke_dr_doraemon_2x2' \
  --task halfcheetah
```

## Test-Everything Commands

### 1) Standard Combined Suite (all methods, all seeds in one command)

```bash
python -m mountaincar.cli run \
  --task halfcheetah \
  --backend sb3_sac \
  --primary-metric mean_episodic_return \
  --eval-episodes 5 \
  --methods BO,DR,BO+DR,DORAEMON,BO+DORAEMON \
  --seeds 1,2,3,4,5 \
  --profile smoke \
  --init-mode scratch \
  --task-config-json '{"surrogate_success_threshold": 0.0}'
```

### 2) Sequential by Algorithm (run all seeds for one algo, then next algo)

```bash
for method in BO DR BO+DR DORAEMON BO+DORAEMON; do
  python -m mountaincar.cli run \
    --task halfcheetah \
    --backend sb3_sac \
    --primary-metric mean_episodic_return \
    --eval-episodes 5 \
    --methods "$method" \
    --seeds 1,2,3,4,5 \
    --profile smoke \
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

## Notes

- HalfCheetah requires `stable-baselines3` and MuJoCo support (`mujoco` / Gymnasium MuJoCo stack).
- `BO`, `BO+DR`, and `BO+DORAEMON` require `scikit-optimize`.
- DORAEMON variants require `scipy`.
