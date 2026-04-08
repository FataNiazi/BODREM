# BODREM Experiments

This repository now contains two top-level experiment areas:

- `manipulation/`
- `mountaincar&cheetah/`

A compatibility symlink `mountaincar -> mountaincar&cheetah` is provided so module commands such as `python -m mountaincar.cli ...` keep working.

## Setup

From repo root:

```bash
cd /Users/kevaanbuch/Desktop/Uni/CSC415/BODREM
./"mountaincar&cheetah"/scripts/setup_env.sh
source ./"mountaincar&cheetah"/.venv/bin/activate
```

If TinySim cannot be installed from pip:

```bash
./"mountaincar&cheetah"/scripts/setup_env.sh --tinysim-path ../TinySim
source ./"mountaincar&cheetah"/.venv/bin/activate
```

## Manipulation

Run the manipulation ablations from repo root:

```bash
python3 manipulation/run_ablations.py
```

`run_ablations.py` executes the experiments defined in its `EXPERIMENTS` dictionary.

## MountainCar + HalfCheetah

See:

- `mountaincar&cheetah/README.md`

Quick check:

```bash
python -m mountaincar.cli list-methods
```
