# BODREM Experiments

This repository contains two top-level experiment areas:

- `manipulation/`
- `mountaincar&cheetah/`

A compatibility symlink `mountaincar -> mountaincar&cheetah` is included, so module commands such as `python -m mountaincar.cli ...` work from repo root.

## Working Directory

Run commands from repo root:

```bash
cd <repo-root>
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

## Manipulation

```bash
python3 manipulation/run_ablations.py
```

`run_ablations.py` executes the experiments declared in its `EXPERIMENTS` dictionary.

## MountainCar + HalfCheetah

See:

- `mountaincar&cheetah/README.md`

Quick check:

```bash
python -m mountaincar.cli list-methods
```
