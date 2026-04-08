# BODREM Experiments

This repository contains two top-level experiment areas:

- `manipulation/`
- `mountaincar/`

## Working Directory

Run commands from repo root:

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"
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

- `mountaincar/README.md`

Quick check:

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
(cd "$REPO_ROOT" && "$REPO_ROOT/.venv/bin/python" -m mountaincar.cli list-methods)
```
