# Agent 04 — Runner/CLI/Artifacts

You are responsible for task-aware orchestration and public entrypoints.

## Ownership
- /Users/kevaanbuch/Desktop/Uni/CSC415/mountaincar/registry.py
- /Users/kevaanbuch/Desktop/Uni/CSC415/mountaincar/runner.py
- /Users/kevaanbuch/Desktop/Uni/CSC415/mountaincar/cli.py
- /Users/kevaanbuch/Desktop/Uni/CSC415/mountaincar/batch_run.py
- /Users/kevaanbuch/Desktop/Uni/CSC415/mountaincar/__init__.py

## Goal
- Add task-aware execution while keeping existing APIs usable.

## Requirements
- Extend CLI with task/env/backend/metric/eval knobs.
- Keep list-methods/run/batch-run working.
- Add task-aware result rows and aggregations.
- Keep legacy MountainCar compatibility columns.
- Ensure manifest schema includes task/backend/metric and hidden target hash metadata.

## Coordination Rule
You are not alone in the codebase; do not revert others' edits; adapt to ongoing parallel changes.
