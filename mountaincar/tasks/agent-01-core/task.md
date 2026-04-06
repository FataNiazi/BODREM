# Agent 01 - core-interfaces

## Objective
Define shared core interfaces and configuration for the rewritten modular MountainCar package.

## Ownership (ONLY edit these)
- /Users/kevaanbuch/Desktop/Uni/CSC415/mountaincar/types.py
- /Users/kevaanbuch/Desktop/Uni/CSC415/mountaincar/config.py
- /Users/kevaanbuch/Desktop/Uni/CSC415/mountaincar/utils/io.py
- /Users/kevaanbuch/Desktop/Uni/CSC415/mountaincar/utils/__init__.py

## Requirements
- Python 3.11+ type hints.
- Implement method enum/name aliases for: BO, DR, BO+DR, DORAEMON, BO+DORAEMON.
- Define dataclasses / TypedDicts for:
  - RunConfig pieces (suite config, method config, profile budgets).
  - MethodResult, SeedResult, SuiteResult.
  - Uniform artifact schema (run_dir, history, checkpoints, metadata).
- Include profile presets for smoke/prototype/edge with consistent method budgets.
- Provide helper in io.py to create timestamped run directories and JSON/CSV writes.
- Keep code minimal, documented, and import-safe.

## Coordination rule
You are not alone in the codebase. Do not revert edits made by others. Adjust your implementation to accommodate concurrent changes.

## Output
At completion, list exactly which files you changed and summarize any assumptions.
