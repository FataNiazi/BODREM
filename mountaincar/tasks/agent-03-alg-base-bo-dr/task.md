# Agent 03 - alg-base-bo-dr

## Objective
Create the abstract Algorithm superclass and implement BO and DR subclasses.

## Ownership (ONLY edit these)
- /Users/kevaanbuch/Desktop/Uni/CSC415/mountaincar/algorithms/base.py
- /Users/kevaanbuch/Desktop/Uni/CSC415/mountaincar/algorithms/bo.py
- /Users/kevaanbuch/Desktop/Uni/CSC415/mountaincar/algorithms/dr.py
- /Users/kevaanbuch/Desktop/Uni/CSC415/mountaincar/algorithms/__init__.py

## Requirements
- `Algorithm` base should own lifecycle:
  - run dir setup
  - warmstart/fresh init
  - target evaluation
  - history/checkpoint flush
  - best model tracking
- Define template hooks for subclasses.
- Implement:
  - BOAlgorithm (BO over center mu; fixed-center training per iteration)
  - DRAlgorithm (uniform random params, periodic target eval)
- Keep method results in uniform schema.
- Depend only on shared modules from `config/types/rl/utils`.

## Coordination rule
You are not alone in the codebase. Do not revert edits made by others. Adjust your implementation to accommodate concurrent changes.

## Output
At completion, list exactly which files you changed and summarize any assumptions.
