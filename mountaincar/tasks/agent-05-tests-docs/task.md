# Agent 05 — Tests and Docs

You are responsible for tests and documentation updates for the multitask extension.

## Ownership
- /Users/kevaanbuch/Desktop/Uni/CSC415/mountaincar/tests/
- /Users/kevaanbuch/Desktop/Uni/CSC415/mountaincar/README.md

## Goal
- Add coverage for new task/backend interfaces and keep existing coverage intact.

## Requirements
- Add unit tests for config defaults and validation for task/backend/metric fields.
- Add tests for halfcheetah param mapping/clipping and hidden target redaction.
- Add smoke tests for run_method across MountainCar + HalfCheetah using dummy adapters/backend stubs.
- Keep integration tests stable; mark Mujoco-dependent tests as optional/skipped when deps missing.
- Update README usage examples for HalfCheetah-v4 + SB3 SAC.

## Coordination Rule
You are not alone in the codebase; do not revert others' edits; adapt to ongoing parallel changes.
