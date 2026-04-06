# Agent 01 — Core Multitask Interfaces

You are responsible for core typed interfaces and config defaults for multi-task support.

## Ownership
- /Users/kevaanbuch/Desktop/Uni/CSC415/mountaincar/types.py
- /Users/kevaanbuch/Desktop/Uni/CSC415/mountaincar/config.py
- /Users/kevaanbuch/Desktop/Uni/CSC415/mountaincar/utils/io.py (only if schema/version helper constants needed)

## Goal
- Add task/backend abstractions while preserving MountainCar backward compatibility.
- Add task-aware config fields needed by HalfCheetah-v4 + SB3 SAC.

## Requirements
- Introduce task/backend enums and typed aliases.
- Extend SuiteConfig/MethodRunConfig with:
  - task
  - env_id
  - backend
  - primary_metric
  - eval_episodes
  - task_config (dict for task-specific settings)
- Preserve existing fields and behavior for current tests and MountainCar defaults.
- Add validation/normalization helpers in config.py.
- Add sane defaults:
  - MountainCar task default + current behavior
  - HalfCheetah default env_id = HalfCheetah-v4
  - backend default for HalfCheetah = sb3_sac
  - primary metric default for HalfCheetah = mean_episodic_return

## Coordination Rule
You are not alone in the codebase; do not revert others' edits; adapt to ongoing parallel changes.
