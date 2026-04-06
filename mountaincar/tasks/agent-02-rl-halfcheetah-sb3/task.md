# Agent 02 — RL HalfCheetah + SB3

You are responsible for RL/env extension modules for HalfCheetah-v4 and hidden target lifecycle.

## Ownership
- /Users/kevaanbuch/Desktop/Uni/CSC415/mountaincar/rl/halfcheetah_params.py
- /Users/kevaanbuch/Desktop/Uni/CSC415/mountaincar/rl/halfcheetah_env.py
- /Users/kevaanbuch/Desktop/Uni/CSC415/mountaincar/rl/target_context.py
- /Users/kevaanbuch/Desktop/Uni/CSC415/mountaincar/rl/envs.py
- /Users/kevaanbuch/Desktop/Uni/CSC415/mountaincar/rl/eval.py
- /Users/kevaanbuch/Desktop/Uni/CSC415/mountaincar/rl/checkpoints.py
- /Users/kevaanbuch/Desktop/Uni/CSC415/mountaincar/rl/__init__.py

## Goal
- Add same-env surrogate/target evaluation pipeline for HalfCheetah-v4 with mean episodic return.
- Support SB3 SAC backend with optional dependency handling.

## Requirements
- Implement 7D parameter schema and bounds:
  - back_thigh_mass [0.08, 2.99]
  - back_shin_mass [0.08, 3.08]
  - back_foot_mass [0.05, 2.08]
  - front_thigh_mass [0.07, 2.78]
  - front_shin_mass [0.06, 2.30]
  - front_foot_mass [0.04, 1.66]
  - surface_friction [0.02, 0.78]
- Map params to MuJoCo bodies/geoms (bthigh/bshin/bfoot/fthigh/fshin/ffoot/floor).
- Add hidden-target manager that samples target once per run and provides hashed public metadata.
- Add generic evaluate_mean_return helper.
- Keep MountainCar helpers intact.
- Add backend/env metadata fields in checkpoint payload utilities where appropriate.

## Coordination Rule
You are not alone in the codebase; do not revert others' edits; adapt to ongoing parallel changes.
