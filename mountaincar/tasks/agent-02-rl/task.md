# Agent 02 - rl-shared

## Objective
Extract and unify RL shared primitives for DQN training and TinySim evaluation.

## Ownership (ONLY edit these)
- /Users/kevaanbuch/Desktop/Uni/CSC415/mountaincar/rl/agent.py
- /Users/kevaanbuch/Desktop/Uni/CSC415/mountaincar/rl/envs.py
- /Users/kevaanbuch/Desktop/Uni/CSC415/mountaincar/rl/eval.py
- /Users/kevaanbuch/Desktop/Uni/CSC415/mountaincar/rl/checkpoints.py
- /Users/kevaanbuch/Desktop/Uni/CSC415/mountaincar/rl/__init__.py

## Requirements
- Implement reusable DQN stack: ReplayBuffer, QNetwork, DQNConfig, DQNAgent.
- Implement env factories:
  - Gym MountainCar env with custom force/gravity.
  - TinySim target eval helper.
- Implement checkpoint I/O helpers:
  - make_agent_from_hparams
  - load_warmstart_agent
  - save_agent_checkpoint
  - clone_agent
- Keep pure utility-level logic only (no method-specific BO/DR loops).
- Ensure deterministic seed handling helpers are available.

## Coordination rule
You are not alone in the codebase. Do not revert edits made by others. Adjust your implementation to accommodate concurrent changes.

## Output
At completion, list exactly which files you changed and summarize any assumptions.
