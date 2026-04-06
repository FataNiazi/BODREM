# Agent 03 — Algorithm ND Generalization

You are responsible for algorithm-side multi-task/ND context support.

## Ownership
- /Users/kevaanbuch/Desktop/Uni/CSC415/mountaincar/algorithms/base.py
- /Users/kevaanbuch/Desktop/Uni/CSC415/mountaincar/algorithms/bo.py
- /Users/kevaanbuch/Desktop/Uni/CSC415/mountaincar/algorithms/dr.py
- /Users/kevaanbuch/Desktop/Uni/CSC415/mountaincar/algorithms/bo_dr.py
- /Users/kevaanbuch/Desktop/Uni/CSC415/mountaincar/algorithms/doraemon.py
- /Users/kevaanbuch/Desktop/Uni/CSC415/mountaincar/algorithms/bo_doraemon.py
- /Users/kevaanbuch/Desktop/Uni/CSC415/mountaincar/algorithms/doraemon_math.py
- Optional new folder for task-specific algorithm implementations if needed.

## Goal
- Keep all 5 methods, add HalfCheetah support, and preserve MountainCar behavior.
- Make method execution rely on task-aware adapter hooks and score interface.

## Requirements
- Base lifecycle remains owner of I/O, checkpoints, history, manifests.
- Use task-aware scoring; for HalfCheetah primary score is mean episodic return.
- For artifacts, do not expose raw hidden target params.
- Ensure methods can run under both tasks.
- Preserve existing MountainCar output compatibility fields where possible.

## Coordination Rule
You are not alone in the codebase; do not revert others' edits; adapt to ongoing parallel changes.
