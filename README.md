# BORDEM: Bayesian Optimization Centered Domain Randomization via Entropy Maximization

Final project for **CSC415: Introduction to Reinforcement Learning**, University of Toronto (March 2026).
By Suhaima Ahmad, Fata Asgharzadniazi, and Kevaan Buch.

📄 [Full report](./report.pdf)

## Overview

Domain Randomization (DR) trains RL policies across a distribution of simulated dynamics so they transfer to a real (or held-out target) environment. In practice, the shape of that distribution matters a lot: too narrow and the policy overfits, too wide and training becomes unstable. Prior methods typically optimize either the *center* of the distribution — using Bayesian Optimization (BO) to align training with a target environment — or its *spread* — using entropy maximization for robustness, as in [DORAEMON](https://arxiv.org/abs/2310.09303) — but rarely both at once.

**BORDEM** decomposes the training distribution into these two components and optimizes them jointly:

- **Center** — moved toward the target environment via Bayesian Optimization over sparse target-domain rollouts.
- **Spread** — expanded via entropy maximization in a cheap surrogate simulator, subject to a minimum success-rate constraint.

This separation lets the training distribution stay aligned with the target domain while remaining robust to dynamics it hasn't directly seen.

## Results

We benchmarked BORDEM against BO-only, DR-only, BO+DR, and DORAEMON-only baselines on MountainCar (3 edge-profile variants), HalfCheetah, and a MuJoCo push-block manipulation task.

**Push-block manipulation** — best empirical success rate at target physics parameters (mean ± std, 5 seeds):

| Method     | Easy              | Hard              |
|------------|-------------------|-------------------|
| BO         | 0.195 ± 0.249     | 0.399 ± 0.211     |
| DORAEMON   | 0.423 ± 0.285     | 0.510 ± 0.160     |
| **BORDEM** | **0.800 ± 0.400** | **0.800 ± 0.400** |

BORDEM roughly doubled the strongest baseline on this task. Results on MountainCar and HalfCheetah were more mixed — the full [report](./report.pdf) (Sections 5–6) covers the complete picture, including settings where simpler baselines outperformed the combined method.

## Repository Structure

```
.
├── manipulation/              # Push-block manipulation experiments (MuJoCo + SAC)
│   ├── pipeline.py             # SAC agent, DORAEMON Beta-DR adaptation, BO loop, training entry points
│   └── run_ablations.py        # Ablation runner: BO / DORAEMON / BORDEM across easy/medium/hard difficulty
├── domain_randomization/      # Bayesian domain randomization for MountainCar
│   └── bayesian_dr.py
├── agents/                    # DQN agent, Q-network, replay buffer (MountainCar)
├── scripts/
│   └── train.py                # DQN + Bayesian DR training entry point for MountainCar
├── mountaincar_methods.py     # Unified runner across all 5 methods for MountainCar benchmarking
├── *.ipynb                    # Experiment notebooks (MountainCar sim2sim runs, BO vs. DR comparison, policy demo)
└── report.pdf                 # Full write-up
```

## Setup

```bash
pip install -r requirements.txt
```

The manipulation experiments additionally require `mujoco` and `tinysim_mujoco`, a MuJoCo-based push-block environment supplied as course material for CSC415 (not on PyPI).

## Running Experiments

**Push-block manipulation:**
```bash
python3 manipulation/run_ablations.py
```
Runs every experiment defined in the `EXPERIMENTS` dict in `run_ablations.py` (method × difficulty), prints an episode-budget summary before execution, and writes per-run configs, histories, and progress plots to `runs_manipulation/`.

**MountainCar:**
See `mountaincar_all_methods_benchmark.ipynb` for the benchmarking driver, or `scripts/train.py` for a single DQN + Bayesian-DR training run.

## Team Contributions

- **Fata Asgharzadniazi** — push-block manipulation experiments, GitHub, writing
- **Kevaan Buch** — MountainCar and HalfCheetah experiments, GitHub, writing
- **Suhaima Ahmad** — related work, writing, proofreading

---

*This was completed as a course final project and is not a peer-reviewed publication. The report uses the ICLR LaTeX template per the course's formatting requirements.*
