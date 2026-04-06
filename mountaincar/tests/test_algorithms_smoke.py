from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mountaincar import runner
from mountaincar.config import build_method_run_config, build_suite_config
from mountaincar.types import ArtifactPaths, MethodName, MethodResult


@dataclass
class DummyAlgorithm:
    cfg: object

    def run(self, seed: int):
        run_dir = Path(self.cfg.output_root) / f"dummy_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = run_dir / "best.pt"
        checkpoint.write_text("ok", encoding="utf-8")
        return MethodResult(
            method=MethodName.DR,
            seed=int(seed),
            profile="smoke",
            artifacts=ArtifactPaths(
                run_dir=str(run_dir),
                history_json=str(run_dir / "history.json"),
                history_csv=str(run_dir / "history.csv"),
                best_checkpoint=str(checkpoint),
                last_checkpoint=str(checkpoint),
            ),
            history=[{"step": 0, "target_solve_rate": 1.0}],
            best_score=1.0,
            metadata={"seed": int(seed)},
        )


def test_run_method_smoke_with_dummy_algorithm():
    original_loader = runner.load_algorithm_class
    runner.load_algorithm_class = lambda method: DummyAlgorithm
    try:
        with tempfile.TemporaryDirectory() as tmp:
            result = runner.run_method(
                method="DR",
                seed=3,
                config={"methods": ["DR"], "seeds": [3], "profile": "smoke", "output_root": tmp},
            )
            assert result.best_score == 1.0
            assert result.history
            assert Path(result.artifacts.best_checkpoint).exists()
    finally:
        runner.load_algorithm_class = original_loader


def test_halfcheetah_nd_vector_handling_for_legacy_defaults():
    try:
        from mountaincar.algorithms.bo_dr import BODRAlgorithm
        from mountaincar.algorithms.bo_doraemon import BODoraemonAlgorithm
        from mountaincar.algorithms.doraemon import DoraemonAlgorithm
    except Exception:
        return

    class _FakeEnv:
        def __init__(self, seed=None):
            self._step = 0

        def reset(self, seed=None):
            self._step = 0
            return np.zeros(4, dtype=np.float32), {}

        def step(self, action):
            self._step += 1
            done = self._step >= 3
            return np.zeros(4, dtype=np.float32), 1.0, done, False, {}

        def close(self):
            return None

    class _FakeAgent:
        def set_env(self, env):
            self._env = env

        def learn(self, total_timesteps=1, reset_num_timesteps=False, progress_bar=False):
            return self

        def predict(self, obs, deterministic=True):
            return np.zeros(1, dtype=np.float32), None

        def save(self, path):
            Path(path).write_text("fake-agent", encoding="utf-8")

    def _fake_env_factory(seed=None, context=None):
        return _FakeEnv(seed=seed)

    with tempfile.TemporaryDirectory() as tmp:
        suite_cfg = build_suite_config(
            seeds=[1],
            methods=["DORAEMON"],
            task="halfcheetah",
            backend="sb3_sac",
            primary_metric="mean_episodic_return",
            profile="smoke",
            output_root=tmp,
            task_config={
                "agent_factory": lambda **kwargs: _FakeAgent(),
                "train_env_factory": _fake_env_factory,
                "target_env_factory": _fake_env_factory,
                "eval_env_factory": _fake_env_factory,
                "deterministic_eval": True,
                "surrogate_success_threshold": 0.0,
            },
        )

        checks = [
            (MethodName.DORAEMON, DoraemonAlgorithm, {"adapt_iters": 1, "blocks": 1, "episodes_per_block": 1, "B_r": 1}),
            (
                MethodName.BO_DORAEMON,
                BODoraemonAlgorithm,
                {
                    "T": 1,
                    "blocks": 1,
                    "episodes_per_block": 1,
                    "B_r": 1,
                    "bo_init_random_points": 0,
                    "stabilization_blocks": 0,
                    "center_hold_rounds": 1,
                },
            ),
            (MethodName.BO_DR, BODRAlgorithm, {"T": 1, "K": 1, "B_r": 1, "surrogate_eval_episodes": 1}),
        ]

        for method, cls, overrides in checks:
            method_cfg = build_method_run_config(suite_cfg, method)
            algo = cls(method_cfg)
            algo.budget.update(overrides)
            result = algo.run(seed=1)
            assert isinstance(result.best_score, float)
