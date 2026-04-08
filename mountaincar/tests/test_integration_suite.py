from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mountaincar import runner
from mountaincar.types import ArtifactPaths, MethodName, MethodResult


@dataclass
class DummyAlgorithm:
    cfg: object

    def run(self, seed: int):
        method = MethodName.from_any(self.cfg.method)
        run_dir = Path(self.cfg.output_root) / f"{method.value.lower().replace('+', '_')}_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = run_dir / "best.pt"
        checkpoint.write_text(f"{method.value}:{seed}", encoding="utf-8")
        metric_value = float(seed) + 0.25
        return MethodResult(
            method=method,
            seed=int(seed),
            profile=self.cfg.profile,
            artifacts=ArtifactPaths(
                run_dir=str(run_dir),
                history_json=str(run_dir / "history.json"),
                history_csv=str(run_dir / "history.csv"),
                best_checkpoint=str(checkpoint),
                last_checkpoint=str(checkpoint),
            ),
            history=[{"step": 0, "mean_episodic_return": metric_value, "target_solve_rate": metric_value}],
            best_score=metric_value,
            metadata={
                "seed": int(seed),
                "method": method.value,
                "task": getattr(self.cfg, "task", "mountaincar").value if hasattr(getattr(self.cfg, "task", None), "value") else getattr(self.cfg, "task", "mountaincar"),
                "env_id": getattr(self.cfg, "env_id", ""),
                "backend": getattr(self.cfg, "backend", "native_dqn").value if hasattr(getattr(self.cfg, "backend", None), "value") else getattr(self.cfg, "backend", "native_dqn"),
                "primary_metric": getattr(self.cfg, "primary_metric", "target_solve_rate"),
                "eval_episodes": getattr(self.cfg, "eval_episodes", 0),
            },
        )


def test_run_suite_writes_artifacts():
    original_loader = runner.load_algorithm_class
    runner.load_algorithm_class = lambda method: DummyAlgorithm
    try:
        with tempfile.TemporaryDirectory() as tmp:
            result = runner.run_suite(
                {
                    "methods": ["BO", "DR"],
                    "seeds": [1, 2],
                    "task": "halfcheetah",
                    "backend": "sb3_sac",
                    "primary_metric": "mean_episodic_return",
                    "eval_episodes": 4,
                    "profile": "smoke",
                    "output_root": tmp,
                }
            )
            assert Path(result.run_dir).exists()
            assert Path(result.master_csv).exists()
            assert Path(result.master_json).exists()
            assert Path(result.aggregate_csv).exists()
            assert Path(result.manifest_json).exists()
            assert len(result.rows) == 4
            assert {row["method"] for row in result.rows} == {"BO", "DR"}
            sample_row = result.rows[0]
            assert sample_row["task"] == "halfcheetah"
            assert sample_row["env_id"] == "HalfCheetah-v4"
            assert sample_row["backend"] == "sb3_sac"
            assert sample_row["primary_metric"] == "mean_episodic_return"
            assert sample_row["final_primary_metric"] == sample_row["final_target_solve_rate"]
    finally:
        runner.load_algorithm_class = original_loader
