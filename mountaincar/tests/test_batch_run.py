from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mountaincar import runner
from mountaincar.types import ArtifactPaths, MethodName, MethodResult


@dataclass
class StableDummyAlgorithm:
    cfg: object

    def run(self, seed: int):
        method = MethodName.from_any(self.cfg.method)
        run_dir = Path(self.cfg.output_root) / f"{method.value.lower().replace('+', '_')}_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = run_dir / "best.pt"
        checkpoint.write_text(f"{method.value}:{seed}", encoding="utf-8")
        metric_value = float(seed) + 0.5
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
                "target_family": "halfcheetah_hidden_target_v1",
                "target_hash": "abc123",
                "hidden_target": True,
            },
        )


@dataclass
class FailSomeDummyAlgorithm:
    cfg: object

    def run(self, seed: int):
        method = MethodName.from_any(self.cfg.method)
        if method == MethodName.DR and int(seed) == 1:
            raise RuntimeError("intentional failure")
        return StableDummyAlgorithm(self.cfg).run(seed)


def test_run_batch_writes_artifacts():
    original_loader = runner.load_algorithm_class
    runner.load_algorithm_class = lambda method: StableDummyAlgorithm
    try:
        with tempfile.TemporaryDirectory() as tmp:
            result = runner.run_batch(
                {
                    "methods": ["BO", "DR"],
                    "seeds": [1, 2],
                    "task": "halfcheetah",
                    "backend": "sb3_sac",
                    "primary_metric": "mean_episodic_return",
                    "eval_episodes": 3,
                    "profile": "smoke",
                    "output_root": tmp,
                },
                repetitions=2,
            )
            assert Path(result.run_dir).exists()
            assert Path(result.attempts_csv).exists()
            assert Path(result.attempts_json).exists()
            assert Path(result.aggregate_by_method_csv).exists()
            assert Path(result.aggregate_by_method_seed_csv).exists()
            assert Path(result.manifest_json).exists()
            assert result.total_tasks == 8
            assert len(result.rows) == 8
            assert result.succeeded == 8
            assert result.failed == 0
            sample_row = result.rows[0]
            assert sample_row["task"] == "halfcheetah"
            assert sample_row["env_id"] == "HalfCheetah-v4"
            assert sample_row["backend"] == "sb3_sac"
            assert sample_row["primary_metric"] == "mean_episodic_return"
            assert "final_primary_metric" in sample_row
            assert sample_row["final_primary_metric"] == sample_row["final_target_solve_rate"]
            manifest = json.loads(Path(result.manifest_json).read_text(encoding="utf-8"))
            assert manifest["task"] == "halfcheetah"
            assert manifest["backend"] == "sb3_sac"
            assert manifest["primary_metric"] == "mean_episodic_return"
    finally:
        runner.load_algorithm_class = original_loader


def test_run_batch_duplicate_seeds_keep_distinct_seed_index():
    original_loader = runner.load_algorithm_class
    runner.load_algorithm_class = lambda method: StableDummyAlgorithm
    try:
        with tempfile.TemporaryDirectory() as tmp:
            result = runner.run_batch(
                {
                    "methods": ["BO"],
                    "seeds": [1, 1],
                    "profile": "smoke",
                    "output_root": tmp,
                },
                repetitions=2,
            )
            assert len(result.rows) == 4
            pairs = {(int(r["seed_index"]), int(r["repetition"])) for r in result.rows}
            assert len(pairs) == 4
            assert {int(r["seed_index"]) for r in result.rows} == {0, 1}
    finally:
        runner.load_algorithm_class = original_loader


def test_run_batch_continue_on_error_records_failure_and_continues():
    original_loader = runner.load_algorithm_class
    runner.load_algorithm_class = lambda method: FailSomeDummyAlgorithm
    try:
        with tempfile.TemporaryDirectory() as tmp:
            result = runner.run_batch(
                {
                    "methods": ["BO", "DR"],
                    "seeds": [1, 2],
                    "profile": "smoke",
                    "output_root": tmp,
                },
                repetitions=1,
                on_error="continue",
            )
            assert len(result.rows) == 4
            failed = [r for r in result.rows if r["status"] == "failed"]
            assert len(failed) == 1
            assert failed[0]["method"] == "DR"
            assert int(failed[0]["seed"]) == 1
            assert result.succeeded == 3
            assert result.failed == 1
    finally:
        runner.load_algorithm_class = original_loader


def test_run_batch_resume_skips_completed_and_failed_unless_rerun_enabled():
    original_loader = runner.load_algorithm_class
    try:
        with tempfile.TemporaryDirectory() as tmp:
            runner.load_algorithm_class = lambda method: FailSomeDummyAlgorithm
            first = runner.run_batch(
                {
                    "methods": ["BO", "DR"],
                    "seeds": [1, 2],
                    "profile": "smoke",
                    "output_root": tmp,
                },
                repetitions=1,
                on_error="fail-fast",
            )
            assert len(first.rows) == 2

            runner.load_algorithm_class = lambda method: StableDummyAlgorithm
            resumed = runner.run_batch(
                {
                    "methods": ["BO", "DR"],
                    "seeds": [1, 2],
                    "profile": "smoke",
                    "output_root": tmp,
                },
                repetitions=1,
                resume_from=first.run_dir,
                rerun_failed=False,
                on_error="continue",
            )

            assert len(resumed.rows) == 4
            by_task: dict[str, list[dict[str, object]]] = {}
            for row in resumed.rows:
                by_task.setdefault(str(row["task_id"]), []).append(row)

            assert len(by_task["BO|seed=1|idx=0|rep=0"]) == 1
            assert len(by_task["DR|seed=1|idx=0|rep=0"]) == 1
            assert len(by_task["BO|seed=2|idx=1|rep=0"]) == 1
            assert len(by_task["DR|seed=2|idx=1|rep=0"]) == 1
            assert resumed.succeeded == 3
            assert resumed.failed == 1
    finally:
        runner.load_algorithm_class = original_loader
