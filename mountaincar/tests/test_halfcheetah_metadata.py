from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mountaincar.algorithms.base import Algorithm
from mountaincar.config import build_method_run_config, build_suite_config
from mountaincar.registry import list_methods
from mountaincar.rl.target_context import make_halfcheetah_target_manager
from mountaincar.types import TaskName


@dataclass
class DummyHalfCheetahAgent:
    def save(self, path: str) -> None:
        Path(path).write_text("dummy-halfcheetah-agent", encoding="utf-8")


class ProbeAlgorithm(Algorithm):
    def execute(self) -> None:
        return


def _make_suite_config(tmp: str, seed: int = 7):
    return build_suite_config(
        seeds=[seed],
        methods=["BO"],
        task=TaskName.HALFCHEETAH,
        init_mode="scratch",
        output_root=tmp,
        task_config={
            "agent_factory": lambda **kwargs: DummyHalfCheetahAgent(),
            "target_manager": make_halfcheetah_target_manager(base_seed=seed),
        },
    )


def test_halfcheetah_hidden_target_redaction_and_hash_metadata():
    with tempfile.TemporaryDirectory() as tmp:
        for method in list_methods():
            suite_cfg = _make_suite_config(tmp, seed=7)
            method_cfg = build_method_run_config(suite_cfg, method)
            algo = ProbeAlgorithm(method_cfg)
            result = algo.run(seed=7)

            manifest_path = Path(result.artifacts.run_dir) / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

            assert manifest["task"] == "halfcheetah"
            assert manifest["target_params"] == {}
            assert manifest["target_public_metadata"]["task"] == "halfcheetah"
            assert manifest["target_public_metadata"]["env_id"] == "HalfCheetah-v4"
            assert manifest["target_public_metadata"]["target_family"] == "halfcheetah_hidden_target_v1"
            assert manifest["target_public_metadata"]["hidden_target"] is True
            assert len(str(manifest["target_public_metadata"]["target_hash"])) == 64
            assert set(manifest["target_public_metadata"].keys()) == {
                "task",
                "env_id",
                "target_family",
                "target_hash",
                "hidden_target",
            }

            second_cfg = build_method_run_config(_make_suite_config(tmp, seed=7), method)
            second_result = ProbeAlgorithm(second_cfg).run(seed=7)
            second_manifest = json.loads((Path(second_result.artifacts.run_dir) / "manifest.json").read_text(encoding="utf-8"))
            assert second_manifest["target_public_metadata"]["target_hash"] == manifest["target_public_metadata"]["target_hash"]
