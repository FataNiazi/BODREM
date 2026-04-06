from __future__ import annotations

from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mountaincar import cli, config, registry
from mountaincar.types import MethodName, RLBackend, TaskName


def test_method_normalization_handles_aliases():
    assert registry.normalize_method_name("bo") == MethodName.BO
    assert registry.normalize_method_name("BO+DR") == MethodName.BO_DR
    assert registry.normalize_method_name("bo_dr") == MethodName.BO_DR
    assert registry.normalize_method_name("doraemon") == MethodName.DORAEMON
    assert registry.normalize_method_name("bo+doraemon") == MethodName.BO_DORAEMON


def test_cli_parses_methods_and_seeds():
    args = cli.parse_args(
        ["run", "--methods", "bo,dr", "--seeds", "1,2", "--profile", "smoke", "--init-mode", "scratch"]
    )
    assert args.command == "run"
    assert args.methods == ["bo", "dr"]
    assert args.seeds == [1, 2]
    assert args.init_mode == "scratch"


def test_cli_parses_batch_run():
    args = cli.parse_args(
        [
            "batch-run",
            "--methods",
            "bo,doraemon",
            "--seeds",
            "3,4",
            "--repetitions",
            "2",
            "--on-error",
            "fail-fast",
            "--dedupe-seeds",
            "--resume-from",
            "/tmp/demo",
            "--rerun-failed",
            "--init-mode",
            "warmstart",
            "--task",
            "halfcheetah",
            "--backend",
            "sb3_sac",
            "--primary-metric",
            "mean_episodic_return",
            "--eval-episodes",
            "7",
            "--task-config-json",
            "{\"foo\": 1}",
        ]
    )
    assert args.command == "batch-run"
    assert args.methods == ["bo", "doraemon"]
    assert args.seeds == [3, 4]
    assert args.repetitions == 2
    assert args.on_error == "fail-fast"
    assert args.dedupe_seeds is True
    assert args.resume_from == "/tmp/demo"
    assert args.rerun_failed is True
    assert args.init_mode == "warmstart"
    assert args.task == "halfcheetah"
    assert args.backend == "sb3_sac"
    assert args.primary_metric == "mean_episodic_return"
    assert args.eval_episodes == 7
    assert args.task_config == {"foo": 1}


def test_cli_parses_run_task_flags():
    args = cli.parse_args(
        [
            "run",
            "--methods",
            "bo",
            "--seeds",
            "1",
            "--task",
            "halfcheetah",
            "--backend",
            "sb3_sac",
            "--primary-metric",
            "mean_episodic_return",
            "--eval-episodes",
            "9",
            "--task-config-json",
            "{\"policy_kwargs\": {\"net_arch\": [64, 64]}}",
        ]
    )
    assert args.task == "halfcheetah"
    assert args.backend == "sb3_sac"
    assert args.primary_metric == "mean_episodic_return"
    assert args.eval_episodes == 9
    assert args.task_config == {"policy_kwargs": {"net_arch": [64, 64]}}


def test_list_methods_is_stable():
    assert [m.value for m in registry.list_methods()] == ["BO", "DR", "BO+DR", "DORAEMON", "BO+DORAEMON"]


def test_build_suite_config_defaults():
    with tempfile.TemporaryDirectory() as tmp:
        suite_cfg = config.build_suite_config(seeds=[3], methods=["BO", "DR"], output_root=tmp)
        assert suite_cfg.task == TaskName.MOUNTAINCAR
        assert suite_cfg.env_id == config.DEFAULT_MOUNTAINCAR_ENV_ID
        assert suite_cfg.backend == RLBackend.NATIVE_DQN
        assert suite_cfg.primary_metric == "target_solve_rate"
        assert suite_cfg.eval_episodes == config.DEFAULT_EVAL_EPISODES
        assert suite_cfg.profile == "smoke"
        assert suite_cfg.init_mode == "auto"
        assert suite_cfg.methods == [MethodName.BO, MethodName.DR]
        assert suite_cfg.output_root == Path(tmp)
        assert suite_cfg.episode_steps == config.DEFAULT_EPISODE_STEPS


def test_build_suite_config_halfcheetah_defaults_and_validation():
    with tempfile.TemporaryDirectory() as tmp:
        suite_cfg = config.build_suite_config(seeds=[11], methods=["BO"], task="halfcheetah", output_root=tmp)
        assert suite_cfg.task == TaskName.HALFCHEETAH
        assert suite_cfg.env_id == config.DEFAULT_HALFCHEETAH_ENV_ID
        assert suite_cfg.backend == RLBackend.SB3_SAC
        assert suite_cfg.primary_metric == "mean_episodic_return"
        assert suite_cfg.eval_episodes == config.DEFAULT_EVAL_EPISODES

    with tempfile.TemporaryDirectory() as tmp:
        try:
            config.build_suite_config(
                seeds=[1],
                methods=["BO"],
                task="mountaincar",
                backend="sb3_sac",
                output_root=tmp,
            )
            assert False, "Expected MountainCar backend validation to fail"
        except ValueError as exc:
            assert "native_dqn" in str(exc)

        try:
            config.build_suite_config(
                seeds=[1],
                methods=["BO"],
                task="mountaincar",
                primary_metric="mean_episodic_return",
                output_root=tmp,
            )
            assert False, "Expected MountainCar primary metric validation to fail"
        except ValueError as exc:
            assert "target_solve_rate" in str(exc)
