from __future__ import annotations

from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mountaincar import registry
from mountaincar.config import build_method_run_config, build_suite_config


def _make_probe_algorithm_class():
    try:
        from mountaincar.algorithms.base import Algorithm
    except Exception:
        return None

    class InitProbeAlgorithm(Algorithm):
        def execute(self) -> None:
            return

    return InitProbeAlgorithm


def _patch_agent_loaders():
    try:
        import mountaincar.algorithms.base as base_module
    except Exception:
        return None

    original_make = base_module.make_agent_from_hparams
    original_load = base_module.load_warmstart_agent

    calls = {"make": 0, "load": 0}

    def fake_make(hparams, device="cpu"):
        calls["make"] += 1
        return {"kind": "scratch", "device": device, "hparams": hparams}

    def fake_load(checkpoint_path, device="cpu"):
        calls["load"] += 1
        return {"kind": "warmstart", "device": device, "checkpoint_path": str(checkpoint_path)}

    base_module.make_agent_from_hparams = fake_make
    base_module.load_warmstart_agent = fake_load
    return base_module, original_make, original_load, calls


def test_all_methods_support_scratch_init_mode():
    patched = _patch_agent_loaders()
    InitProbeAlgorithm = _make_probe_algorithm_class()
    if patched is None or InitProbeAlgorithm is None:
        return

    base_module, original_make, original_load, calls = patched
    try:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "warm.pt"
            checkpoint.write_text("ok", encoding="utf-8")

            for method in registry.list_methods():
                suite_cfg = build_suite_config(
                    seeds=[1],
                    methods=[method.value],
                    output_root=tmp,
                    init_mode="scratch",
                    checkpoint_path=str(checkpoint),
                )
                method_cfg = build_method_run_config(suite_cfg, method)
                algo = InitProbeAlgorithm(method_cfg)
                _, used_warmstart = algo._init_agent(seed=1)
                assert used_warmstart is False

            assert calls["make"] == len(registry.list_methods())
            assert calls["load"] == 0
    finally:
        base_module.make_agent_from_hparams = original_make
        base_module.load_warmstart_agent = original_load


def test_all_methods_support_warmstart_init_mode_when_checkpoint_exists():
    patched = _patch_agent_loaders()
    InitProbeAlgorithm = _make_probe_algorithm_class()
    if patched is None or InitProbeAlgorithm is None:
        return

    base_module, original_make, original_load, calls = patched
    try:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "warm.pt"
            checkpoint.write_text("ok", encoding="utf-8")

            for method in registry.list_methods():
                suite_cfg = build_suite_config(
                    seeds=[1],
                    methods=[method.value],
                    output_root=tmp,
                    init_mode="warmstart",
                    checkpoint_path=str(checkpoint),
                )
                method_cfg = build_method_run_config(suite_cfg, method)
                algo = InitProbeAlgorithm(method_cfg)
                _, used_warmstart = algo._init_agent(seed=1)
                assert used_warmstart is True

            assert calls["load"] == len(registry.list_methods())
            assert calls["make"] == 0
    finally:
        base_module.make_agent_from_hparams = original_make
        base_module.load_warmstart_agent = original_load


def test_warmstart_mode_requires_existing_checkpoint_path():
    InitProbeAlgorithm = _make_probe_algorithm_class()
    if InitProbeAlgorithm is None:
        return

    with tempfile.TemporaryDirectory() as tmp:
        for method in registry.list_methods():
            suite_cfg = build_suite_config(
                seeds=[1],
                methods=[method.value],
                output_root=tmp,
                init_mode="warmstart",
                checkpoint_path=str(Path(tmp) / "missing.pt"),
            )
            method_cfg = build_method_run_config(suite_cfg, method)
            algo = InitProbeAlgorithm(method_cfg)

            try:
                algo._init_agent(seed=1)
                assert False, "Expected ValueError for missing warmstart checkpoint"
            except ValueError as exc:
                assert "warmstart" in str(exc)


def test_bo_doraemon_no_direct_loader_bypass_in_algorithm_file():
    source = Path(__file__).resolve().parents[1] / "algorithms" / "bo_doraemon.py"
    content = source.read_text(encoding="utf-8")
    assert "load_warmstart_agent(" not in content
