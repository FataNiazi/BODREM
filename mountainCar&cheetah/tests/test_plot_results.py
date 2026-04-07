from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mountaincar.plot_results import (  # noqa: E402
    RunSeries,
    _set_metric_ylim,
    discover_run_directories,
    generate_param_pairs,
    generate_plots,
    load_run_series,
    summarize_by_method,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def _create_run(
    run_dir: Path,
    *,
    manifest: dict[str, object],
    history_rows: list[dict[str, object]],
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "manifest.json", manifest)
    _write_csv(run_dir / "history.csv", history_rows)


def test_discover_from_manifest_and_directory_walk() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        suite = root / "suite"

        run1 = suite / "bo_seed1_20260101_000001"
        _create_run(
            run1,
            manifest={"run_dir": str(run1), "task": "mountaincar", "env_id": "MountainCar-v0", "seed": 1},
            history_rows=[
                {
                    "method": "BO",
                    "step": 0,
                    "mu_force": 0.001,
                    "mu_gravity": 0.003,
                    "target_solve_rate": 0.1,
                }
            ],
        )

        master = suite / "master_results.json"
        _write_json(master, [{"method": "BO", "run_dir": str(run1)}])
        _write_json(suite / "manifest.json", {"master_json": str(master), "run_dir": str(suite)})

        run_dirs = discover_run_directories(suite / "manifest.json")
        assert [p.resolve() for p in run_dirs] == [run1.resolve()]

        run2 = root / "nested" / "dr_seed2_20260101_000002"
        _create_run(
            run2,
            manifest={"run_dir": str(run2), "task": "mountaincar", "env_id": "MountainCar-v0", "seed": 2},
            history_rows=[{"method": "DR", "step": 0, "force": 0.001, "gravity": 0.003, "target_solve_rate": 0.2}],
        )

        scanned = discover_run_directories(root)
        assert run1.resolve() in scanned
        assert run2.resolve() in scanned


def test_load_run_series_normalizes_mountaincar_and_halfcheetah() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        mc_run = root / "bo_doraemon_seed3_20260101_010101"
        _create_run(
            mc_run,
            manifest={
                "run_dir": str(mc_run),
                "task": "mountaincar",
                "env_id": "MountainCar-v0",
                "primary_metric": "target_solve_rate",
                "seed": 3,
            },
            history_rows=[
                {
                    "method": "BO+DORAEMON",
                    "step": 0,
                    "mu_force": 0.001,
                    "mu_gravity": 0.003,
                    "conc_end_force": 18.0,
                    "conc_end_gravity": 18.0,
                    "g_last": 0.4,
                    "target_solve_rate": 0.1,
                }
            ],
        )

        hc_run = root / "doraemon_seed4_20260101_020202"
        _create_run(
            hc_run,
            manifest={
                "run_dir": str(hc_run),
                "task": "halfcheetah",
                "env_id": "HalfCheetah-v4",
                "primary_metric": "mean_episodic_return",
                "seed": 4,
            },
            history_rows=[
                {
                    "method": "DORAEMON",
                    "step": 0,
                    "center_back_thigh_mass": 1.0,
                    "center_back_shin_mass": 1.1,
                    "center_back_foot_mass": 1.2,
                    "center_front_thigh_mass": 1.3,
                    "center_front_shin_mass": 1.4,
                    "center_front_foot_mass": 1.5,
                    "center_surface_friction": 0.5,
                    "conc_end_back_thigh_mass": 18.0,
                    "conc_end_back_shin_mass": 18.0,
                    "conc_end_back_foot_mass": 18.0,
                    "conc_end_front_thigh_mass": 18.0,
                    "conc_end_front_shin_mass": 18.0,
                    "conc_end_front_foot_mass": 18.0,
                    "conc_end_surface_friction": 18.0,
                    "g_last": 0.7,
                    "mean_episodic_return": 10.0,
                }
            ],
        )

        series = load_run_series(root)
        assert len(series) == 2

        by_task = {s.task: s for s in series}
        mc = by_task["mountaincar"]
        hc = by_task["halfcheetah"]

        assert mc.method == "BO+DORAEMON"
        assert mc.primary_metric_name == "target_solve_rate"
        assert mc.center_names == ["force", "gravity"]
        assert mc.spread_names == ["force", "gravity"]

        assert hc.method == "DORAEMON"
        assert hc.primary_metric_name == "mean_episodic_return"
        assert len(hc.center_names) == 7
        assert len(hc.spread_names) == 7


def test_summarize_by_method_mean_std_with_uneven_steps() -> None:
    run_a = RunSeries(
        method="BO",
        method_slug="bo",
        variant="default",
        task="mountaincar",
        env_id="MountainCar-v0",
        seed=1,
        run_dir="/tmp/a",
        primary_metric_name="target_solve_rate",
        step=np.array([0, 1, 2], dtype=int),
        primary_metric=np.array([1.0, 2.0, 3.0]),
        g_last=np.array([0.2, 0.5, 0.8]),
        center_names=["force", "gravity"],
        center=np.array([[1.0, 2.0], [1.1, 2.1], [1.2, 2.2]]),
        spread_kind=None,
        spread_names=[],
        spread=np.zeros((3, 0), dtype=float),
        episode_x=np.array([0.0, 360.0, 720.0]),
        budget=720.0,
    )
    run_b = RunSeries(
        method="BO",
        method_slug="bo",
        variant="default",
        task="mountaincar",
        env_id="MountainCar-v0",
        seed=2,
        run_dir="/tmp/b",
        primary_metric_name="target_solve_rate",
        step=np.array([0, 2], dtype=int),
        primary_metric=np.array([3.0, 5.0]),
        g_last=np.array([0.4, 1.0]),
        center_names=["force", "gravity"],
        center=np.array([[1.4, 2.4], [1.6, 2.6]]),
        spread_kind=None,
        spread_names=[],
        spread=np.zeros((2, 0), dtype=float),
        episode_x=np.array([0.0, 720.0]),
        budget=720.0,
    )

    summaries = summarize_by_method([run_a, run_b])
    key = next(k for k in summaries if k.startswith("BO__"))
    summary = summaries[key]
    assert summary.metric_steps.tolist() == [0, 1, 2]
    assert np.allclose(summary.metric_mean, np.array([2.0, 2.0, 4.0]))
    assert np.allclose(summary.metric_std, np.array([1.0, 0.0, 1.0]))


def test_generate_param_pairs_for_seven_dims() -> None:
    names = [
        "back_thigh_mass",
        "back_shin_mass",
        "back_foot_mass",
        "front_thigh_mass",
        "front_shin_mass",
        "front_foot_mass",
        "surface_friction",
    ]
    pairs = generate_param_pairs(names)
    assert len(pairs) == 21
    assert len(set(pairs)) == 21


def test_variant_inference_from_run_directory_name() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        run = root / "bo_v2_seed7_20260101_010101"
        _create_run(
            run,
            manifest={
                "run_dir": str(run),
                "task": "mountaincar",
                "env_id": "MountainCar-v0",
                "seed": 7,
            },
            history_rows=[
                {"method": "BO", "step": 0, "mu_force": 0.001, "mu_gravity": 0.003, "target_solve_rate": 0.2}
            ],
        )

        series = load_run_series(root)
        assert len(series) == 1
        assert series[0].variant == "v2"


def test_manifest_variant_is_not_collapsed() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        run = root / "bo_seed7_20260101_010101"
        _create_run(
            run,
            manifest={
                "run_dir": str(run),
                "task": "mountaincar",
                "env_id": "MountainCar-v0",
                "variant": "v2_baseline",
                "seed": 7,
            },
            history_rows=[
                {"method": "BO", "step": 0, "mu_force": 0.001, "mu_gravity": 0.003, "target_solve_rate": 0.2}
            ],
        )

        series = load_run_series(root)
        assert len(series) == 1
        assert series[0].variant == "v2_baseline"


def test_set_metric_ylim_constant_series_respects_task_cap() -> None:
    fig, ax = plt.subplots(figsize=(4, 3))
    try:
        _set_metric_ylim(ax, np.array([0.2, 0.2, 0.2], dtype=float), task="mountaincar")
        _, ymax_mc = ax.get_ylim()
        assert np.isclose(float(ymax_mc), 1.0)

        _set_metric_ylim(ax, np.array([320.0, 320.0], dtype=float), task="halfcheetah")
        _, ymax_hc = ax.get_ylim()
        assert np.isclose(float(ymax_hc), 5000.0)
    finally:
        plt.close(fig)


def test_generate_plots_smoke_outputs_manifest_and_files() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        run1 = root / "bo_seed1_20260101_010101"
        run2 = root / "bo_seed2_20260101_010102"
        run3 = root / "doraemon_seed3_20260101_020201"

        _create_run(
            run1,
            manifest={"run_dir": str(run1), "task": "mountaincar", "env_id": "MountainCar-v0", "seed": 1},
            history_rows=[
                {"method": "BO", "step": 0, "mu_force": 0.9, "mu_gravity": 1.0, "target_solve_rate": 0.2},
                {"method": "BO", "step": 1, "mu_force": 1.0, "mu_gravity": 1.1, "target_solve_rate": 0.4},
            ],
        )
        _create_run(
            run2,
            manifest={"run_dir": str(run2), "task": "mountaincar", "env_id": "MountainCar-v0", "seed": 2},
            history_rows=[
                {"method": "BO", "step": 0, "mu_force": 1.1, "mu_gravity": 1.2, "target_solve_rate": 0.3},
                {"method": "BO", "step": 1, "mu_force": 1.2, "mu_gravity": 1.3, "target_solve_rate": 0.5},
            ],
        )
        _create_run(
            run3,
            manifest={
                "run_dir": str(run3),
                "task": "halfcheetah",
                "env_id": "HalfCheetah-v4",
                "primary_metric": "mean_episodic_return",
                "seed": 3,
            },
            history_rows=[
                {
                    "method": "DORAEMON",
                    "step": 0,
                    "center_back_thigh_mass": 1.0,
                    "center_back_shin_mass": 1.1,
                    "center_back_foot_mass": 1.2,
                    "center_front_thigh_mass": 1.3,
                    "center_front_shin_mass": 1.4,
                    "center_front_foot_mass": 1.5,
                    "center_surface_friction": 0.5,
                    "mean_episodic_return": 10.0,
                },
                {
                    "method": "DORAEMON",
                    "step": 1,
                    "center_back_thigh_mass": 1.1,
                    "center_back_shin_mass": 1.2,
                    "center_back_foot_mass": 1.3,
                    "center_front_thigh_mass": 1.4,
                    "center_front_shin_mass": 1.5,
                    "center_front_foot_mass": 1.6,
                    "center_surface_friction": 0.55,
                    "mean_episodic_return": 12.0,
                },
            ],
        )

        out = root / "plots"
        manifest = generate_plots(input_path=root, output_dir=out)

        manifest_path = out / "plot_manifest.json"
        assert manifest_path.exists()
        assert manifest["generated_file_count"] > 0
        assert manifest["generated_file_count"] == len(manifest["generated_files"])
        for file_path in manifest["generated_files"]:
            assert Path(file_path).exists()

        assert any("center_pairs_page_01" in Path(p).name for p in manifest["generated_files"])
