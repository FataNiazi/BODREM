from __future__ import annotations

import argparse
from pathlib import Path

from mountaincar.config import build_suite_config
from mountaincar.runner import run_suite
from mountaincar.types import MethodName

MOUNTAINCAR_SWEEP = {
    MethodName.BO: [
        ("v1_baseline", {"bo_iters": 20, "bo_train_episodes_per_iter": 180}),  # 3600
        ("v2_more_outer", {"bo_iters": 30, "bo_train_episodes_per_iter": 120}),  # 3600
        ("v3_fewer_outer", {"bo_iters": 10, "bo_train_episodes_per_iter": 360}),  # 3600
    ],
    MethodName.DR: [
        ("v1_baseline", {"dr_total_episodes": 3600, "dr_eval_every": 180}),
        ("v2_more_evals", {"dr_total_episodes": 3600, "dr_eval_every": 120}),
        ("v3_fewer_evals", {"dr_total_episodes": 3600, "dr_eval_every": 360}),
    ],
    MethodName.BO_DR: [
        ("v1_baseline", {"T": 20, "K": 180}),  # 3600
        ("v2_more_outer", {"T": 30, "K": 120}),  # 3600
        ("v3_fewer_outer", {"T": 10, "K": 360}),  # 3600
    ],
    MethodName.DORAEMON: [
        ("v1_baseline", {"adapt_iters": 20, "blocks": 30, "episodes_per_block": 6}),  # 3600
        ("v2_more_outer", {"adapt_iters": 20, "blocks": 15, "episodes_per_block": 12}),  # 3600
        ("v3_fewer_outer", {"adapt_iters": 10, "blocks": 15, "episodes_per_block": 24}),  # 3600
    ],
    MethodName.BO_DORAEMON: [
        (
            "v1_baseline",
            {
                "T": 20,
                "blocks": 30,
                "episodes_per_block": 6,
                "stabilization_blocks": 0,
                "bo_init_random_points": 0,
                "center_hold_rounds": 1,
                "competence_extra_rounds": 0,
            },
        ),  # 3600
        (
            "v2_more_outer",
            {
                "T": 20,
                "blocks": 15,
                "episodes_per_block": 12,
                "stabilization_blocks": 0,
                "bo_init_random_points": 0,
                "center_hold_rounds": 1,
                "competence_extra_rounds": 0,
            },
        ),  # 3600
        (
            "v3_fewer_outer",
            {
                "T": 10,
                "blocks": 15,
                "episodes_per_block": 24,
                "stabilization_blocks": 0,
                "bo_init_random_points": 0,
                "center_hold_rounds": 1,
                "competence_extra_rounds": 0,
            },
        ),  # 3600
    ],
}

HALFCHEETAH_METHOD_OVERRIDES = {
    MethodName.BO: {"bo_iters": 10, "bo_train_episodes_per_iter": 180},  # 1800
    MethodName.DR: {"dr_total_episodes": 1800, "dr_eval_every": 180},  # 1800
    MethodName.BO_DR: {"T": 10, "K": 180},  # 1800
    MethodName.DORAEMON: {"adapt_iters": 10, "blocks": 15, "episodes_per_block": 12},  # 1800
    MethodName.BO_DORAEMON: {
        "T": 10,
        "blocks": 15,
        "episodes_per_block": 12,  # 1800
        "stabilization_blocks": 0,
        "bo_init_random_points": 0,
        "center_hold_rounds": 1,
        "competence_extra_rounds": 0,
    },
}

METHODS = [MethodName.BO, MethodName.DR, MethodName.BO_DR, MethodName.DORAEMON, MethodName.BO_DORAEMON]


def run_mountaincar(order: str) -> None:
    seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    base_out = Path(__file__).resolve().parent / "runs" / "mountaincar_edge_v123"

    if order == "method-first":
        for method, variants in MOUNTAINCAR_SWEEP.items():
            for variant_name, override in variants:
                cfg = build_suite_config(
                    seeds=seeds,
                    methods=[method],
                    task="mountaincar",
                    profile="edge",
                    init_mode="scratch",
                    output_root=base_out / method.value.replace("+", "plus").lower() / variant_name,
                    method_overrides={method: override},
                )
                result = run_suite(cfg)
                print(method.value, variant_name, result.run_dir)
        return

    for seed in seeds:
        for method, variants in MOUNTAINCAR_SWEEP.items():
            for variant_name, override in variants:
                cfg = build_suite_config(
                    seeds=[seed],
                    methods=[method],
                    task="mountaincar",
                    profile="edge",
                    init_mode="scratch",
                    output_root=base_out / f"seed_{seed}" / method.value.replace("+", "plus").lower() / variant_name,
                    method_overrides={method: override},
                )
                result = run_suite(cfg)
                print(f"seed={seed}", method.value, variant_name, result.run_dir)


def run_halfcheetah(order: str) -> None:
    seeds = [1, 2, 3, 4, 5]
    base_out = Path(__file__).resolve().parent / "runs" / "halfcheetah_edge_1800"

    if order == "method-first":
        for method in METHODS:
            cfg = build_suite_config(
                seeds=seeds,
                methods=[method],
                task="halfcheetah",
                backend="sb3_sac",
                primary_metric="mean_episodic_return",
                eval_episodes=5,
                profile="edge",
                init_mode="scratch",
                task_config={"surrogate_success_threshold": 0.0},
                output_root=base_out / method.value.replace("+", "plus").lower(),
                method_overrides={method: HALFCHEETAH_METHOD_OVERRIDES[method]},
            )
            result = run_suite(cfg)
            print(method.value, result.run_dir)
        return

    for seed in seeds:
        for method in METHODS:
            cfg = build_suite_config(
                seeds=[seed],
                methods=[method],
                task="halfcheetah",
                backend="sb3_sac",
                primary_metric="mean_episodic_return",
                eval_episodes=5,
                profile="edge",
                init_mode="scratch",
                task_config={"surrogate_success_threshold": 0.0},
                output_root=base_out / f"seed_{seed}" / method.value.replace("+", "plus").lower(),
                method_overrides={method: HALFCHEETAH_METHOD_OVERRIDES[method]},
            )
            result = run_suite(cfg)
            print(f"seed={seed}", method.value, result.run_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run edge sweeps for MountainCar/HalfCheetah")
    parser.add_argument("--task", choices=["mountaincar", "halfcheetah"], required=True)
    parser.add_argument("--order", choices=["method-first", "seed-first"], default="method-first")
    args = parser.parse_args()

    if args.task == "mountaincar":
        run_mountaincar(order=args.order)
    else:
        run_halfcheetah(order=args.order)


if __name__ == "__main__":
    main()
