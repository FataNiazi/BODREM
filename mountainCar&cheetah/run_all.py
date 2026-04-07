from pathlib import Path
from config import build_suite_config
from runner import run_suite
from types import MethodName

seeds = [1, 2, 3, 4, 5]
base_out = Path("/Users/kevaanbuch/Desktop/Uni/CSC415/mountaincar/runs/all_algo_sweep_equiv3600")

sweep = {
    MethodName.BO: [
        ("v1_baseline",    {"bo_iters": 20, "bo_train_episodes_per_iter": 180}),  # 3600
        ("v2_more_outer",  {"bo_iters": 30, "bo_train_episodes_per_iter": 120}),  # 3600
        ("v3_fewer_outer", {"bo_iters": 10, "bo_train_episodes_per_iter": 360}),  # 3600
    ],
    MethodName.DR: [
        ("v1_baseline",    {"dr_total_episodes": 3600, "dr_eval_every": 180}),
        ("v2_more_evals",  {"dr_total_episodes": 3600, "dr_eval_every": 120}),
        ("v3_fewer_evals", {"dr_total_episodes": 3600, "dr_eval_every": 360}),
    ],
    MethodName.BO_DR: [
        ("v1_baseline",    {"T": 20, "K": 180}),  # 3600
        ("v2_more_outer",  {"T": 30, "K": 120}),  # 3600
        ("v3_fewer_outer", {"T": 10, "K": 360}),  # 3600
    ],
    MethodName.DORAEMON: [
        ("v1_baseline",    {"adapt_iters": 20, "blocks": 30, "episodes_per_block": 6}),   # 3600
        ("v2_more_outer",  {"adapt_iters": 20, "blocks": 15, "episodes_per_block": 12}),  # 3600
        ("v3_fewer_outer", {"adapt_iters": 10, "blocks": 15, "episodes_per_block": 24}),  # 3600
    ],
    MethodName.BO_DORAEMON: [
        ("v1_baseline", {
            "T": 20, "blocks": 30, "episodes_per_block": 6,
            "stabilization_blocks": 0, "bo_init_random_points": 0,
            "center_hold_rounds": 1, "competence_extra_rounds": 0,
        }),  # 3600
        ("v2_more_outer", {
            "T": 20, "blocks": 15, "episodes_per_block": 12,
            "stabilization_blocks": 0, "bo_init_random_points": 0,
            "center_hold_rounds": 1, "competence_extra_rounds": 0,
        }),  # 3600
        ("v3_fewer_outer", {
            "T": 10, "blocks": 15, "episodes_per_block": 24,
            "stabilization_blocks": 0, "bo_init_random_points": 0,
            "center_hold_rounds": 1, "competence_extra_rounds": 0,
        }),  # 3600
    ],
}

def train_budget(method, o):
    if method == MethodName.BO:
        return int(o["bo_iters"]) * int(o["bo_train_episodes_per_iter"])
    if method == MethodName.DR:
        return int(o["dr_total_episodes"])
    if method == MethodName.BO_DR:
        return int(o["T"]) * int(o["K"])
    if method == MethodName.DORAEMON:
        return int(o["adapt_iters"]) * int(o["blocks"]) * int(o["episodes_per_block"])
    if method == MethodName.BO_DORAEMON:
        return int(o["T"]) * int(o["blocks"]) * int(o["episodes_per_block"])
    raise ValueError(method)

total_runs = 0
ok_runs = 0
failed_runs = 0
failures = []

for method, variants in sweep.items():
    for variant_name, override in variants:
        total_runs += len(seeds)
        try:
            budget = train_budget(method, override)
            if budget != 3600:
                raise ValueError(f"budget={budget} != 3600")

            out_dir = base_out / method.value.replace("+", "plus").lower() / variant_name
            cfg = build_suite_config(
                seeds=seeds,
                methods=[method],
                profile="edge",
                init_mode="scratch",
                output_root=out_dir,
                method_overrides={method: override},
            )
            result = run_suite(cfg)
            ok_runs += len(seeds)
            print(f"OK     | {method.value} | {variant_name} | train_budget={budget} | {result.run_dir}")
        except Exception as e:
            failed_runs += len(seeds)
            failures.append((method.value, variant_name, str(e)))
            print(f"FAILED | {method.value} | {variant_name} | {e}")

print(f"done: planned method-seed runs = {total_runs}, succeeded={ok_runs}, failed={failed_runs}")
if failures:
    print("failure summary:")
    for method_name, variant_name, err in failures:
        print(f"- {method_name} | {variant_name} | {err}")
