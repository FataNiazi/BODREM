from __future__ import annotations

import argparse
import json
from typing import Any

from .config import build_suite_config
from .runner import list_methods, run_batch, run_suite


def _split_csv(values: list[str] | None) -> list[str]:
    if not values:
        return []
    parts: list[str] = []
    for value in values:
        parts.extend([p.strip() for p in str(value).split(",") if p.strip()])
    return parts


def _parse_task_config_json(raw: str | None) -> dict[str, Any] | None:
    if raw is None:
        return None
    token = str(raw).strip()
    if not token:
        return None
    data = json.loads(token)
    if not isinstance(data, dict):
        raise ValueError("--task-config-json must decode to a JSON object.")
    return dict(data)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mountaincar", description="Modular MountainCar runner")
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="Run one suite across methods and seeds")
    run_parser.add_argument("--methods", nargs="+", default=None, help="Comma-separated or repeated method names")
    run_parser.add_argument("--seeds", nargs="+", required=True, help="Comma-separated or repeated integer seeds")
    run_parser.add_argument("--profile", choices=["smoke", "prototype", "edge"], default="smoke")
    run_parser.add_argument("--init-mode", choices=["auto", "scratch", "warmstart"], default="auto")
    run_parser.add_argument("--task", default="mountaincar")
    run_parser.add_argument("--env-id", dest="env_id", default=None)
    run_parser.add_argument("--backend", default=None)
    run_parser.add_argument("--primary-metric", dest="primary_metric", default=None)
    run_parser.add_argument("--eval-episodes", dest="eval_episodes", type=int, default=None)
    run_parser.add_argument("--task-config-json", dest="task_config_json", default=None)
    run_parser.add_argument("--checkpoint-path", dest="checkpoint_path", default=None)
    run_parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    run_parser.add_argument("--output-root", dest="output_root", default=None)

    batch_parser = sub.add_parser("batch-run", help="Run repeated method x seed attempts")
    batch_parser.add_argument("--methods", nargs="+", default=None, help="Comma-separated or repeated method names")
    batch_parser.add_argument("--seeds", nargs="+", required=True, help="Comma-separated or repeated integer seeds")
    batch_parser.add_argument("--repetitions", type=int, required=True, help="Number of repetitions per method/seed")
    batch_parser.add_argument("--profile", choices=["smoke", "prototype", "edge"], default="smoke")
    batch_parser.add_argument("--init-mode", choices=["auto", "scratch", "warmstart"], default="auto")
    batch_parser.add_argument("--task", default="mountaincar")
    batch_parser.add_argument("--env-id", dest="env_id", default=None)
    batch_parser.add_argument("--backend", default=None)
    batch_parser.add_argument("--primary-metric", dest="primary_metric", default=None)
    batch_parser.add_argument("--eval-episodes", dest="eval_episodes", type=int, default=None)
    batch_parser.add_argument("--task-config-json", dest="task_config_json", default=None)
    batch_parser.add_argument("--checkpoint-path", dest="checkpoint_path", default=None)
    batch_parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    batch_parser.add_argument("--output-root", dest="output_root", default=None)
    batch_parser.add_argument("--on-error", choices=["continue", "fail-fast"], default="continue")
    batch_parser.add_argument("--dedupe-seeds", action="store_true")
    batch_parser.add_argument("--resume-from", default=None)
    batch_parser.add_argument("--rerun-failed", action="store_true")

    list_parser = sub.add_parser("list-methods", help="Print supported methods")
    list_parser.add_argument("--json", action="store_true", help="Print methods as JSON")

    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command in {"run", "batch-run"}:
        args.methods = _split_csv(args.methods)
        args.seeds = [int(s) for s in _split_csv(args.seeds)]
        args.task_config = _parse_task_config_json(getattr(args, "task_config_json", None))
    return args


def main(argv: list[str] | None = None) -> Any:
    args = parse_args(argv)

    if args.command == "list-methods":
        methods = list_methods()
        if args.json:
            print(json.dumps([m.value for m in methods], indent=2))
        else:
            for method in methods:
                print(method.value)
        return methods

    if args.command == "run":
        suite_cfg = build_suite_config(
            seeds=args.seeds,
            methods=args.methods,
            task=args.task,
            env_id=args.env_id,
            backend=args.backend,
            profile=args.profile,
            init_mode=args.init_mode,
            primary_metric=args.primary_metric,
            eval_episodes=args.eval_episodes or 5,
            task_config=args.task_config,
            checkpoint_path=args.checkpoint_path,
            device=args.device,
            output_root=args.output_root,
        )
        result = run_suite(suite_cfg)
        print(result.run_dir)
        return result

    if args.command == "batch-run":
        suite_cfg = build_suite_config(
            seeds=args.seeds,
            methods=args.methods,
            task=args.task,
            env_id=args.env_id,
            backend=args.backend,
            profile=args.profile,
            init_mode=args.init_mode,
            primary_metric=args.primary_metric,
            eval_episodes=args.eval_episodes or 5,
            task_config=args.task_config,
            checkpoint_path=args.checkpoint_path,
            device=args.device,
            output_root=args.output_root,
        )
        result = run_batch(
            suite_cfg,
            repetitions=int(args.repetitions),
            on_error=args.on_error,
            dedupe_seeds=bool(args.dedupe_seeds),
            resume_from=args.resume_from,
            rerun_failed=bool(args.rerun_failed),
        )
        print(result.run_dir)
        return result

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    main()
