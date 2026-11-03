"""python -m benchmark prepare|run|publish|bundle"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--work", type=Path, required=True)
    prepare.add_argument("--seed", type=int, default=42)
    run = sub.add_parser("run")
    run.add_argument("--work", type=Path, required=True)
    run.add_argument(
        "--models",
        nargs="+",
        choices=["resnet18", "phikon-v2", "uni2-h"],
        default=["resnet18", "phikon-v2", "uni2-h"],
    )
    run.add_argument(
        "--batch-size",
        type=int,
        help="Override all model batch sizes; try 1 for low memory",
    )
    run.add_argument("--workers", type=int, default=2)
    run.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    publish = sub.add_parser("publish")
    publish.add_argument("runs", nargs="+", type=Path)
    publish.add_argument("--docs", type=Path, default=Path("docs"))
    bundle = sub.add_parser("bundle")
    bundle.add_argument("runs", nargs="+", type=Path)
    bundle.add_argument("--output", type=Path, default=Path("dashboard-results.zip"))
    args = parser.parse_args()
    if args.command == "prepare":
        from .data import prepare

        prepare(args.work, seed=args.seed)
    elif args.command == "run":
        from .run import run

        batch = {m: args.batch_size for m in args.models} if args.batch_size else None
        run(
            args.work,
            models=args.models,
            batch_sizes=batch,
            workers=args.workers,
            precision=args.precision,
        )
    elif args.command == "publish":
        from .publish import publish

        publish(args.runs, args.docs)
    else:
        from .publish import bundle

        print(bundle(args.runs, args.output))


if __name__ == "__main__":
    main()
