"""Command-line interface: ``search``, ``compare`` and ``report`` subcommands."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from typing import List, Optional

from .evaluate import available_strategies, compare_strategies, run_search
from .objectives import demo_objective
from .report import render_report, write_report
from .spaces import parse_space


def _load_space(ref: str) -> dict:
    """Parse ``--space`` as inline JSON or ``@path`` to a JSON file."""
    if ref.startswith("@"):
        with open(ref[1:], "r", encoding="utf-8") as fh:
            spec = json.load(fh)
    else:
        spec = json.loads(ref)
    return parse_space(spec)


def _load_objective(ref: str):
    """Resolve ``--objective`` to a callable: ``demo`` or ``module:function``."""
    if ref == "demo":
        return demo_objective
    module_path, sep, attr = ref.partition(":")
    if not sep:
        raise ValueError(
            f"objective must be 'demo' or 'module:function', got {ref!r}"
        )
    module = importlib.import_module(module_path)
    if not hasattr(module, attr):
        raise ValueError(f"module {module_path!r} has no attribute {attr!r}")
    return getattr(module, attr)


def _positive_int(value: str) -> int:
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return n


def _eta(value: str) -> int:
    n = int(value)
    if n < 2:
        raise argparse.ArgumentTypeError("eta must be an integer >= 2")
    return n


def _hyperband_kwargs(args) -> dict:
    kwargs = {
        "eta": args.eta,
        "min_resource": args.min_resource,
        "max_resource": args.max_resource,
    }
    if getattr(args, "n_candidates", None) is not None:
        kwargs["n_candidates"] = args.n_candidates
    return kwargs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hyperopt-kit",
        description=(
            "Hyperparameter optimization toolkit: grid, random, bayesian, "
            "TPE and Hyperband / successive-halving search."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    strategy_names = available_strategies()

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--space", required=True, help="JSON space spec, or @path/to/space.json"
        )
        p.add_argument(
            "--objective", required=True, help="'demo' or a 'module:function' path"
        )
        p.add_argument(
            "--budget", type=_positive_int, required=True,
            help="number of evaluations per strategy",
        )
        p.add_argument("--seed", type=int, default=None, help="seed for reproducibility")
        p.add_argument(
            "--strategies", default="grid,random,bayesian,tpe,hyperband",
            help="comma-separated strategies to run",
        )
        p.add_argument(
            "--gamma", type=float, default=0.25,
            help="TPE quantile: fraction of observations modeled by l(x)",
        )
        p.add_argument(
            "--eta", type=_eta, default=3,
            help="Hyperband / successive-halving downsampling rate (integer >= 2)",
        )
        p.add_argument(
            "--min-resource", type=_positive_int, default=1,
            help="minimum resource units for multi-fidelity search",
        )
        p.add_argument(
            "--max-resource", type=_positive_int, default=9,
            help="maximum resource units (full fidelity) for multi-fidelity search",
        )

    p_search = sub.add_parser("search", help="run a single search strategy")
    add_common(p_search)
    p_search.add_argument(
        "--strategy", default="bayesian", choices=strategy_names
    )
    p_search.add_argument(
        "--xi", type=float, default=0.01,
        help="exploration/exploitation weight for bayesian search",
    )
    p_search.add_argument(
        "--n-initial", type=int, default=None,
        help="number of random points before the surrogate kicks in",
    )
    p_search.add_argument(
        "--n-candidates", type=_positive_int, default=None,
        help="TPE / bayesian candidate pool, or successive_halving starting configs",
    )
    p_search.add_argument(
        "--floor", type=float, default=None, help="stop when the best score reaches this"
    )
    p_search.add_argument(
        "--patience", type=int, default=None,
        help="stop after this many consecutive evaluations without improvement",
    )

    p_compare = sub.add_parser("compare", help="compare strategies on a common budget")
    add_common(p_compare)

    p_report = sub.add_parser("report", help="compare strategies and write a markdown report")
    add_common(p_report)
    p_report.add_argument("--out", default="hyperopt_report.md", help="output markdown path")
    p_report.add_argument("--title", default="Hyperparameter search comparison")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        space = _load_space(args.space)
        objective = _load_objective(args.objective)
    except Exception as exc:
        parser.error(str(exc))
    strategies = tuple(s.strip() for s in args.strategies.split(",") if s.strip())
    if not strategies:
        parser.error("--strategies must not be empty")

    mf_kwargs = _hyperband_kwargs(args)
    strategy_kwargs = {
        "hyperband": {
            "eta": mf_kwargs["eta"],
            "min_resource": mf_kwargs["min_resource"],
            "max_resource": mf_kwargs["max_resource"],
        },
        "successive_halving": dict(mf_kwargs),
        "tpe": {"gamma": args.gamma},
    }

    if args.command == "search":
        extra = {
            "xi": args.xi,
            "n_initial": args.n_initial,
            "gamma": args.gamma,
            "eta": mf_kwargs["eta"],
            "min_resource": mf_kwargs["min_resource"],
            "max_resource": mf_kwargs["max_resource"],
        }
        if args.n_candidates is not None:
            extra["n_candidates"] = args.n_candidates
        result = run_search(
            args.strategy,
            space,
            objective,
            args.budget,
            seed=args.seed,
            floor=args.floor,
            patience=args.patience,
            **extra,
        )
        print(f"strategy: {result.strategy}")
        print(f"trials: {len(result.trials)}")
        print(f"best score: {result.best_score:.6g}")
        print(
            "best parameters: "
            + json.dumps(result.best_params, default=str, sort_keys=True)
        )
        return 0

    if args.command == "compare":
        results = compare_strategies(
            space,
            objective,
            args.budget,
            strategies=strategies,
            seed=args.seed,
            strategy_kwargs=strategy_kwargs,
        )
        print("| strategy | trials | best score |")
        print("| --- | ---: | ---: |")
        for s, r in results.items():
            print(f"| {s} | {len(r.trials)} | {r.best_score:.6g} |")
        return 0

    if args.command == "report":
        results = compare_strategies(
            space,
            objective,
            args.budget,
            strategies=strategies,
            seed=args.seed,
            strategy_kwargs=strategy_kwargs,
        )
        text = render_report(
            results,
            space,
            title=args.title,
            objective_name=args.objective,
            budget=args.budget,
        )
        path = write_report(text, args.out)
        print(f"wrote report to {path}")
        return 0

    parser.error(f"unknown command {args.command!r}")


if __name__ == "__main__":
    sys.exit(main())
