"""Command-line interface: ``search``, ``compare``, ``report``, ``pareto`` and ``sensitivity`` subcommands."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from typing import List, Optional

from .evaluate import available_strategies, compare_strategies, run_search
from .multi_objective import hypervolume, nsga2_search, pareto_trials
from .objectives import demo_multi_objective, demo_objective, zdt1
from .report import render_report, write_report
from .sensitivity import format_sobol_table, surrogate_sobol_importance
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


def _population_size(value: str) -> int:
    n = int(value)
    if n < 2:
        raise argparse.ArgumentTypeError("population-size must be an integer >= 2")
    return n


def _unit_fraction(value: str) -> float:
    x = float(value)
    if not 0.0 <= x <= 1.0:
        raise argparse.ArgumentTypeError("must be a float in [0, 1]")
    return x


def _top_n_percent(value: str) -> int:
    n = int(value)
    if n < 1 or n > 99:
        raise argparse.ArgumentTypeError("top-n-percent must be an integer in 1..99")
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
            "TPE, CMA-ES, population-based training, Hyperband / "
            "successive-halving, BOHB, and random search with "
            "successive-halving early stopping."
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
            "--strategies",
            default="grid,random,bayesian,tpe,cmaes,hyperband,bohb",
            help="comma-separated strategies to run",
        )
        p.add_argument(
            "--gamma", type=float, default=0.25,
            help="TPE quantile: fraction of observations modeled by l(x)",
        )
        p.add_argument(
            "--population-size", type=_population_size, default=None,
            help=(
                "CMA-ES offspring per generation (default: 4 + floor(3 log n)), "
                "or PBT population size (default: 4)"
            ),
        )
        p.add_argument(
            "--sigma0", type=float, default=0.3,
            help="CMA-ES initial step-size on the unit cube",
        )
        p.add_argument(
            "--eta", type=_eta, default=3,
            help=(
                "downsampling rate for Hyperband, successive halving, "
                "random successive-halving, and PBT rungs (integer >= 2)"
            ),
        )
        p.add_argument(
            "--min-resource", type=_positive_int, default=1,
            help="minimum resource units for multi-fidelity search",
        )
        p.add_argument(
            "--max-resource", type=_positive_int, default=9,
            help="maximum resource units (full fidelity) for multi-fidelity search",
        )
        p.add_argument(
            "--top-n-percent", type=_top_n_percent, default=15,
            help="BOHB: percent of observations at a fidelity treated as good (l(x))",
        )
        p.add_argument(
            "--random-fraction", type=_unit_fraction, default=1.0 / 3.0,
            help="BOHB: fraction of proposals drawn uniformly instead of from the KDE",
        )
        p.add_argument(
            "--exploit-interval", type=_positive_int, default=1,
            help="PBT: completed population steps between exploit and explore",
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
        help=(
            "TPE / bayesian / BOHB candidate pool, or successive_halving starting configs"
        ),
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

    p_pareto = sub.add_parser(
        "pareto", help="multi-objective NSGA-II search; prints the Pareto front and hypervolume"
    )
    p_pareto.add_argument("--space", required=True, help="JSON space spec, or @path/to/space.json")
    p_pareto.add_argument(
        "--objective", default="demo",
        help="'demo' (loss vs. a+b cost), 'zdt1', or a 'module:function' returning a sequence",
    )
    p_pareto.add_argument("--budget", type=_positive_int, required=True, help="number of evaluations")
    p_pareto.add_argument("--population-size", type=_population_size, default=None)
    p_pareto.add_argument("--seed", type=int, default=None)
    p_pareto.add_argument(
        "--ref", default=None,
        help="comma-separated hypervolume reference point (default: per-objective max of all trials + 10%% of range)",
    )
    p_pareto.add_argument("--json", action="store_true", help="print the front as JSON")

    p_sens = sub.add_parser(
        "sensitivity",
        help="run a search, then rank hyperparameters by Sobol indices on a GP surrogate",
    )
    p_sens.add_argument("--space", required=True, help="JSON space spec, or @path/to/space.json")
    p_sens.add_argument("--objective", default="demo", help="'demo' or 'module:function'")
    p_sens.add_argument("--budget", type=_positive_int, required=True, help="number of evaluations")
    p_sens.add_argument("--strategy", default="random", choices=available_strategies())
    p_sens.add_argument("--seed", type=int, default=None)
    p_sens.add_argument(
        "--n-samples", type=_positive_int, default=1024,
        help="Saltelli base sample size (surrogate is queried n*(d+2) times)",
    )
    p_sens.add_argument(
        "--bootstrap", type=int, default=0, help="bootstrap resamples for confidence intervals"
    )
    p_sens.add_argument("--json", action="store_true", help="print indices as JSON")
    return parser


def _run_sensitivity(args, parser) -> int:
    import numpy as np

    try:
        space = _load_space(args.space)
        objective = _load_objective(args.objective)
    except Exception as exc:
        parser.error(str(exc))
    if args.n_samples < 2:
        parser.error("--n-samples must be >= 2")
    if args.bootstrap < 0:
        parser.error("--bootstrap must be >= 0")
    result = run_search(args.strategy, space, objective, args.budget, seed=args.seed)
    try:
        sobol = surrogate_sobol_importance(
            result.trials,
            space,
            n=args.n_samples,
            rng=np.random.default_rng(args.seed),
            n_bootstrap=args.bootstrap,
        )
    except ValueError as exc:
        parser.error(str(exc))
    if args.json:
        print(json.dumps({
            "strategy": result.strategy,
            "trials": len(result.trials),
            "best_score": result.best_score,
            "variance": sobol.variance,
            "indices": sobol.as_dict(),
        }, default=str))
        return 0
    print(f"strategy: {result.strategy}")
    print(f"trials: {len(result.trials)} (surrogate fitted on {sobol.extra['n_trials']})")
    print(f"best score: {result.best_score:.6g}")
    print(format_sobol_table(sobol))
    return 0


def _run_pareto(args, parser) -> int:
    import numpy as np

    try:
        space = _load_space(args.space)
        if args.objective == "demo":
            objective = demo_multi_objective
        elif args.objective == "zdt1":
            objective = zdt1
        else:
            objective = _load_objective(args.objective)
    except Exception as exc:
        parser.error(str(exc))
    rng = np.random.default_rng(args.seed)
    try:
        trials = nsga2_search(space, objective, args.budget, rng, population_size=args.population_size)
    except ValueError as exc:
        parser.error(str(exc))
    front = pareto_trials(trials)
    F = np.array([t.objectives for t in trials])
    if args.ref:
        ref = [float(v) for v in args.ref.split(",")]
        if len(ref) != F.shape[1]:
            parser.error(f"--ref needs {F.shape[1]} comma-separated values")
    else:
        span = F.max(axis=0) - F.min(axis=0)
        ref = (F.max(axis=0) + 0.1 * np.where(span > 0, span, 1.0)).tolist()
    hv = hypervolume([t.objectives for t in front], ref)
    if args.json:
        print(json.dumps({
            "trials": len(trials),
            "hypervolume": hv,
            "ref": ref,
            "front": [{"iteration": t.iteration, "params": t.params, "objectives": list(t.objectives)} for t in front],
        }, default=str))
        return 0
    print(f"trials: {len(trials)}")
    print(f"pareto front: {len(front)} configurations")
    print(f"hypervolume: {hv:.6g} (ref {', '.join(f'{r:.4g}' for r in ref)})")
    m = F.shape[1]
    print("| iter | " + " | ".join(f"f{j + 1}" for j in range(m)) + " | params |")
    print("| ---: | " + " | ".join("---:" for _ in range(m)) + " | --- |")
    for t in front:
        vals = " | ".join(f"{v:.5g}" for v in t.objectives)
        print(f"| {t.iteration} | {vals} | {json.dumps(t.params, default=str, sort_keys=True)} |")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "pareto":
        return _run_pareto(args, parser)
    if args.command == "sensitivity":
        return _run_sensitivity(args, parser)
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
        "random_successive_halving": {
            "eta": mf_kwargs["eta"],
            "min_resource": mf_kwargs["min_resource"],
            "max_resource": mf_kwargs["max_resource"],
        },
        "tpe": {"gamma": args.gamma},
        "cmaes": {"sigma0": args.sigma0},
        "bohb": {
            "eta": mf_kwargs["eta"],
            "min_resource": mf_kwargs["min_resource"],
            "max_resource": mf_kwargs["max_resource"],
            "top_n_percent": args.top_n_percent,
            "random_fraction": args.random_fraction,
        },
        "pbt": {
            "eta": mf_kwargs["eta"],
            "min_resource": mf_kwargs["min_resource"],
            "max_resource": mf_kwargs["max_resource"],
            "exploit_interval": args.exploit_interval,
        },
    }
    if args.population_size is not None:
        strategy_kwargs["cmaes"]["population_size"] = args.population_size
        strategy_kwargs["pbt"]["population_size"] = args.population_size

    if args.command == "search":
        extra = {
            "xi": args.xi,
            "n_initial": args.n_initial,
            "gamma": args.gamma,
            "sigma0": args.sigma0,
            "eta": mf_kwargs["eta"],
            "min_resource": mf_kwargs["min_resource"],
            "max_resource": mf_kwargs["max_resource"],
            "top_n_percent": args.top_n_percent,
            "random_fraction": args.random_fraction,
            "exploit_interval": args.exploit_interval,
        }
        if args.population_size is not None:
            extra["population_size"] = args.population_size
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
