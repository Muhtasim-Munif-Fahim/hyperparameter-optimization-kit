"""Demo: compare grid, random, bayesian, TPE, CMA-ES, Hyperband and BOHB on a noisy toy.

The demo objective has a known minimum of 0 at (a, b) = (1, 2) plus gaussian
noise, so you can eyeball how close each strategy gets on the same budget.
Hyperband and BOHB additionally see a cheaper, noisier fidelity of the same surface.

Usage: python examples/run_demo.py [--budget 30] [--seed 7]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
)

from hyperopt_kit.evaluate import compare_strategies
from hyperopt_kit.objectives import demo_objective
from hyperopt_kit.report import learning_curve_table, render_report, write_report
from hyperopt_kit.spaces import FloatRange


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--out",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "output", "demo_report.md"
        ),
    )
    args = parser.parse_args(argv)

    space = {"a": FloatRange(0.05, 5.0), "b": FloatRange(0.05, 5.0)}
    print(
        f"optimizing demo_objective (known minimum 0 at a=1, b=2) with "
        f"budget={args.budget} per strategy, seed={args.seed}"
    )
    results = compare_strategies(space, demo_objective, args.budget, seed=args.seed)

    print()
    print("| strategy | trials | best score | best parameters |")
    print("| --- | ---: | ---: | --- |")
    for name, result in results.items():
        params = json.dumps(result.best_params, default=str)
        print(f"| {name} | {len(result.trials)} | {result.best_score:.6g} | {params} |")

    print()
    print("learning curve (best-so-far score after each evaluation):")
    print(learning_curve_table(results))

    path = write_report(
        render_report(
            results,
            space,
            title="Hyperparameter search demo",
            objective_name="demo_objective (noisy, minimize)",
            budget=args.budget,
        ),
        args.out,
    )
    print()
    print(f"wrote report to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
