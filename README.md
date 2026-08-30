# hyperparameter-optimization-kit

A small, dependency-light toolkit for hyperparameter and configuration search
over machine-learning models, built on pure Python and numpy. It ships three
search strategies — grid search, random search, and a Gaussian-process
bayesian optimizer with expected-improvement acquisition — plus utilities for
comparing strategies on a common evaluation budget and rendering markdown
reports of the results.

## Features

- **Search-space API** (`hyperopt_kit.spaces`): `Categorical`, `FloatRange`
  and `IntRange` spaces with seeded sampling, one-step mutation, log-scale
  support, and normalize/denormalize mappings for surrogate models.
- **Bayesian search** (`hyperopt_kit.searchers`): a squared-exponential
  Gaussian-process surrogate with a ridge-regularized solve and
  expected-improvement acquisition that exposes an exploration/exploitation
  tradeoff parameter.
- **Evaluation** (`hyperopt_kit.evaluate`): a trial runner that tracks the
  best configuration so far, supports early stopping via a target floor or a
  patience budget, and compares strategies on a common budget with
  reproducible per-strategy random streams.
- **Reports** (`hyperopt_kit.report`): markdown renderer with best configs,
  learning-curve tables, and per-parameter importance aggregations over the
  best configurations.
- **CLI** (`hyperopt_kit.cli`): `search`, `compare` and `report` subcommands.
- No dependencies beyond numpy.

## Install

```bash
pip install -r requirements.txt
# or, to import the package from anywhere:
pip install -e .
```

## Quickstart (Python API)

```python
import numpy as np
from hyperopt_kit.evaluate import compare_strategies
from hyperopt_kit.report import render_report, write_report
from hyperopt_kit.spaces import FloatRange

space = {
    "learning_rate": FloatRange(1e-3, 1.0, log_scale=True),
    "units": IntRange(16, 256),
}

def objective(params):
    # minimize convention: lower is better
    lr, u = params["learning_rate"], params["units"]
    return (np.log(lr) + 0.5) ** 2 + (u - 128.0) ** 2 / 1e4

results = compare_strategies(space, objective, budget=30, seed=7)
for strategy, result in results.items():
    print(strategy, result.best_score, result.best_params)

write_report(render_report(results, space, objective_name="toy"), "report.md")
```

Single strategies are available directly: `grid_search`, `random_search` and
`bayesian_search` each return a list of `Trial`s (`iteration`, `params`,
`score`), and `run_search` wraps one of them with early-stopping support.

## CLI

```bash
# run one strategy
python -m hyperopt_kit.cli search \
  --space '{"learning_rate": [0.001, 1.0, "log"], "units": [16, 256, "int"]}' \
  --objective demo --budget 30 --strategy bayesian --seed 7

# compare strategies on a common budget
python -m hyperopt_kit.cli compare --space '{"a": [0.1, 5.0], "b": [0.1, 5.0]}' \
  --objective demo --budget 30 --seed 7

# write a markdown report
python -m hyperopt_kit.cli report --space '{"a": [0.1, 5.0], "b": [0.1, 5.0]}' \
  --objective demo --budget 30 --seed 7 --out report.md
```

`--objective` accepts the built-in `demo` objective or any `module:function`
path. `--space` accepts inline JSON or `@path/to/space.json`.

## Search strategies

| strategy | how it picks the next config | strengths | watch out |
| --- | --- | --- | --- |
| `grid` | exhausts a cartesian product, at most `grid_points_per_dim` values per continuous/int parameter | deterministic, no tuning | cost grows with the product of per-dimension resolution |
| `random` | uniform samples from the spaces | trivial, robust to sharp peaks | wastes evaluations in flat regions |
| `bayesian` | GP surrogate on normalized inputs + expected improvement | sample-efficient on smooth objectives | fragile with nominal categoricals and non-smooth surfaces |

## Space spec format

Spaces can be built from classes or parsed from a compact dict:

```python
from hyperopt_kit.spaces import parse_space

space = parse_space({
    "learning_rate": [0.001, 0.1, "log"],   # log-scaled float
    "depth":         [2, 8, "int"],         # integer range
    "units":         [16, 256, "int", "log"],
    "kernel":        ["rbf", "linear"],     # categorical
})
```

Equivalent dict form: `{"type": "float"|"int", "low": ..., "high": ...,
"log": bool}` and `{"type": "categorical", "choices": [...]}`.

## Demo

```bash
python examples/run_demo.py --budget 30 --seed 7
```

Runs grid, random and bayesian search on a noisy two-parameter objective with
a known minimum of 0 at `(a, b) = (1, 2)`, prints the best scores and a
learning-curve table, and writes `examples/output/demo_report.md`.

## Caveats

- The Gaussian-process surrogate treats categorical parameters as ordered,
  which distorts distances for nominal categories.
- Bayesian search starts from a few random points and only becomes
  sample-efficient once the surrogate has enough evaluations; give it a
  budget of at least ~10 per dimension.
- Results on noisy objectives are stochastic — rerun with different seeds to
  gauge stability.
- Grid search cost grows with the product of per-dimension resolution; keep
  it on low-dimensional spaces.
- Log-scale ranges are sampled and mutated on the log axis, so equal steps in
  normalized space are multiplicative on the raw scale.
- Early stopping (floor target or patience) can return fewer than `budget`
  trials; comparison reports leave the strategy's best value in place for
  later rows.

## Tests

```bash
python -m pytest tests -q -c pyproject.toml
```

## License

MIT — see [LICENSE](LICENSE).
