# hyperparameter-optimization-kit

A small, dependency-light toolkit for hyperparameter and configuration search
over machine-learning models, built on pure Python and numpy. It ships grid
search, random search, a Gaussian-process bayesian optimizer with
expected-improvement acquisition, Tree-structured Parzen Estimator (TPE)
search, CMA-ES (covariance-matrix adaptation), Hyperband / successive
halving, BOHB (Bayesian Optimization Hyperband), and random search with
successive-halving early stopping — plus utilities for comparing strategies
on a common evaluation budget and rendering markdown reports of the results.

Gaussian-process expected improvement is already in the toolkit
(`GaussianProcess`, `expected_improvement`, `bayesian_search`). The searcher
added beside it is `random_successive_halving`: uniform random configurations
that are stopped early unless they stay in the top `1/eta` of their current
fidelity rung.

## Features

- **Search-space API** (`hyperopt_kit.spaces`): `Categorical`, `FloatRange`
  and `IntRange` spaces with seeded sampling, one-step mutation, log-scale
  support, and normalize/denormalize mappings for surrogate models.
- **Bayesian search** (`hyperopt_kit.searchers`): a squared-exponential
  Gaussian-process surrogate with a ridge-regularized solve and
  expected-improvement acquisition that exposes an exploration/exploitation
  tradeoff parameter.
- **TPE search** (`hyperopt_kit.searchers`): Tree-structured Parzen
  Estimators that split observations at a score quantile, fit independent
  densities `l(x)` (good) and `g(x)` (bad) on the search-space API, and
  pick the candidate that maximizes the `l(x)/g(x)` density ratio.
- **CMA-ES search** (`hyperopt_kit.searchers`): Hansen CMA-ES on the
  normalized unit cube. A multivariate Gaussian `N(m, σ²C)` is sampled
  each generation; the mean, isotropic step-size and covariance are
  adapted from the ranked offspring. Designed for `FloatRange` /
  `IntRange` (including log-scaled ranges); categoricals are encoded as
  ordered coordinates.
- **Hyperband / successive-halving** (`hyperopt_kit.searchers`): multi-fidelity
  brackets that evaluate many configurations at a cheap resource, discard the
  worst, and promote survivors to higher fidelity. Objectives may accept a
  normalized `resource` (or `fidelity`) in `(0, 1]`; single-argument
  objectives still work.
- **Random search with successive-halving early stopping**
  (`random_successive_halving`): the same uniform draws as `random_search`,
  but each configuration starts at the cheapest fidelity and continues only
  while it ranks in the running top `1/eta` of that rung (ASHA-style). This
  is not a second GP-EI optimizer, and it is not the closed cohort inside
  `successive_halving`.
- **BOHB** (`hyperopt_kit.searchers`): the same Hyperband brackets, but new
  configurations are proposed by a multivariate product-kernel density
  (Gaussian on numeric coordinates, Aitchison–Aitken on categoricals) fit to
  the best observations at the largest fidelity with enough data. Candidates
  are drawn from the "good" density and ranked by `l(x)/g(x)`. A fraction of
  proposals stays uniformly random. Implemented in numpy only — no
  statsmodels or ConfigSpace dependency.
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
from hyperopt_kit.spaces import FloatRange, IntRange

space = {
    "learning_rate": FloatRange(1e-3, 1.0, log_scale=True),
    "units": IntRange(16, 256),
}

def objective(params, resource=1.0):
    # minimize convention: lower is better
    # resource in (0, 1] is the Hyperband / BOHB fidelity (1.0 = full)
    lr, u = params["learning_rate"], params["units"]
    noise = 0.05 / np.sqrt(max(resource, 1e-12))
    return (np.log(lr) + 0.5) ** 2 + (u - 128.0) ** 2 / 1e4 + noise

results = compare_strategies(space, objective, budget=30, seed=7)
for strategy, result in results.items():
    print(strategy, result.best_score, result.best_params)

write_report(render_report(results, space, objective_name="toy"), "report.md")
```

Single strategies are available directly: `grid_search`, `random_search`,
`bayesian_search`, `tpe_search`, `cmaes_search`, `hyperband_search`,
`bohb_search`, `successive_halving` and `random_successive_halving` each
return a list of `Trial`s (`iteration`, `params`, `score`, optional
`resource`), and `run_search` wraps one of them with early-stopping support.

## CLI

```bash
# run one strategy
python -m hyperopt_kit.cli search \
  --space '{"learning_rate": [0.001, 1.0, "log"], "units": [16, 256, "int"]}' \
  --objective demo --budget 30 --strategy cmaes --seed 7

# compare strategies on a common budget
python -m hyperopt_kit.cli compare --space '{"a": [0.1, 5.0], "b": [0.1, 5.0]}' \
  --objective demo --budget 30 --seed 7

# write a markdown report
python -m hyperopt_kit.cli report --space '{"a": [0.1, 5.0], "b": [0.1, 5.0]}' \
  --objective demo --budget 30 --seed 7 --out report.md
```

`--objective` accepts the built-in `demo` objective or any `module:function`
path. `--space` accepts inline JSON or `@path/to/space.json`. Multi-fidelity
flags `--eta`, `--min-resource` and `--max-resource` apply to `hyperband`,
`successive_halving`, `random_successive_halving` and `bohb`. `--gamma` is
the TPE quantile (fraction of observations modeled by `l(x)`); `--n-initial`
and `--n-candidates` apply to
TPE and bayesian search, and `--n-candidates` also sets the BOHB candidate
pool (default 64). `--population-size` and `--sigma0` configure CMA-ES
(Hansen's `4 + floor(3 log n)` offspring and unit-cube step-size `0.3` by
default). `--top-n-percent` (default 15) is the BOHB good-set percentage and
`--random-fraction` (default 1/3) is the share of BOHB proposals drawn
uniformly instead of from the kernel density.

## Search strategies

| strategy | how it picks the next config | strengths | watch out |
| --- | --- | --- | --- |
| `grid` | exhausts a cartesian product, at most `grid_points_per_dim` values per continuous/int parameter | deterministic, no tuning | cost grows with the product of per-dimension resolution |
| `random` | uniform samples from the spaces | trivial, robust to sharp peaks | wastes evaluations in flat regions |
| `bayesian` | GP surrogate on normalized inputs + expected improvement | sample-efficient on smooth objectives | fragile with nominal categoricals and non-smooth surfaces |
| `tpe` | Parzen densities `l(x)` / `g(x)` on good vs bad observations; next point maximizes `l/g` | handles mixed numeric/categorical spaces; numpy-only | factorized per parameter, so it misses interactions |
| `cmaes` | sample `N(m, σ²C)` in normalized space; adapt mean, step-size and covariance from ranked offspring | models parameter interactions via `C`; strong on smooth continuous / integer ranges | not a natural fit for nominal categoricals; needs a few generations to adapt |
| `hyperband` | several successive-halving brackets with different `(n, r)` allocations | cheap fidelities discard losers early | needs a fidelity-aware objective to save real work; ranking can change across rungs |
| `bohb` | Hyperband schedule, but new configs maximize `l(x)/g(x)` under a product-kernel density of the best observations at the largest usable fidelity | multi-fidelity sample efficiency; the joint kernel can express interactions that factorized TPE misses | model is idle until both good and bad sets have more points than the dimension; still needs a fidelity-aware objective |
| `successive_halving` | one Hyperband bracket (aggressive early-stop), repeated to fill the budget | simpler than full Hyperband | same fidelity caveats; fewer full-fidelity evaluations |
| `random_successive_halving` | uniform random configs; continue one only when it is in the running top `1/eta` at its current fidelity | stops poor random trials before full fidelity; no closed cohort to wait for | needs a fidelity-aware objective to save real work; distinct from bracket `successive_halving` |

`compare_strategies` and the CLI `compare` / `report` commands share one
evaluation budget across `grid`, `random`, `bayesian`, `tpe`, `cmaes`,
`hyperband` and `bohb` by default. `random_successive_halving` is registered
for `--strategy` / `--strategies` and is left out of that default set, same
as `successive_halving`. Each Hyperband, BOHB, or early-stopping random
evaluation counts as one trial, matching the other searchers; reported
winners use the highest-resource score so a noisy cheap rung cannot beat a
full-fidelity result.

## Multi-fidelity objectives

Hyperband, successive halving, random successive-halving and BOHB call
`objective(params, resource=r)` when the callable accepts
`resource` or `fidelity` (a float in `(0, 1]`, where `1.0` is full fidelity).
A one-argument `objective(params)` is still valid — every rung then sees the
same full evaluation, so successive-halving only re-queries survivors.

The built-in `demo` objective inflates gaussian noise at `resource < 1` and
keeps the original `sigma = 0.05` surface at `resource = 1`, so comparisons
against random / grid / GP stay on a shared full-fidelity scale.

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

Runs grid, random, bayesian, TPE, CMA-ES, Hyperband and BOHB search on a noisy
two-parameter objective with a known minimum of 0 at `(a, b) = (1, 2)`,
prints the best scores and a learning-curve table, and writes
`examples/output/demo_report.md`.

## Caveats

- The Gaussian-process surrogate treats categorical parameters as ordered,
  which distorts distances for nominal categories.
- Bayesian search starts from a few random points and only becomes
  sample-efficient once the surrogate has enough evaluations; give it a
  budget of at least ~10 per dimension.
- TPE likewise starts from random points, then samples from `l(x)` and ranks
  by `l(x)/g(x)`. Densities are factorized per parameter (using normalize /
  denormalize on each `Space`), so strong interactions are not modeled.
- CMA-ES searches the unit cube from `Space.normalize` / `denormalize`, so
  log-scaled floats and integers stay continuous internally. It adapts a
  full covariance and therefore can follow interactions that factorized
  TPE misses. Give it at least a couple of generations (default
  population `4 + floor(3 log n)`). Categorical parameters are treated as
  ordered, same as the GP surrogate.
- Results on noisy objectives are stochastic — rerun with different seeds to
  gauge stability.
- Grid search cost grows with the product of per-dimension resolution; keep
  it on low-dimensional spaces.
- Log-scale ranges are sampled and mutated on the log axis, so equal steps in
  normalized space are multiplicative on the raw scale.
- Early stopping (floor target or patience) can return fewer than `budget`
  trials; comparison reports leave the strategy's best value in place for
  later rows.
- Hyperband's cheap rungs are noisier (or otherwise cheaper) approximations.
  The comparison table reports the best score among the highest-resource
  evaluations. Without a fidelity argument, Hyperband cannot save work — it
  only re-evaluates survivors.
- Default Hyperband uses `eta=3`, `min_resource=1`, `max_resource=9` (rungs
  at fidelity `1/9`, `1/3`, `1`). Increase `max_resource` when the real
  training budget has more geometric rungs.
- `random_successive_halving` uses that same fidelity ladder, but it does
  not wait for a bracket. A new configuration is drawn uniformly and
  evaluated at the cheapest rung. It is promoted only after the rung has at
  least `eta` scores and the configuration is still among the best
  `floor(n / eta)` (ties keep the earlier observation). Otherwise it is
  stopped and the next evaluation is either a better survivor or a fresh
  random configuration. With `max_resource == min_resource` the ladder has
  one rung, nothing is stopped early, and the samples match `random_search`
  on the same generator. Pass it explicitly (`--strategy
  random_successive_halving` or `strategies=("random_successive_halving",)`);
  it is not in the default comparison set. GP-EI was already available, so
  this is the method added in its place.
- BOHB (Falkner, Klein, Hutter, 2018) keeps that schedule and replaces
  uniform sampling with a product-kernel density estimator. The kernel is
  the product of per-coordinate factors centered on each joint observation
  (Gaussian for `FloatRange` / `IntRange` after `normalize`, Aitchison–Aitken
  for `Categorical`), so it can represent interactions that factorized TPE
  cannot. Bandwidths use Scott's normal-reference rule,
  `h = 1.06 σ n^{-1/(4+d)}`, floored at `min_bandwidth` (default `1e-3`);
  categorical bandwidths are capped so the kernel stays a valid probability.
  The model is built only on the largest fidelity where both the good slice
  (default best 15%) and the following worse slice each have more points
  than the number of parameters. `min_points_in_model` defaults to
  `dimension + 1`. One third of proposals (`random_fraction`) are uniform.
  Sampling widens continuous bandwidths by `bandwidth_factor` (default 3);
  scoring `l(x)/g(x)` uses the unwidened kernel. Until the model can be
  fit, BOHB samples uniformly, so it behaves like Hyperband on a short
  budget.

## Tests

```bash
python -m pytest tests -q -c pyproject.toml
```

## License

MIT — see [LICENSE](LICENSE).
