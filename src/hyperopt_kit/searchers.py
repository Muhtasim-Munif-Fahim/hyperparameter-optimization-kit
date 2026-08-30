"""Search strategies over a ``{name: Space}`` configuration space.

``grid_search`` and ``random_search`` are self-explanatory baselines;
``bayesian_search`` (in the same module) fits a squared-exponential
Gaussian-process surrogate and picks the next configuration by maximizing
expected improvement.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .spaces import Categorical, FloatRange, IntRange, Space

ScoreFn = Callable[[Dict[str, Any]], float]
Stopper = Optional[Callable[[float, int], bool]]


@dataclass(frozen=True)
class Trial:
    """One evaluated configuration."""

    iteration: int
    params: Dict[str, Any]
    score: float


def _validate_space(space: Dict[str, Space]) -> None:
    if not isinstance(space, dict) or len(space) == 0:
        raise ValueError("space must be a non-empty mapping of name -> Space")
    for name, sp in space.items():
        if not isinstance(sp, Space):
            raise ValueError(f"parameter {name!r} is not a Space instance")


def _sample_config(space: Dict[str, Space], rng: np.random.Generator) -> Dict[str, Any]:
    return {name: sp.sample(rng) for name, sp in space.items()}


def _best_score(trials: List[Trial]) -> float:
    if not trials:
        raise ValueError("no trials available")
    return min(t.score for t in trials)


def _grid_axis(sp: Space, per_dim: int) -> List[Any]:
    """Grid points for one parameter, at most ``per_dim`` of them."""
    if isinstance(sp, Categorical):
        return list(sp.choices)
    if isinstance(sp, IntRange):
        n = sp.n_values
        if n <= per_dim:
            return list(range(sp.low, sp.high + 1))
        values = np.linspace(sp.low, sp.high, per_dim)
        return list(dict.fromkeys(int(round(v)) for v in values))
    if isinstance(sp, FloatRange):
        xs = np.linspace(0.0, 1.0, per_dim)
        return [sp.denormalize(float(x)) for x in xs]
    raise TypeError(f"unsupported space type {type(sp).__name__}")


def grid_search(
    space: Dict[str, Space],
    objective: ScoreFn,
    budget: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    stop_when: Stopper = None,
    grid_points_per_dim: int = 4,
) -> List[Trial]:
    """Deterministic grid search over a cartesian product of the spaces.

    The grid is built lazily with at most ``grid_points_per_dim`` points per
    continuous/int parameter (categoricals contribute every choice), so a
    budget larger than the grid simply exhausts it.
    """
    _validate_space(space)
    if budget is None:
        budget = float("inf")
    elif budget < 1:
        raise ValueError("budget must be >= 1")
    if grid_points_per_dim < 1:
        raise ValueError("grid_points_per_dim must be >= 1")

    names = list(space)
    axes = [_grid_axis(sp, grid_points_per_dim) for sp in space.values()]
    trials: List[Trial] = []
    best = float("inf")
    for k, combo in enumerate(itertools.product(*axes)):
        if k >= budget:
            break
        params = dict(zip(names, combo))
        score = objective(params)
        trials.append(Trial(k, params, score))
        best = min(best, score)
        if stop_when is not None and stop_when(best, len(trials)):
            break
    return trials


def random_search(
    space: Dict[str, Space],
    objective: ScoreFn,
    budget: int,
    rng: Optional[np.random.Generator] = None,
    stop_when: Stopper = None,
) -> List[Trial]:
    """Uniform random sampling of configurations for ``budget`` evaluations."""
    _validate_space(space)
    if budget < 1:
        raise ValueError("budget must be >= 1")
    rng = rng if rng is not None else np.random.default_rng()

    trials: List[Trial] = []
    best = float("inf")
    for i in range(budget):
        params = _sample_config(space, rng)
        score = objective(params)
        trials.append(Trial(i, params, score))
        best = min(best, score)
        if stop_when is not None and stop_when(best, len(trials)):
            break
    return trials
