"""Trial runner, early stopping and strategy comparison."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np

from .searchers import (
    Trial,
    bayesian_search,
    grid_search,
    random_search,
)
from .spaces import Space

_STRATEGIES = {
    "grid": grid_search,
    "random": random_search,
    "bayesian": bayesian_search,
}


@dataclass(frozen=True)
class RunResult:
    """Outcome of running one strategy over a budget."""

    strategy: str
    trials: List[Trial]
    best_score: float
    best_params: Dict


def best_trial(trials: List[Trial]) -> Trial:
    """The trial with the lowest score (minimize convention)."""
    if not trials:
        raise ValueError("no trials available")
    return min(trials, key=lambda t: t.score)


def best_score(trials: List[Trial]) -> float:
    return best_trial(trials).score


def best_params(trials: List[Trial]) -> Dict:
    return best_trial(trials).params


def learning_curve(trials: List[Trial]) -> List[float]:
    """Best-so-far score after each evaluation (monotone non-increasing)."""
    best = float("inf")
    curve: List[float] = []
    for trial in trials:
        best = min(best, trial.score)
        curve.append(best)
    return curve


class EarlyStopper:
    """Stops a search when a target floor is reached or the best score stays
    stale for ``patience`` consecutive evaluations."""

    def __init__(self, floor: Optional[float] = None, patience: Optional[int] = None):
        self.floor = floor
        self.patience = patience
        self._best = float("inf")
        self._stale = 0

    def __call__(self, best: float, n_evaluated: int) -> bool:
        if best < self._best - 1e-12:
            self._best = best
            self._stale = 0
        else:
            self._stale += 1
        if self.floor is not None and best <= self.floor:
            return True
        if self.patience is not None and self._stale >= self.patience:
            return True
        return False


def make_stopper(
    floor: Optional[float] = None, patience: Optional[int] = None
) -> Optional[EarlyStopper]:
    if floor is None and patience is None:
        return None
    return EarlyStopper(floor=floor, patience=patience)


def run_search(
    strategy: str,
    space: Dict[str, Space],
    objective,
    budget: int,
    rng: Optional[np.random.Generator] = None,
    *,
    seed: Optional[int] = None,
    floor: Optional[float] = None,
    patience: Optional[int] = None,
    **kwargs,
) -> RunResult:
    """Run one strategy against ``objective`` for up to ``budget`` trials."""
    if strategy not in _STRATEGIES:
        raise ValueError(
            f"unknown strategy {strategy!r}; expected one of {sorted(_STRATEGIES)}"
        )
    if budget < 1:
        raise ValueError("budget must be >= 1")
    rng = rng if rng is not None else np.random.default_rng(seed)
    stopper = make_stopper(floor=floor, patience=patience)
    searcher = _STRATEGIES[strategy]
    if strategy == "bayesian":
        allowed = {k: kwargs[k] for k in ("xi", "n_initial", "n_candidates") if k in kwargs}
        trials = searcher(space, objective, budget, rng=rng, stop_when=stopper, **allowed)
    else:
        trials = searcher(space, objective, budget, rng=rng, stop_when=stopper)
    return RunResult(
        strategy=strategy,
        trials=trials,
        best_score=best_score(trials),
        best_params=best_params(trials),
    )


def compare_strategies(
    space: Dict[str, Space],
    objective,
    budget: int,
    strategies: Iterable[str] = ("grid", "random", "bayesian"),
    rng: Optional[np.random.Generator] = None,
    *,
    seed: Optional[int] = None,
    floor: Optional[float] = None,
    patience: Optional[int] = None,
    strategy_kwargs: Optional[Dict] = None,
) -> Dict[str, RunResult]:
    """Run every strategy on a common evaluation budget.

    Each strategy gets its own independent random stream derived from
    ``seed`` (one generator per strategy), so passing a seed makes the whole
    comparison reproducible and fair.
    """
    strategies = list(strategies)
    for s in strategies:
        if s not in _STRATEGIES:
            raise ValueError(
                f"unknown strategy {s!r}; expected one of {sorted(_STRATEGIES)}"
            )
    if budget < 1:
        raise ValueError("budget must be >= 1")
    if seed is None:
        entropy = rng if rng is not None else np.random.default_rng()
        seed = int(entropy.integers(0, 2**31 - 1))
    strategy_kwargs = strategy_kwargs or {}
    results: Dict[str, RunResult] = {}
    for i, s in enumerate(strategies):
        kwargs = strategy_kwargs.get(s, {})
        results[s] = run_search(
            s,
            space,
            objective,
            budget,
            seed=seed + i,
            floor=floor,
            patience=patience,
            **kwargs,
        )
    return results
