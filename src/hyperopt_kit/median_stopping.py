"""Random search with the median stopping rule (Golovin et al., 2017).

The median stopping rule is the default early-stopping policy in Google
Vizier and in many tuning services. It is model-free: no learning-curve
model or bracket structure is needed.

A configuration is trained up a ladder of fidelities (the same geometric
ladder as :func:`successive_halving_rungs`). After its ``k``-th rung it is
stopped when its best objective so far is *strictly worse* than the median,
over earlier configurations that reached rung ``k``, of their running
averages ``mean(score_0 .. score_k)``. Lower scores are better, as in every
other searcher here.

Rungs before ``grace_rungs`` are never pruned, and no decision is made
until at least ``min_samples`` reference curves reached the rung.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .searchers import (
    ScoreFn,
    Stopper,
    Trial,
    _best_score,
    _resource_caller,
    _sample_config,
    _validate_hyperband_params,
    _validate_space,
    successive_halving_rungs,
)
from .spaces import Space

__all__ = [
    "MedianStoppingDecision",
    "median_stopping_decision",
    "median_stopping_search",
]


@dataclass(frozen=True)
class MedianStoppingDecision:
    """Outcome of :func:`median_stopping_decision`."""

    stop: bool
    rung: int
    best_so_far: float
    median_reference: Optional[float]
    n_reference: int


def median_stopping_decision(
    curve: Sequence[float],
    reference_curves: Sequence[Sequence[float]],
    *,
    grace_rungs: int = 1,
    min_samples: int = 3,
) -> MedianStoppingDecision:
    """Apply the median stopping rule to ``curve`` at its latest rung.

    ``curve`` holds the scores of the running configuration at rungs
    ``0..k``. ``reference_curves`` are the score sequences of earlier
    configurations; only those with at least ``k + 1`` entries count.
    """
    if len(curve) == 0:
        raise ValueError("curve must contain at least one score")
    if grace_rungs < 0:
        raise ValueError("grace_rungs must be >= 0")
    if min_samples < 1:
        raise ValueError("min_samples must be >= 1")
    k = len(curve) - 1
    best = float(np.min(np.asarray(curve, dtype=float)))
    running = [float(np.mean(np.asarray(c[: k + 1], dtype=float))) for c in reference_curves if len(c) > k]
    if k < grace_rungs or len(running) < min_samples:
        return MedianStoppingDecision(False, k, best, None, len(running))
    med = float(np.median(running))
    return MedianStoppingDecision(best > med, k, best, med, len(running))


@dataclass
class _Curve:
    params: Dict[str, Any]
    scores: List[float] = field(default_factory=list)


def median_stopping_search(
    space: Dict[str, Space],
    objective: ScoreFn,
    budget: int,
    rng: Optional[np.random.Generator] = None,
    stop_when: Stopper = None,
    *,
    eta: int = 3,
    min_resource: int = 1,
    max_resource: int = 9,
    grace_rungs: int = 1,
    min_samples: int = 3,
) -> List[Trial]:
    """Random search whose configurations are pruned by the median rule.

    Each sampled configuration is evaluated at successive normalized
    fidelities from :func:`successive_halving_rungs` (``eta``,
    ``min_resource``, ``max_resource``) until it reaches full fidelity or
    :func:`median_stopping_decision` stops it. Every evaluation is one
    :class:`Trial` with its ``resource`` set, and ``budget`` counts
    evaluations, as with the other multi-fidelity searchers.

    Unlike successive halving, no fixed fraction is discarded: a
    configuration continues as long as it is at least as good as the
    typical earlier configuration at the same point.
    """
    _validate_space(space)
    if budget < 1:
        raise ValueError("budget must be >= 1")
    _validate_hyperband_params(eta, min_resource, max_resource)
    if grace_rungs < 0:
        raise ValueError("grace_rungs must be >= 0")
    if min_samples < 1:
        raise ValueError("min_samples must be >= 1")
    rng = rng if rng is not None else np.random.default_rng()
    rungs = successive_halving_rungs(max_resource, eta, min_resource)
    evaluate = _resource_caller(objective)
    trials: List[Trial] = []
    history: List[List[float]] = []

    while len(trials) < budget:
        cur = _Curve(_sample_config(space, rng))
        for fidelity in rungs:
            if len(trials) >= budget:
                break
            score = float(evaluate(dict(cur.params), float(fidelity)))
            cur.scores.append(score)
            trials.append(Trial(len(trials), dict(cur.params), score, resource=float(fidelity)))
            if stop_when is not None and stop_when(_best_score(trials), len(trials)):
                history.append(cur.scores)
                return trials
            decision = median_stopping_decision(
                cur.scores, history, grace_rungs=grace_rungs, min_samples=min_samples
            )
            if decision.stop:
                break
        history.append(cur.scores)
    return trials
