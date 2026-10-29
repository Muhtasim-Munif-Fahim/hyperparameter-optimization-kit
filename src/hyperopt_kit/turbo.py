"""TuRBO-1: trust-region Bayesian optimization (Eriksson et al., 2019).

Global Bayesian optimization with one Gaussian process over the whole cube
over-explores in more than a handful of dimensions: the surrogate is too
smooth everywhere and the acquisition keeps proposing points at the
boundary. TuRBO keeps the GP but restricts proposals to a *trust region*, a
hypercube of side ``L`` centred on the best point of the current run:

* candidates are drawn inside the box, each one perturbing a random subset
  of coordinates (probability ``min(1, 20 / d)``, as in the reference
  implementation) so high-dimensional searches still move along few axes;
* the next point is the minimiser of one joint Thompson sample of the GP
  posterior over those candidates;
* after ``success_tolerance`` consecutive improvements the box doubles (up
  to ``length_max``); after ``failure_tolerance`` consecutive failures it
  halves. When it falls below ``length_min`` the run has converged
  locally and TuRBO *restarts* with a fresh Latin hypercube design and a
  new GP, keeping the global best.

The kit's :class:`~hyperopt_kit.searchers.GaussianProcess` is isotropic, so
the trust region is a cube (the paper stretches it by ARD length scales).
Inputs are the normalized coordinates of the configurations actually
evaluated, so integer and categorical parameters work with the same caveat
as ``bayesian_search``. NumPy only; minimization convention.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Dict, List, Optional

import numpy as np

from .acquisition import thompson_sample
from .quasi_random import latin_hypercube
from .searchers import GaussianProcess, Stopper, Trial, _validate_space
from .spaces import Space


@dataclass(frozen=True)
class TurboState:
    """Trust-region bookkeeping of one TuRBO run."""

    dims: int
    length: float = 0.8
    length_min: float = 0.5 ** 7
    length_max: float = 1.6
    success_counter: int = 0
    failure_counter: int = 0
    success_tolerance: int = 3
    failure_tolerance: int = 4
    restart_triggered: bool = False


def turbo_initial_state(
    dims: int,
    *,
    length_init: float = 0.8,
    length_min: float = 0.5 ** 7,
    length_max: float = 1.6,
    success_tolerance: int = 3,
    failure_tolerance: Optional[int] = None,
) -> TurboState:
    """Fresh :class:`TurboState`; ``failure_tolerance`` defaults to ``max(4, dims)``."""
    if isinstance(dims, bool) or not isinstance(dims, int) or dims < 1:
        raise ValueError("dims must be a positive integer")
    if not 0.0 < length_min < length_init <= length_max:
        raise ValueError("need 0 < length_min < length_init <= length_max")
    if success_tolerance < 1:
        raise ValueError("success_tolerance must be >= 1")
    if failure_tolerance is None:
        failure_tolerance = max(4, dims)
    if failure_tolerance < 1:
        raise ValueError("failure_tolerance must be >= 1")
    return TurboState(
        dims=dims,
        length=float(length_init),
        length_min=float(length_min),
        length_max=float(length_max),
        success_tolerance=int(success_tolerance),
        failure_tolerance=int(failure_tolerance),
    )


def update_turbo_state(state: TurboState, improved: bool) -> TurboState:
    """Apply one success / failure to the trust region (expand, shrink or flag a restart).

    ``improved`` should be true when the new observation beats the run's
    best by a small relative margin (``1e-3 |best|`` in :func:`turbo_search`).
    """
    if improved:
        succ, fail = state.success_counter + 1, 0
    else:
        succ, fail = 0, state.failure_counter + 1
    length = state.length
    if succ >= state.success_tolerance:
        length, succ = min(2.0 * length, state.length_max), 0
    elif fail >= state.failure_tolerance:
        length, fail = length / 2.0, 0
    return replace(
        state,
        length=length,
        success_counter=succ,
        failure_counter=fail,
        restart_triggered=length < state.length_min,
    )


def trust_region_bounds(center: np.ndarray, length: float) -> tuple:
    """Box ``[center - L/2, center + L/2]`` clipped to the unit cube.

    The box keeps its side length when the centre is near a face (it is
    shifted inwards rather than truncated), as long as ``L <= 1``.
    """
    center = np.asarray(center, dtype=float)
    side = min(float(length), 1.0)
    lb = np.clip(center - side / 2.0, 0.0, 1.0 - side)
    ub = lb + side
    return lb, ub


def turbo_candidates(
    center: np.ndarray,
    length: float,
    n_candidates: int,
    rng: np.random.Generator,
    perturbation_probability: Optional[float] = None,
) -> np.ndarray:
    """Candidates inside the trust region that perturb a random subset of coordinates.

    Each coordinate is replaced by a uniform draw from the box with
    probability ``perturbation_probability`` (default ``min(1, 20/d)``); at
    least one coordinate always changes.
    """
    center = np.asarray(center, dtype=float)
    d = center.shape[0]
    lb, ub = trust_region_bounds(center, length)
    prob = min(1.0, 20.0 / d) if perturbation_probability is None else float(perturbation_probability)
    if not 0.0 < prob <= 1.0:
        raise ValueError("perturbation_probability must lie in (0, 1]")
    draws = lb + (ub - lb) * rng.random((n_candidates, d))
    mask = rng.random((n_candidates, d)) <= prob
    empty = ~mask.any(axis=1)
    if empty.any():
        mask[np.flatnonzero(empty), rng.integers(0, d, size=int(empty.sum()))] = True
    return np.where(mask, draws, center[None, :])


def _encode(space: Dict[str, Space], names: List[str], params: Dict) -> np.ndarray:
    return np.array([space[n].normalize(params[n]) for n in names], dtype=float)


def turbo_search(
    space: Dict[str, Space],
    objective,
    budget: int,
    rng: Optional[np.random.Generator] = None,
    stop_when: Stopper = None,
    *,
    n_initial: Optional[int] = None,
    n_candidates: Optional[int] = None,
    length_init: float = 0.8,
    length_min: float = 0.5 ** 7,
    length_max: float = 1.6,
    success_tolerance: int = 3,
    failure_tolerance: Optional[int] = None,
    max_gp_points: int = 200,
    history: Optional[list] = None,
) -> List[Trial]:
    """TuRBO-1 trust-region Bayesian optimization (minimization).

    Each run starts with ``n_initial`` Latin-hypercube points (default
    ``max(2, min(2 d, 10))``, capped by the remaining budget), then
    repeatedly fits the GP on that run's observations (the most recent
    ``max_gp_points``), draws ``n_candidates`` points (default
    ``min(100 d, 500)``) in the trust region around the run's best point and
    evaluates the Thompson-sample minimiser. The trust region is updated
    with :func:`update_turbo_state`; when it collapses below
    ``length_min`` a new run starts. Non-finite scores count as failures
    and are kept out of the GP.

    ``history``, if given, is extended with one dict per evaluation:
    ``iteration``, ``run`` (restart counter), ``length`` (trust-region side
    when the point was proposed, ``None`` for initial-design points) and
    ``restart`` (true when that evaluation triggered a restart).
    """
    _validate_space(space)
    if budget < 1:
        raise ValueError("budget must be >= 1")
    if max_gp_points < 2:
        raise ValueError("max_gp_points must be >= 2")
    names = list(space)
    d = len(names)
    # validates the trust-region settings up front
    template = turbo_initial_state(
        d,
        length_init=length_init,
        length_min=length_min,
        length_max=length_max,
        success_tolerance=success_tolerance,
        failure_tolerance=failure_tolerance,
    )
    if n_initial is None:
        n_initial = max(2, min(2 * d, 10))
    n_initial = int(n_initial)
    if n_initial < 1:
        raise ValueError("n_initial must be >= 1")
    if n_candidates is None:
        n_candidates = min(100 * d, 500)
    n_candidates = int(n_candidates)
    if n_candidates < 1:
        raise ValueError("n_candidates must be >= 1")
    rng = rng if rng is not None else np.random.default_rng()

    trials: List[Trial] = []
    global_best = float("inf")
    run = 0

    def evaluate(params: Dict, length: Optional[float]) -> float:
        nonlocal global_best
        score = float(objective(params))
        trials.append(Trial(len(trials), params, score))
        if np.isfinite(score):
            global_best = min(global_best, score)
        if history is not None:
            history.append(
                {"iteration": len(trials) - 1, "run": run, "length": length, "restart": False}
            )
        return score

    def stopped() -> bool:
        return stop_when is not None and stop_when(global_best, len(trials))

    while len(trials) < budget:
        state = template
        X: List[np.ndarray] = []
        y: List[float] = []
        design = latin_hypercube(min(n_initial, budget - len(trials)), d, rng=rng)
        for u in design:
            params = {n: space[n].denormalize(float(u[j])) for j, n in enumerate(names)}
            score = evaluate(params, None)
            if np.isfinite(score):
                X.append(_encode(space, names, params))
                y.append(score)
            if stopped():
                return trials
        while len(trials) < budget:
            if len(y) == 0:
                break  # nothing finite to centre on: restart with a new design
            best_idx = int(np.argmin(y))
            run_best = y[best_idx]
            center = X[best_idx]
            Xa = np.asarray(X[-max_gp_points:])
            ya = np.asarray(y[-max_gp_points:])
            cand = turbo_candidates(center, state.length, n_candidates, rng)
            if len(ya) >= 2:
                gp = GaussianProcess()
                gp.fit(Xa, ya)
                sample = thompson_sample(gp, cand, rng)
                pick = cand[int(np.argmin(sample))]
            else:
                pick = cand[int(rng.integers(0, len(cand)))]
            params = {n: space[n].denormalize(float(pick[j])) for j, n in enumerate(names)}
            score = evaluate(params, state.length)
            improved = bool(np.isfinite(score) and score < run_best - 1e-3 * abs(run_best))
            if np.isfinite(score):
                X.append(_encode(space, names, params))
                y.append(score)
            state = update_turbo_state(state, improved)
            if stopped():
                return trials
            if state.restart_triggered:
                if history is not None:
                    history[-1]["restart"] = True
                break
        run += 1
    return trials
