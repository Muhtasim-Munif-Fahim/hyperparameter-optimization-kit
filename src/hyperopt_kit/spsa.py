"""Simultaneous Perturbation Stochastic Approximation (Spall, 1992).

SPSA estimates the full gradient from only **two** objective evaluations per
iteration, regardless of dimension. With a Rademacher direction
``delta in {-1, +1}^d`` and perturbation size ``c_k``::

    g_k = (f(x + c_k delta) - f(x - c_k delta)) / (2 c_k delta)
    x_{k+1} = x_k - a_k g_k

and the standard gain sequences ``a_k = a / (k + 1 + A)^alpha`` and
``c_k = c / (k + 1)^gamma`` (``alpha = 0.602``, ``gamma = 0.101``, Spall 1998).
Because the gradient is a difference quotient, SPSA tolerates noisy
objectives (stochastic training runs) better than pattern searches.

The search runs in the normalized unit cube of each numeric
:class:`~hyperopt_kit.spaces.Space` and clips iterates to ``[0, 1]``.
Categoricals stay fixed while one SPSA run descends. A run restarts from a
fresh random configuration (which redraws categoricals) after ``patience``
iterations without improving the best score seen in that run.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np

from .searchers import (
    ScoreFn,
    Stopper,
    Trial,
    _is_real_number,
    _is_strict_int,
    _sample_config,
    _validate_space,
    random_search,
)
from .spaces import Categorical, Space

__all__ = ["spsa_gains", "spsa_search"]


def spsa_gains(k: int, a: float, c: float, A: float, alpha: float = 0.602, gamma: float = 0.101):
    """Return the SPSA gains ``(a_k, c_k)`` for zero-based iteration ``k``."""
    return a / (k + 1 + A) ** alpha, c / (k + 1) ** gamma


class _Done(Exception):
    pass


def spsa_search(
    space: Dict[str, Space],
    objective: ScoreFn,
    budget: int,
    rng: Optional[np.random.Generator] = None,
    stop_when: Stopper = None,
    *,
    c: float = 0.1,
    initial_step: float = 0.1,
    alpha: float = 0.602,
    gamma: float = 0.101,
    stability: Optional[float] = None,
    evaluate_iterate: bool = True,
    patience: int = 10,
    tol: float = 1e-4,
) -> List[Trial]:
    """Minimize ``objective`` with SPSA in the normalized search space.

    Parameters
    ----------
    c:
        Initial perturbation half-width in the unit cube (``c_0``).
    initial_step:
        Target size of the first update (per coordinate). The step gain ``a``
        is calibrated from the first gradient estimate so that
        ``a_0 * mean|g_0| = initial_step`` (Spall's practical rule).
    alpha, gamma:
        Gain-sequence exponents; the defaults are Spall's recommendations.
    stability:
        The stability constant ``A`` (default: 10% of the expected number of
        iterations).
    evaluate_iterate:
        Also evaluate the new iterate ``x_{k+1}`` each iteration (3 evaluations
        per iteration instead of 2), so the best trial tracks the iterate.
    patience:
        Iterations without improvement before restarting from a random point.
    tol:
        Minimum relative improvement ``tol * max(1, |best|)`` that resets the
        patience counter, so a converged run restarts instead of creeping.

    Scores are minimized; non-finite scores count as ``+inf`` and make that
    iteration's gradient estimate zero. ``budget`` counts objective calls.
    """
    _validate_space(space)
    if not _is_strict_int(budget) or int(budget) < 1:
        raise ValueError("budget must be >= 1")
    budget = int(budget)
    for name, value in (("c", c), ("initial_step", initial_step)):
        if not _is_real_number(value) or not 0.0 < float(value) <= 1.0:
            raise ValueError(f"{name} must be in (0, 1]")
    if not _is_real_number(alpha) or not 0.0 < float(alpha) <= 1.0:
        raise ValueError("alpha must be in (0, 1]")
    if not _is_real_number(gamma) or not 0.0 < float(gamma) < float(alpha):
        raise ValueError("gamma must be in (0, alpha)")
    if stability is not None and (not _is_real_number(stability) or float(stability) < 0.0):
        raise ValueError("stability must be a non-negative number or None")
    if not _is_strict_int(patience) or int(patience) < 1:
        raise ValueError("patience must be a positive integer")
    if not _is_real_number(tol) or float(tol) < 0.0:
        raise ValueError("tol must be a non-negative number")

    rng = rng if rng is not None else np.random.default_rng()
    numeric = [n for n, sp in space.items() if not isinstance(sp, Categorical)]
    if not numeric:
        return random_search(space, objective, budget, rng=rng, stop_when=stop_when)

    evals_per_iter = 3 if evaluate_iterate else 2
    A = float(stability) if stability is not None else 0.1 * max(1.0, budget / evals_per_iter)
    trials: List[Trial] = []
    best = float("inf")

    def to_params(x: np.ndarray, cats: Dict[str, Any]) -> Dict[str, Any]:
        out = {name: space[name].denormalize(float(v)) for name, v in zip(numeric, x)}
        out.update(cats)
        return {name: out[name] for name in space}

    def evaluate(x: np.ndarray, cats: Dict[str, Any]) -> float:
        nonlocal best
        if len(trials) >= budget:
            raise _Done
        params = to_params(x, cats)
        raw = float(objective(params))
        trials.append(Trial(len(trials), params, raw))
        score = raw if math.isfinite(raw) else float("inf")
        best = min(best, score)
        if stop_when is not None and stop_when(best, len(trials)):
            raise _Done
        if len(trials) >= budget:
            raise _Done
        return score

    try:
        while True:
            start = _sample_config(space, rng)
            x = np.array([space[n].normalize(start[n]) for n in numeric], dtype=float)
            cats = {n: start[n] for n in space if n not in numeric}
            run_best = evaluate(x, cats)
            a_gain: Optional[float] = None
            stall = 0
            k = 0
            while stall < patience:
                _, ck = spsa_gains(k, 1.0, float(c), A, float(alpha), float(gamma))
                delta = rng.choice([-1.0, 1.0], size=x.size)
                xp = np.clip(x + ck * delta, 0.0, 1.0)
                xm = np.clip(x - ck * delta, 0.0, 1.0)
                yp = evaluate(xp, cats)
                ym = evaluate(xm, cats)
                width = xp - xm
                if math.isfinite(yp) and math.isfinite(ym):
                    g = np.divide(yp - ym, width, out=np.zeros_like(x), where=width != 0)
                else:
                    g = np.zeros_like(x)
                if a_gain is None:
                    scale = float(np.mean(np.abs(g)))
                    if scale > 0:
                        a_gain = float(initial_step) * (1 + A) ** float(alpha) / scale
                ak = 0.0 if a_gain is None else spsa_gains(k, a_gain, float(c), A, float(alpha), float(gamma))[0]
                x = np.clip(x - ak * g, 0.0, 1.0)
                threshold = run_best - float(tol) * max(1.0, abs(run_best)) if math.isfinite(run_best) else run_best
                new_best = min(yp, ym)
                if evaluate_iterate:
                    new_best = min(new_best, evaluate(x, cats))
                improved = new_best < threshold
                run_best = min(run_best, new_best)
                stall = 0 if improved else stall + 1
                k += 1
    except _Done:
        pass
    return trials
