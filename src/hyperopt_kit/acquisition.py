"""Gaussian-process acquisition portfolio and batch Bayesian optimization.

``bayesian_search`` only uses expected improvement. This module adds the
other standard GP acquisition functions and two searchers built on them:

* :func:`probability_of_improvement` (Kushner, 1964),
* :func:`lower_confidence_bound`: GP-LCB, the minimization form of GP-UCB,
  with the Srinivas et al. (2010) ``beta_t`` schedule from :func:`gp_ucb_beta`,
* :func:`thompson_sample`: a draw from the joint GP posterior over a
  candidate set,
* :func:`gp_hedge_search`: the GP-Hedge portfolio (Hoffman, Brochu & de
  Freitas, 2011). Every round each acquisition nominates a point, one
  nominee is picked with softmax(eta * gains), and every acquisition's gain
  is credited with the posterior mean at its own nominee (lower is better),
* :func:`batch_bayesian_search`: q-point batches proposed sequentially
  with the Kriging-believer or constant-liar heuristic (Ginsbourger, Le
  Riche & Carraro, 2010). Pending points are "hallucinated" so that the
  rest of the batch spreads out.

All functions follow the kit's minimization convention.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .searchers import (
    GaussianProcess,
    Stopper,
    Trial,
    _norm_cdf,
    _norm_pdf,
    _sample_config,
    _validate_space,
    expected_improvement,
)
from .spaces import Space

ACQUISITIONS = ("ei", "pi", "lcb", "ts")


def probability_of_improvement(mu, sigma, best: float, xi: float = 0.01) -> np.ndarray:
    """``P(f(x) < best - xi)`` under the Gaussian predictive distribution."""
    mu = np.asarray(mu, dtype=float)
    sigma = np.maximum(np.asarray(sigma, dtype=float), 1e-12)
    return _norm_cdf((best - xi - mu) / sigma)


def gp_ucb_beta(t: int, dims: int, delta: float = 0.1) -> float:
    """Srinivas et al. (2010) Theorem 1 schedule ``2 log(|D| t^2 pi^2 / 6 delta)``.

    ``|D|`` is taken as ``10^dims``, a discretization of the unit cube.
    ``t`` counts iterations from 1.
    """
    if t < 1:
        raise ValueError("t must be >= 1")
    if dims < 1:
        raise ValueError("dims must be >= 1")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie in (0, 1)")
    log_d = dims * math.log(10.0)
    return 2.0 * (log_d + 2.0 * math.log(t) + math.log(math.pi ** 2 / (6.0 * delta)))


def lower_confidence_bound(mu, sigma, beta: float = 4.0) -> np.ndarray:
    """GP-LCB score ``-(mu - sqrt(beta) * sigma)``: bigger is better, like EI."""
    if beta < 0:
        raise ValueError("beta must be >= 0")
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    return -(mu - math.sqrt(beta) * sigma)


def posterior_covariance(gp: GaussianProcess, Xstar: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Joint predictive mean and covariance of a fitted GP at ``Xstar``."""
    if gp._alpha is None or gp._X is None:
        raise RuntimeError("GaussianProcess must be fit first")
    Xstar = np.asarray(Xstar, dtype=float).reshape(-1, gp._X.shape[1])
    ls = gp.length_scale
    kstar = gp._kernel(gp._X, Xstar, ls)
    K = gp._gram(gp._X, ls)
    mu = (kstar.T @ gp._alpha) * gp._y_std + gp._y_mean
    cov = gp._kernel(Xstar, Xstar, ls) - kstar.T @ np.linalg.solve(K, kstar)
    cov = 0.5 * (cov + cov.T) * gp._y_std ** 2
    return mu, cov


def thompson_sample(gp: GaussianProcess, Xstar: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One joint posterior draw of ``f`` at ``Xstar`` (eigen-decomposition, PSD-clipped)."""
    mu, cov = posterior_covariance(gp, Xstar)
    w, V = np.linalg.eigh(cov)
    w = np.clip(w, 0.0, None)
    z = rng.standard_normal(len(w))
    return mu + V @ (np.sqrt(w) * z)


def _acq_scores(name, gp, cand, best, rng, *, xi, beta):
    if name == "ts":
        return -thompson_sample(gp, cand, rng)
    mu, var = gp.predict(cand)
    sigma = np.sqrt(var)
    if name == "ei":
        return expected_improvement(mu, sigma, best, xi)
    if name == "pi":
        return probability_of_improvement(mu, sigma, best, xi)
    if name == "lcb":
        return lower_confidence_bound(mu, sigma, beta)
    raise ValueError(f"unknown acquisition {name!r}; expected one of {ACQUISITIONS}")


def _initial(space, objective, n_initial, rng, stop_when):
    trials: List[Trial] = []
    best = float("inf")
    for i in range(n_initial):
        params = _sample_config(space, rng)
        score = objective(params)
        trials.append(Trial(i, params, score))
        best = min(best, score)
        if stop_when is not None and stop_when(best, len(trials)):
            return trials, True
    return trials, False


def _resolve_n_initial(n_initial, dims, budget):
    if n_initial is None:
        n_initial = max(2, dims + 1)
    return min(max(1, int(n_initial)), budget)


def _beta_for(beta, t, dims):
    return gp_ucb_beta(t, dims) if beta is None else float(beta)


def gp_hedge_search(
    space: Dict[str, Space],
    objective,
    budget: int,
    rng: Optional[np.random.Generator] = None,
    stop_when: Stopper = None,
    *,
    acquisitions: Sequence[str] = ("ei", "pi", "lcb"),
    eta: float = 1.0,
    xi: float = 0.01,
    beta: Optional[float] = None,
    n_initial: Optional[int] = None,
    n_candidates: int = 200,
    history: Optional[list] = None,
) -> List[Trial]:
    """GP-Hedge portfolio Bayesian optimization (Hoffman et al., 2011).

    ``beta=None`` uses :func:`gp_ucb_beta`. If ``history`` is a list, it is
    filled with ``{"chosen", "probabilities", "gains"}`` for every round.
    """
    _validate_space(space)
    if budget < 1:
        raise ValueError("budget must be >= 1")
    acquisitions = tuple(acquisitions)
    if not acquisitions:
        raise ValueError("acquisitions must be non-empty")
    for a in acquisitions:
        if a not in ACQUISITIONS:
            raise ValueError(f"unknown acquisition {a!r}; expected one of {ACQUISITIONS}")
    if eta <= 0:
        raise ValueError("eta must be > 0")
    rng = rng if rng is not None else np.random.default_rng()
    names = list(space)
    dims = len(names)
    trials, stopped = _initial(space, objective, _resolve_n_initial(n_initial, dims, budget), rng, stop_when)
    if stopped:
        return trials
    X = np.array([[space[n].normalize(t.params[n]) for n in names] for t in trials])
    y = np.array([t.score for t in trials])
    gains = np.zeros(len(acquisitions))
    gp = GaussianProcess()
    gp.fit(X, y)
    t = 1
    while len(trials) < budget:
        best = float(np.min(y))
        cand = rng.random((n_candidates, dims))
        nominees = []
        for a in acquisitions:
            s = _acq_scores(a, gp, cand, best, rng, xi=xi, beta=_beta_for(beta, t, dims))
            nominees.append(cand[int(np.nanargmax(s))])
        logits = eta * (gains - gains.max())
        probs = np.exp(logits) / np.exp(logits).sum()
        k = int(rng.choice(len(acquisitions), p=probs))
        x = nominees[k]
        params = {n: space[n].denormalize(float(x[j])) for j, n in enumerate(names)}
        score = objective(params)
        trials.append(Trial(len(trials), params, score))
        X = np.vstack([X, x[None, :]])
        y = np.append(y, score)
        gp.fit(X, y)
        mu_nom, _ = gp.predict(np.array(nominees))
        # reward = -posterior mean (minimization); standardize so eta is scale-free
        gains = gains - (mu_nom - gp._y_mean) / gp._y_std
        if history is not None:
            history.append(
                {"chosen": acquisitions[k], "probabilities": probs.tolist(), "gains": gains.tolist()}
            )
        t += 1
        if stop_when is not None and stop_when(float(np.min(y)), len(trials)):
            break
    return trials


def batch_bayesian_search(
    space: Dict[str, Space],
    objective,
    budget: int,
    rng: Optional[np.random.Generator] = None,
    stop_when: Stopper = None,
    *,
    batch_size: int = 4,
    acquisition: str = "ei",
    lie: str = "kriging_believer",
    xi: float = 0.01,
    beta: Optional[float] = None,
    n_initial: Optional[int] = None,
    n_candidates: int = 200,
) -> List[Trial]:
    """Synchronous batch BO: propose ``batch_size`` points, then evaluate them.

    Within a batch each chosen point gets a fake observation before the next
    point is chosen. ``lie`` is one of ``"kriging_believer"`` (the GP mean),
    ``"cl_min"``, ``"cl_mean"`` or ``"cl_max"`` (constant liar with the
    min/mean/max of the observed scores). ``Trial.resource`` is left as
    ``None``. Each trial's iteration index counts evaluations; batch ``b``
    covers iterations ``n_initial + b*q ... n_initial + (b+1)*q - 1``.
    """
    _validate_space(space)
    if budget < 1:
        raise ValueError("budget must be >= 1")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if acquisition not in ACQUISITIONS:
        raise ValueError(f"unknown acquisition {acquisition!r}; expected one of {ACQUISITIONS}")
    lies = ("kriging_believer", "cl_min", "cl_mean", "cl_max")
    if lie not in lies:
        raise ValueError(f"lie must be one of {lies}")
    rng = rng if rng is not None else np.random.default_rng()
    names = list(space)
    dims = len(names)
    trials, stopped = _initial(space, objective, _resolve_n_initial(n_initial, dims, budget), rng, stop_when)
    if stopped:
        return trials
    X = np.array([[space[n].normalize(t.params[n]) for n in names] for t in trials])
    y = np.array([t.score for t in trials])
    t = 1
    while len(trials) < budget:
        q = min(batch_size, budget - len(trials))
        batch = propose_batch(X, y, q, rng, acquisition=acquisition, lie=lie, xi=xi,
                              beta=_beta_for(beta, t, dims), n_candidates=n_candidates)
        stop = False
        for x in batch:
            params = {n: space[n].denormalize(float(x[j])) for j, n in enumerate(names)}
            score = objective(params)
            trials.append(Trial(len(trials), params, score))
            X = np.vstack([X, x[None, :]])
            y = np.append(y, score)
            if stop_when is not None and stop_when(float(np.min(y)), len(trials)):
                stop = True
                break
        if stop:
            break
        t += 1
    return trials


def propose_batch(
    X: np.ndarray,
    y: np.ndarray,
    q: int,
    rng: np.random.Generator,
    *,
    acquisition: str = "ei",
    lie: str = "kriging_believer",
    xi: float = 0.01,
    beta: float = 4.0,
    n_candidates: int = 200,
) -> np.ndarray:
    """Return ``q`` normalized points chosen by sequential hallucination."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if q < 1:
        raise ValueError("q must be >= 1")
    dims = X.shape[1]
    const = {"cl_min": np.min(y), "cl_mean": np.mean(y), "cl_max": np.max(y)}.get(lie)
    Xf, yf = X.copy(), y.copy()
    gp = GaussianProcess()
    gp.fit(Xf, yf)  # length scale is chosen once, on real data only
    out = []
    for _ in range(q):
        cand = rng.random((n_candidates, dims))
        s = _acq_scores(acquisition, gp, cand, float(np.min(yf)), rng, xi=xi, beta=beta)
        x = cand[int(np.nanargmax(s))]
        out.append(x)
        fake = float(gp.predict(x[None, :])[0][0]) if const is None else float(const)
        Xf = np.vstack([Xf, x[None, :]])
        yf = np.append(yf, fake)
        gp.fit(Xf, yf, fit_hyperparams=False)
    return np.array(out)
