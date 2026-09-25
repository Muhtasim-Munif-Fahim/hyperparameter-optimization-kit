"""Search strategies over a ``{name: Space}`` configuration space.

``grid_search`` and ``random_search`` are self-explanatory baselines;
``bayesian_search`` fits a squared-exponential Gaussian-process surrogate
and picks the next configuration by maximizing expected improvement;
``tpe_search`` models good vs bad observations with Parzen estimators and
picks the candidate that maximizes the ``l(x)/g(x)`` density ratio;
``cmaes_search`` maintains a multivariate Gaussian in normalized
FloatRange / IntRange space and adapts its mean, step-size and covariance
from ranked offspring (Hansen CMA-ES);
``hyperband_search`` / ``successive_halving`` run multi-fidelity brackets
that spend cheap evaluations to discard poor configurations early;
``random_successive_halving`` draws the same uniform configurations as
``random_search`` and stops them with that successive-halving rule on an
open-ended stream (a configuration continues only while it stays in the
top ``1/eta`` of its current rung. Gaussian-process expected improvement
is already provided by ``bayesian_search``; the early-stopping random
search is the method added beside it.
``bohb_search`` keeps that Hyperband schedule but proposes configurations
from a multivariate product-kernel density of the best observations
(Falkner, Klein, Hutter, 2018) instead of sampling them uniformly.
``pbt_search`` trains a population in parallel and, every few resource
rungs, copies weights and hyperparameters from better members into worse
ones, then perturbs those hyperparameters (Jaderberg et al., 2017).
"""

from __future__ import annotations

import inspect
import itertools
import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .spaces import Categorical, FloatRange, IntRange, Space

ScoreFn = Callable[..., float]
Stopper = Optional[Callable[[float, int], bool]]


@dataclass(frozen=True)
class Trial:
    """One evaluated configuration.

    ``resource`` is the normalized fidelity in ``(0, 1]`` used for this
    evaluation (``1.0`` is full fidelity). Single-fidelity searchers leave
    it as ``None``.
    """

    iteration: int
    params: Dict[str, Any]
    score: float
    resource: Optional[float] = None


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


_ERF = np.vectorize(math.erf)


def _norm_pdf(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + _ERF(x / np.sqrt(2.0)))


class GaussianProcess:
    """Squared-exponential Gaussian process with a ridge-regularized solve.

    Works on normalized inputs in ``[0, 1]^d`` and standardized targets. The
    length scale is picked from a small candidate set by maximizing the log
    marginal likelihood, which keeps the surrogate stable with the few dozen
    points typical of hyperparameter search.
    """

    def __init__(
        self,
        signal_variance: float = 1.0,
        noise_variance: float = 1e-6,
        ridge: float = 1e-8,
        length_scale_candidates: tuple = (0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5),
    ):
        self.signal_variance = signal_variance
        self.noise_variance = noise_variance
        self.ridge = ridge
        self.length_scale_candidates = tuple(length_scale_candidates)
        self.length_scale = 0.5
        self._X: Optional[np.ndarray] = None
        self._y: Optional[np.ndarray] = None
        self._alpha: Optional[np.ndarray] = None
        self._y_mean = 0.0
        self._y_std = 1.0

    def _kernel(self, X: np.ndarray, Y: np.ndarray, length_scale: float) -> np.ndarray:
        sq = (
            np.sum(X * X, axis=1)[:, None]
            + np.sum(Y * Y, axis=1)[None, :]
            - 2.0 * X @ Y.T
        )
        sq = np.maximum(sq, 0.0)
        return self.signal_variance * np.exp(-0.5 * sq / (length_scale * length_scale))

    def _gram(self, X: np.ndarray, length_scale: float) -> np.ndarray:
        return self._kernel(X, X, length_scale) + (
            self.noise_variance + self.ridge
        ) * np.eye(X.shape[0])

    def _log_marginal_likelihood(
        self, X: np.ndarray, y: np.ndarray, length_scale: float
    ) -> float:
        K = self._gram(X, length_scale)
        try:
            alpha = np.linalg.solve(K, y)
            sign, logdet = np.linalg.slogdet(K)
        except np.linalg.LinAlgError:
            return -np.inf
        if sign <= 0:
            return -np.inf
        return (
            -0.5 * float(y @ alpha)
            - 0.5 * float(logdet)
            - 0.5 * X.shape[0] * np.log(2.0 * np.pi)
        )

    def fit(self, X: np.ndarray, y: np.ndarray, fit_hyperparams: bool = True) -> None:
        """Fit the surrogate on observed (normalized) inputs and targets."""
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        y = np.asarray(y, dtype=float)
        if X.ndim != 2 or X.shape[0] == 0:
            raise ValueError("X must be a non-empty 2-d array")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows")
        self._X = X
        self._y = y
        self._y_mean = float(np.mean(y))
        y_std = float(np.std(y))
        self._y_std = y_std if y_std > 0.0 else 1.0
        y_norm = (y - self._y_mean) / self._y_std
        if fit_hyperparams and X.shape[0] >= 3:
            self.length_scale = max(
                self.length_scale_candidates,
                key=lambda ls: self._log_marginal_likelihood(X, y_norm, ls),
            )
        self._alpha = np.linalg.solve(self._gram(X, self.length_scale), y_norm)

    def predict(self, Xstar: np.ndarray) -> tuple:
        """Predictive mean and variance at normalized query points."""
        if self._alpha is None or self._X is None:
            raise RuntimeError("GaussianProcess must be fit before predict")
        Xstar = np.asarray(Xstar, dtype=float).reshape(-1, self._X.shape[1])
        kstar = self._kernel(self._X, Xstar, self.length_scale)
        K = self._gram(self._X, self.length_scale)
        K_inv_kstar = np.linalg.solve(K, kstar)
        mu_norm = kstar.T @ self._alpha
        var_norm = np.clip(
            np.diag(self._kernel(Xstar, Xstar, self.length_scale) - kstar.T @ K_inv_kstar),
            0.0,
            None,
        )
        mu = mu_norm * self._y_std + self._y_mean
        var = var_norm * self._y_std * self._y_std
        return mu, var


def expected_improvement(
    mu: np.ndarray, sigma: np.ndarray, best: float, xi: float = 0.01
) -> np.ndarray:
    """Expected improvement over ``best`` (minimize convention).

    ``xi`` trades exploration against exploitation: larger values push the
    acquisition towards regions of high predictive uncertainty.
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    sigma = np.maximum(sigma, 1e-12)
    z = (best - xi - mu) / sigma
    return (best - xi - mu) * _norm_cdf(z) + sigma * _norm_pdf(z)


def bayesian_search(
    space: Dict[str, Space],
    objective: ScoreFn,
    budget: int,
    rng: Optional[np.random.Generator] = None,
    *,
    xi: float = 0.01,
    n_initial: Optional[int] = None,
    n_candidates: int = 200,
    stop_when: Stopper = None,
) -> List[Trial]:
    """Bayesian optimization with a Gaussian-process surrogate.

    Starts from ``n_initial`` random configurations, then proposes the next
    configuration by maximizing expected improvement over a random set of
    candidate points in normalized space.
    """
    _validate_space(space)
    if budget < 1:
        raise ValueError("budget must be >= 1")
    rng = rng if rng is not None else np.random.default_rng()
    names = list(space)
    dims = len(names)

    if n_initial is None:
        n_initial = max(2, dims + 1)
    n_initial = max(1, int(n_initial))
    n_initial = min(n_initial, budget)

    trials: List[Trial] = []
    best = float("inf")
    for i in range(n_initial):
        params = _sample_config(space, rng)
        score = objective(params)
        trials.append(Trial(i, params, score))
        best = min(best, score)
        if stop_when is not None and stop_when(best, len(trials)):
            return trials

    if len(trials) < budget:
        X = np.array(
            [[space[n].normalize(t.params[n]) for n in names] for t in trials]
        )
        y = np.array([t.score for t in trials])
        gp = GaussianProcess()
        gp.fit(X, y)
        while len(trials) < budget:
            candidates = rng.random((n_candidates, dims))
            mu, var = gp.predict(candidates)
            ei = expected_improvement(mu, np.sqrt(var), best, xi)
            idx = int(np.argmax(ei))
            if not np.isfinite(ei[idx]) or ei[idx] <= 0.0:
                idx = int(rng.integers(0, n_candidates))
            params = {
                n: space[n].denormalize(float(candidates[idx, j]))
                for j, n in enumerate(names)
            }
            score = objective(params)
            trials.append(Trial(len(trials), params, score))
            best = min(best, score)
            if stop_when is not None and stop_when(best, len(trials)):
                break
            X = np.vstack([X, candidates[idx][None, :]])
            y = np.append(y, score)
            gp.fit(X, y)
    return trials


def _logsumexp(a: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable log-sum-exp along ``axis``."""
    a = np.asarray(a, dtype=float)
    keep = np.max(a, axis=axis, keepdims=True)
    safe = np.where(np.isfinite(keep), keep, 0.0)
    summed = np.log(np.maximum(np.sum(np.exp(a - safe), axis=axis, keepdims=True), 0.0))
    out = np.squeeze(summed + keep, axis=axis)
    finite = np.isfinite(np.squeeze(keep, axis=axis))
    return np.where(finite, out, -np.inf)


def tpe_split(scores, gamma: float = 0.25) -> Tuple[np.ndarray, np.ndarray]:
    """Split observation indices into below-threshold (good) and above (bad).

    ``gamma`` is the TPE quantile: roughly that fraction of the lowest scores
    form ``l(x)``. At least one observation is kept on each side whenever
    ``len(scores) >= 2``.
    """
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or scores.size == 0:
        raise ValueError("scores must be a non-empty 1-d array")
    if not (0.0 < float(gamma) < 1.0):
        raise ValueError("gamma must be in (0, 1)")
    n = int(scores.size)
    if n == 1:
        return np.array([0], dtype=int), np.array([], dtype=int)
    n_below = int(np.floor(float(gamma) * n))
    n_below = min(max(n_below, 1), n - 1)
    order = np.argsort(scores, kind="mergesort")
    return order[:n_below].astype(int), order[n_below:].astype(int)


def adaptive_parzen(
    observations: np.ndarray,
    *,
    prior_mu: float = 0.5,
    prior_sigma: float = 1.0,
    prior_weight: float = 1.0,
    bandwidth_factor: float = 1.0,
    low: float = 0.0,
    high: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """1-d adaptive Parzen estimator (Bergstra et al., 2011).

    Returns ``(weights, means, stds)`` of a Gaussian mixture. Each observation
    is a kernel whose bandwidth is the greater of the distances to its sorted
    neighbours, mixed with a prior component so empty regions still have mass.
    """
    xs = np.asarray(observations, dtype=float).reshape(-1)
    if prior_weight <= 0.0:
        raise ValueError("prior_weight must be > 0")
    if bandwidth_factor <= 0.0:
        raise ValueError("bandwidth_factor must be > 0")
    span = max(float(high) - float(low), 1e-12)
    prior_sigma = max(float(prior_sigma), 1e-12)

    if xs.size == 0:
        return (
            np.array([1.0], dtype=float),
            np.array([float(prior_mu)], dtype=float),
            np.array([prior_sigma], dtype=float),
        )

    order = np.argsort(xs)
    sorted_xs = xs[order]
    sorted_sigma = np.empty(xs.size, dtype=float)
    if xs.size == 1:
        sorted_sigma[0] = prior_sigma
    else:
        sorted_sigma[0] = sorted_xs[1] - sorted_xs[0]
        sorted_sigma[-1] = sorted_xs[-1] - sorted_xs[-2]
        if xs.size > 2:
            left = sorted_xs[1:-1] - sorted_xs[:-2]
            right = sorted_xs[2:] - sorted_xs[1:-1]
            sorted_sigma[1:-1] = np.maximum(left, right)
    sigma = np.empty(xs.size, dtype=float)
    sigma[order] = sorted_sigma * float(bandwidth_factor)
    minsigma = span / max(100.0, 1.0 + float(xs.size))
    sigma = np.clip(sigma, minsigma, span)

    weights = np.ones(xs.size + 1, dtype=float)
    weights[-1] = float(prior_weight)
    weights /= weights.sum()
    means = np.append(xs, float(prior_mu))
    stds = np.append(sigma, prior_sigma)
    return weights, means, stds


def categorical_probs(
    space: Categorical, values, prior_weight: float = 1.0
) -> np.ndarray:
    """Dirichlet-smoothed categorical probabilities over ``space.choices``."""
    if prior_weight <= 0.0:
        raise ValueError("prior_weight must be > 0")
    counts = np.zeros(len(space.choices), dtype=float)
    for value in values:
        counts[space.choices.index(value)] += 1.0
    probs = counts + float(prior_weight)
    return probs / probs.sum()


def gmm_logpdf(
    x: np.ndarray,
    weights: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    low: float = 0.0,
    high: float = 1.0,
) -> np.ndarray:
    """Log-density of a truncated 1-d Gaussian mixture on ``[low, high]``."""
    x = np.asarray(x, dtype=float).reshape(-1)
    weights = np.asarray(weights, dtype=float)
    means = np.asarray(means, dtype=float)
    stds = np.maximum(np.asarray(stds, dtype=float), 1e-12)
    z = (x[:, None] - means[None, :]) / stds[None, :]
    log_normal = np.log(np.maximum(_norm_pdf(z) / stds[None, :], 1e-300))
    mass = _norm_cdf((high - means) / stds) - _norm_cdf((low - means) / stds)
    mass = np.maximum(mass, 1e-12)
    log_comp = np.log(np.maximum(weights, 1e-300)) + log_normal - np.log(mass)
    out = _logsumexp(log_comp, axis=1)
    return np.where((x < low) | (x > high), -np.inf, out)


def _truncated_normal(
    rng: np.random.Generator,
    mu: float,
    sigma: float,
    low: float,
    high: float,
    size: int,
) -> np.ndarray:
    sigma = max(float(sigma), 1e-12)
    collected: List[np.ndarray] = []
    remaining = int(size)
    for _ in range(32):
        draw = rng.normal(mu, sigma, size=max(remaining * 4, 8))
        accepted = draw[(draw >= low) & (draw <= high)]
        if accepted.size:
            take = min(remaining, int(accepted.size))
            collected.append(accepted[:take])
            remaining -= take
        if remaining <= 0:
            return np.concatenate(collected)[:size]
    collected.append(np.clip(rng.normal(mu, sigma, size=remaining), low, high))
    return np.concatenate(collected)[:size]


def gmm_sample(
    rng: np.random.Generator,
    n: int,
    weights: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    low: float = 0.0,
    high: float = 1.0,
) -> np.ndarray:
    """Draw ``n`` samples from a truncated 1-d Gaussian mixture."""
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    means = np.asarray(means, dtype=float)
    stds = np.asarray(stds, dtype=float)
    comps = rng.choice(len(weights), size=n, p=weights)
    out = np.empty(n, dtype=float)
    for k in range(len(weights)):
        idx = np.flatnonzero(comps == k)
        if idx.size == 0:
            continue
        out[idx] = _truncated_normal(rng, float(means[k]), float(stds[k]), low, high, idx.size)
    return np.clip(out, low, high)


def _fit_parzen(
    sp: Space, values: List[Any], prior_weight: float, bandwidth_factor: float
):
    if isinstance(sp, Categorical):
        return ("cat", categorical_probs(sp, values, prior_weight))
    xs = np.array([sp.normalize(v) for v in values], dtype=float)
    weights, means, stds = adaptive_parzen(
        xs, prior_weight=prior_weight, bandwidth_factor=bandwidth_factor
    )
    return ("gmm", weights, means, stds)


def _sample_from_parzen(sp: Space, model, rng: np.random.Generator, n: int):
    kind = model[0]
    if kind == "cat":
        probs = model[1]
        idx = rng.choice(len(sp.choices), size=n, p=probs)
        values = [sp.choices[int(i)] for i in idx]
        logp = np.log(np.maximum(probs[idx], 1e-300))
        return values, logp
    weights, means, stds = model[1], model[2], model[3]
    xs = gmm_sample(rng, n, weights, means, stds)
    logp = gmm_logpdf(xs, weights, means, stds)
    return [sp.denormalize(float(x)) for x in xs], logp


def _parzen_logpdf(sp: Space, model, values: List[Any]) -> np.ndarray:
    kind = model[0]
    if kind == "cat":
        probs = model[1]
        idx = np.array([sp.choices.index(v) for v in values], dtype=int)
        return np.log(np.maximum(probs[idx], 1e-300))
    weights, means, stds = model[1], model[2], model[3]
    xs = np.array([sp.normalize(v) for v in values], dtype=float)
    return gmm_logpdf(xs, weights, means, stds)


def _fit_tpe_models(
    space: Dict[str, Space],
    names: List[str],
    below: List[Trial],
    above: List[Trial],
    prior_weight: float,
    bandwidth_factor: float,
):
    models_l = {}
    models_g = {}
    for name in names:
        sp = space[name]
        models_l[name] = _fit_parzen(
            sp, [t.params[name] for t in below], prior_weight, bandwidth_factor
        )
        models_g[name] = _fit_parzen(
            sp, [t.params[name] for t in above], prior_weight, bandwidth_factor
        )
    return models_l, models_g


def tpe_log_density_ratio(
    params: Dict[str, Any],
    space: Dict[str, Space],
    trials: List[Trial],
    *,
    gamma: float = 0.25,
    prior_weight: float = 1.0,
    bandwidth_factor: float = 1.0,
) -> float:
    """``log l(x) - log g(x)`` for ``params`` given observed ``trials``.

    ``l`` is the Parzen density of observations below the ``gamma`` quantile
    of scores; ``g`` is the density of the rest. TPE maximizes this ratio
    (equivalently expected improvement under the TPE model).
    """
    _validate_space(space)
    if len(trials) < 2:
        raise ValueError("tpe_log_density_ratio needs at least two trials")
    names = list(space)
    below_idx, above_idx = tpe_split([t.score for t in trials], gamma=gamma)
    below = [trials[int(i)] for i in below_idx]
    above = [trials[int(i)] for i in above_idx]
    models_l, models_g = _fit_tpe_models(
        space, names, below, above, prior_weight, bandwidth_factor
    )
    log_ratio = 0.0
    for name in names:
        log_ratio += float(
            _parzen_logpdf(space[name], models_l[name], [params[name]])[0]
            - _parzen_logpdf(space[name], models_g[name], [params[name]])[0]
        )
    return log_ratio


def _propose_tpe(
    space: Dict[str, Space],
    names: List[str],
    below: List[Trial],
    above: List[Trial],
    rng: np.random.Generator,
    n_candidates: int,
    prior_weight: float,
    bandwidth_factor: float,
) -> Dict[str, Any]:
    models_l, models_g = _fit_tpe_models(
        space, names, below, above, prior_weight, bandwidth_factor
    )
    samples: Dict[str, List[Any]] = {}
    log_ratio = np.zeros(n_candidates, dtype=float)
    for name in names:
        sp = space[name]
        sampled, log_l = _sample_from_parzen(sp, models_l[name], rng, n_candidates)
        log_g = _parzen_logpdf(sp, models_g[name], sampled)
        samples[name] = sampled
        log_ratio += log_l - log_g
    idx = int(np.argmax(log_ratio))
    if not np.isfinite(log_ratio[idx]):
        idx = int(rng.integers(0, n_candidates))
    return {name: samples[name][idx] for name in names}


def tpe_search(
    space: Dict[str, Space],
    objective: ScoreFn,
    budget: int,
    rng: Optional[np.random.Generator] = None,
    stop_when: Stopper = None,
    *,
    gamma: float = 0.25,
    n_initial: Optional[int] = None,
    n_candidates: int = 24,
    prior_weight: float = 1.0,
    bandwidth_factor: float = 1.0,
) -> List[Trial]:
    """Tree-structured Parzen Estimator search (Bergstra et al., 2011).

    After ``n_initial`` random evaluations, observations are split by the
    ``gamma`` quantile of scores. Independent Parzen estimators ``l(x)``
    (below-threshold / good) and ``g(x)`` (above / bad) are fit on each
    parameter using :meth:`Space.normalize` / :meth:`Space.denormalize`.
    Candidates are drawn from ``l`` and the one maximizing ``l(x)/g(x)``
    is evaluated. Expected improvement under this model is a monotone
    function of that density ratio.
    """
    _validate_space(space)
    if budget < 1:
        raise ValueError("budget must be >= 1")
    if not (0.0 < float(gamma) < 1.0):
        raise ValueError("gamma must be in (0, 1)")
    if int(n_candidates) != n_candidates or n_candidates < 1:
        raise ValueError("n_candidates must be an integer >= 1")
    if prior_weight <= 0.0:
        raise ValueError("prior_weight must be > 0")
    if bandwidth_factor <= 0.0:
        raise ValueError("bandwidth_factor must be > 0")
    rng = rng if rng is not None else np.random.default_rng()
    names = list(space)
    dims = len(names)

    if n_initial is None:
        n_initial = max(5, dims + 1)
    n_initial = max(1, int(n_initial))
    n_initial = min(n_initial, budget)

    trials: List[Trial] = []
    best = float("inf")
    for i in range(n_initial):
        params = _sample_config(space, rng)
        score = float(objective(params))
        trials.append(Trial(i, params, score))
        best = min(best, score)
        if stop_when is not None and stop_when(best, len(trials)):
            return trials

    while len(trials) < budget:
        if len(trials) < 2:
            params = _sample_config(space, rng)
        else:
            below_idx, above_idx = tpe_split([t.score for t in trials], gamma=gamma)
            below = [trials[int(i)] for i in below_idx]
            above = [trials[int(i)] for i in above_idx]
            params = _propose_tpe(
                space,
                names,
                below,
                above,
                rng,
                int(n_candidates),
                prior_weight,
                bandwidth_factor,
            )
        score = float(objective(params))
        trials.append(Trial(len(trials), params, score))
        best = min(best, score)
        if stop_when is not None and stop_when(best, len(trials)):
            break
    return trials


def _validate_hyperband_params(eta: int, min_resource: int, max_resource: int) -> None:
    if int(eta) != eta or eta < 2:
        raise ValueError("eta must be an integer >= 2")
    if int(min_resource) != min_resource or min_resource < 1:
        raise ValueError("min_resource must be an integer >= 1")
    if int(max_resource) != max_resource or max_resource < min_resource:
        raise ValueError("max_resource must be an integer >= min_resource")


def _resource_caller(objective: ScoreFn) -> Callable[[Dict[str, Any], float], float]:
    """Adapt ``objective(params)`` or ``objective(params, resource=...)``."""

    def _single(params: Dict[str, Any], resource: float) -> float:
        return float(objective(params))

    try:
        signature = inspect.signature(objective)
    except (TypeError, ValueError):
        return _single

    parameters = signature.parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return lambda params, resource: float(objective(params, resource=resource))
    if "resource" in parameters:
        return lambda params, resource: float(objective(params, resource=resource))
    if "fidelity" in parameters:
        return lambda params, resource: float(objective(params, fidelity=resource))

    positional = [
        p
        for p in parameters.values()
        if p.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    if len(positional) >= 2:
        return lambda params, resource: float(objective(params, resource))
    return _single


def hyperband_brackets(
    max_resource: int = 9,
    eta: int = 3,
    min_resource: int = 1,
) -> List[Tuple[int, int, float]]:
    """Return Hyperband brackets as ``(s, n, r)`` from largest ``s`` to 0.

    ``s`` is the number of successive-halving rungs minus one, ``n`` is how
    many configurations the bracket samples, and ``r`` is the starting
    resource (before multiplying by ``eta`` at each rung). ``max_resource``
    is the paper's ``R``; ``eta`` is the downsampling rate.
    """
    _validate_hyperband_params(eta, min_resource, max_resource)
    s_max = int(math.floor(math.log(max_resource / min_resource, eta)))
    budget_units = (s_max + 1) * max_resource
    brackets: List[Tuple[int, int, float]] = []
    for s in range(s_max, -1, -1):
        n = int(math.ceil(budget_units / max_resource * (eta**s) / (s + 1)))
        r = max(float(min_resource), float(max_resource) * (eta ** (-s)))
        brackets.append((s, n, r))
    return brackets


def _run_sh_bracket(
    *,
    space: Dict[str, Space],
    evaluate: Callable[[Dict[str, Any], float], float],
    rng: np.random.Generator,
    n: int,
    r: float,
    s: int,
    eta: int,
    max_resource: int,
    trials: List[Trial],
    budget: int,
    stop_when: Stopper,
) -> bool:
    """Run one successive-halving bracket.

    Returns True when the search should stop (budget exhausted or
    ``stop_when`` fired).
    """
    if n < 1 or len(trials) >= budget:
        return len(trials) >= budget

    configs: List[Dict[str, Any]] = [_sample_config(space, rng) for _ in range(int(n))]
    for i in range(s + 1):
        if not configs or len(trials) >= budget:
            break
        n_i = min(len(configs), max(1, int(math.floor(n * (eta ** (-i))))))
        r_i = min(float(max_resource), r * (eta**i))
        fidelity = float(r_i) / float(max_resource)
        configs = configs[:n_i]
        scores: List[float] = []
        for params in configs:
            if len(trials) >= budget:
                return True
            score = float(evaluate(params, fidelity))
            trials.append(
                Trial(len(trials), dict(params), score, resource=fidelity)
            )
            scores.append(score)
            if stop_when is not None and stop_when(_best_score(trials), len(trials)):
                return True
        if i < s and scores:
            k = max(1, int(math.floor(len(configs) / eta)))
            k = min(k, len(configs))
            order = sorted(range(len(scores)), key=lambda j: scores[j])
            configs = [configs[j] for j in order[:k]]
    return len(trials) >= budget


def successive_halving(
    space: Dict[str, Space],
    objective: ScoreFn,
    budget: int,
    rng: Optional[np.random.Generator] = None,
    stop_when: Stopper = None,
    *,
    eta: int = 3,
    min_resource: int = 1,
    max_resource: int = 9,
    n_candidates: Optional[int] = None,
) -> List[Trial]:
    """Successive Halving: one multi-fidelity bracket, repeated to fill ``budget``.

    Configurations are sampled from ``space`` and evaluated at geometrically
    increasing resources. After each rung the worst ``1 - 1/eta`` fraction is
    discarded. Resource is passed to ``objective`` as a normalized fidelity
    in ``(0, 1]`` (``1.0`` = full fidelity) when the callable accepts a
    ``resource`` / ``fidelity`` argument; otherwise the objective is called
    with params only.

    ``budget`` counts objective evaluations, matching grid / random / GP.
    """
    _validate_space(space)
    if budget < 1:
        raise ValueError("budget must be >= 1")
    _validate_hyperband_params(eta, min_resource, max_resource)
    rng = rng if rng is not None else np.random.default_rng()

    brackets = hyperband_brackets(max_resource, eta, min_resource)
    s, n_default, r = brackets[0]
    n = int(n_candidates) if n_candidates is not None else n_default
    if n < 1:
        raise ValueError("n_candidates must be >= 1")

    evaluate = _resource_caller(objective)
    trials: List[Trial] = []
    while len(trials) < budget:
        remaining = budget - len(trials)
        n_use = min(n, remaining)
        stopped = _run_sh_bracket(
            space=space,
            evaluate=evaluate,
            rng=rng,
            n=n_use,
            r=r,
            s=s,
            eta=eta,
            max_resource=max_resource,
            trials=trials,
            budget=budget,
            stop_when=stop_when,
        )
        if stopped:
            break
        if n_use < 1:
            break
    return trials


def successive_halving_rungs(
    max_resource: int = 9,
    eta: int = 3,
    min_resource: int = 1,
) -> List[float]:
    """Normalized fidelities of the most aggressive Hyperband bracket.

    Low to high, ending at ``1.0``. This is the ladder used by
    :func:`successive_halving` (one closed cohort) and by
    :func:`random_successive_halving` (open-ended early stopping).
    """
    brackets = hyperband_brackets(max_resource, eta, min_resource)
    s, _n, r = brackets[0]
    fidelities: List[float] = []
    for i in range(s + 1):
        r_i = min(float(max_resource), r * (eta**i))
        fidelities.append(float(r_i) / float(max_resource))
    return fidelities


def select_halving_survivors(scores, eta: int = 3) -> List[int]:
    """Indices successive halving would keep, best (lowest) score first.

    Returns an empty list until at least ``eta`` scores have been observed,
    so early stopping stays off while the rung is too small to rank. After
    that, the best ``floor(n / eta)`` indices are kept (at least one). Ties
    break toward the earlier index.
    """
    scores_arr = np.asarray(scores, dtype=float)
    if scores_arr.ndim != 1:
        raise ValueError("scores must be a 1-d array")
    if isinstance(eta, bool) or int(eta) != eta or int(eta) < 2:
        raise ValueError("eta must be an integer >= 2")
    n = int(scores_arr.size)
    n_keep = 0 if n < int(eta) else max(1, n // int(eta))
    if n_keep <= 0:
        return []
    order = np.argsort(scores_arr, kind="mergesort")
    return [int(i) for i in order[:n_keep]]


@dataclass
class _RungEntry:
    """One configuration's completed evaluation at a single fidelity rung."""

    config_id: int
    params: Dict[str, Any]
    score: float
    promoted: bool = False


def _next_survivor(entries: List[_RungEntry], eta: int) -> Optional[int]:
    """Index of the best not-yet-promoted survivor, or ``None``."""
    scores = [entry.score for entry in entries]
    for idx in select_halving_survivors(scores, eta):
        if not entries[idx].promoted:
            return idx
    return None


def random_successive_halving(
    space: Dict[str, Space],
    objective: ScoreFn,
    budget: int,
    rng: Optional[np.random.Generator] = None,
    stop_when: Stopper = None,
    *,
    eta: int = 3,
    min_resource: int = 1,
    max_resource: int = 9,
) -> List[Trial]:
    """Random search with successive-halving early stopping.

    Configurations are drawn uniformly from ``space``, as in
    :func:`random_search`. Each one starts at the cheapest fidelity on the
    ladder from :func:`successive_halving_rungs` and is continued to the
    next fidelity only when :func:`select_halving_survivors` still ranks it
    among the best ``floor(n / eta)`` observations at its current rung.
    Until a rung has ``eta`` observations, nobody is promoted. When nothing
    is waiting for promotion, a fresh random configuration starts at the
    bottom rung.

    That is the successive-halving early-stopping rule on an open-ended
    stream of random configurations (the synchronous form of ASHA; Li et
    al., 2020), not a closed cohort. :func:`successive_halving` samples a
    fixed bracket, evaluates the whole cohort, then discards the worst of
    that cohort. Here the survivor set is the running top ``1/eta``, and a
    configuration that was stopped can still be promoted later if later,
    worse observations move it into that set.

    Gaussian-process expected improvement is already implemented by
    :class:`GaussianProcess`, :func:`expected_improvement` and
    :func:`bayesian_search`. This searcher is the method added beside those,
    rather than a second GP-EI optimizer.

    ``budget`` counts objective evaluations, matching the other searchers.
    Fidelity is a normalized ``resource`` in ``(0, 1]`` (``1.0`` = full
    fidelity) when the objective accepts ``resource`` or ``fidelity``;
    otherwise the objective is called with params only. With
    ``max_resource == min_resource`` every evaluation is full fidelity and
    the sampled configurations match :func:`random_search` on the same
    generator.
    """
    _validate_space(space)
    if budget < 1:
        raise ValueError("budget must be >= 1")
    _validate_hyperband_params(eta, min_resource, max_resource)
    rng = rng if rng is not None else np.random.default_rng()
    rungs = successive_halving_rungs(max_resource, eta, min_resource)
    evaluate = _resource_caller(objective)
    trials: List[Trial] = []
    rung_entries: List[List[_RungEntry]] = [[] for _ in rungs]
    next_id = 0

    while len(trials) < budget:
        chosen_id: Optional[int] = None
        chosen_params: Optional[Dict[str, Any]] = None
        rung_idx = 0
        for candidate_rung in range(len(rungs) - 2, -1, -1):
            survivor = _next_survivor(rung_entries[candidate_rung], eta)
            if survivor is None:
                continue
            entry = rung_entries[candidate_rung][survivor]
            entry.promoted = True
            chosen_id = entry.config_id
            chosen_params = dict(entry.params)
            rung_idx = candidate_rung + 1
            break
        if chosen_params is None:
            chosen_id = next_id
            next_id += 1
            chosen_params = _sample_config(space, rng)
            rung_idx = 0
        fidelity = float(rungs[rung_idx])
        params = dict(chosen_params)
        score = float(evaluate(dict(params), fidelity))
        trials.append(Trial(len(trials), dict(params), score, resource=fidelity))
        rung_entries[rung_idx].append(
            _RungEntry(int(chosen_id), dict(params), score)
        )
        if stop_when is not None and stop_when(_best_score(trials), len(trials)):
            break
    return trials


def hyperband_search(
    space: Dict[str, Space],
    objective: ScoreFn,
    budget: int,
    rng: Optional[np.random.Generator] = None,
    stop_when: Stopper = None,
    *,
    eta: int = 3,
    min_resource: int = 1,
    max_resource: int = 9,
) -> List[Trial]:
    """Hyperband: several Successive Halving brackets on a shared eval budget.

    Follows Li et al. (2018): ``s_max = floor(log_eta(R / r_min))`` brackets
    with different ``(n, r)`` allocations, from many cheap evaluations to a
    few full-fidelity ones. Rounds repeat until ``budget`` evaluations are
    used, so the strategy is comparable to random / grid / GP on the same
    number of objective calls.

    Fidelity is a normalized ``resource`` in ``(0, 1]``; see
    :func:`successive_halving`.
    """
    _validate_space(space)
    if budget < 1:
        raise ValueError("budget must be >= 1")
    _validate_hyperband_params(eta, min_resource, max_resource)
    rng = rng if rng is not None else np.random.default_rng()

    brackets = hyperband_brackets(max_resource, eta, min_resource)
    evaluate = _resource_caller(objective)
    trials: List[Trial] = []
    while len(trials) < budget:
        progressed = False
        for s, n, r in brackets:
            remaining = budget - len(trials)
            if remaining <= 0:
                break
            n_use = min(int(n), remaining)
            before = len(trials)
            stopped = _run_sh_bracket(
                space=space,
                evaluate=evaluate,
                rng=rng,
                n=n_use,
                r=r,
                s=s,
                eta=eta,
                max_resource=max_resource,
                trials=trials,
                budget=budget,
                stop_when=stop_when,
            )
            if len(trials) > before:
                progressed = True
            if stopped:
                return trials
        if not progressed:
            while len(trials) < budget:
                params = _sample_config(space, rng)
                score = float(evaluate(params, 1.0))
                trials.append(Trial(len(trials), params, score, resource=1.0))
                if stop_when is not None and stop_when(_best_score(trials), len(trials)):
                    return trials
            break
    return trials


_SCOTT_FACTOR = 1.06


class ProductKernelDensity:
    """Multivariate product-kernel density used by BOHB.

    Each observation contributes one kernel, and that kernel is the product
    of a per-coordinate factor: a Gaussian on continuous (normalized)
    coordinates and an Aitchison–Aitken kernel on unordered categoricals.
    Because the factors are centered on the joint observation, the density
    can put mass on correlated combinations. That is the BOHB surrogate
    (Falkner et al., 2018), and it is distinct from TPE, which multiplies
    independent per-parameter densities.

    Bandwidths follow the multivariate normal-reference rule used by the
    reference implementation (statsmodels ``KDEMultivariate`` with
    ``bw='normal_reference'``): ``h_j = 1.06 σ_j n^{-1/(4+d)}`` with the
    population standard deviation ``σ_j``. Continuous bandwidths are floored
    at ``min_bandwidth``. Categorical bandwidths are also capped at
    ``(c-1)/c``, the value that makes the Aitchison–Aitken kernel uniform, so
    the kernel stays a valid probability.
    """

    def __init__(
        self,
        samples: np.ndarray,
        kinds: str,
        n_levels: np.ndarray,
        bandwidth: np.ndarray,
        min_bandwidth: float,
    ):
        self.samples = np.asarray(samples, dtype=float)
        self.kinds = str(kinds)
        self.n_levels = np.asarray(n_levels, dtype=int)
        self.bandwidth = np.asarray(bandwidth, dtype=float)
        self.min_bandwidth = float(min_bandwidth)

    @classmethod
    def fit(
        cls,
        samples: np.ndarray,
        kinds: str,
        n_levels,
        *,
        min_bandwidth: float = 1e-3,
    ) -> "ProductKernelDensity":
        """Fit Scott / normal-reference bandwidths on ``samples``."""
        samples = np.asarray(samples, dtype=float)
        if samples.ndim != 2 or samples.shape[0] == 0:
            raise ValueError("samples must be a non-empty array of shape (n, d)")
        n, d = samples.shape
        kinds = str(kinds)
        if len(kinds) != d:
            raise ValueError("kinds must have one character per column")
        if any(kind not in ("c", "u") for kind in kinds):
            raise ValueError("kinds must contain only 'c' (continuous) and 'u' (unordered)")
        n_levels = np.asarray(n_levels, dtype=int).reshape(-1)
        if n_levels.shape != (d,):
            raise ValueError("n_levels must have one entry per column")
        if not np.isfinite(min_bandwidth) or not 0.0 < float(min_bandwidth) <= 1.0:
            raise ValueError("min_bandwidth must be in (0, 1]")
        for j, kind in enumerate(kinds):
            if kind == "u" and int(n_levels[j]) < 1:
                raise ValueError("unordered coordinates need n_levels >= 1")
        std = np.std(samples, axis=0)
        std = np.where(np.isfinite(std), std, 0.0)
        bandwidth = _SCOTT_FACTOR * std * (float(n) ** (-1.0 / (4.0 + d)))
        bandwidth = np.maximum(bandwidth, float(min_bandwidth))
        for j, kind in enumerate(kinds):
            if kind != "u":
                continue
            levels = int(n_levels[j])
            if levels <= 1:
                bandwidth[j] = float(min_bandwidth)
                continue
            uniform_lambda = (levels - 1) / float(levels)
            bandwidth[j] = min(float(bandwidth[j]), uniform_lambda)
        return cls(samples, kinds, n_levels, bandwidth, float(min_bandwidth))

    def logpdf(self, X: np.ndarray) -> np.ndarray:
        """Log-density at each row of ``X`` (encoded like ``samples``)."""
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2 or X.shape[1] != self.samples.shape[1]:
            raise ValueError("X must have one column per parameter")
        n = self.samples.shape[0]
        log_comp = np.zeros((X.shape[0], n), dtype=float)
        for j, kind in enumerate(self.kinds):
            h = max(float(self.bandwidth[j]), 1e-12)
            column = self.samples[:, j]
            query = X[:, j]
            if kind == "c":
                z = (query[:, None] - column[None, :]) / h
                log_comp += -0.5 * np.log(2.0 * np.pi) - 0.5 * z * z - np.log(h)
            else:
                levels = int(self.n_levels[j])
                if levels <= 1:
                    continue
                match = np.isclose(query[:, None], column[None, :])
                log_match = math.log(max(1.0 - h, 1e-300))
                log_other = math.log(max(h / (levels - 1), 1e-300))
                log_comp += np.where(match, log_match, log_other)
        return _logsumexp(log_comp, axis=1) - math.log(n)

    def sample(
        self,
        rng: np.random.Generator,
        n: int,
        bandwidth_factor: float = 1.0,
    ) -> np.ndarray:
        """Draw ``n`` rows from the kernel.

        Continuous coordinates are sampled from a normal truncated to
        ``[0, 1]``, with the bandwidth widened by ``bandwidth_factor`` (BOHB
        uses this only when proposing, not when scoring ``l(x)/g(x)``).
        Categorical coordinates follow the Aitchison–Aitken kernel: keep the
        observed level with probability ``1 - λ``, otherwise draw uniformly
        from the other levels.
        """
        if int(n) != n or int(n) < 1:
            raise ValueError("n must be an integer >= 1")
        if not np.isfinite(bandwidth_factor) or float(bandwidth_factor) <= 0.0:
            raise ValueError("bandwidth_factor must be a positive finite number")
        choice = rng.integers(0, self.samples.shape[0], size=int(n))
        drawn = self.samples[choice].copy()
        for j, kind in enumerate(self.kinds):
            # Bandwidths were already floored (and categorical ones capped)
            # in fit(); do not raise a categorical λ back above (c - 1) / c.
            h = float(self.bandwidth[j])
            if kind == "c":
                scale = max(h, self.min_bandwidth) * float(bandwidth_factor)
                for i in range(int(n)):
                    drawn[i, j] = _truncated_normal(
                        rng, float(drawn[i, j]), scale, 0.0, 1.0, 1
                    )[0]
            else:
                levels = int(self.n_levels[j])
                if levels <= 1:
                    drawn[:, j] = 0.0
                    continue
                keep_prob = 1.0 - h
                for i in range(int(n)):
                    current = int(np.clip(round(float(drawn[i, j])), 0, levels - 1))
                    if float(rng.random()) < keep_prob:
                        drawn[i, j] = current
                    else:
                        alt = int(rng.integers(0, levels - 1))
                        if alt >= current:
                            alt += 1
                        drawn[i, j] = float(alt)
        return drawn


def _validate_bohb_params(
    dimension: int,
    top_n_percent: int,
    min_points_in_model: Optional[int],
    n_candidates: int,
    random_fraction: float,
    bandwidth_factor: float,
    min_bandwidth: float,
) -> int:
    """Validate BOHB knobs and return the resolved ``min_points_in_model``."""
    if int(dimension) != dimension or int(dimension) < 1:
        raise ValueError("dimension must be an integer >= 1")
    if isinstance(top_n_percent, bool) or not isinstance(top_n_percent, (int, np.integer)):
        raise ValueError("top_n_percent must be an integer in 1..99")
    if not 1 <= int(top_n_percent) <= 99:
        raise ValueError("top_n_percent must be an integer in 1..99")
    minimum = int(dimension) + 1
    if min_points_in_model is None:
        min_points = minimum
    else:
        if isinstance(min_points_in_model, bool) or not isinstance(
            min_points_in_model, (int, np.integer)
        ):
            raise ValueError("min_points_in_model must be an integer >= dimension + 1")
        min_points = int(min_points_in_model)
        if min_points < minimum:
            raise ValueError(
                f"min_points_in_model must be >= dimension + 1 ({minimum})"
            )
    if isinstance(n_candidates, bool) or not isinstance(n_candidates, (int, np.integer)):
        raise ValueError("n_candidates must be an integer >= 1")
    if int(n_candidates) < 1:
        raise ValueError("n_candidates must be an integer >= 1")
    if not np.isfinite(random_fraction) or not 0.0 <= float(random_fraction) <= 1.0:
        raise ValueError("random_fraction must be in [0, 1]")
    if not np.isfinite(bandwidth_factor) or float(bandwidth_factor) <= 0.0:
        raise ValueError("bandwidth_factor must be a positive finite number")
    if not np.isfinite(min_bandwidth) or not 0.0 < float(min_bandwidth) <= 1.0:
        raise ValueError("min_bandwidth must be in (0, 1]")
    return min_points


def _resource_key(resource: Optional[float]) -> float:
    if resource is None:
        return 1.0
    return round(float(resource), 10)


def _bohb_split_pool(
    pool: List[Trial],
    dimension: int,
    top_n_percent: int,
    min_points: int,
) -> Optional[Tuple[List[Trial], List[Trial]]]:
    """Good/bad split for one fidelity, or ``None`` when a KDE cannot be fit.

    ``n_good`` is the larger of ``min_points`` and ``top_n_percent`` percent
    of the pool (integer arithmetic, as in HpBandSter). ``n_bad`` is the
    larger of ``min_points`` and the complementary percent, taken from the
    observations immediately after the good set. Both slices must contain
    more points than there are parameters.
    """
    n = len(pool)
    if n < min_points:
        return None
    n_good = max(min_points, (int(top_n_percent) * n) // 100)
    n_bad = max(min_points, ((100 - int(top_n_percent)) * n) // 100)
    order = sorted(range(n), key=lambda i: pool[i].score)
    good_idx = order[:n_good]
    bad_idx = order[n_good : n_good + n_bad]
    if len(good_idx) <= dimension or len(bad_idx) <= dimension:
        return None
    return [pool[i] for i in good_idx], [pool[i] for i in bad_idx]


def bohb_select_trials(
    trials: List[Trial],
    dimension: int,
    *,
    top_n_percent: int = 15,
    min_points_in_model: Optional[int] = None,
) -> Optional[Tuple[List[Trial], List[Trial]]]:
    """Largest fidelity whose good/bad sets can both fit a product KDE.

    Losses at different fidelities are not pooled: a cheap rung and a full
    run do not share a scale. Among fidelities with enough points, the
    largest one is used. Returns ``(good, bad)`` or ``None`` when every
    fidelity is still too small. ``min_points_in_model`` defaults to
    ``dimension + 1``.
    """
    if int(dimension) != dimension or int(dimension) < 1:
        raise ValueError("dimension must be an integer >= 1")
    min_points = _validate_bohb_params(
        int(dimension),
        top_n_percent,
        min_points_in_model,
        n_candidates=1,
        random_fraction=0.0,
        bandwidth_factor=1.0,
        min_bandwidth=1e-3,
    )
    groups: Dict[float, List[Trial]] = {}
    for trial in trials:
        groups.setdefault(_resource_key(trial.resource), []).append(trial)
    for key in sorted(groups, reverse=True):
        split = _bohb_split_pool(groups[key], int(dimension), int(top_n_percent), min_points)
        if split is not None:
            return split
    return None


def _space_layout(space: Dict[str, Space], names: List[str]) -> Tuple[str, np.ndarray]:
    kinds: List[str] = []
    n_levels: List[int] = []
    for name in names:
        sp = space[name]
        if isinstance(sp, Categorical):
            kinds.append("u")
            n_levels.append(len(sp.choices))
        else:
            kinds.append("c")
            n_levels.append(0)
    return "".join(kinds), np.asarray(n_levels, dtype=int)


def _encode_params(
    space: Dict[str, Space], names: List[str], params: Dict[str, Any]
) -> np.ndarray:
    row = np.empty(len(names), dtype=float)
    for j, name in enumerate(names):
        sp = space[name]
        if isinstance(sp, Categorical):
            row[j] = float(sp.choices.index(params[name]))
        else:
            row[j] = float(sp.normalize(params[name]))
    return row


def _encode_trials(
    space: Dict[str, Space], names: List[str], trials: List[Trial]
) -> np.ndarray:
    return np.vstack([_encode_params(space, names, trial.params) for trial in trials])


def _decode_vector(
    space: Dict[str, Space], names: List[str], vector: np.ndarray
) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    for j, name in enumerate(names):
        sp = space[name]
        if isinstance(sp, Categorical):
            idx = int(np.clip(round(float(vector[j])), 0, len(sp.choices) - 1))
            params[name] = sp.choices[idx]
        else:
            params[name] = sp.denormalize(float(vector[j]))
    return params


def _fit_bohb_kdes(
    space: Dict[str, Space],
    names: List[str],
    good: List[Trial],
    bad: List[Trial],
    min_bandwidth: float,
) -> Tuple[ProductKernelDensity, ProductKernelDensity]:
    kinds, n_levels = _space_layout(space, names)
    good_kde = ProductKernelDensity.fit(
        _encode_trials(space, names, good), kinds, n_levels, min_bandwidth=min_bandwidth
    )
    bad_kde = ProductKernelDensity.fit(
        _encode_trials(space, names, bad), kinds, n_levels, min_bandwidth=min_bandwidth
    )
    return good_kde, bad_kde


def bohb_log_density_ratio(
    params: Dict[str, Any],
    space: Dict[str, Space],
    trials: List[Trial],
    *,
    top_n_percent: int = 15,
    min_points_in_model: Optional[int] = None,
    min_bandwidth: float = 1e-3,
) -> float:
    """``log l(x) - log g(x)`` for ``params`` under the BOHB product KDEs.

    ``l`` is the product-kernel density of the best observations at the
    largest fidelity that can support a model; ``g`` is the density of the
    following (worse) slice. BOHB ranks candidates by this ratio, which is
    the TPE acquisition written with a joint kernel instead of factorized
    Parzen estimators. Raises ``ValueError`` when no fidelity has enough
    points to fit both densities.
    """
    _validate_space(space)
    names = list(space)
    selected = bohb_select_trials(
        trials,
        len(names),
        top_n_percent=top_n_percent,
        min_points_in_model=min_points_in_model,
    )
    if selected is None:
        raise ValueError(
            "not enough observations at any fidelity to fit a BOHB model"
        )
    good, bad = selected
    _validate_bohb_params(
        len(names),
        top_n_percent,
        min_points_in_model if min_points_in_model is not None else len(names) + 1,
        n_candidates=1,
        random_fraction=0.0,
        bandwidth_factor=1.0,
        min_bandwidth=min_bandwidth,
    )
    good_kde, bad_kde = _fit_bohb_kdes(space, names, good, bad, min_bandwidth)
    encoded = _encode_params(space, names, params).reshape(1, -1)
    return float(good_kde.logpdf(encoded)[0] - bad_kde.logpdf(encoded)[0])


def bohb_propose(
    space: Dict[str, Space],
    trials: List[Trial],
    rng: np.random.Generator,
    *,
    top_n_percent: int = 15,
    min_points_in_model: Optional[int] = None,
    n_candidates: int = 64,
    random_fraction: float = 1.0 / 3.0,
    bandwidth_factor: float = 3.0,
    min_bandwidth: float = 1e-3,
) -> Dict[str, Any]:
    """Propose one configuration from the BOHB model, or uniformly at random.

    Whenever no fidelity has enough data, the proposal is uniform. Once a
    model can be fit, a further ``random_fraction`` of proposals (drawn only
    in that case) are uniform as well. Otherwise ``n_candidates``
    configurations are drawn from the good-set product kernel, continuous
    bandwidths widened by ``bandwidth_factor``, and the candidate maximizing
    ``l(x)/g(x)`` is returned.
    """
    _validate_space(space)
    names = list(space)
    _validate_bohb_params(
        len(names),
        top_n_percent,
        min_points_in_model,
        n_candidates,
        random_fraction,
        bandwidth_factor,
        min_bandwidth,
    )
    selected = bohb_select_trials(
        trials,
        len(names),
        top_n_percent=top_n_percent,
        min_points_in_model=min_points_in_model,
    )
    if (
        selected is None
        or float(random_fraction) >= 1.0
        or (
            float(random_fraction) > 0.0
            and float(rng.random()) < float(random_fraction)
        )
    ):
        return _sample_config(space, rng)
    good, bad = selected
    good_kde, bad_kde = _fit_bohb_kdes(space, names, good, bad, min_bandwidth)
    samples = good_kde.sample(rng, int(n_candidates), bandwidth_factor=bandwidth_factor)
    log_l = good_kde.logpdf(samples)
    log_g = bad_kde.logpdf(samples)
    ratio = log_l - log_g
    if not np.any(np.isfinite(ratio)):
        finite_l = np.flatnonzero(np.isfinite(log_l))
        if finite_l.size == 0:
            return _sample_config(space, rng)
        idx = int(finite_l[0])
    else:
        idx = int(np.argmax(np.where(np.isfinite(ratio), ratio, -np.inf)))
    return _decode_vector(space, names, samples[idx])


def _run_bohb_bracket(
    *,
    evaluate: Callable[[Dict[str, Any], float], float],
    propose: Callable[[], Dict[str, Any]],
    n: int,
    r: float,
    s: int,
    eta: int,
    max_resource: int,
    trials: List[Trial],
    budget: int,
    stop_when: Stopper,
) -> bool:
    """One successive-halving bracket with model-based proposals on rung 0.

    Later rungs only promote survivors, matching Hyperband. New
    configurations are drawn one at a time so each proposal sees the
    evaluations already recorded in ``trials``.
    """
    if n < 1 or len(trials) >= budget:
        return len(trials) >= budget

    configs: List[Dict[str, Any]] = []
    for i in range(s + 1):
        if len(trials) >= budget:
            break
        n_i = max(1, int(math.floor(n * (eta ** (-i)))))
        r_i = min(float(max_resource), r * (eta**i))
        fidelity = float(r_i) / float(max_resource)
        scores: List[float] = []
        if i == 0:
            n_i = min(int(n), n_i)
            configs = []
            for _ in range(n_i):
                if len(trials) >= budget:
                    return True
                params = dict(propose())
                score = float(evaluate(params, fidelity))
                trials.append(Trial(len(trials), params, score, resource=fidelity))
                configs.append(params)
                scores.append(score)
                if stop_when is not None and stop_when(_best_score(trials), len(trials)):
                    return True
        else:
            if not configs:
                break
            n_i = min(len(configs), n_i)
            configs = configs[:n_i]
            for params in configs:
                if len(trials) >= budget:
                    return True
                score = float(evaluate(dict(params), fidelity))
                trials.append(
                    Trial(len(trials), dict(params), score, resource=fidelity)
                )
                scores.append(score)
                if stop_when is not None and stop_when(_best_score(trials), len(trials)):
                    return True
        if i < s and scores:
            k = max(1, int(math.floor(len(configs) / eta)))
            k = min(k, len(configs))
            order = sorted(range(len(scores)), key=lambda j: scores[j])
            configs = [configs[j] for j in order[:k]]
    return len(trials) >= budget


def bohb_search(
    space: Dict[str, Space],
    objective: ScoreFn,
    budget: int,
    rng: Optional[np.random.Generator] = None,
    stop_when: Stopper = None,
    *,
    eta: int = 3,
    min_resource: int = 1,
    max_resource: int = 9,
    top_n_percent: int = 15,
    min_points_in_model: Optional[int] = None,
    n_candidates: int = 64,
    random_fraction: float = 1.0 / 3.0,
    bandwidth_factor: float = 3.0,
    min_bandwidth: float = 1e-3,
) -> List[Trial]:
    """BOHB: Hyperband brackets with a product-kernel configuration model.

    The bracket schedule is the same as :func:`hyperband_search` (Li et al.,
    2018, via Falkner et al., 2018). Configurations that start a bracket are
    not drawn uniformly. Once the largest fidelity with enough evaluations
    can fit two product-kernel densities — the best ``top_n_percent`` and a
    worse slice — candidates are sampled from the good density and the one
    with the largest ``l(x)/g(x)`` is evaluated. A ``random_fraction`` of
    proposals (default one third) stays uniform so the model cannot collapse
    the search. ``min_points_in_model`` defaults to ``dimension + 1``.

    ``budget`` counts objective calls, matching the other searchers. Fidelity
    is a normalized ``resource`` in ``(0, 1]``; see :func:`successive_halving`.
    No dependencies beyond numpy: the KDE replaces statsmodels'
    ``KDEMultivariate``.
    """
    _validate_space(space)
    if budget < 1:
        raise ValueError("budget must be >= 1")
    _validate_hyperband_params(eta, min_resource, max_resource)
    names = list(space)
    _validate_bohb_params(
        len(names),
        top_n_percent,
        min_points_in_model,
        n_candidates,
        random_fraction,
        bandwidth_factor,
        min_bandwidth,
    )
    rng = rng if rng is not None else np.random.default_rng()
    brackets = hyperband_brackets(max_resource, eta, min_resource)
    evaluate = _resource_caller(objective)
    trials: List[Trial] = []

    def propose() -> Dict[str, Any]:
        return bohb_propose(
            space,
            trials,
            rng,
            top_n_percent=top_n_percent,
            min_points_in_model=min_points_in_model,
            n_candidates=n_candidates,
            random_fraction=random_fraction,
            bandwidth_factor=bandwidth_factor,
            min_bandwidth=min_bandwidth,
        )

    while len(trials) < budget:
        progressed = False
        for s, n, r in brackets:
            remaining = budget - len(trials)
            if remaining <= 0:
                break
            n_use = min(int(n), remaining)
            before = len(trials)
            stopped = _run_bohb_bracket(
                evaluate=evaluate,
                propose=propose,
                n=n_use,
                r=r,
                s=s,
                eta=eta,
                max_resource=max_resource,
                trials=trials,
                budget=budget,
                stop_when=stop_when,
            )
            if len(trials) > before:
                progressed = True
            if stopped:
                return trials
        if not progressed:
            while len(trials) < budget:
                params = propose()
                score = float(evaluate(params, 1.0))
                trials.append(Trial(len(trials), params, score, resource=1.0))
                if stop_when is not None and stop_when(_best_score(trials), len(trials)):
                    return trials
            break
    return trials


def _chi_expectation(n: int) -> float:
    """Approximate ``E[||N(0, I_n)||]`` used by CMA-ES step-size control."""
    n = max(int(n), 1)
    return math.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n))


def cmaes_weights(n_parents: int) -> np.ndarray:
    """Positive recombination weights ``w_i ∝ log(μ + 1/2) - log(i)``.

    Hansen's default: the ``μ`` best offspring get logarithmically decaying
    weights that sum to one. ``μ_eff = 1 / Σ w_i²`` follows from these.
    """
    if int(n_parents) != n_parents or n_parents < 1:
        raise ValueError("n_parents must be an integer >= 1")
    ranks = np.arange(1, int(n_parents) + 1, dtype=float)
    weights = np.log(n_parents + 0.5) - np.log(ranks)
    weights = np.maximum(weights, 0.0)
    total = float(weights.sum())
    if total <= 0.0:
        return np.ones(int(n_parents), dtype=float) / float(n_parents)
    return weights / total


@dataclass
class CMAESParameters:
    """Hansen default population size, weights and learning rates."""

    population_size: int
    n_parents: int
    weights: np.ndarray
    mu_eff: float
    c_sigma: float
    d_sigma: float
    c_c: float
    c_1: float
    c_mu: float


def cmaes_parameters(
    dimension: int, population_size: Optional[int] = None
) -> CMAESParameters:
    """Recommended CMA-ES constants for a ``dimension``-d search.

    ``population_size`` defaults to ``4 + floor(3 log n)``. Learning rates
    follow Hansen's tutorial so rank-1 / rank-μ covariance updates and
    cumulative step-size adaptation stay stable on the small continuous
    spaces typical of hyperparameter search.
    """
    if int(dimension) != dimension or dimension < 1:
        raise ValueError("dimension must be an integer >= 1")
    n = int(dimension)
    if population_size is None:
        lam = 4 + int(math.floor(3.0 * math.log(n)))
    else:
        if int(population_size) != population_size or population_size < 2:
            raise ValueError("population_size must be an integer >= 2")
        lam = int(population_size)
    lam = max(2, lam)
    mu = max(1, lam // 2)
    weights = cmaes_weights(mu)
    mu_eff = float(1.0 / np.sum(weights * weights))
    c_sigma = (mu_eff + 2.0) / (n + mu_eff + 5.0)
    d_sigma = (
        1.0
        + 2.0 * max(0.0, math.sqrt((mu_eff - 1.0) / (n + 1.0)) - 1.0)
        + c_sigma
    )
    c_c = (4.0 + mu_eff / n) / (n + 4.0 + 2.0 * mu_eff / n)
    c_1 = 2.0 / ((n + 1.3) ** 2 + mu_eff)
    c_mu = min(
        1.0 - c_1,
        2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((n + 2.0) ** 2 + mu_eff),
    )
    c_mu = max(float(c_mu), 0.0)
    return CMAESParameters(
        population_size=lam,
        n_parents=mu,
        weights=weights,
        mu_eff=mu_eff,
        c_sigma=float(c_sigma),
        d_sigma=float(d_sigma),
        c_c=float(c_c),
        c_1=float(c_1),
        c_mu=float(c_mu),
    )


def _eigendecompose_covariance(C: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(B, D)`` such that ``C ≈ B diag(D²) B.T`` with ``D > 0``."""
    C = 0.5 * (C + C.T)
    eigvals, B = np.linalg.eigh(C)
    eigvals = np.maximum(eigvals, 1e-14)
    return B, np.sqrt(eigvals)


def _decode_cmaes_vector(
    x: np.ndarray, names: List[str], space: Dict[str, Space]
) -> Dict[str, Any]:
    return {
        name: space[name].denormalize(float(x[j])) for j, name in enumerate(names)
    }


def cmaes_search(
    space: Dict[str, Space],
    objective: ScoreFn,
    budget: int,
    rng: Optional[np.random.Generator] = None,
    stop_when: Stopper = None,
    *,
    population_size: Optional[int] = None,
    sigma0: float = 0.3,
) -> List[Trial]:
    """CMA-ES search on normalized ``FloatRange`` / ``IntRange`` space.

    Samples offspring from ``N(m, σ² C)`` in the unit cube obtained from
    :meth:`Space.normalize`, evaluates them, then updates the mean,
    isotropic step-size (cumulative step-size adaptation) and covariance
    (rank-1 + rank-μ) from the ``μ`` best points (Hansen, 2016). Integer
    and log-scaled ranges are handled by denormalize / normalize, so the
    internal search stays continuous. Categorical parameters are encoded
    as ordered unit-interval coordinates (same caveat as the GP surrogate).

    ``budget`` counts objective evaluations, matching grid / random / GP /
    TPE. ``population_size`` defaults to Hansen's ``4 + floor(3 log n)``
    and is capped by ``budget`` so a short run still completes at least
    one generation. ``sigma0`` is the initial step-size on the unit cube.
    """
    _validate_space(space)
    if budget < 1:
        raise ValueError("budget must be >= 1")
    if not np.isfinite(sigma0) or sigma0 <= 0.0:
        raise ValueError("sigma0 must be a positive finite number")
    rng = rng if rng is not None else np.random.default_rng()
    names = list(space)
    n = len(names)
    params = cmaes_parameters(n, population_size=population_size)
    lam = params.population_size
    if population_size is None:
        lam = min(lam, max(2, int(budget)))
        if lam != params.population_size:
            params = cmaes_parameters(n, population_size=lam)
    mu = params.n_parents
    weights = params.weights
    mu_eff = params.mu_eff
    c_sigma = params.c_sigma
    d_sigma = params.d_sigma
    c_c = params.c_c
    c_1 = params.c_1
    c_mu = params.c_mu
    chi_n = _chi_expectation(n)

    mean = np.full(n, 0.5, dtype=float)
    sigma = float(sigma0)
    C = np.eye(n, dtype=float)
    p_sigma = np.zeros(n, dtype=float)
    p_c = np.zeros(n, dtype=float)
    B, D = _eigendecompose_covariance(C)

    trials: List[Trial] = []
    best = float("inf")
    generation = 0
    pending_y: List[np.ndarray] = []
    pending_scores: List[float] = []

    def _update_state(ys: List[np.ndarray], scores: List[float]) -> None:
        nonlocal mean, sigma, C, p_sigma, p_c, B, D, generation
        if len(ys) < mu:
            return
        order = np.argsort(np.asarray(scores, dtype=float), kind="mergesort")
        y_sel = np.stack([ys[int(i)] for i in order[:mu]], axis=0)
        y_w = np.sum(weights[:, None] * y_sel, axis=0)
        mean = np.clip(mean + sigma * y_w, 0.0, 1.0)
        # C^{-1/2} y_w = B D^{-1} B^T y_w
        z_w = (B.T @ y_w) / D
        invsqrt_y = B @ z_w
        p_sigma = (1.0 - c_sigma) * p_sigma + (
            math.sqrt(c_sigma * (2.0 - c_sigma) * mu_eff) * invsqrt_y
        )
        p_sigma_norm = float(np.linalg.norm(p_sigma))
        generation += 1
        denom = math.sqrt(max(1.0 - (1.0 - c_sigma) ** (2.0 * generation), 1e-12))
        h_sigma = (
            1.0
            if p_sigma_norm / denom < (1.4 + 2.0 / (n + 1.0)) * chi_n
            else 0.0
        )
        p_c = (1.0 - c_c) * p_c + (
            h_sigma * math.sqrt(c_c * (2.0 - c_c) * mu_eff) * y_w
        )
        rank_mu = (y_sel * weights[:, None]).T @ y_sel
        delta_h = (1.0 - h_sigma) * c_c * (2.0 - c_c)
        C = (
            (1.0 - c_1 - c_mu + c_1 * delta_h) * C
            + c_1 * np.outer(p_c, p_c)
            + c_mu * rank_mu
        )
        C = 0.5 * (C + C.T)
        B, D = _eigendecompose_covariance(C)
        sigma *= math.exp((c_sigma / d_sigma) * (p_sigma_norm / chi_n - 1.0))
        sigma = float(np.clip(sigma, 1e-12, 1.0))

    while len(trials) < budget:
        z = rng.standard_normal(n)
        y = B @ (D * z)
        x = np.clip(mean + sigma * y, 0.0, 1.0)
        # Use the feasible step so clipped samples still update C honestly.
        y = (x - mean) / max(sigma, 1e-12)
        config = _decode_cmaes_vector(x, names, space)
        score = float(objective(config))
        trials.append(Trial(len(trials), config, score))
        best = min(best, score)
        pending_y.append(y)
        pending_scores.append(score)
        if stop_when is not None and stop_when(best, len(trials)):
            break
        if len(pending_y) >= lam:
            _update_state(pending_y, pending_scores)
            pending_y = []
            pending_scores = []

    return trials


def _is_strict_int(value: Any) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


def _is_real_number(value: Any) -> bool:
    if isinstance(value, bool) or isinstance(value, (str, bytes)):
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number)


def _validate_pbt_perturbation(
    perturbation_factors: Tuple[float, ...],
    resample_probability: float,
) -> Tuple[float, ...]:
    try:
        factors = tuple(float(factor) for factor in perturbation_factors)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "perturbation_factors must be a non-empty sequence of positive finite numbers"
        ) from exc
    if len(factors) < 1 or any(not math.isfinite(factor) or factor <= 0.0 for factor in factors):
        raise ValueError(
            "perturbation_factors must be a non-empty sequence of positive finite numbers"
        )
    if (
        not _is_real_number(resample_probability)
        or not 0.0 <= float(resample_probability) <= 1.0
    ):
        raise ValueError("resample_probability must be in [0, 1]")
    return factors


def _validate_pbt_params(
    population_size: Optional[int],
    exploit_interval: int,
    quantile: float,
    perturbation_factors: Tuple[float, ...],
    resample_probability: float,
    eta: int,
    min_resource: int,
    max_resource: int,
) -> Tuple[float, ...]:
    if population_size is not None and (
        not _is_strict_int(population_size) or int(population_size) < 2
    ):
        raise ValueError("population_size must be an integer >= 2")
    if not _is_strict_int(exploit_interval) or int(exploit_interval) < 1:
        raise ValueError("exploit_interval must be an integer >= 1")
    if not _is_real_number(quantile) or not 0.0 < float(quantile) <= 0.5:
        raise ValueError("quantile must be in (0, 0.5]")
    factors = _validate_pbt_perturbation(perturbation_factors, resample_probability)
    _validate_hyperband_params(eta, min_resource, max_resource)
    return factors


def pbt_perturb(
    space: Dict[str, Space],
    params: Dict[str, Any],
    rng: np.random.Generator,
    *,
    perturbation_factors: Tuple[float, ...] = (0.8, 1.2),
    resample_probability: float = 0.25,
) -> Dict[str, Any]:
    """Perturb one configuration (the PBT explore step).

    Each continuous or integer parameter is multiplied by a factor drawn
    from ``perturbation_factors`` and clipped to its bounds (Jaderberg et
    al., 2017). If that product does not move a non-degenerate integer, or
    a float already sitting on the boundary, a one-step
    :meth:`Space.mutate` is applied so explore is not a no-op. Each
    categorical parameter is redrawn from the other choices with
    probability ``resample_probability``. The input mapping is not mutated.
    """
    _validate_space(space)
    factors = _validate_pbt_perturbation(perturbation_factors, resample_probability)
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be a numpy Generator")
    missing = [name for name in space if name not in params]
    if missing:
        raise ValueError(f"params missing keys: {missing}")

    perturbed = dict(params)
    factor_array = np.asarray(factors, dtype=float)
    for name, sp in space.items():
        if isinstance(sp, Categorical):
            if len(sp.choices) > 1 and float(resample_probability) > 0.0:
                if float(rng.random()) < float(resample_probability):
                    choices = [choice for choice in sp.choices if choice != params[name]]
                    perturbed[name] = choices[int(rng.integers(0, len(choices)))]
            continue
        factor = float(rng.choice(factor_array))
        if isinstance(sp, FloatRange):
            current = float(params[name])
            value = float(np.clip(current * factor, sp.low, sp.high))
            if (
                abs(value - current) <= 1e-15
                and abs(factor - 1.0) > 1e-12
                and not sp.degenerate
            ):
                value = float(sp.mutate(current, rng))
            perturbed[name] = value
        elif isinstance(sp, IntRange):
            current = int(params[name])
            value = int(np.clip(int(round(current * factor)), sp.low, sp.high))
            if value == current and sp.n_values > 1 and abs(factor - 1.0) > 1e-12:
                value = int(sp.mutate(current, rng))
            perturbed[name] = value
        else:
            raise TypeError(f"unsupported space type {type(sp).__name__}")
    return perturbed


def _interpret_pbt_result(result: Any) -> Tuple[float, Any, bool]:
    """Return ``(score, weights, explicit)``.

    A ``(score, weights)`` tuple is an explicit checkpoint. Any other real
    number is a scalar score and the searcher keeps its own checkpoint.
    """
    if isinstance(result, tuple) and len(result) == 2 and _is_real_number(result[0]):
        return float(result[0]), result[1], True
    if not _is_real_number(result):
        raise TypeError("PBT objective must return a float score, or (score, weights)")
    return float(result), None, False


def _pbt_caller(
    objective: ScoreFn,
) -> Callable[[Dict[str, Any], float, Any], Tuple[float, Any, bool]]:
    """Adapt ``objective(params)``, ``(..., resource=)`` and ``(..., weights=)``."""
    try:
        signature = inspect.signature(objective)
    except (TypeError, ValueError):

        def _bare(params: Dict[str, Any], resource: float, weights: Any):
            return _interpret_pbt_result(objective(params))

        return _bare

    parameters = signature.parameters
    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values())
    names = set(parameters)
    positional = [
        p
        for p in parameters.values()
        if p.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]

    def _call(params: Dict[str, Any], resource: float, weights: Any):
        kwargs: Dict[str, Any] = {}
        if has_var_kw or "resource" in names:
            kwargs["resource"] = resource
        elif "fidelity" in names:
            kwargs["fidelity"] = resource
        elif len(positional) >= 2 and positional[1].name not in ("weights", "checkpoint"):
            kwargs[positional[1].name] = resource
        if has_var_kw or "weights" in names:
            kwargs["weights"] = weights
        elif "checkpoint" in names:
            kwargs["checkpoint"] = weights
        elif len(positional) >= 3:
            kwargs[positional[2].name] = weights
        return _interpret_pbt_result(objective(params, **kwargs))

    return _call


@dataclass
class _PBTMember:
    """One member of a population-based training population."""

    params: Dict[str, Any]
    weights: Any = None
    score: float = float("inf")
    rung: int = -1
    eval_count: int = 0


def _pbt_generation_ready(members: List[_PBTMember], exploit_interval: int) -> bool:
    """True when every member has just finished a synchronous exploit step."""
    counts = [member.eval_count for member in members]
    if not counts or any(count <= 0 for count in counts):
        return False
    if any(count != counts[0] for count in counts):
        return False
    return counts[0] % int(exploit_interval) == 0


def _pbt_exploit_explore(
    members: List[_PBTMember],
    space: Dict[str, Space],
    rng: np.random.Generator,
    quantile: float,
    perturbation_factors: Tuple[float, ...],
    resample_probability: float,
) -> None:
    """Truncation selection: copy elites into the bottom quantile, then perturb.

    The bottom ``floor(quantile * n)`` members (at least one, and at most
    half the population) each copy weights, hyperparameters, score and rung
    from a uniformly chosen member of the matching top quantile. Explore
    then perturbs only the copied hyperparameters.
    """
    n = len(members)
    cutoff = max(1, int(math.floor(float(quantile) * n)))
    cutoff = min(cutoff, n // 2)
    if cutoff < 1:
        return
    order = sorted(range(n), key=lambda i: (members[i].score, i))
    elites = order[:cutoff]
    for idx in order[-cutoff:]:
        donor = members[int(elites[int(rng.integers(0, len(elites)))])]
        child = members[idx]
        child.params = pbt_perturb(
            space,
            deepcopy(donor.params),
            rng,
            perturbation_factors=perturbation_factors,
            resample_probability=resample_probability,
        )
        child.weights = deepcopy(donor.weights)
        child.score = float(donor.score)
        child.rung = int(donor.rung)


def pbt_best_trial(trials: List[Trial]) -> Trial:
    """Lowest-score trial in a PBT history.

    When trials record ``resource``, the winner is chosen among the
    highest-resource evaluations, matching
    :func:`hyperopt_kit.evaluate.best_trial`, so a lucky cheap rung cannot
    beat a full-fidelity score.
    """
    if len(trials) == 0:
        raise ValueError("no trials available")
    resources = [
        trial.resource for trial in trials if getattr(trial, "resource", None) is not None
    ]
    if resources:
        r_max = max(resources)
        candidates = [
            trial
            for trial in trials
            if getattr(trial, "resource", None) is not None
            and trial.resource >= r_max - 1e-12
        ]
        if candidates:
            return min(candidates, key=lambda trial: trial.score)
    return min(trials, key=lambda trial: trial.score)


class PBTHistory(list):
    """Evaluation history returned by :func:`pbt_search`.

    The list itself is every :class:`Trial` in evaluation order. ``best``
    is the trial :func:`pbt_best_trial` would pick from that history.
    """

    @property
    def best(self) -> Trial:
        return pbt_best_trial(self)

    @property
    def history(self) -> List[Trial]:
        return list(self)


def _pbt_checkpoint(params: Dict[str, Any], resource: float, step: int, parent: Any) -> Dict[str, Any]:
    """Checkpoint stored when the objective returns a scalar score."""
    return {
        "params": deepcopy(params),
        "resource": float(resource),
        "step": int(step),
        "parent": parent,
    }


def pbt_search(
    space: Dict[str, Space],
    objective: ScoreFn,
    budget: int,
    rng: Optional[np.random.Generator] = None,
    stop_when: Stopper = None,
    *,
    population_size: Optional[int] = None,
    exploit_interval: int = 1,
    quantile: float = 0.25,
    perturbation_factors: Tuple[float, ...] = (0.8, 1.2),
    resample_probability: float = 0.25,
    eta: int = 3,
    min_resource: int = 1,
    max_resource: int = 9,
) -> PBTHistory:
    """Population-Based Training (Jaderberg et al., 2017).

    A population of configurations is sampled from ``space`` and stepped
    together along the fidelity ladder from :func:`successive_halving_rungs`.
    Each objective call is one resource rung (a normalized ``resource`` in
    ``(0, 1]``, ``1.0`` at full fidelity). After the last rung, later steps
    stay at full fidelity so training can continue. ``budget`` counts
    objective calls, matching :func:`tpe_search`, :func:`bohb_search` and
    :func:`cmaes_search`.

    Every ``exploit_interval`` completed generations, truncation selection
    replaces the worst ``quantile`` fraction of the population. Each
    replaced member copies the donor's weights and hyperparameters (and the
    donor's rung, since those weights have already been trained that far),
    then :func:`pbt_perturb` explores by perturbing continuous, integer and
    categorical dimensions. Members are evaluated in population order, so
    trial ``t`` belongs to member ``t % population_size``.

    The objective may return a float, or ``(score, weights)`` to own the
    checkpoint that exploit copies. Checkpoints are passed back as
    ``weights`` (or ``checkpoint``) when the callable accepts that argument;
    ``resource`` / ``fidelity`` is passed the same way as
    :func:`hyperband_search`. A scalar return is stored as a checkpoint of
    the params, resource and step so exploit still has weights to copy.
    A one-argument ``objective(params)`` remains valid.

    ``population_size`` defaults to 4 (or to the budget, when the budget is
    smaller, but never below 2). ``perturbation_factors`` defaults to
    ``(0.8, 1.2)``. ``quantile`` is the truncation fraction in ``(0, 0.5]``.

    Returns a :class:`PBTHistory`: the list is the full history, and
    ``history.best`` is the best trial (lowest score at the highest
    resource).
    """
    _validate_space(space)
    if budget < 1:
        raise ValueError("budget must be >= 1")
    factors = _validate_pbt_params(
        population_size,
        exploit_interval,
        quantile,
        perturbation_factors,
        resample_probability,
        eta,
        min_resource,
        max_resource,
    )
    rng = rng if rng is not None else np.random.default_rng()
    if population_size is None:
        size = 4 if budget >= 4 else max(2, int(budget))
    else:
        size = int(population_size)

    rungs = successive_halving_rungs(max_resource, eta, min_resource)
    evaluate = _pbt_caller(objective)
    members = [_PBTMember(params=_sample_config(space, rng)) for _ in range(size)]
    trials = PBTHistory()
    best = float("inf")

    while len(trials) < budget:
        progressed = False
        for member in members:
            if len(trials) >= budget:
                break
            next_rung = min(member.rung + 1, len(rungs) - 1)
            fidelity = float(rungs[next_rung])
            params = dict(member.params)
            score, returned, explicit = evaluate(params, fidelity, member.weights)
            member.eval_count += 1
            member.rung = next_rung
            member.score = float(score)
            if explicit:
                member.weights = returned
            else:
                member.weights = _pbt_checkpoint(
                    params, fidelity, member.eval_count, member.weights
                )
            trials.append(Trial(len(trials), dict(params), float(score), resource=fidelity))
            best = min(best, float(score))
            progressed = True
            if stop_when is not None and stop_when(best, len(trials)):
                return trials
        if not progressed:
            break
        if _pbt_generation_ready(members, exploit_interval):
            _pbt_exploit_explore(
                members,
                space,
                rng,
                float(quantile),
                factors,
                float(resample_probability),
            )
    return trials


population_based_training = pbt_search
