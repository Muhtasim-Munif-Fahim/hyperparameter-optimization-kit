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
that spend cheap evaluations to discard poor configurations early.
"""

from __future__ import annotations

import inspect
import itertools
import math
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
