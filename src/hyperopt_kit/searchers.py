"""Search strategies over a ``{name: Space}`` configuration space.

``grid_search`` and ``random_search`` are self-explanatory baselines;
``bayesian_search`` fits a squared-exponential Gaussian-process surrogate
and picks the next configuration by maximizing expected improvement.
"""

from __future__ import annotations

import itertools
import math
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
