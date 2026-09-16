"""Search strategies over a ``{name: Space}`` configuration space.

``grid_search`` and ``random_search`` are self-explanatory baselines;
``bayesian_search`` fits a squared-exponential Gaussian-process surrogate
and picks the next configuration by maximizing expected improvement;
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
