"""Multi-objective search: Pareto dominance, NSGA-II and quality indicators.

Many tuning problems trade off competing goals, such as validation loss
against latency, model size or training cost, where no single best
configuration exists. This module provides:

* :func:`dominates`, :func:`non_dominated_sort` (Deb et al.'s fast
  non-dominated sort), :func:`pareto_front_mask` and
  :func:`crowding_distance`;
* :func:`hypervolume`, the exact dominated hypervolume with respect to a
  reference point (2-D sweep, recursive slicing for 3 or more objectives),
  plus :func:`igd` / :func:`generational_distance` against a known front;
* :func:`nsga2_search`, NSGA-II (Deb, Pratap, Agarwal & Meyarivan, 2002) on
  the normalised unit cube of a :class:`~hyperopt_kit.spaces.Space` mapping.
  It uses binary tournament on (rank, crowding), simulated binary crossover
  (SBX), polynomial mutation and elitist (mu + lambda) survival. Floats,
  ints, log ranges and categoricals all work through ``Space.denormalize``.

All objectives are **minimised**. The objective callable takes a parameter
dict and returns a sequence of floats, one per objective.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .spaces import Space

__all__ = [
    "MOTrial",
    "crowding_distance",
    "dominates",
    "generational_distance",
    "hypervolume",
    "igd",
    "non_dominated_sort",
    "nsga2_search",
    "pareto_front_mask",
    "pareto_trials",
    "polynomial_mutation",
    "sbx_crossover",
]

MultiObjectiveFn = Callable[[Dict[str, Any]], Sequence[float]]


@dataclass
class MOTrial:
    """One evaluated configuration with a vector of objective values."""

    iteration: int
    params: Dict[str, Any]
    objectives: Tuple[float, ...]


def _as_points(F, name: str = "F") -> np.ndarray:
    A = np.asarray(F, dtype=float)
    if A.ndim == 1:
        A = A[None, :]
    if A.ndim != 2 or A.shape[1] == 0:
        raise ValueError(f"{name} must be a (n_points, n_objectives) array")
    if not np.all(np.isfinite(A)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return A


def dominates(a: Sequence[float], b: Sequence[float]) -> bool:
    """True if ``a`` Pareto-dominates ``b``: no worse anywhere, better somewhere."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return bool(np.all(a <= b) and np.any(a < b))


def non_dominated_sort(F) -> List[np.ndarray]:
    """Fast non-dominated sort. Returns index arrays, front 0 first.

    Duplicate points share a front, since neither dominates the other.
    """
    A = _as_points(F)
    n = A.shape[0]
    if n == 0:
        return []
    le = np.all(A[:, None, :] <= A[None, :, :], axis=2)
    lt = np.any(A[:, None, :] < A[None, :, :], axis=2)
    dom = le & lt                        # dom[i, j]: i dominates j
    count = dom.sum(axis=0)              # number of points dominating j
    fronts: List[np.ndarray] = []
    current = np.flatnonzero(count == 0)
    while current.size:
        fronts.append(current)
        count = count - dom[current].sum(axis=0)
        count[current] = -1
        current = np.flatnonzero(count == 0)
    return fronts


def pareto_front_mask(F) -> np.ndarray:
    """Boolean mask of the non-dominated points."""
    A = _as_points(F)
    mask = np.zeros(A.shape[0], dtype=bool)
    if A.shape[0]:
        mask[non_dominated_sort(A)[0]] = True
    return mask


def crowding_distance(F) -> np.ndarray:
    """NSGA-II crowding distance of the points of a single front.

    For each objective, the points are sorted and the two boundary points get
    ``inf``. Each interior point adds ``(f[next] - f[prev]) / (f_max - f_min)``,
    and an objective with zero spread adds nothing. Fronts with two or fewer
    points are all ``inf``.
    """
    A = _as_points(F)
    n, m = A.shape
    if n <= 2:
        return np.full(n, np.inf)
    dist = np.zeros(n)
    for j in range(m):
        order = np.argsort(A[:, j], kind="stable")
        f = A[order, j]
        span = f[-1] - f[0]
        dist[order[0]] = np.inf
        dist[order[-1]] = np.inf
        if span <= 0:
            continue
        dist[order[1:-1]] += (f[2:] - f[:-2]) / span
    return dist


def _hv2(P: np.ndarray, ref: np.ndarray) -> float:
    P = P[np.argsort(P[:, 0], kind="stable")]
    vol, best_y = 0.0, ref[1]
    for x, y in P:
        if y < best_y:
            vol += (ref[0] - x) * (best_y - y)
            best_y = y
    return vol


def _hv(P: np.ndarray, ref: np.ndarray) -> float:
    if P.shape[0] == 0:
        return 0.0
    d = P.shape[1]
    if d == 1:
        return float(ref[0] - P[:, 0].min())
    if d == 2:
        return _hv2(P, ref)
    P = P[np.argsort(P[:, -1], kind="stable")]
    vol = 0.0
    for i in range(P.shape[0]):
        upper = P[i + 1, -1] if i + 1 < P.shape[0] else ref[-1]
        height = upper - P[i, -1]
        if height > 0:
            sub = P[: i + 1, :-1]
            sub = sub[pareto_front_mask(sub)]
            vol += _hv(sub, ref[:-1]) * height
    return vol


def hypervolume(F, ref) -> float:
    """Exact hypervolume dominated by ``F`` and bounded above by ``ref``.

    Points that do not strictly dominate ``ref`` in every objective add
    nothing. Two objectives use an O(n log n) sweep. Three or more use
    recursive slicing along the last objective, which is exact and suits
    the few-hundred-point fronts that come out of hyperparameter search.
    """
    A = _as_points(F)
    r = np.asarray(ref, dtype=float).ravel()
    if r.size != A.shape[1]:
        raise ValueError(f"ref has {r.size} entries but F has {A.shape[1]} objectives")
    A = A[np.all(A < r, axis=1)]
    if A.shape[0] == 0:
        return 0.0
    A = A[pareto_front_mask(A)]
    return float(_hv(A, r))


def _min_dists(From: np.ndarray, To: np.ndarray) -> np.ndarray:
    d = np.sqrt(((From[:, None, :] - To[None, :, :]) ** 2).sum(axis=2))
    return d.min(axis=1)


def igd(F, reference_front) -> float:
    """Inverted generational distance: mean distance from each reference
    point to its nearest point in ``F`` (lower is better)."""
    A = _as_points(F)
    R = _as_points(reference_front, "reference_front")
    if R.shape[1] != A.shape[1]:
        raise ValueError("F and reference_front must have the same number of objectives")
    return float(_min_dists(R, A).mean())


def generational_distance(F, reference_front) -> float:
    """Generational distance: mean distance from each point of ``F`` to the
    nearest reference point (lower is better)."""
    A = _as_points(F)
    R = _as_points(reference_front, "reference_front")
    if R.shape[1] != A.shape[1]:
        raise ValueError("F and reference_front must have the same number of objectives")
    return float(_min_dists(A, R).mean())


def sbx_crossover(
    p1: np.ndarray,
    p2: np.ndarray,
    rng: np.random.Generator,
    eta: float = 15.0,
    prob_var: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Bounded simulated binary crossover (Deb & Agrawal, 1995) on [0, 1]^d.

    Each variable is recombined with probability ``prob_var`` (when the
    parents differ). The spread factor uses Deb's bounded form, so children
    stay inside the unit cube. Children are then swapped per variable with
    probability 1/2.
    """
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    c1, c2 = p1.copy(), p2.copy()
    for j in range(p1.size):
        if rng.random() > prob_var or abs(p1[j] - p2[j]) <= 1e-14:
            continue
        y1, y2 = min(p1[j], p2[j]), max(p1[j], p2[j])
        u = rng.random()
        children = []
        for bound_gap in (y1 - 0.0, 1.0 - y2):
            beta = 1.0 + 2.0 * bound_gap / (y2 - y1)
            alpha = 2.0 - beta ** (-(eta + 1.0))
            if u <= 1.0 / alpha:
                betaq = (u * alpha) ** (1.0 / (eta + 1.0))
            else:
                betaq = (1.0 / (2.0 - u * alpha)) ** (1.0 / (eta + 1.0))
            children.append(betaq)
        a = 0.5 * ((y1 + y2) - children[0] * (y2 - y1))
        b = 0.5 * ((y1 + y2) + children[1] * (y2 - y1))
        a, b = float(np.clip(a, 0.0, 1.0)), float(np.clip(b, 0.0, 1.0))
        if rng.random() < 0.5:
            a, b = b, a
        c1[j], c2[j] = a, b
    return c1, c2


def polynomial_mutation(
    x: np.ndarray,
    rng: np.random.Generator,
    eta: float = 20.0,
    prob: Optional[float] = None,
) -> np.ndarray:
    """Bounded polynomial mutation (Deb & Goyal, 1996) on [0, 1]^d.

    Each variable mutates with probability ``prob`` (default ``1/d``).
    """
    y = np.asarray(x, dtype=float).copy()
    d = y.size
    pm = 1.0 / d if prob is None else float(prob)
    mut_pow = 1.0 / (eta + 1.0)
    for j in range(d):
        if rng.random() >= pm:
            continue
        v = y[j]
        d1, d2 = v - 0.0, 1.0 - v
        u = rng.random()
        if u < 0.5:
            xy = 1.0 - d1
            val = 2.0 * u + (1.0 - 2.0 * u) * xy ** (eta + 1.0)
            dq = val ** mut_pow - 1.0
        else:
            xy = 1.0 - d2
            val = 2.0 * (1.0 - u) + 2.0 * (u - 0.5) * xy ** (eta + 1.0)
            dq = 1.0 - val ** mut_pow
        y[j] = float(np.clip(v + dq, 0.0, 1.0))
    return y


def _rank_and_crowding(F: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    rank = np.empty(F.shape[0], dtype=int)
    crowd = np.empty(F.shape[0])
    for r, front in enumerate(non_dominated_sort(F)):
        rank[front] = r
        crowd[front] = crowding_distance(F[front])
    return rank, crowd


def _survive(F: np.ndarray, n_keep: int) -> np.ndarray:
    keep: List[int] = []
    for front in non_dominated_sort(F):
        if len(keep) + front.size <= n_keep:
            keep.extend(front.tolist())
            continue
        cd = crowding_distance(F[front])
        order = np.argsort(-cd, kind="stable")
        keep.extend(front[order[: n_keep - len(keep)]].tolist())
        break
    return np.asarray(keep, dtype=int)


def _tournament(rank, crowd, rng) -> int:
    i, j = rng.integers(0, rank.size, size=2)
    if rank[i] != rank[j]:
        return int(i if rank[i] < rank[j] else j)
    if crowd[i] != crowd[j]:
        return int(i if crowd[i] > crowd[j] else j)
    return int(i if rng.random() < 0.5 else j)


def nsga2_search(
    space: Dict[str, Space],
    objective: MultiObjectiveFn,
    budget: int,
    rng: Optional[np.random.Generator] = None,
    *,
    population_size: Optional[int] = None,
    crossover_prob: float = 0.9,
    eta_crossover: float = 15.0,
    eta_mutation: float = 20.0,
    mutation_prob: Optional[float] = None,
) -> List[MOTrial]:
    """NSGA-II multi-objective search over ``space`` (minimise every objective).

    ``budget`` counts objective evaluations. ``population_size`` defaults to
    ``min(max(8, 4 * n_dim), budget)`` and is rounded up to an even number
    (capped at ``budget``). Returns every evaluated trial in order. Use
    :func:`pareto_trials` to get the non-dominated set.
    """
    if not isinstance(space, dict) or not space:
        raise ValueError("space must be a non-empty mapping of name -> Space")
    for name, sp in space.items():
        if not isinstance(sp, Space):
            raise ValueError(f"parameter {name!r} is not a Space instance")
    if budget < 2:
        raise ValueError("budget must be >= 2")
    if not 0.0 <= crossover_prob <= 1.0:
        raise ValueError("crossover_prob must be in [0, 1]")
    if eta_crossover <= 0 or eta_mutation <= 0:
        raise ValueError("eta_crossover and eta_mutation must be positive")
    if mutation_prob is not None and not 0.0 <= mutation_prob <= 1.0:
        raise ValueError("mutation_prob must be in [0, 1]")
    rng = rng if rng is not None else np.random.default_rng()
    names = list(space)
    d = len(names)
    size = int(population_size) if population_size is not None else max(8, 4 * d)
    if size < 2:
        raise ValueError("population_size must be >= 2")
    size = min(size + (size % 2), int(budget))

    trials: List[MOTrial] = []
    n_obj: List[int] = []

    def evaluate(x: np.ndarray) -> np.ndarray:
        params = {name: space[name].denormalize(float(x[j])) for j, name in enumerate(names)}
        vals = np.asarray(objective(params), dtype=float).ravel()
        if vals.size < 2:
            raise ValueError("multi-objective functions must return at least 2 values")
        if n_obj and vals.size != n_obj[0]:
            raise ValueError("objective returned a different number of values than before")
        if not np.all(np.isfinite(vals)):
            raise ValueError(f"objective returned non-finite values {vals.tolist()}")
        n_obj[:1] = [vals.size]
        trials.append(MOTrial(len(trials), params, tuple(float(v) for v in vals)))
        return vals

    X = rng.random((size, d))
    F = np.vstack([evaluate(x) for x in X])
    while len(trials) < budget:
        rank, crowd = _rank_and_crowding(F)
        n_children = min(size, budget - len(trials))
        kids: List[np.ndarray] = []
        while len(kids) < n_children:
            a = X[_tournament(rank, crowd, rng)]
            b = X[_tournament(rank, crowd, rng)]
            if rng.random() < crossover_prob:
                c1, c2 = sbx_crossover(a, b, rng, eta=eta_crossover)
            else:
                c1, c2 = a.copy(), b.copy()
            kids.append(polynomial_mutation(c1, rng, eta=eta_mutation, prob=mutation_prob))
            if len(kids) < n_children:
                kids.append(polynomial_mutation(c2, rng, eta=eta_mutation, prob=mutation_prob))
        Xk = np.vstack(kids)
        Fk = np.vstack([evaluate(x) for x in Xk])
        X_all = np.vstack([X, Xk])
        F_all = np.vstack([F, Fk])
        keep = _survive(F_all, size)
        X, F = X_all[keep], F_all[keep]
    return trials


def pareto_trials(trials: Sequence[MOTrial], unique: bool = True) -> List[MOTrial]:
    """The non-dominated trials, sorted by objectives (first objective first).

    With ``unique=True`` (the default), trials whose objective vectors are
    identical (for example, the same configuration re-evaluated) are
    collapsed, keeping the earliest.
    """
    if not trials:
        return []
    F = np.array([t.objectives for t in trials], dtype=float)
    idx = np.flatnonzero(pareto_front_mask(F))
    idx = idx[np.lexsort(np.vstack([idx, F[idx].T[::-1]]))]
    if unique:
        seen = set()
        kept = []
        for i in idx:
            key = tuple(F[i])
            if key not in seen:
                seen.add(key)
                kept.append(i)
        idx = np.asarray(kept, dtype=int)
    return [trials[i] for i in idx]
