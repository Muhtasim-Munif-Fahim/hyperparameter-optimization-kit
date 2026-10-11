"""Low-discrepancy (quasi-random) designs and a quasi-random search baseline.

Bergstra & Bengio (2012) showed that random search beats grid search in
hyperparameter optimization because only a few dimensions usually matter.
They also noted that *low-discrepancy* sequences cover the space more
evenly than i.i.d. uniform draws at the same budget. This module provides:

* :func:`halton_sequence`: the Halton sequence (radical inverses in the
  first ``d`` prime bases), optionally scrambled with independent random
  digit permutations per base and digit position (Braaten-Weller /
  Mascagni-Chi style). Scrambling removes the strong correlations that
  plain Halton shows between high-dimensional coordinate pairs;
* :func:`latin_hypercube`: Latin hypercube sampling (McKay, Beckman and
  Conover, 1979), optionally the most space-filling (maximin) of several
  random designs;
* :func:`centered_discrepancy`: Hickernell's squared centered
  L2-discrepancy, a quantitative measure of uniformity (lower is better);
* :func:`quasi_random_search`: a drop-in replacement for ``random_search``
  that evaluates a Halton or LHS design mapped through the search space.

Everything is NumPy only.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np

from .searchers import ScoreFn, Stopper, Trial, _validate_space
from .spaces import Categorical, FloatRange, IntRange, Space

QUASI_RANDOM_METHODS = ("halton", "lhs")


def first_primes(n: int) -> List[int]:
    """The first ``n`` prime numbers."""
    if not isinstance(n, (int, np.integer)) or isinstance(n, bool) or n < 0:
        raise ValueError("n must be a non-negative integer")
    primes: List[int] = []
    candidate = 2
    while len(primes) < n:
        if all(candidate % p for p in primes if p * p <= candidate):
            primes.append(candidate)
        candidate += 1
    return primes


def _check_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _radical_inverse(indices: np.ndarray, base: int, permutations: Optional[np.ndarray]) -> np.ndarray:
    """Van der Corput radical inverse of ``indices`` in ``base``.

    With ``permutations`` (shape ``(n_digits, base)``) every digit position
    ``k`` maps its digit through ``permutations[k]``. All ``n_digits``
    positions are used, including the zero digits beyond an index's length,
    so the scrambled point is fully randomized to double precision.
    """
    result = np.zeros(indices.shape, dtype=float)
    remaining = indices.astype(np.int64).copy()
    n_digits = permutations.shape[0] if permutations is not None else None
    scale = 1.0 / base
    k = 0
    while True:
        if permutations is None:
            if not np.any(remaining):
                break
        elif k >= n_digits:
            break
        digits = remaining % base
        if permutations is not None:
            digits = permutations[k][digits]
        result += digits * scale
        remaining //= base
        scale /= base
        k += 1
    return result


def halton_sequence(
    n: int,
    d: int,
    *,
    scramble: bool = True,
    rng: Optional[np.random.Generator] = None,
    start_index: int = 0,
) -> np.ndarray:
    """``n`` points of the ``d``-dimensional Halton sequence in ``[0, 1)^d``.

    Point ``i`` (counting from ``start_index``) has coordinate ``j`` equal to
    the radical inverse of ``i`` in the ``j``-th prime base. Unscrambled, the
    sequence starts at the origin, and it matches
    ``scipy.stats.qmc.Halton(d, scramble=False)``. ``scramble=True`` draws an
    independent random permutation of ``{0..b-1}`` for every base and digit
    position (enough positions to reach double precision).
    """
    n = _check_positive_int(n, "n")
    d = _check_positive_int(d, "d")
    if isinstance(start_index, bool) or not isinstance(start_index, (int, np.integer)) or start_index < 0:
        raise ValueError("start_index must be a non-negative integer")
    rng = rng if rng is not None else np.random.default_rng()
    indices = np.arange(start_index, start_index + n, dtype=np.int64)
    points = np.empty((n, d), dtype=float)
    for j, base in enumerate(first_primes(d)):
        perms = None
        if scramble:
            n_digits = int(math.ceil(53 * math.log(2) / math.log(base)))
            perms = np.stack([rng.permutation(base) for _ in range(n_digits)])
        points[:, j] = _radical_inverse(indices, base, perms)
    return np.clip(points, 0.0, np.nextafter(1.0, 0.0))


def _min_pairwise_distance(points: np.ndarray) -> float:
    if points.shape[0] < 2:
        return float("inf")
    diff = points[:, None, :] - points[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1))
    iu = np.triu_indices(points.shape[0], k=1)
    return float(dist[iu].min())


def latin_hypercube(
    n: int,
    d: int,
    *,
    rng: Optional[np.random.Generator] = None,
    centered: bool = False,
    n_candidates: int = 1,
) -> np.ndarray:
    """Latin hypercube sample of ``n`` points in ``[0, 1)^d``.

    Each coordinate puts exactly one point in each of the ``n`` strata
    ``[k/n, (k+1)/n)``. Within a stratum the point is uniform, or at the
    stratum midpoint with ``centered=True``. With ``n_candidates > 1``,
    that many independent designs are drawn and the one with the largest
    minimum pairwise distance is returned (maximin LHS).
    """
    n = _check_positive_int(n, "n")
    d = _check_positive_int(d, "d")
    n_candidates = _check_positive_int(n_candidates, "n_candidates")
    rng = rng if rng is not None else np.random.default_rng()
    best = None
    best_dist = -1.0
    for _ in range(n_candidates):
        strata = np.stack([rng.permutation(n) for _ in range(d)], axis=1).astype(float)
        offsets = 0.5 if centered else rng.random((n, d))
        design = (strata + offsets) / n
        dist = _min_pairwise_distance(design) if n_candidates > 1 else 0.0
        if best is None or dist > best_dist:
            best, best_dist = design, dist
    return best


def centered_discrepancy(points) -> float:
    """Hickernell's squared centered L2-discrepancy of points in ``[0, 1]^d``.

    Lower values mean more uniform coverage. Equals
    ``scipy.stats.qmc.discrepancy(points, method="CD")``.
    """
    x = np.asarray(points, dtype=float)
    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError("points must be a non-empty (n, d) array")
    if np.any(x < 0.0) or np.any(x > 1.0) or not np.all(np.isfinite(x)):
        raise ValueError("points must lie in the unit hypercube")
    n, d = x.shape
    a = np.abs(x - 0.5)
    term1 = (13.0 / 12.0) ** d
    term2 = (2.0 / n) * np.sum(np.prod(1.0 + 0.5 * a - 0.5 * a * a, axis=1))
    pair = (
        1.0
        + 0.5 * a[:, None, :]
        + 0.5 * a[None, :, :]
        - 0.5 * np.abs(x[:, None, :] - x[None, :, :])
    )
    term3 = np.sum(np.prod(pair, axis=2)) / (n * n)
    return float(term1 - term2 + term3)


def unit_to_config(u, names: List[str], space: Dict[str, Space]) -> Dict[str, Any]:
    """Map a point of ``[0, 1)^d`` to a configuration, preserving uniformity.

    Unlike ``Space.denormalize`` (which rounds, so the end values of an
    integer or categorical range get half the mass), equal-width cells of
    ``u`` map to each integer and to each categorical choice. This keeps
    the stratification of a low-discrepancy design. Log-scaled integers use
    the same log-uniform map as ``IntRange.sample``.
    """
    config: Dict[str, Any] = {}
    for j, name in enumerate(names):
        sp = space[name]
        x = float(min(max(u[j], 0.0), np.nextafter(1.0, 0.0)))
        if isinstance(sp, Categorical):
            config[name] = sp.choices[min(int(x * len(sp.choices)), len(sp.choices) - 1)]
        elif isinstance(sp, IntRange):
            if sp.log_scale and sp.n_values > 1:
                v = math.exp(math.log(sp.low) + x * (math.log(sp.high) - math.log(sp.low)))
                config[name] = int(min(max(round(v), sp.low), sp.high))
            else:
                config[name] = sp.low + min(int(x * sp.n_values), sp.n_values - 1)
        elif isinstance(sp, FloatRange):
            config[name] = sp.denormalize(x)
        else:
            config[name] = sp.denormalize(x)
    return config


def quasi_random_search(
    space: Dict[str, Space],
    objective: ScoreFn,
    budget: int,
    rng: Optional[np.random.Generator] = None,
    stop_when: Stopper = None,
    *,
    method: str = "halton",
    scramble: bool = True,
    n_candidates: int = 1,
) -> List[Trial]:
    """Evaluate a low-discrepancy design of ``budget`` configurations.

    ``method="halton"`` (default) uses the scrambled Halton sequence.
    ``method="lhs"`` uses a Latin hypercube; set ``n_candidates > 1`` for
    maximin LHS. Points are mapped with :func:`unit_to_config`. Like
    ``random_search`` it needs no model, so it is a stronger baseline at
    the same cost. The whole design is generated up front (LHS needs to
    know ``budget``), and ``stop_when`` can still end the run early.
    """
    _validate_space(space)
    if isinstance(budget, bool) or not isinstance(budget, (int, np.integer)) or budget < 1:
        raise ValueError("budget must be >= 1")
    if method not in QUASI_RANDOM_METHODS:
        raise ValueError(f"method must be one of {QUASI_RANDOM_METHODS}")
    rng = rng if rng is not None else np.random.default_rng()
    names = list(space)
    if method == "halton":
        design = halton_sequence(int(budget), len(names), scramble=scramble, rng=rng)
    else:
        design = latin_hypercube(int(budget), len(names), rng=rng, n_candidates=n_candidates)

    trials: List[Trial] = []
    best = float("inf")
    for i in range(int(budget)):
        params = unit_to_config(design[i], names, space)
        score = float(objective(params))
        trials.append(Trial(i, params, score))
        best = min(best, score)
        if stop_when is not None and stop_when(best, len(trials)):
            break
    return trials


def halton_search(space, objective, budget, rng=None, stop_when=None, *, scramble=True):
    """``quasi_random_search`` with ``method="halton"``."""
    return quasi_random_search(
        space, objective, budget, rng, stop_when, method="halton", scramble=scramble
    )


def lhs_search(space, objective, budget, rng=None, stop_when=None, *, n_candidates=1):
    """``quasi_random_search`` with ``method="lhs"``."""
    return quasi_random_search(
        space, objective, budget, rng, stop_when, method="lhs", n_candidates=n_candidates
    )
