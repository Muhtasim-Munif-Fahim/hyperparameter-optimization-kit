"""Variance-based (Sobol) hyperparameter importance.

``param_importance`` in :mod:`hyperopt_kit.report` only groups the best
trials by value. This module answers the global question *"how much of the
variation in the objective is explained by each hyperparameter?"* using
Sobol sensitivity indices:

* the **first-order** index ``S1_i = Var(E[f | x_i]) / Var(f)`` is the share
  of variance explained by ``x_i`` alone;
* the **total-order** index ``ST_i = E[Var(f | x_~i)] / Var(f)`` adds every
  interaction ``x_i`` takes part in, so ``ST_i - S1_i`` measures how much of
  its influence comes from interactions.

Indices are estimated with the Saltelli (2010) first-order and Jansen (1999)
total-order Monte Carlo estimators over the unit hypercube, optionally with
bootstrap confidence intervals. :func:`surrogate_sobol_importance` applies
the estimator to a Gaussian-process surrogate fitted on completed trials, so
the importance analysis needs no extra objective evaluations.

References
----------
Saltelli, A. et al. (2010). Variance based sensitivity analysis of model
output. Design and estimator for the total sensitivity index. *Computer
Physics Communications*, 181(2), 259-270.
Jansen, M. J. W. (1999). Analysis of variance designs for model output.
*Computer Physics Communications*, 117(1-2), 35-43.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .searchers import GaussianProcess, Trial
from .spaces import Space

ArrayFn = Callable[[np.ndarray], np.ndarray]


@dataclass
class SobolResult:
    """Sobol indices for each input dimension.

    ``first_order`` and ``total_order`` are arrays of length ``d``;
    ``*_ci`` hold ``(low, high)`` bootstrap percentile intervals when
    bootstrap resampling was requested, otherwise ``None``.
    """

    names: List[str]
    first_order: np.ndarray
    total_order: np.ndarray
    variance: float
    n_evaluations: int
    first_order_ci: Optional[np.ndarray] = None
    total_order_ci: Optional[np.ndarray] = None
    extra: Dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Dict[str, object]]:
        out: Dict[str, Dict[str, object]] = {}
        for i, name in enumerate(self.names):
            row: Dict[str, object] = {
                "first_order": float(self.first_order[i]),
                "total_order": float(self.total_order[i]),
                "interaction": float(self.total_order[i] - self.first_order[i]),
            }
            if self.first_order_ci is not None:
                row["first_order_ci"] = [float(v) for v in self.first_order_ci[i]]
            if self.total_order_ci is not None:
                row["total_order_ci"] = [float(v) for v in self.total_order_ci[i]]
            out[name] = row
        return out

    def ranking(self, by: str = "total_order") -> List[Tuple[str, float]]:
        """Names ordered from most to least important by ``by``."""
        if by not in ("first_order", "total_order"):
            raise ValueError("by must be 'first_order' or 'total_order'")
        values = getattr(self, by)
        order = np.argsort(-values, kind="stable")
        return [(self.names[i], float(values[i])) for i in order]


def saltelli_matrices(
    n: int, d: int, rng: Optional[np.random.Generator] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``A``, ``B`` (``n x d`` uniform) and ``AB`` (``d x n x d``).

    ``AB[i]`` is ``A`` with column ``i`` replaced by column ``i`` of ``B``.
    """
    if n < 2:
        raise ValueError("n must be >= 2")
    if d < 1:
        raise ValueError("d must be >= 1")
    rng = rng if rng is not None else np.random.default_rng()
    base = rng.random((n, 2 * d))
    A, B = base[:, :d], base[:, d:]
    AB = np.repeat(A[None, :, :], d, axis=0)
    for i in range(d):
        AB[i, :, i] = B[:, i]
    return A, B, AB


def _indices(fA: np.ndarray, fB: np.ndarray, fAB: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    var = float(np.var(np.concatenate([fA, fB])))
    if var <= 0.0:
        d = fAB.shape[0]
        return np.zeros(d), np.zeros(d), 0.0
    # Centering fB leaves the estimator unbiased (E[fAB - fA] = 0) but
    # removes the variance contributed by a large output mean.
    centre = float(np.mean(np.concatenate([fA, fB])))
    s1 = np.mean((fB - centre)[None, :] * (fAB - fA[None, :]), axis=1) / var
    st = 0.5 * np.mean((fA[None, :] - fAB) ** 2, axis=1) / var
    return s1, st, var


def sobol_indices(
    func: ArrayFn,
    d: int,
    n: int = 2048,
    rng: Optional[np.random.Generator] = None,
    *,
    names: Optional[Sequence[str]] = None,
    n_bootstrap: int = 0,
    confidence: float = 0.95,
) -> SobolResult:
    """Estimate first- and total-order Sobol indices of ``func`` on ``[0, 1]^d``.

    ``func`` maps an ``(m, d)`` array to ``m`` outputs. The estimator costs
    ``n * (d + 2)`` function evaluations. With ``n_bootstrap > 0`` the rows
    of the Saltelli design are resampled to give percentile intervals.
    """
    if names is None:
        names = [f"x{i}" for i in range(d)]
    names = list(names)
    if len(names) != d:
        raise ValueError("names must have length d")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    if n_bootstrap < 0:
        raise ValueError("n_bootstrap must be >= 0")
    rng = rng if rng is not None else np.random.default_rng()
    A, B, AB = saltelli_matrices(n, d, rng)
    stacked = np.vstack([A, B, AB.reshape(d * n, d)])
    out = np.asarray(func(stacked), dtype=float).reshape(-1)
    if out.shape[0] != stacked.shape[0]:
        raise ValueError("func must return one value per input row")
    if not np.all(np.isfinite(out)):
        raise ValueError("func returned non-finite values")
    fA, fB = out[:n], out[n:2 * n]
    fAB = out[2 * n:].reshape(d, n)
    s1, st, var = _indices(fA, fB, fAB)
    s1_ci = st_ci = None
    if n_bootstrap > 0:
        s1_boot = np.empty((n_bootstrap, d))
        st_boot = np.empty((n_bootstrap, d))
        for b in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            s1_boot[b], st_boot[b], _ = _indices(fA[idx], fB[idx], fAB[:, idx])
        lo, hi = 50.0 * (1.0 - confidence), 50.0 * (1.0 + confidence)
        s1_ci = np.stack([np.percentile(s1_boot, lo, axis=0), np.percentile(s1_boot, hi, axis=0)], axis=1)
        st_ci = np.stack([np.percentile(st_boot, lo, axis=0), np.percentile(st_boot, hi, axis=0)], axis=1)
    return SobolResult(
        names=names,
        first_order=s1,
        total_order=st,
        variance=var,
        n_evaluations=int(stacked.shape[0]),
        first_order_ci=s1_ci,
        total_order_ci=st_ci,
    )


def _encode(space: Dict[str, Space], names: List[str], trials: Sequence[Trial]) -> np.ndarray:
    return np.array(
        [[space[name].normalize(t.params[name]) for name in names] for t in trials],
        dtype=float,
    )


def _snap(space: Dict[str, Space], names: List[str], X: np.ndarray) -> np.ndarray:
    """Round normalized points to values the space can actually take."""
    Y = np.empty_like(X)
    for j, name in enumerate(names):
        sp = space[name]
        Y[:, j] = [sp.normalize(sp.denormalize(float(x))) for x in X[:, j]]
    return Y


def surrogate_sobol_importance(
    trials: Sequence[Trial],
    space: Dict[str, Space],
    n: int = 1024,
    rng: Optional[np.random.Generator] = None,
    *,
    n_bootstrap: int = 0,
    confidence: float = 0.95,
    snap: bool = True,
) -> SobolResult:
    """Sobol importance of each hyperparameter, estimated on a GP surrogate.

    A :class:`~hyperopt_kit.searchers.GaussianProcess` is fitted to the
    normalized parameters and scores of ``trials``; Sobol indices are then
    estimated on its posterior mean, so no extra objective calls are made.
    For multi-fidelity runs only trials at the highest observed resource are
    used. With ``snap=True`` integer and categorical parameters are rounded
    to admissible values before the surrogate is queried.
    """
    trials = [t for t in trials if np.isfinite(t.score)]
    resources = [t.resource for t in trials if t.resource is not None]
    if resources:
        top = max(resources)
        trials = [t for t in trials if t.resource is None or t.resource == top]
    if len(trials) < 3:
        raise ValueError("need at least 3 finite trials to fit the surrogate")
    names = list(space)
    missing = [nm for nm in names if any(nm not in t.params for t in trials)]
    if missing:
        raise ValueError(f"trials are missing parameters: {missing}")
    X = _encode(space, names, trials)
    y = np.array([t.score for t in trials], dtype=float)
    gp = GaussianProcess()
    gp.fit(X, y)

    def surrogate(Z: np.ndarray) -> np.ndarray:
        if snap:
            Z = _snap(space, names, Z)
        return gp.predict(Z)[0]

    result = sobol_indices(
        surrogate,
        len(names),
        n,
        rng,
        names=names,
        n_bootstrap=n_bootstrap,
        confidence=confidence,
    )
    result.extra.update({"n_trials": len(trials), "length_scale": gp.length_scale})
    return result


def format_sobol_table(result: SobolResult) -> str:
    """Markdown table sorted by total-order index."""
    lines = [
        "| parameter | S1 (first order) | ST (total) | ST - S1 |",
        "| --- | ---: | ---: | ---: |",
    ]
    order = np.argsort(-result.total_order, kind="stable")
    for i in order:
        s1, st = result.first_order[i], result.total_order[i]
        s1_txt, st_txt = f"{s1:.3f}", f"{st:.3f}"
        if result.first_order_ci is not None:
            lo, hi = result.first_order_ci[i]
            s1_txt += f" [{lo:.3f}, {hi:.3f}]"
        if result.total_order_ci is not None:
            lo, hi = result.total_order_ci[i]
            st_txt += f" [{lo:.3f}, {hi:.3f}]"
        lines.append(f"| {result.names[i]} | {s1_txt} | {st_txt} | {st - s1:.3f} |")
    return "\n".join(lines)
