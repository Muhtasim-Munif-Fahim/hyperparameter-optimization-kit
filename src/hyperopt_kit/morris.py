"""Morris elementary-effects screening for hyperparameters.

Sobol indices (:mod:`hyperopt_kit.sensitivity`) need ``n * (d + 2)``
evaluations, which is usually too many to spend on the real objective.
The Morris (1991) method is the classic cheap *screening* design: it costs
``r * (d + 1)`` evaluations and tells which parameters matter at all and
whether their effect is linear/additive or non-linear/interacting.

The unit cube is discretised to a ``p``-level grid ``{0, 1/(p-1), ..., 1}``
with step ``delta = p / (2 (p - 1))``. Each of ``r`` *trajectories* starts at
a random grid point and moves one coordinate at a time (in random order) by
``+/- delta``; every move gives one elementary effect

    EE_i = (f(x + delta e_i) - f(x)) / delta.

Per parameter we report

* ``mu`` — mean effect (signed; effects of opposite sign can cancel),
* ``mu_star`` — mean absolute effect (Campolongo, Cariboni & Saltelli 2007),
  the recommended importance measure,
* ``sigma`` — standard deviation of the effects (``ddof=1``): large values
  flag non-linearity or interactions.

With ``n_candidates > r`` the trajectories are chosen from a larger random
pool by Campolongo et al.'s spread criterion (greedily maximising the sum
of pairwise trajectory distances), which improves coverage of the cube.

References
----------
Morris, M. D. (1991). Factorial sampling plans for preliminary computational
experiments. *Technometrics*, 33(2), 161-174.
Campolongo, F., Cariboni, J. & Saltelli, A. (2007). An effective screening
design for sensitivity analysis of large models. *Environmental Modelling &
Software*, 22(10), 1509-1518.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .spaces import Space

ArrayFn = Callable[[np.ndarray], np.ndarray]

__all__ = [
    "MorrisResult",
    "morris_trajectories",
    "morris_elementary_effects",
    "morris_screening",
    "morris_hyperparameter_screening",
    "format_morris_table",
]


@dataclass
class MorrisResult:
    """Morris screening statistics per input dimension."""

    names: List[str]
    mu: np.ndarray
    mu_star: np.ndarray
    sigma: np.ndarray
    effects: np.ndarray  # (r, d) elementary effects
    n_evaluations: int
    mu_star_ci: Optional[np.ndarray] = None
    extra: Dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Dict[str, object]]:
        out: Dict[str, Dict[str, object]] = {}
        for i, name in enumerate(self.names):
            row: Dict[str, object] = {
                "mu": float(self.mu[i]),
                "mu_star": float(self.mu_star[i]),
                "sigma": float(self.sigma[i]),
            }
            if self.mu_star_ci is not None:
                row["mu_star_ci"] = [float(v) for v in self.mu_star_ci[i]]
            out[name] = row
        return out

    def ranking(self, by: str = "mu_star") -> List[Tuple[str, float]]:
        """Names ordered from most to least influential by ``by``."""
        if by not in ("mu_star", "sigma", "abs_mu"):
            raise ValueError("by must be 'mu_star', 'sigma' or 'abs_mu'")
        values = np.abs(self.mu) if by == "abs_mu" else getattr(self, by)
        order = np.argsort(-values, kind="stable")
        return [(self.names[i], float(values[i])) for i in order]


def _check_int(value: Any, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _one_trajectory(d: int, p: int, rng: np.random.Generator) -> np.ndarray:
    delta = p / (2.0 * (p - 1))
    levels = np.arange(p) / (p - 1)
    x = levels[rng.integers(0, p, size=d)]
    traj = np.empty((d + 1, d))
    traj[0] = x
    for step, i in enumerate(rng.permutation(d), start=1):
        x = x.copy()
        up_ok = x[i] + delta <= 1.0 + 1e-12
        down_ok = x[i] - delta >= -1e-12
        if up_ok and down_ok:
            x[i] += delta if rng.random() < 0.5 else -delta
        elif up_ok:
            x[i] += delta
        else:
            x[i] -= delta
        x[i] = min(max(x[i], 0.0), 1.0)
        traj[step] = x
    return traj


def _trajectory_distance(a: np.ndarray, b: np.ndarray) -> float:
    diff = a[:, None, :] - b[None, :, :]
    return float(np.sqrt((diff ** 2).sum(-1)).sum())


def morris_trajectories(
    r: int,
    d: int,
    p: int = 4,
    rng: Optional[np.random.Generator] = None,
    *,
    n_candidates: Optional[int] = None,
) -> np.ndarray:
    """Return ``r`` Morris trajectories, shape ``(r, d + 1, d)``.

    ``p`` must be even (so ``delta = p / (2 (p - 1))`` keeps every move on the
    grid). With ``n_candidates > r`` the ``r`` most spread trajectories are
    selected greedily from that many random ones (Campolongo et al. 2007).
    """
    r = _check_int(r, "r", 2)
    d = _check_int(d, "d", 1)
    p = _check_int(p, "p", 2)
    if p % 2:
        raise ValueError("p (number of grid levels) must be even")
    rng = rng if rng is not None else np.random.default_rng()
    m = r if n_candidates is None else _check_int(n_candidates, "n_candidates", r)
    pool = np.stack([_one_trajectory(d, p, rng) for _ in range(m)])
    if m == r:
        return pool
    dist = np.zeros((m, m))
    for a in range(m):
        for b in range(a + 1, m):
            dist[a, b] = dist[b, a] = _trajectory_distance(pool[a], pool[b])
    # Greedy: start from the most distant pair, then add the trajectory that
    # maximises the total spread of the chosen set.
    a, b = np.unravel_index(np.argmax(dist), dist.shape)
    chosen = [int(a), int(b)]
    while len(chosen) < r:
        rest = [k for k in range(m) if k not in chosen]
        gains = [dist[k, chosen].sum() for k in rest]
        chosen.append(rest[int(np.argmax(gains))])
    return pool[sorted(chosen)]


def morris_elementary_effects(trajectories: np.ndarray, outputs: np.ndarray) -> np.ndarray:
    """Elementary effects ``(r, d)`` from trajectories and their outputs ``(r, d + 1)``."""
    T = np.asarray(trajectories, dtype=float)
    Y = np.asarray(outputs, dtype=float)
    if T.ndim != 3 or T.shape[1] != T.shape[2] + 1:
        raise ValueError("trajectories must have shape (r, d + 1, d)")
    if Y.shape != T.shape[:2]:
        raise ValueError("outputs must have shape (r, d + 1)")
    r, _, d = T.shape
    effects = np.empty((r, d))
    for t in range(r):
        step = np.diff(T[t], axis=0)  # (d, d), one non-zero per row
        idx = np.argmax(np.abs(step), axis=1)
        move = step[np.arange(d), idx]
        if np.any(np.abs(move) < 1e-12) or len(set(idx.tolist())) != d:
            raise ValueError("each trajectory step must move exactly one new coordinate")
        effects[t, idx] = np.diff(Y[t]) / move
    return effects


def _summarise(effects: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = effects.mean(axis=0)
    mu_star = np.abs(effects).mean(axis=0)
    sigma = effects.std(axis=0, ddof=1) if effects.shape[0] > 1 else np.zeros(effects.shape[1])
    return mu, mu_star, sigma


def morris_screening(
    func: ArrayFn,
    d: int,
    r: int = 10,
    p: int = 4,
    rng: Optional[np.random.Generator] = None,
    *,
    names: Optional[Sequence[str]] = None,
    n_candidates: Optional[int] = None,
    n_bootstrap: int = 0,
    confidence: float = 0.95,
) -> MorrisResult:
    """Morris screening of ``func`` on ``[0, 1]^d`` (``r * (d + 1)`` evaluations).

    ``func`` maps an ``(m, d)`` array to ``m`` outputs. With
    ``n_bootstrap > 0`` trajectories are resampled for a percentile interval
    on ``mu_star``.
    """
    d = _check_int(d, "d", 1)
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
    T = morris_trajectories(r, d, p, rng, n_candidates=n_candidates)
    flat = T.reshape(-1, d)
    out = np.asarray(func(flat), dtype=float).reshape(-1)
    if out.shape[0] != flat.shape[0]:
        raise ValueError("func must return one value per input row")
    if not np.all(np.isfinite(out)):
        raise ValueError("func returned non-finite values")
    effects = morris_elementary_effects(T, out.reshape(T.shape[0], d + 1))
    mu, mu_star, sigma = _summarise(effects)
    ci = None
    if n_bootstrap > 0:
        boot = np.empty((n_bootstrap, d))
        for b in range(n_bootstrap):
            idx = rng.integers(0, effects.shape[0], size=effects.shape[0])
            boot[b] = np.abs(effects[idx]).mean(axis=0)
        lo, hi = 50.0 * (1.0 - confidence), 50.0 * (1.0 + confidence)
        ci = np.stack([np.percentile(boot, lo, axis=0), np.percentile(boot, hi, axis=0)], axis=1)
    return MorrisResult(
        names=names, mu=mu, mu_star=mu_star, sigma=sigma, effects=effects,
        n_evaluations=int(flat.shape[0]), mu_star_ci=ci,
        extra={"p": p, "delta": p / (2.0 * (p - 1)), "r": int(T.shape[0])},
    )


def morris_hyperparameter_screening(
    space: Dict[str, Space],
    objective: Callable[[Dict[str, Any]], float],
    r: int = 10,
    p: int = 4,
    rng: Optional[np.random.Generator] = None,
    *,
    n_candidates: Optional[int] = None,
    n_bootstrap: int = 0,
) -> MorrisResult:
    """Run Morris screening directly on ``objective(params)`` over ``space``.

    Each grid point in ``[0, 1]^d`` is mapped through ``Space.denormalize``,
    so effects are per unit of *normalised* range (log-scaled for log
    floats). Integer/categorical parameters are snapped to admissible values,
    so a move can occasionally leave the parameter unchanged and give a zero
    effect. The completed evaluations are stored in ``result.extra["trials"]``
    as ``(params, score)`` pairs.
    """
    if not space:
        raise ValueError("space must not be empty")
    names = list(space)
    trials: List[Tuple[Dict[str, Any], float]] = []

    def func(U: np.ndarray) -> np.ndarray:
        scores = []
        for row in U:
            params = {nm: space[nm].denormalize(float(u)) for nm, u in zip(names, row)}
            score = float(objective(params))
            trials.append((params, score))
            scores.append(score)
        return np.array(scores)

    result = morris_screening(
        func, len(names), r, p, rng, names=names,
        n_candidates=n_candidates, n_bootstrap=n_bootstrap,
    )
    result.extra["trials"] = trials
    return result


def format_morris_table(result: MorrisResult) -> str:
    """Markdown table sorted by ``mu_star``."""
    lines = [
        "| parameter | mu* | mu | sigma | sigma / mu* |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for i in np.argsort(-result.mu_star, kind="stable"):
        ms = result.mu_star[i]
        txt = f"{ms:.4g}"
        if result.mu_star_ci is not None:
            lo, hi = result.mu_star_ci[i]
            txt += f" [{lo:.4g}, {hi:.4g}]"
        ratio = f"{result.sigma[i] / ms:.3f}" if ms > 0 else "-"
        lines.append(f"| {result.names[i]} | {txt} | {result.mu[i]:.4g} | {result.sigma[i]:.4g} | {ratio} |")
    return "\n".join(lines)
