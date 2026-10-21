"""Built-in objective functions for the demo and the CLI."""

from __future__ import annotations

import math
from typing import Callable, Dict

import numpy as np


def make_noisy_objective(base: Callable[[Dict], float], noise: float = 0.05, seed: int = 7):
    """Wrap a deterministic base objective with additive gaussian noise.

    The noise stream is owned by the returned closure, so each objective
    instance reproduces the same sequence of observations when evaluations
    happen in the same order.
    """
    rng = np.random.default_rng(seed)

    def objective(params, resource: float = 1.0):
        sigma = float(noise) / math.sqrt(max(float(resource), 1e-12))
        return float(base(params)) + float(rng.normal(0.0, sigma))

    return objective


def demo_objective(params: Dict, resource: float = 1.0) -> float:
    """Noisy toy objective with an interaction term.

    The noiseless part ``(a*b - 2)**2 + (a - 1)**2 + (b - 2)**2`` has a known
    global minimum of 0 at ``(a, b) = (1, 2)``. A gaussian pseudo-noise term
    (sigma = 0.05 at full fidelity) is derived deterministically from the
    config, so the surface is bumpy but every evaluation is reproducible
    across runs.

    ``resource`` is a normalized fidelity in ``(0, 1]``. Lower values inflate
    the noise (a cheap, less accurate estimate); ``resource=1.0`` matches the
    original single-fidelity surface and is what grid / random / GP use.
    """
    a = float(params["a"])
    b = float(params["b"])
    base = (a * b - 2.0) ** 2 + (a - 1.0) ** 2 + (b - 2.0) ** 2
    seed = (int(abs(a) * 1e6) * 31 + int(abs(b) * 1e6)) % (2**31 - 1)
    resource = max(float(resource), 1e-12)
    if abs(resource - 1.0) > 1e-12:
        seed = (seed + int(round(resource * 1e6))) % (2**31 - 1)
    sigma = 0.05 / math.sqrt(resource)
    noise = float(np.random.default_rng(seed).normal(0.0, sigma))
    return base + noise


def demo_multi_objective(params: Dict, resource: float = 1.0):
    """Two competing objectives for the ``pareto`` CLI demo (minimise both).

    The first is the noise-free surface of :func:`demo_objective`, which is 0
    at ``(a, b) = (1, 2)``. The second is a "cost" ``a + b`` that pulls
    toward small values. Each objective has its own optimum, so the
    trade-off between them forms a 1-D Pareto front.
    """
    a = float(params["a"])
    b = float(params["b"])
    loss = (a * b - 2.0) ** 2 + (a - 1.0) ** 2 + (b - 2.0) ** 2
    return (loss, a + b)


def zdt1(params: Dict):
    """ZDT1 benchmark (Zitzler, Deb & Thiele, 2000) on parameters in [0, 1].

    The parameters are taken in sorted-name order. The Pareto front is
    ``f2 = 1 - sqrt(f1)`` for ``f1`` in [0, 1], reached when every parameter
    except the first is 0.
    """
    x = np.array([float(params[k]) for k in sorted(params)])
    if x.size < 2:
        raise ValueError("zdt1 needs at least 2 parameters")
    f1 = x[0]
    g = 1.0 + 9.0 * x[1:].sum() / (x.size - 1)
    return (float(f1), float(g * (1.0 - math.sqrt(f1 / g))))
