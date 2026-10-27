import json
import math

import numpy as np
import pytest

from hyperopt_kit import (
    FloatRange,
    sobol_indices,
    saltelli_matrices,
    surrogate_sobol_importance,
    format_sobol_table,
)
from hyperopt_kit.cli import main
from hyperopt_kit.searchers import Trial
from hyperopt_kit.spaces import Categorical, IntRange


def ishigami(X, a=7.0, b=0.1):
    Z = -math.pi + 2.0 * math.pi * X
    return np.sin(Z[:, 0]) + a * np.sin(Z[:, 1]) ** 2 + b * Z[:, 2] ** 4 * np.sin(Z[:, 0])


def ishigami_truth(a=7.0, b=0.1):
    pi = math.pi
    V = a**2 / 8 + b * pi**4 / 5 + b**2 * pi**8 / 18 + 0.5
    V1 = 0.5 * (1 + b * pi**4 / 5) ** 2
    V2 = a**2 / 8
    V13 = b**2 * pi**8 * (1 / 18 - 1 / 50)
    s1 = np.array([V1, V2, 0.0]) / V
    st = np.array([V1 + V13, V2, V13]) / V
    return s1, st


def test_saltelli_matrices_structure():
    A, B, AB = saltelli_matrices(16, 3, np.random.default_rng(0))
    assert A.shape == B.shape == (16, 3) and AB.shape == (3, 16, 3)
    for i in range(3):
        np.testing.assert_array_equal(AB[i][:, i], B[:, i])
        others = [j for j in range(3) if j != i]
        np.testing.assert_array_equal(AB[i][:, others], A[:, others])
    with pytest.raises(ValueError):
        saltelli_matrices(1, 3)


def test_ishigami_matches_analytic_indices():
    s1_true, st_true = ishigami_truth()
    res = sobol_indices(ishigami, 3, n=20000, rng=np.random.default_rng(1))
    np.testing.assert_allclose(res.first_order, s1_true, atol=0.04)
    np.testing.assert_allclose(res.total_order, st_true, atol=0.04)
    assert res.n_evaluations == 20000 * 5
    assert [n for n, _ in res.ranking("total_order")][0] == "x0"


def test_additive_linear_function_has_no_interactions():
    w = np.array([1.0, 2.0, 0.0])
    res = sobol_indices(lambda X: X @ w, 3, n=8000, rng=np.random.default_rng(2))
    truth = w**2 / np.sum(w**2)
    np.testing.assert_allclose(res.first_order, truth, atol=0.03)
    np.testing.assert_allclose(res.total_order, truth, atol=0.03)
    assert abs(res.total_order[2]) < 1e-12


def test_bootstrap_ci_covers_truth_and_constant_function():
    s1_true, st_true = ishigami_truth()
    res = sobol_indices(ishigami, 3, n=4000, rng=np.random.default_rng(3), n_bootstrap=200)
    assert res.first_order_ci.shape == (3, 2)
    assert np.all(res.total_order_ci[:, 0] - 0.02 <= st_true)
    assert np.all(st_true <= res.total_order_ci[:, 1] + 0.02)
    const = sobol_indices(lambda X: np.ones(len(X)), 2, n=64, rng=np.random.default_rng(0))
    assert const.variance == 0.0 and np.all(const.first_order == 0)


def test_input_validation():
    with pytest.raises(ValueError):
        sobol_indices(lambda X: X[:, 0], 2, n=10, names=["a"])
    with pytest.raises(ValueError):
        sobol_indices(lambda X: X[:1, 0], 2, n=10)
    with pytest.raises(ValueError):
        sobol_indices(lambda X: np.full(len(X), np.nan), 2, n=10)
    with pytest.raises(ValueError):
        sobol_indices(lambda X: X[:, 0], 2, n=10, confidence=1.5)


def _trials(space, f, n, seed=0, resource=None):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        p = {k: sp.sample(rng) for k, sp in space.items()}
        out.append(Trial(i, p, f(p), resource))
    return out


def test_surrogate_identifies_dominant_and_irrelevant_parameters():
    space = {"lr": FloatRange(0.0, 1.0), "depth": IntRange(1, 10), "noise": FloatRange(0.0, 1.0)}
    f = lambda p: 5.0 * (p["lr"] - 0.3) ** 2 + 0.05 * p["depth"] / 10
    trials = _trials(space, f, 60)
    res = surrogate_sobol_importance(trials, space, n=2048, rng=np.random.default_rng(0))
    order = [n for n, _ in res.ranking()]
    assert order[0] == "lr"
    assert res.total_order[0] > 0.9
    assert res.total_order[2] < 0.05
    assert res.extra["n_trials"] == 60
    table = format_sobol_table(res)
    assert table.splitlines()[2].startswith("| lr |")


def test_surrogate_uses_top_fidelity_and_categoricals():
    space = {"opt": Categorical(["sgd", "adam"]), "x": FloatRange(0.0, 1.0)}
    f = lambda p: (3.0 if p["opt"] == "sgd" else 0.0) + 0.1 * p["x"]
    full = _trials(space, f, 30, seed=1, resource=1.0)
    low = _trials(space, lambda p: 100.0 * p["x"], 30, seed=2, resource=0.1)
    res = surrogate_sobol_importance(full + low, space, n=1024, rng=np.random.default_rng(0))
    assert res.extra["n_trials"] == 30
    assert res.ranking()[0][0] == "opt"
    with pytest.raises(ValueError):
        surrogate_sobol_importance(full[:2], space)
    with pytest.raises(ValueError):
        surrogate_sobol_importance([Trial(0, {"opt": "sgd"}, 1.0)] * 3, space)


def test_cli_sensitivity(capsys):
    spec = json.dumps({"a": {"type": "float", "low": -2, "high": 4}, "b": {"type": "float", "low": -2, "high": 4}})
    assert main(["sensitivity", "--space", spec, "--budget", "25", "--seed", "0", "--n-samples", "256", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["trials"] == 25 and set(data["indices"]) == {"a", "b"}
    assert main(["sensitivity", "--space", spec, "--budget", "25", "--seed", "0", "--n-samples", "128", "--bootstrap", "20"]) == 0
    out = capsys.readouterr().out
    assert "| parameter | S1 (first order)" in out and "[" in out
