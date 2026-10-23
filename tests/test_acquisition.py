import math

import numpy as np
import pytest

from hyperopt_kit import run_search
from hyperopt_kit.acquisition import (
    batch_bayesian_search,
    gp_hedge_search,
    gp_ucb_beta,
    lower_confidence_bound,
    posterior_covariance,
    probability_of_improvement,
    propose_batch,
    thompson_sample,
)
from hyperopt_kit.searchers import GaussianProcess, random_search
from hyperopt_kit.spaces import FloatRange


def _space():
    return {"x": FloatRange(-2.0, 2.0), "y": FloatRange(-2.0, 2.0)}


def _sphere(p):
    return (p["x"] - 0.5) ** 2 + (p["y"] + 0.3) ** 2


def test_probability_of_improvement_closed_form():
    pi = probability_of_improvement(np.array([0.0, 1.0]), np.array([1.0, 1.0]), best=0.0, xi=0.0)
    assert pi[0] == pytest.approx(0.5)
    assert pi[1] == pytest.approx(0.5 * (1 + math.erf(-1 / math.sqrt(2))))


def test_lcb_prefers_low_mean_and_high_sigma():
    s = lower_confidence_bound(np.array([1.0, 0.0, 1.0]), np.array([0.0, 0.0, 1.0]), beta=4.0)
    assert s.tolist() == pytest.approx([-1.0, 0.0, 1.0])
    with pytest.raises(ValueError):
        lower_confidence_bound([0.0], [1.0], beta=-1)


def test_gp_ucb_beta_formula_and_growth():
    b = gp_ucb_beta(3, 2, delta=0.1)
    assert b == pytest.approx(2 * math.log(100 * 9 * math.pi ** 2 / 0.6))
    assert gp_ucb_beta(10, 2) > gp_ucb_beta(2, 2)
    with pytest.raises(ValueError):
        gp_ucb_beta(0, 2)


def test_posterior_covariance_matches_predict_and_is_symmetric():
    rng = np.random.default_rng(0)
    X = rng.random((8, 2))
    y = np.sin(3 * X[:, 0]) + X[:, 1]
    gp = GaussianProcess()
    gp.fit(X, y)
    Q = rng.random((6, 2))
    mu, cov = posterior_covariance(gp, Q)
    mu2, var2 = gp.predict(Q)
    assert np.allclose(mu, mu2)
    assert np.allclose(np.clip(np.diag(cov), 0, None), var2, atol=1e-10)
    assert np.allclose(cov, cov.T)


def test_thompson_sample_moments():
    rng = np.random.default_rng(1)
    X = rng.random((5, 1))
    gp = GaussianProcess()
    gp.fit(X, X[:, 0] ** 2)
    Q = np.array([[0.2], [0.9]])
    draws = np.array([thompson_sample(gp, Q, rng) for _ in range(4000)])
    mu, cov = posterior_covariance(gp, Q)
    assert np.allclose(draws.mean(axis=0), mu, atol=0.05 * max(1.0, np.sqrt(cov.max())))
    assert np.allclose(np.cov(draws.T), cov, atol=0.1 * max(cov.max(), 1e-3))


def test_gp_hedge_search_beats_random_and_records_history():
    hist = []
    wins = 0
    for seed in range(4):
        tr = gp_hedge_search(_space(), _sphere, 25, rng=np.random.default_rng(seed),
                             acquisitions=("ei", "pi", "lcb", "ts"), history=hist if seed == 0 else None)
        rs = random_search(_space(), _sphere, 25, rng=np.random.default_rng(seed))
        wins += min(t.score for t in tr) <= min(t.score for t in rs)
        assert len(tr) == 25
    assert wins >= 3
    assert len(hist) == 25 - 3
    assert all(abs(sum(h["probabilities"]) - 1) < 1e-9 for h in hist)
    assert {h["chosen"] for h in hist} <= {"ei", "pi", "lcb", "ts"}


def test_gp_hedge_validation():
    with pytest.raises(ValueError):
        gp_hedge_search(_space(), _sphere, 5, acquisitions=("nope",))
    with pytest.raises(ValueError):
        gp_hedge_search(_space(), _sphere, 5, eta=0)


@pytest.mark.parametrize("lie", ["kriging_believer", "cl_min", "cl_mean", "cl_max"])
def test_propose_batch_spreads_points(lie):
    rng = np.random.default_rng(3)
    X = rng.random((6, 2))
    y = ((X - 0.5) ** 2).sum(axis=1)
    B = propose_batch(X, y, 4, rng, lie=lie)
    assert B.shape == (4, 2)
    d = np.sqrt(((B[:, None] - B[None]) ** 2).sum(-1))[np.triu_indices(4, 1)]
    assert d.min() > 1e-6  # hallucination keeps batch points distinct


def test_batch_bayesian_search_budget_and_quality():
    tr = batch_bayesian_search(_space(), _sphere, 23, rng=np.random.default_rng(5), batch_size=5)
    assert len(tr) == 23
    assert [t.iteration for t in tr] == list(range(23))
    assert min(t.score for t in tr) < 0.15
    with pytest.raises(ValueError):
        batch_bayesian_search(_space(), _sphere, 5, lie="bad")
    with pytest.raises(ValueError):
        batch_bayesian_search(_space(), _sphere, 5, batch_size=0)


def test_strategies_registered_and_stopper():
    r = run_search("gp_hedge", _space(), _sphere, 12, seed=0, eta=2.0)
    assert len(r.trials) == 12
    r = run_search("batch_bayesian", _space(), _sphere, 12, seed=0, batch_size=3, lie="cl_max")
    assert len(r.trials) == 12
    r = run_search("batch_bayesian", _space(), _sphere, 50, seed=0, floor=10.0)
    assert len(r.trials) == 1
