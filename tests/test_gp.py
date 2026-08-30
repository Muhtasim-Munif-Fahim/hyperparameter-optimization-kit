import numpy as np
import pytest

from hyperopt_kit.searchers import (
    GaussianProcess,
    bayesian_search,
    expected_improvement,
    random_search,
)
from hyperopt_kit.spaces import Categorical, FloatRange, parse_space


def test_gp_predict_shape():
    rng = np.random.default_rng(0)
    X = rng.random((10, 2))
    y = np.sin(X[:, 0]) + X[:, 1]
    gp = GaussianProcess()
    gp.fit(X, y)
    mu, var = gp.predict(X[:5])
    assert mu.shape == (5,)
    assert var.shape == (5,)


def test_gp_interpolates_training_points():
    rng = np.random.default_rng(1)
    X = rng.random((8, 1))
    y = 2.0 * X[:, 0] + 1.0
    gp = GaussianProcess()
    gp.fit(X, y)
    mu, _ = gp.predict(X)
    assert np.allclose(mu, y, atol=1e-3)


def test_gp_predictive_variance_is_non_negative():
    rng = np.random.default_rng(2)
    X = rng.random((6, 2))
    y = np.sum(X, axis=1)
    gp = GaussianProcess()
    gp.fit(X, y)
    mu, var = gp.predict(rng.random((20, 2)))
    assert np.all(var >= 0.0)
    assert np.all(np.isfinite(mu))
    assert np.all(np.isfinite(var))


def test_gp_handles_single_point():
    gp = GaussianProcess()
    gp.fit(np.array([[0.3, 0.7]]), np.array([1.5]))
    mu, var = gp.predict(np.array([[0.1, 0.9], [0.5, 0.5]]))
    assert mu.shape == (2,)
    assert np.all(np.isfinite(mu))
    assert np.all(var >= 0.0)


def test_gp_handles_identical_points():
    X = np.ones((5, 2))
    y = np.ones(5)
    gp = GaussianProcess()
    gp.fit(X, y)
    mu, var = gp.predict(np.ones((3, 2)))
    assert np.all(np.isfinite(mu))
    assert np.all(var >= 0.0)


def test_gp_accepts_1d_input():
    gp = GaussianProcess()
    gp.fit(np.array([0.0, 0.5, 1.0]), np.array([0.0, 0.25, 1.0]))
    mu, _ = gp.predict(np.array([[0.25]]))
    assert mu.shape == (1,)


def test_gp_rejects_empty_fit():
    gp = GaussianProcess()
    with pytest.raises(ValueError):
        gp.fit(np.zeros((0, 2)), np.zeros(0))


def test_gp_rejects_mismatched_shapes():
    gp = GaussianProcess()
    with pytest.raises(ValueError):
        gp.fit(np.zeros((3, 2)), np.zeros(4))


def test_gp_predict_before_fit_raises():
    gp = GaussianProcess()
    with pytest.raises(RuntimeError):
        gp.predict(np.zeros((2, 2)))


def test_gp_fit_hyperparams_picks_positive_length_scale():
    rng = np.random.default_rng(3)
    X = rng.random((20, 1))
    y = np.cos(3.0 * X[:, 0])
    gp = GaussianProcess()
    gp.fit(X, y)
    assert 0.0 < gp.length_scale <= max(gp.length_scale_candidates)


def test_expected_improvement_prefers_lower_mean():
    mu = np.array([0.5, 0.0, 1.0])
    sigma = np.ones(3) * 0.2
    ei = expected_improvement(mu, sigma, best=0.0, xi=0.01)
    assert ei[1] > ei[0]
    assert ei[1] > ei[2]


def test_expected_improvement_explores_high_variance():
    mu = np.array([0.5, 0.5])
    sigma = np.array([0.1, 0.5])
    ei = expected_improvement(mu, sigma, best=0.0, xi=0.01)
    assert ei[1] > ei[0]


def test_expected_improvement_zero_at_best_point():
    ei = expected_improvement(np.array([0.0]), np.array([0.0]), best=0.0, xi=0.0)
    assert ei[0] < 1e-6


def test_xi_weights_exploration_towards_high_variance_points():
    exploit = np.array([0.0]), np.array([0.3])
    explore = np.array([0.4]), np.array([1.0])

    def ratio(xi):
        ei_exploit = expected_improvement(exploit[0], exploit[1], best=0.0, xi=xi)[0]
        ei_explore = expected_improvement(explore[0], explore[1], best=0.0, xi=xi)[0]
        return ei_explore / ei_exploit

    assert ratio(0.5) > ratio(0.01)
    assert ratio(0.5) > 2.0


def test_bayesian_search_respects_budget():
    space = parse_space({"x": [0.0, 1.0], "y": [0.0, 1.0]})
    trials = bayesian_search(
        space, lambda p: p["x"] * p["y"], budget=15, rng=np.random.default_rng(0)
    )
    assert len(trials) == 15
    assert trials[0].iteration == 0
    assert trials[-1].iteration == 14


def test_bayesian_search_deterministic_with_seed():
    space = parse_space({"x": [0.0, 1.0]})
    objective = lambda p: (p["x"] - 0.5) ** 2
    a = bayesian_search(space, objective, 12, rng=np.random.default_rng(7))
    b = bayesian_search(space, objective, 12, rng=np.random.default_rng(7))
    assert [t.params for t in a] == [t.params for t in b]
    assert [t.score for t in a] == [t.score for t in b]


def test_bayesian_search_budget_smaller_than_initial_points():
    space = parse_space({"x": [0.0, 1.0], "y": [0.0, 1.0], "z": [0.0, 1.0]})
    trials = bayesian_search(
        space, lambda p: sum(p.values()), budget=2, rng=np.random.default_rng(1)
    )
    assert len(trials) == 2


def test_bayesian_search_finds_convex_minimum():
    space = parse_space({"x": [0.0, 1.0]})
    objective = lambda p: (p["x"] - 0.5) ** 2
    rng = np.random.default_rng(7)
    trials = bayesian_search(space, objective, 12, rng=rng)
    best = min(t.score for t in trials)
    assert best < 0.02
    random_best = min(
        t.score
        for t in random_search(space, objective, 12, rng=np.random.default_rng(7))
    )
    assert best < random_best


def test_bayesian_search_beats_random_on_demo_like_objective():
    space = parse_space({"a": [0.1, 5.0], "b": [0.1, 5.0]})
    objective = lambda p: (p["a"] * p["b"] - 2.0) ** 2 + (p["a"] - 1.0) ** 2 + (p["b"] - 2.0) ** 2
    bayes = bayesian_search(space, objective, 20, rng=np.random.default_rng(3))
    rnd = random_search(space, objective, 20, rng=np.random.default_rng(3))
    assert min(t.score for t in bayes) < min(t.score for t in rnd)


def test_bayesian_search_on_categorical_only_space():
    space = parse_space({"x": ["a", "b", "c"]})
    objective = lambda p: {"a": 0.0, "b": 1.0, "c": 4.0}[p["x"]]
    trials = bayesian_search(space, objective, 6, rng=np.random.default_rng(5))
    best_params = min(trials, key=lambda t: t.score).params
    assert best_params["x"] == "a"


def test_bayesian_search_rejects_bad_budget():
    with pytest.raises(ValueError):
        bayesian_search(parse_space({"x": [0.0, 1.0]}), lambda p: 0.0, 0)


def test_bayesian_search_empty_space_rejected():
    with pytest.raises(ValueError):
        bayesian_search({}, lambda p: 0.0, 5)


def test_bayesian_search_early_stop_on_floor():
    space = parse_space({"x": [0.0, 1.0]})
    objective = lambda p: (p["x"] - 0.5) ** 2

    def stopper(best, n):
        return best <= 0.01

    trials = bayesian_search(space, objective, 20, rng=np.random.default_rng(2), stop_when=stopper)
    assert len(trials) < 20
    assert min(t.score for t in trials) <= 0.01


def test_bayesian_search_categorical_in_mixed_space():
    space = parse_space({"x": [0.0, 1.0], "mode": ["fast", "slow"]})
    objective = lambda p: (p["x"] - 0.5) ** 2 + (0.0 if p["mode"] == "fast" else 0.5)
    trials = bayesian_search(space, objective, 15, rng=np.random.default_rng(9))
    assert all(t.params["mode"] in ("fast", "slow") for t in trials)
    assert all(0.0 <= t.params["x"] <= 1.0 for t in trials)
