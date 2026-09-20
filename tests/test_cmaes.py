import numpy as np
import pytest

from hyperopt_kit.evaluate import compare_strategies, run_search
from hyperopt_kit.searchers import (
    cmaes_parameters,
    cmaes_search,
    cmaes_weights,
    random_search,
)
from hyperopt_kit.spaces import parse_space


def _space():
    return parse_space({"x": [0.0, 1.0], "y": [0.0, 1.0]})


def _obj(params):
    return float(params["x"]) + float(params.get("y", 0.0))


def _mixed_space():
    return parse_space(
        {
            "x": [0.0, 1.0],
            "y": [0, 10, "int"],
            "z": ["a", "b", "c"],
        }
    )


def test_cmaes_weights_sum_to_one_and_decrease():
    w = cmaes_weights(5)
    assert w.shape == (5,)
    assert w.sum() == pytest.approx(1.0)
    assert np.all(w > 0.0)
    assert np.all(np.diff(w) < 0.0)


def test_cmaes_weights_single_parent():
    w = cmaes_weights(1)
    assert w == pytest.approx([1.0])


def test_cmaes_weights_rejects_bad_n():
    with pytest.raises(ValueError):
        cmaes_weights(0)
    with pytest.raises(ValueError):
        cmaes_weights(1.5)


def test_cmaes_parameters_hansen_defaults():
    p = cmaes_parameters(2)
    assert p.population_size == 4 + int(np.floor(3.0 * np.log(2)))
    assert p.n_parents == p.population_size // 2
    assert p.weights.shape == (p.n_parents,)
    assert p.weights.sum() == pytest.approx(1.0)
    assert p.mu_eff == pytest.approx(1.0 / float(np.sum(p.weights**2)))
    for rate in (p.c_sigma, p.c_c, p.c_1, p.c_mu):
        assert 0.0 < rate <= 1.0
    assert p.d_sigma > 0.0


def test_cmaes_parameters_custom_population():
    p = cmaes_parameters(3, population_size=10)
    assert p.population_size == 10
    assert p.n_parents == 5


def test_cmaes_parameters_rejects_bad_dimension_and_population():
    with pytest.raises(ValueError):
        cmaes_parameters(0)
    with pytest.raises(ValueError):
        cmaes_parameters(2, population_size=1)


def test_cmaes_search_respects_budget():
    trials = cmaes_search(_space(), _obj, budget=18, rng=np.random.default_rng(0))
    assert len(trials) == 18
    for i, trial in enumerate(trials):
        assert trial.iteration == i
        assert isinstance(trial.score, float)
        assert set(trial.params) == {"x", "y"}
        assert trial.resource is None
        assert 0.0 <= trial.params["x"] <= 1.0
        assert 0.0 <= trial.params["y"] <= 1.0


def test_cmaes_search_deterministic_with_seed():
    a = cmaes_search(_space(), _obj, 16, rng=np.random.default_rng(11))
    b = cmaes_search(_space(), _obj, 16, rng=np.random.default_rng(11))
    assert [t.params for t in a] == [t.params for t in b]
    assert [t.score for t in a] == [t.score for t in b]


def test_cmaes_search_differs_across_seeds():
    a = cmaes_search(_space(), _obj, 16, rng=np.random.default_rng(11))
    b = cmaes_search(_space(), _obj, 16, rng=np.random.default_rng(99))
    assert [t.params for t in a] != [t.params for t in b]


def test_cmaes_search_budget_smaller_than_population():
    trials = cmaes_search(
        _space(), _obj, budget=3, rng=np.random.default_rng(1), population_size=10
    )
    assert len(trials) == 3


def test_cmaes_search_rejects_bad_budget_and_space():
    with pytest.raises(ValueError):
        cmaes_search(_space(), _obj, 0, rng=np.random.default_rng(0))
    with pytest.raises(ValueError):
        cmaes_search({}, _obj, 5, rng=np.random.default_rng(0))
    with pytest.raises(ValueError):
        cmaes_search(_space(), _obj, 5, sigma0=0.0, rng=np.random.default_rng(0))
    with pytest.raises(ValueError):
        cmaes_search(_space(), _obj, 5, population_size=1, rng=np.random.default_rng(0))


def test_cmaes_search_finds_convex_minimum():
    space = parse_space({"x": [0.0, 1.0]})
    objective = lambda p: (p["x"] - 0.5) ** 2
    trials = cmaes_search(
        space, objective, 16, rng=np.random.default_rng(7), population_size=4
    )
    best = min(trials, key=lambda t: t.score)
    assert best.score < 0.02
    assert best.params["x"] == pytest.approx(0.5, abs=0.15)
    later = np.array([t.params["x"] for t in trials[4:]])
    assert np.mean(np.abs(later - 0.5)) < 0.20


def test_cmaes_search_beats_random_on_demo_like_objective():
    space = parse_space({"a": [0.1, 5.0], "b": [0.1, 5.0]})
    objective = (
        lambda p: (p["a"] * p["b"] - 2.0) ** 2 + (p["a"] - 1.0) ** 2 + (p["b"] - 2.0) ** 2
    )
    cma = cmaes_search(space, objective, 24, rng=np.random.default_rng(3))
    rnd = random_search(space, objective, 24, rng=np.random.default_rng(3))
    assert min(t.score for t in cma) < min(t.score for t in rnd)


def test_cmaes_search_int_range_values():
    space = parse_space({"n": [0, 20, "int"]})
    objective = lambda p: (p["n"] - 7) ** 2
    trials = cmaes_search(space, objective, 16, rng=np.random.default_rng(4))
    assert all(isinstance(t.params["n"], int) for t in trials)
    assert all(0 <= t.params["n"] <= 20 for t in trials)
    assert min(t.score for t in trials) <= 4.0


def test_cmaes_search_mixed_space():
    trials = cmaes_search(_mixed_space(), _obj, budget=14, rng=np.random.default_rng(3))
    assert all(t.params["z"] in ("a", "b", "c") for t in trials)
    assert all(isinstance(t.params["y"], int) for t in trials)
    assert all(0.0 <= t.params["x"] <= 1.0 for t in trials)


def test_cmaes_search_log_scale_space():
    space = parse_space({"lr": [1e-3, 1.0, "log"]})
    objective = lambda p: (np.log10(p["lr"]) + 1.0) ** 2
    trials = cmaes_search(space, objective, 16, rng=np.random.default_rng(4))
    assert all(1e-3 <= t.params["lr"] <= 1.0 for t in trials)
    assert min(t.score for t in trials) < 0.05


def test_cmaes_search_early_stop_on_floor():
    space = parse_space({"x": [0.0, 1.0]})
    objective = lambda p: (p["x"] - 0.5) ** 2

    def stopper(best, n):
        return best <= 0.01

    trials = cmaes_search(
        space, objective, 20, rng=np.random.default_rng(2), stop_when=stopper
    )
    assert len(trials) < 20
    assert min(t.score for t in trials) <= 0.01


def test_cmaes_search_with_none_rng():
    trials = cmaes_search(parse_space({"x": [0.0, 1.0]}), _obj, 8)
    assert len(trials) == 8


def test_cmaes_search_concentration_around_1d_optimum():
    space = parse_space({"x": [0.0, 1.0]})
    objective = lambda p: (p["x"] - 0.25) ** 2
    trials = cmaes_search(
        space, objective, 20, rng=np.random.default_rng(5), population_size=4
    )
    later = np.array([t.params["x"] for t in trials[8:]])
    assert np.mean(np.abs(later - 0.25)) < 0.20
    assert min(t.score for t in trials) < 0.01


def test_run_search_cmaes():
    result = run_search("cmaes", _space(), _obj, budget=15, seed=3)
    assert result.strategy == "cmaes"
    assert len(result.trials) == 15
    assert result.best_score == min(t.score for t in result.trials)


def test_run_search_forwarded_cmaes_kwargs():
    a = run_search("cmaes", _space(), _obj, 18, seed=1, sigma0=0.1, population_size=4)
    b = run_search("cmaes", _space(), _obj, 18, seed=1, sigma0=0.8, population_size=4)
    assert len(a.trials) == 18
    assert [t.params for t in a.trials] != [t.params for t in b.trials]


def test_compare_strategies_includes_cmaes():
    results = compare_strategies(_space(), _obj, budget=10, seed=7)
    assert set(results) >= {"grid", "random", "bayesian", "tpe", "cmaes", "hyperband"}
    assert len(results["cmaes"].trials) == 10


def test_compare_strategies_cmaes_subset():
    results = compare_strategies(
        _space(), _obj, 8, strategies=("random", "cmaes"), seed=2
    )
    assert set(results) == {"random", "cmaes"}
    for result in results.values():
        assert len(result.trials) == 8
