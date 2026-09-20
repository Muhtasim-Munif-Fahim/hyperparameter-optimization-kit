import numpy as np
import pytest

from hyperopt_kit.evaluate import compare_strategies, run_search
from hyperopt_kit.searchers import (
    Trial,
    adaptive_parzen,
    categorical_probs,
    gmm_logpdf,
    gmm_sample,
    random_search,
    tpe_log_density_ratio,
    tpe_search,
    tpe_split,
)
from hyperopt_kit.spaces import Categorical, parse_space


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


def test_tpe_split_gamma_quantile():
    scores = np.array([4.0, 1.0, 3.0, 2.0, 5.0, 0.0, 6.0, 7.0])
    below, above = tpe_split(scores, gamma=0.25)
    assert len(below) == 2
    assert len(above) == 6
    assert set(below) | set(above) == set(range(8))
    assert set(below).isdisjoint(set(above))
    assert set(scores[below]) == {0.0, 1.0}


def test_tpe_split_keeps_at_least_one_on_each_side():
    scores = np.array([1.0, 2.0])
    below, above = tpe_split(scores, gamma=0.25)
    assert len(below) == 1
    assert len(above) == 1
    below, above = tpe_split(scores, gamma=0.9)
    assert len(below) == 1
    assert len(above) == 1


def test_tpe_split_single_observation():
    below, above = tpe_split(np.array([3.0]), gamma=0.25)
    assert list(below) == [0]
    assert list(above) == []


def test_tpe_split_rejects_bad_gamma_and_empty():
    with pytest.raises(ValueError):
        tpe_split(np.array([1.0, 2.0]), gamma=0.0)
    with pytest.raises(ValueError):
        tpe_split(np.array([1.0, 2.0]), gamma=1.0)
    with pytest.raises(ValueError):
        tpe_split(np.array([]), gamma=0.25)


def test_adaptive_parzen_weights_sum_to_one():
    w, mu, sig = adaptive_parzen(np.array([0.2, 0.3, 0.8]))
    assert w.shape == mu.shape == sig.shape
    assert w.shape[0] == 4
    assert w.sum() == pytest.approx(1.0)
    assert np.all(sig > 0.0)
    assert np.all(np.isfinite(w))


def test_adaptive_parzen_empty_is_prior_only():
    w, mu, sig = adaptive_parzen(np.array([]), prior_mu=0.4, prior_sigma=0.2)
    assert w == pytest.approx([1.0])
    assert mu == pytest.approx([0.4])
    assert sig == pytest.approx([0.2])


def test_adaptive_parzen_identical_points_have_positive_bandwidth():
    w, mu, sig = adaptive_parzen(np.array([0.5, 0.5, 0.5]))
    assert np.all(sig > 0.0)
    assert w.sum() == pytest.approx(1.0)


def test_gmm_logpdf_higher_near_observations():
    w, mu, sig = adaptive_parzen(np.array([0.2, 0.22, 0.18]))
    near = float(gmm_logpdf(np.array([0.2]), w, mu, sig)[0])
    far = float(gmm_logpdf(np.array([0.9]), w, mu, sig)[0])
    assert near > far
    assert np.isfinite(near) and np.isfinite(far)


def test_gmm_logpdf_outside_bounds_is_neg_inf():
    w, mu, sig = adaptive_parzen(np.array([0.5]))
    out = gmm_logpdf(np.array([-0.1, 1.1]), w, mu, sig)
    assert np.all(np.isneginf(out))


def test_gmm_sample_stays_in_unit_interval():
    rng = np.random.default_rng(0)
    w, mu, sig = adaptive_parzen(np.array([0.1, 0.2, 0.9]))
    xs = gmm_sample(rng, 200, w, mu, sig)
    assert xs.shape == (200,)
    assert np.all((xs >= 0.0) & (xs <= 1.0))


def test_categorical_probs_dirichlet_smoothed():
    sp = Categorical(["a", "b", "c"])
    probs = categorical_probs(sp, ["a", "a", "b"], prior_weight=1.0)
    assert probs.shape == (3,)
    assert probs.sum() == pytest.approx(1.0)
    assert probs[0] > probs[1] > probs[2]
    assert probs[2] > 0.0


def test_tpe_log_density_ratio_prefers_good_region():
    space = parse_space({"x": [0.0, 1.0]})
    trials = [
        Trial(0, {"x": 0.15}, 0.10),
        Trial(1, {"x": 0.20}, 0.12),
        Trial(2, {"x": 0.25}, 0.11),
        Trial(3, {"x": 0.75}, 1.00),
        Trial(4, {"x": 0.80}, 1.10),
        Trial(5, {"x": 0.85}, 0.95),
        Trial(6, {"x": 0.90}, 1.05),
        Trial(7, {"x": 0.70}, 0.90),
    ]
    good = tpe_log_density_ratio({"x": 0.20}, space, trials, gamma=0.25)
    bad = tpe_log_density_ratio({"x": 0.80}, space, trials, gamma=0.25)
    assert good > bad
    assert np.isfinite(good) and np.isfinite(bad)


def test_tpe_log_density_ratio_on_categorical():
    space = parse_space({"k": ["good", "bad", "other"]})
    trials = [
        Trial(0, {"k": "good"}, 0.1),
        Trial(1, {"k": "good"}, 0.2),
        Trial(2, {"k": "bad"}, 1.0),
        Trial(3, {"k": "bad"}, 1.1),
        Trial(4, {"k": "other"}, 0.9),
        Trial(5, {"k": "other"}, 0.8),
    ]
    good = tpe_log_density_ratio({"k": "good"}, space, trials, gamma=0.25)
    bad = tpe_log_density_ratio({"k": "bad"}, space, trials, gamma=0.25)
    assert good > bad


def test_tpe_log_density_ratio_needs_two_trials():
    space = parse_space({"x": [0.0, 1.0]})
    with pytest.raises(ValueError):
        tpe_log_density_ratio({"x": 0.5}, space, [Trial(0, {"x": 0.1}, 1.0)])


def test_tpe_search_respects_budget():
    trials = tpe_search(_space(), _obj, budget=18, rng=np.random.default_rng(0))
    assert len(trials) == 18
    for i, trial in enumerate(trials):
        assert trial.iteration == i
        assert isinstance(trial.score, float)
        assert set(trial.params) == {"x", "y"}
        assert trial.resource is None


def test_tpe_search_deterministic_with_seed():
    a = tpe_search(_space(), _obj, 16, rng=np.random.default_rng(11))
    b = tpe_search(_space(), _obj, 16, rng=np.random.default_rng(11))
    assert [t.params for t in a] == [t.params for t in b]
    assert [t.score for t in a] == [t.score for t in b]


def test_tpe_search_differs_across_seeds():
    a = tpe_search(_space(), _obj, 16, rng=np.random.default_rng(11))
    b = tpe_search(_space(), _obj, 16, rng=np.random.default_rng(99))
    assert [t.params for t in a] != [t.params for t in b]


def test_tpe_search_budget_smaller_than_initial_points():
    trials = tpe_search(
        _space(), _obj, budget=2, rng=np.random.default_rng(1), n_initial=10
    )
    assert len(trials) == 2


def test_tpe_search_rejects_bad_budget_and_space():
    with pytest.raises(ValueError):
        tpe_search(_space(), _obj, 0, rng=np.random.default_rng(0))
    with pytest.raises(ValueError):
        tpe_search({}, _obj, 5, rng=np.random.default_rng(0))
    with pytest.raises(ValueError):
        tpe_search(_space(), _obj, 5, gamma=0.0, rng=np.random.default_rng(0))
    with pytest.raises(ValueError):
        tpe_search(_space(), _obj, 5, n_candidates=0, rng=np.random.default_rng(0))
    with pytest.raises(ValueError):
        tpe_search(_space(), _obj, 5, prior_weight=0.0, rng=np.random.default_rng(0))


def test_tpe_search_finds_convex_minimum():
    space = parse_space({"x": [0.0, 1.0]})
    objective = lambda p: (p["x"] - 0.5) ** 2
    trials = tpe_search(
        space, objective, 16, rng=np.random.default_rng(7), n_initial=5
    )
    best = min(trials, key=lambda t: t.score)
    assert best.score < 0.02
    assert best.params["x"] == pytest.approx(0.5, abs=0.15)
    later = np.array([t.params["x"] for t in trials[5:]])
    assert np.mean(np.abs(later - 0.5)) < 0.25


def test_tpe_search_beats_random_on_demo_like_objective():
    space = parse_space({"a": [0.1, 5.0], "b": [0.1, 5.0]})
    objective = (
        lambda p: (p["a"] * p["b"] - 2.0) ** 2 + (p["a"] - 1.0) ** 2 + (p["b"] - 2.0) ** 2
    )
    tpe = tpe_search(space, objective, 24, rng=np.random.default_rng(3))
    rnd = random_search(space, objective, 24, rng=np.random.default_rng(3))
    assert min(t.score for t in tpe) < min(t.score for t in rnd)


def test_tpe_search_on_categorical_only_space():
    space = parse_space({"x": ["a", "b", "c"]})
    objective = lambda p: {"a": 0.0, "b": 1.0, "c": 4.0}[p["x"]]
    trials = tpe_search(space, objective, 10, rng=np.random.default_rng(5))
    best_params = min(trials, key=lambda t: t.score).params
    assert best_params["x"] == "a"
    assert all(t.params["x"] in ("a", "b", "c") for t in trials)


def test_tpe_search_mixed_space():
    trials = tpe_search(_mixed_space(), _obj, budget=14, rng=np.random.default_rng(3))
    assert all(t.params["z"] in ("a", "b", "c") for t in trials)
    assert all(isinstance(t.params["y"], int) for t in trials)
    assert all(0.0 <= t.params["x"] <= 1.0 for t in trials)


def test_tpe_search_log_scale_space():
    space = parse_space({"lr": [1e-3, 1.0, "log"]})
    objective = lambda p: (np.log10(p["lr"]) + 1.0) ** 2
    trials = tpe_search(space, objective, 14, rng=np.random.default_rng(4))
    assert all(1e-3 <= t.params["lr"] <= 1.0 for t in trials)
    assert min(t.score for t in trials) < 0.05


def test_tpe_search_early_stop_on_floor():
    space = parse_space({"x": [0.0, 1.0]})
    objective = lambda p: (p["x"] - 0.5) ** 2

    def stopper(best, n):
        return best <= 0.01

    trials = tpe_search(
        space, objective, 20, rng=np.random.default_rng(2), stop_when=stopper
    )
    assert len(trials) < 20
    assert min(t.score for t in trials) <= 0.01


def test_tpe_search_with_none_rng():
    trials = tpe_search(parse_space({"x": [0.0, 1.0]}), _obj, 8)
    assert len(trials) == 8


def test_run_search_tpe():
    result = run_search("tpe", _space(), _obj, budget=15, seed=3)
    assert result.strategy == "tpe"
    assert len(result.trials) == 15
    assert result.best_score == min(t.score for t in result.trials)


def test_run_search_forwarded_tpe_kwargs():
    a = run_search("tpe", _space(), _obj, 18, seed=1, gamma=0.1, n_initial=4)
    b = run_search("tpe", _space(), _obj, 18, seed=1, gamma=0.5, n_initial=4)
    assert len(a.trials) == 18
    assert [t.params for t in a.trials] != [t.params for t in b.trials]


def test_compare_strategies_includes_tpe():
    results = compare_strategies(_space(), _obj, budget=10, seed=7)
    assert set(results) >= {"grid", "random", "bayesian", "tpe", "cmaes", "hyperband"}
    assert len(results["tpe"].trials) == 10


def test_compare_strategies_tpe_subset():
    results = compare_strategies(
        _space(), _obj, 8, strategies=("random", "tpe"), seed=2
    )
    assert set(results) == {"random", "tpe"}
    for result in results.values():
        assert len(result.trials) == 8
