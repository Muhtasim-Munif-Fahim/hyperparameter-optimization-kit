import numpy as np
import pytest

from hyperopt_kit import (
    FloatRange,
    median_stopping_decision,
    median_stopping_search,
    run_search,
    successive_halving_rungs,
)


def test_decision_stops_when_best_is_worse_than_median():
    refs = [[1.0, 0.5, 0.2], [1.2, 0.6, 0.3], [0.8, 0.4, 0.1]]
    # running averages at rung 1: 0.75, 0.9, 0.6 -> median 0.75
    d = median_stopping_decision([1.5, 0.9], refs)
    assert d.stop and d.median_reference == pytest.approx(0.75) and d.n_reference == 3
    assert not median_stopping_decision([1.5, 0.7], refs).stop
    # equal to the median is not strictly worse
    assert not median_stopping_decision([0.75, 0.9], refs).stop


def test_decision_respects_grace_and_min_samples():
    refs = [[0.1], [0.2], [0.3]]
    assert not median_stopping_decision([5.0], refs, grace_rungs=1).stop
    assert median_stopping_decision([5.0], refs, grace_rungs=0).stop
    short = [[0.1, 0.1], [0.2]]  # only one reaches rung 1
    d = median_stopping_decision([9.0, 9.0], short, min_samples=2)
    assert not d.stop and d.n_reference == 1 and d.median_reference is None
    with pytest.raises(ValueError):
        median_stopping_decision([], refs)


def _space():
    return {"x": FloatRange(-3.0, 3.0)}


def _objective(params, resource):
    # A noisy-free learning curve: loss shrinks with resource toward (x-1)^2.
    return (params["x"] - 1.0) ** 2 + (1.0 - resource)


def test_search_respects_budget_and_resources():
    trials = median_stopping_search(_space(), _objective, 40, rng=np.random.default_rng(0))
    assert len(trials) == 40
    rungs = set(successive_halving_rungs(9, 3, 1))
    assert {t.resource for t in trials} <= rungs
    assert [t.iteration for t in trials] == list(range(40))


def test_search_prunes_bad_configurations():
    trials = median_stopping_search(_space(), _objective, 60, rng=np.random.default_rng(1))
    full = [t for t in trials if t.resource == 1.0]
    low = [t for t in trials if t.resource < 1.0]
    assert full, "some configurations must reach full fidelity"
    # Pruning means more configurations start than finish.
    starts = sum(1 for t in trials if t.resource == min(t.resource for t in trials))
    assert starts > len(full)
    # Configurations promoted to full fidelity are better than average starters.
    assert np.mean([t.score for t in full]) < np.mean([t.score for t in low])


def test_search_is_reproducible():
    a = median_stopping_search(_space(), _objective, 30, rng=np.random.default_rng(5))
    b = median_stopping_search(_space(), _objective, 30, rng=np.random.default_rng(5))
    assert [t.params for t in a] == [t.params for t in b]


def test_stop_when_and_run_search_integration():
    trials = median_stopping_search(
        _space(), _objective, 50, rng=np.random.default_rng(2), stop_when=lambda best, n: n >= 7
    )
    assert len(trials) == 7
    res = run_search("median_stopping", _space(), _objective, 25, seed=3, grace_rungs=0, min_samples=2)
    assert res.strategy == "median_stopping"
    assert len(res.trials) == 25


def test_validation():
    with pytest.raises(ValueError):
        median_stopping_search(_space(), _objective, 0)
    with pytest.raises(ValueError):
        median_stopping_search(_space(), _objective, 5, grace_rungs=-1)
    with pytest.raises(ValueError):
        median_stopping_search(_space(), _objective, 5, min_samples=0)
