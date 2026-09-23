import numpy as np
import pytest

from hyperopt_kit.evaluate import (
    RunResult,
    available_strategies,
    best_params,
    best_score,
    best_trial,
    compare_strategies,
    learning_curve,
    make_stopper,
    run_search,
)
from hyperopt_kit.spaces import parse_space


def _space():
    return parse_space({"x": [0.0, 1.0], "y": [0.0, 1.0]})


def _objective(params):
    return (params["x"] - 0.5) ** 2 + (params["y"] - 0.5) ** 2


def test_run_search_grid_returns_runresult():
    result = run_search("grid", _space(), _objective, budget=10)
    assert isinstance(result, RunResult)
    assert result.strategy == "grid"
    assert len(result.trials) == 10
    assert result.best_score == min(t.score for t in result.trials)


def test_run_search_random_returns_runresult():
    result = run_search("random", _space(), _objective, budget=15, seed=3)
    assert result.strategy == "random"
    assert len(result.trials) == 15
    assert result.best_score < 0.5


def test_run_search_bayesian_returns_runresult():
    result = run_search("bayesian", _space(), _objective, budget=15, seed=3)
    assert result.strategy == "bayesian"
    assert len(result.trials) == 15
    assert result.best_params["x"] == pytest.approx(0.5, abs=0.2)


def test_run_search_hyperband_returns_runresult():
    result = run_search("hyperband", _space(), _objective, budget=15, seed=3)
    assert result.strategy == "hyperband"
    assert len(result.trials) == 15
    assert all(t.resource is not None for t in result.trials)


def test_run_search_tpe_returns_runresult():
    result = run_search("tpe", _space(), _objective, budget=15, seed=3)
    assert result.strategy == "tpe"
    assert len(result.trials) == 15
    assert result.best_params["x"] == pytest.approx(0.5, abs=0.25)


def test_run_search_cmaes_returns_runresult():
    result = run_search("cmaes", _space(), _objective, budget=15, seed=3)
    assert result.strategy == "cmaes"
    assert len(result.trials) == 15
    assert result.best_params["x"] == pytest.approx(0.5, abs=0.25)


def test_available_strategies_lists_tpe_cmaes_hyperband_and_bohb():
    names = available_strategies()
    assert names == [
        "grid",
        "random",
        "bayesian",
        "tpe",
        "cmaes",
        "hyperband",
        "bohb",
        "successive_halving",
        "random_successive_halving",
    ]


def test_run_search_unknown_strategy_raises():
    with pytest.raises(ValueError):
        run_search("grid-search", _space(), _objective, 5)


def test_run_search_budget_zero_raises():
    with pytest.raises(ValueError):
        run_search("random", _space(), _objective, 0)


def test_run_search_deterministic_with_seed():
    a = run_search("bayesian", _space(), _objective, 12, seed=9)
    b = run_search("bayesian", _space(), _objective, 12, seed=9)
    assert a.best_score == b.best_score
    assert [t.params for t in a.trials] == [t.params for t in b.trials]


def test_run_search_forwarded_bayesian_kwargs():
    a = run_search("bayesian", _space(), _objective, 20, seed=1, xi=0.5, n_initial=4)
    b = run_search("bayesian", _space(), _objective, 20, seed=1, xi=0.01, n_initial=4)
    assert len(a.trials) == 20
    assert a.trials != b.trials


def test_early_stopping_floor_stops_immediately():
    result = run_search(
        "random",
        _space(),
        lambda p: 0.0,
        budget=20,
        seed=0,
        floor=0.5,
    )
    assert len(result.trials) == 1


def test_early_stopping_patience_stops_after_plateau():
    result = run_search(
        "random",
        _space(),
        lambda p: 0.0,
        budget=50,
        seed=0,
        patience=5,
    )
    assert len(result.trials) == 1 + 5


def test_early_stopping_unmet_floor_runs_full_budget():
    result = run_search(
        "random",
        _space(),
        _objective,
        budget=20,
        seed=0,
        floor=-100.0,
    )
    assert len(result.trials) == 20


def test_early_stopping_only_when_configured():
    result = run_search("random", _space(), _objective, budget=20, seed=0)
    assert len(result.trials) == 20


def test_early_stopper_floor_below_best_never_triggers():
    stopper = make_stopper(floor=-1.0)
    assert stopper(5.0, 1) is False
    assert stopper(5.0, 2) is False


def test_make_stopper_none_without_config():
    assert make_stopper() is None
    assert make_stopper(floor=None, patience=None) is None


def test_best_trial_and_helpers():
    trials = [
        type("T", (), {"score": 3.0, "params": {"x": 1}})(),
        type("T", (), {"score": 1.0, "params": {"x": 2}})(),
        type("T", (), {"score": 2.0, "params": {"x": 3}})(),
    ]
    assert best_score(trials) == 1.0
    assert best_params(trials) == {"x": 2}


def test_best_trial_on_empty_raises():
    with pytest.raises(ValueError):
        best_score([])


def test_learning_curve_is_monotone_non_increasing():
    result = run_search("random", _space(), _objective, budget=30, seed=5)
    curve = learning_curve(result.trials)
    assert len(curve) == 30
    assert curve[0] == result.trials[0].score
    assert curve[-1] == result.best_score
    assert all(b <= a for a, b in zip(curve, curve[1:]))


def test_compare_strategies_common_budget():
    results = compare_strategies(_space(), _objective, budget=20, seed=7)
    assert set(results) == {
        "grid",
        "random",
        "bayesian",
        "tpe",
        "cmaes",
        "hyperband",
        "bohb",
    }
    assert len(results["random"].trials) == 20
    assert len(results["bayesian"].trials) == 20
    assert len(results["tpe"].trials) == 20
    assert len(results["cmaes"].trials) == 20
    assert len(results["hyperband"].trials) == 20
    assert len(results["bohb"].trials) == 20
    assert len(results["grid"].trials) <= 20


def test_compare_strategies_deterministic_with_seed():
    a = compare_strategies(_space(), _objective, 20, seed=7)
    b = compare_strategies(_space(), _objective, 20, seed=7)
    assert {s: r.best_score for s, r in a.items()} == {
        s: r.best_score for s, r in b.items()
    }


def test_compare_strategies_unknown_strategy_raises():
    with pytest.raises(ValueError):
        compare_strategies(_space(), _objective, 10, strategies=("grid", "nope"))


def test_compare_strategies_floor_applies_to_all():
    results = compare_strategies(
        _space(), lambda p: 0.0, 20, seed=7, floor=0.5
    )
    for result in results.values():
        assert len(result.trials) == 1


def test_compare_strategies_per_strategy_kwargs():
    results = compare_strategies(
        _space(),
        _objective,
        20,
        seed=7,
        strategy_kwargs={"bayesian": {"xi": 0.5}},
    )
    assert set(results) == {
        "grid",
        "random",
        "bayesian",
        "tpe",
        "cmaes",
        "hyperband",
        "bohb",
    }


def test_compare_strategies_subset():
    results = compare_strategies(_space(), _objective, 10, strategies=("grid", "random"), seed=1)
    assert set(results) == {"grid", "random"}


def test_compare_strategies_rejects_bad_budget():
    with pytest.raises(ValueError):
        compare_strategies(_space(), _objective, 0)
