import numpy as np
import pytest

from hyperopt_kit.evaluate import compare_strategies, run_search
from hyperopt_kit.searchers import (
    Trial,
    hyperband_brackets,
    hyperband_search,
    successive_halving,
)
from hyperopt_kit.spaces import parse_space


def _space():
    return parse_space({"x": [0.0, 1.0], "y": [0.0, 1.0]})


def _obj(params):
    return float(params["x"]) + float(params.get("y", 0.0))


def _fidelity_obj(params, resource=1.0):
    return float(params["x"]) + 0.01 * (1.0 - float(resource))


def test_hyperband_brackets_match_li_et_al_r81_eta3():
    brackets = hyperband_brackets(max_resource=81, eta=3, min_resource=1)
    assert [s for s, _, _ in brackets] == [4, 3, 2, 1, 0]
    ns = [n for _, n, _ in brackets]
    rs = [r for _, _, r in brackets]
    assert ns == [81, 34, 15, 8, 5]
    assert rs == pytest.approx([1.0, 3.0, 9.0, 27.0, 81.0])


def test_hyperband_brackets_default_r9():
    brackets = hyperband_brackets()
    assert [s for s, n, r in brackets] == [2, 1, 0]
    assert [n for s, n, r in brackets] == [9, 5, 3]
    assert [r for s, n, r in brackets] == pytest.approx([1.0, 3.0, 9.0])


def test_hyperband_brackets_s_max_zero_when_no_rungs():
    brackets = hyperband_brackets(max_resource=1, eta=3, min_resource=1)
    assert brackets == [(0, 1, 1.0)]


def test_hyperband_respects_budget():
    trials = hyperband_search(
        _space(), _obj, budget=17, rng=np.random.default_rng(0)
    )
    assert len(trials) == 17
    for i, trial in enumerate(trials):
        assert trial.iteration == i
        assert trial.resource is not None
        assert 0.0 < trial.resource <= 1.0 + 1e-12


def test_successive_halving_respects_budget():
    trials = successive_halving(
        _space(), _obj, budget=11, rng=np.random.default_rng(1)
    )
    assert len(trials) == 11


def test_hyperband_deterministic_with_seed():
    a = hyperband_search(_space(), _obj, 20, rng=np.random.default_rng(11))
    b = hyperband_search(_space(), _obj, 20, rng=np.random.default_rng(11))
    assert [t.params for t in a] == [t.params for t in b]
    assert [t.score for t in a] == [t.score for t in b]
    assert [t.resource for t in a] == [t.resource for t in b]


def test_hyperband_differs_across_seeds():
    a = hyperband_search(_space(), _obj, 20, rng=np.random.default_rng(11))
    b = hyperband_search(_space(), _obj, 20, rng=np.random.default_rng(99))
    assert [t.params for t in a] != [t.params for t in b]


def test_hyperband_rejects_bad_budget_and_space():
    with pytest.raises(ValueError):
        hyperband_search(_space(), _obj, 0, rng=np.random.default_rng(0))
    with pytest.raises(ValueError):
        hyperband_search({}, _obj, 5, rng=np.random.default_rng(0))
    with pytest.raises(ValueError):
        successive_halving(_space(), _obj, 5, eta=1, rng=np.random.default_rng(0))
    with pytest.raises(ValueError):
        hyperband_search(
            _space(), _obj, 5, min_resource=5, max_resource=3, rng=np.random.default_rng(0)
        )


def test_successive_halving_promotes_better_configs():
    space = parse_space({"x": [0.0, 1.0]})
    calls = []

    def obj(params, resource=1.0):
        calls.append((params["x"], resource))
        return float(params["x"])

    successive_halving(
        space,
        obj,
        budget=13,
        rng=np.random.default_rng(0),
        eta=3,
        min_resource=1,
        max_resource=9,
        n_candidates=9,
    )
    low = [x for x, r in calls if abs(r - 1.0 / 9.0) < 1e-9]
    mid = [x for x, r in calls if abs(r - 1.0 / 3.0) < 1e-9]
    high = [x for x, r in calls if abs(r - 1.0) < 1e-9]
    assert len(low) == 9
    assert len(mid) == 3
    assert len(high) == 1
    assert set(high) <= set(mid) <= set(low)
    assert high[0] == min(low)
    assert max(mid) <= sorted(low)[2]


def test_hyperband_uses_multiple_starting_fidelities():
    space = parse_space({"x": [0.0, 1.0]})
    first_resource = {}

    def obj(params, resource=1.0):
        key = round(params["x"], 12)
        first_resource.setdefault(key, resource)
        return float(params["x"])

    hyperband_search(
        space, obj, budget=22, rng=np.random.default_rng(2), eta=3, max_resource=9
    )
    starts = sorted({round(r, 10) for r in first_resource.values()})
    assert starts == pytest.approx([1.0 / 9.0, 1.0 / 3.0, 1.0])


def test_hyperband_passes_resource_to_objective():
    seen = []

    def obj(params, resource=1.0):
        seen.append(resource)
        return 0.0

    hyperband_search(
        parse_space({"x": [0.0, 1.0]}),
        obj,
        budget=8,
        rng=np.random.default_rng(0),
    )
    assert seen
    assert all(0.0 < r <= 1.0 + 1e-12 for r in seen)


def test_hyperband_fidelity_kwarg_alias():
    seen = []

    def obj(params, fidelity=1.0):
        seen.append(fidelity)
        return float(params["x"])

    hyperband_search(
        parse_space({"x": [0.0, 1.0]}),
        obj,
        budget=6,
        rng=np.random.default_rng(0),
    )
    assert seen
    assert any(abs(r - 1.0) > 1e-9 for r in seen)


def test_hyperband_single_arg_objective_still_runs():
    trials = hyperband_search(
        parse_space({"x": [0.0, 1.0]}),
        lambda p: (p["x"] - 0.25) ** 2,
        budget=10,
        rng=np.random.default_rng(4),
    )
    assert len(trials) == 10
    assert min(t.score for t in trials) < 0.2


def test_hyperband_mixed_space():
    space = parse_space({"x": [0.0, 1.0], "z": ["a", "b", "c"], "n": [1, 8, "int"]})
    trials = hyperband_search(space, _obj, budget=12, rng=np.random.default_rng(3))
    assert all(t.params["z"] in ("a", "b", "c") for t in trials)
    assert all(isinstance(t.params["n"], int) for t in trials)


def test_hyperband_early_stop_on_floor():
    def stopper(best, n):
        return best <= 0.0 or n >= 4

    trials = hyperband_search(
        parse_space({"x": [0.0, 1.0]}),
        lambda p, resource=1.0: 0.0,
        budget=20,
        rng=np.random.default_rng(0),
        stop_when=stopper,
    )
    assert len(trials) <= 4


def test_run_search_hyperband():
    result = run_search("hyperband", _space(), _obj, budget=15, seed=3)
    assert result.strategy == "hyperband"
    assert len(result.trials) == 15
    assert result.best_score == min(
        t.score
        for t in result.trials
        if t.resource is not None
        and t.resource >= max(tr.resource for tr in result.trials if tr.resource is not None) - 1e-12
    )


def test_run_search_successive_halving_and_kwargs():
    a = run_search(
        "successive_halving",
        _space(),
        _fidelity_obj,
        budget=12,
        seed=1,
        eta=3,
        n_candidates=6,
    )
    b = run_search(
        "successive_halving",
        _space(),
        _fidelity_obj,
        budget=12,
        seed=1,
        eta=2,
        n_candidates=6,
    )
    assert a.strategy == "successive_halving"
    assert len(a.trials) == 12
    assert [t.resource for t in a.trials] != [t.resource for t in b.trials]


def test_best_trial_prefers_high_resource():
    from hyperopt_kit.evaluate import best_trial

    trials = [
        Trial(0, {"x": 0.0}, score=0.01, resource=0.1),
        Trial(1, {"x": 0.9}, score=0.50, resource=1.0),
        Trial(2, {"x": 0.2}, score=0.20, resource=1.0),
    ]
    winner = best_trial(trials)
    assert winner.params["x"] == 0.2
    assert winner.resource == 1.0


def test_compare_strategies_includes_hyperband():
    results = compare_strategies(_space(), _obj, budget=10, seed=7)
    assert set(results) >= {"grid", "random", "bayesian", "tpe", "cmaes", "hyperband", "bohb"}
    assert len(results["hyperband"].trials) == 10


def test_compare_strategies_hyperband_subset():
    results = compare_strategies(
        _space(),
        _obj,
        8,
        strategies=("random", "hyperband", "successive_halving"),
        seed=2,
    )
    assert set(results) == {"random", "hyperband", "successive_halving"}
    for result in results.values():
        assert len(result.trials) == 8


def test_hyperband_fills_small_budget():
    trials = hyperband_search(
        parse_space({"x": [0.0, 1.0]}), _obj, budget=1, rng=np.random.default_rng(0)
    )
    assert len(trials) == 1


def test_successive_halving_rejects_bad_n_candidates():
    with pytest.raises(ValueError):
        successive_halving(
            _space(), _obj, 5, n_candidates=0, rng=np.random.default_rng(0)
        )


def test_demo_objective_full_fidelity_matches_one_arg_call():
    from hyperopt_kit.objectives import demo_objective

    params = {"a": 1.2, "b": 2.3}
    assert demo_objective(params) == demo_objective(params, resource=1.0)


def test_hyperband_learning_curve_ends_at_reported_best():
    from hyperopt_kit.evaluate import learning_curve

    result = run_search("hyperband", _space(), _fidelity_obj, budget=16, seed=4)
    curve = learning_curve(result.trials)
    assert curve[-1] == result.best_score
