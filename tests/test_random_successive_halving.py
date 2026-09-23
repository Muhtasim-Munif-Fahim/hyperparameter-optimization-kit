import numpy as np
import pytest

from hyperopt_kit.evaluate import compare_strategies, run_search
from hyperopt_kit.searchers import (
    random_search,
    random_successive_halving,
    select_halving_survivors,
    successive_halving,
    successive_halving_rungs,
)
from hyperopt_kit.spaces import parse_space


def _space():
    return parse_space({"x": [0.0, 1.0], "y": [0.0, 1.0]})


def _obj(params, resource=1.0):
    return float(params["x"]) + 0.01 * (1.0 - float(resource))


def test_select_halving_survivors_waits_until_eta():
    assert select_halving_survivors([0.2, 0.1], eta=3) == []
    assert select_halving_survivors([], eta=3) == []


def test_select_halving_survivors_keeps_top_fraction():
    scores = [0.4, 0.1, 0.9, 0.2, 0.8, 0.3]
    assert select_halving_survivors(scores, eta=3) == [1, 3]


def test_select_halving_survivors_tie_breaks_toward_earlier_index():
    assert select_halving_survivors([0.5, 0.5, 0.5, 0.5], eta=2) == [0, 1]


def test_select_halving_survivors_rejects_bad_inputs():
    with pytest.raises(ValueError):
        select_halving_survivors([0.1, 0.2], eta=1)
    with pytest.raises(ValueError):
        select_halving_survivors(np.zeros((2, 2)), eta=2)


def test_rungs_follow_aggressive_hyperband_bracket():
    assert successive_halving_rungs() == pytest.approx([1.0 / 9.0, 1.0 / 3.0, 1.0])
    assert successive_halving_rungs(max_resource=1) == pytest.approx([1.0])
    assert successive_halving_rungs(max_resource=81, eta=3) == pytest.approx(
        [1.0 / 81.0, 1.0 / 27.0, 1.0 / 9.0, 1.0 / 3.0, 1.0]
    )


def test_random_successive_halving_respects_budget():
    trials = random_successive_halving(
        _space(), _obj, budget=17, rng=np.random.default_rng(0)
    )
    assert len(trials) == 17
    rungs = successive_halving_rungs()
    for i, trial in enumerate(trials):
        assert trial.iteration == i
        assert any(abs(trial.resource - rung) < 1e-12 for rung in rungs)


def test_promotes_as_soon_as_a_rung_can_rank():
    trials = random_successive_halving(
        parse_space({"x": [0.0, 1.0]}),
        _obj,
        budget=13,
        rng=np.random.default_rng(0),
    )
    rungs = successive_halving_rungs()
    resources = [t.resource for t in trials]
    assert resources[:4] == pytest.approx([rungs[0], rungs[0], rungs[0], rungs[1]])
    assert any(abs(r - rungs[-1]) < 1e-12 for r in resources)


def test_schedule_differs_from_closed_successive_halving_bracket():
    space = parse_space({"x": [0.0, 1.0]})
    streaming = random_successive_halving(
        space, _obj, budget=12, rng=np.random.default_rng(0)
    )
    bracket = successive_halving(
        space, _obj, budget=12, rng=np.random.default_rng(0), n_candidates=9
    )
    assert [t.resource for t in streaming] != [t.resource for t in bracket]
    assert all(abs(t.resource - successive_halving_rungs()[0]) < 1e-12 for t in bracket[:9])


def test_promotion_follows_running_top_fraction():
    """Replay the early-stopping rule against the recorded trial sequence."""
    eta = 3
    rungs = successive_halving_rungs(eta=eta)
    trials = random_successive_halving(
        _space(), _obj, budget=40, rng=np.random.default_rng(2), eta=eta
    )
    entries = [[] for _ in rungs]

    def rung_index(resource):
        matches = [i for i, rung in enumerate(rungs) if abs(resource - rung) < 1e-9]
        assert matches
        return matches[0]

    def key_of(params):
        return tuple(sorted((name, repr(value)) for name, value in params.items()))

    def unpromoted_survivors(rung_entries):
        scores = [entry["score"] for entry in rung_entries]
        return [
            i
            for i in select_halving_survivors(scores, eta)
            if not rung_entries[i]["promoted"]
        ]

    for trial in trials:
        idx = rung_index(trial.resource)
        key = key_of(trial.params)
        if idx == 0:
            for rung_entries in entries[:-1]:
                assert unpromoted_survivors(rung_entries) == []
            entries[0].append({"key": key, "score": trial.score, "promoted": False})
            continue
        pending = unpromoted_survivors(entries[idx - 1])
        assert pending
        chosen = pending[0]
        assert entries[idx - 1][chosen]["key"] == key
        entries[idx - 1][chosen]["promoted"] = True
        entries[idx].append({"key": key, "score": trial.score, "promoted": False})

    full = [t for t in trials if abs(t.resource - rungs[-1]) < 1e-9]
    base = [t for t in trials if abs(t.resource - rungs[0]) < 1e-9]
    assert full
    assert len(full) < len(base)


def test_single_rung_matches_random_search():
    space = parse_space({"x": [0.0, 1.0], "mode": ["a", "b", "c"]})
    objective = lambda p: (p["x"] - 0.25) ** 2 + (0.0 if p["mode"] == "a" else 0.2)
    halving = random_successive_halving(
        space,
        objective,
        budget=15,
        rng=np.random.default_rng(4),
        min_resource=1,
        max_resource=1,
    )
    plain = random_search(space, objective, 15, rng=np.random.default_rng(4))
    assert [t.params for t in halving] == [t.params for t in plain]
    assert [t.score for t in halving] == [t.score for t in plain]
    assert all(t.resource == pytest.approx(1.0) for t in halving)


def test_deterministic_with_seed_and_differs_across_seeds():
    space = parse_space({"x": [0.0, 1.0]})
    a = random_successive_halving(space, _obj, 16, rng=np.random.default_rng(11))
    b = random_successive_halving(space, _obj, 16, rng=np.random.default_rng(11))
    c = random_successive_halving(space, _obj, 16, rng=np.random.default_rng(99))
    assert [t.params for t in a] == [t.params for t in b]
    assert [t.score for t in a] == [t.score for t in b]
    assert [t.params for t in a] != [t.params for t in c]


def test_rejects_bad_budget_space_and_resources():
    with pytest.raises(ValueError):
        random_successive_halving(_space(), _obj, 0, rng=np.random.default_rng(0))
    with pytest.raises(ValueError):
        random_successive_halving({}, _obj, 5, rng=np.random.default_rng(0))
    with pytest.raises(ValueError):
        random_successive_halving(_space(), _obj, 5, eta=1, rng=np.random.default_rng(0))
    with pytest.raises(ValueError):
        random_successive_halving(
            _space(),
            _obj,
            5,
            min_resource=5,
            max_resource=3,
            rng=np.random.default_rng(0),
        )


def test_early_stop_on_evaluation_count():
    def stopper(best, n):
        return n >= 4

    trials = random_successive_halving(
        parse_space({"x": [0.0, 1.0]}),
        _obj,
        budget=30,
        rng=np.random.default_rng(0),
        stop_when=stopper,
    )
    assert len(trials) == 4


def test_mixed_space_values_stay_valid():
    space = parse_space({"x": [0.0, 1.0], "z": ["a", "b", "c"], "n": [1, 8, "int"]})
    trials = random_successive_halving(space, _obj, budget=12, rng=np.random.default_rng(3))
    assert all(t.params["z"] in ("a", "b", "c") for t in trials)
    assert all(isinstance(t.params["n"], int) for t in trials)
    assert all(0.0 <= t.params["x"] <= 1.0 for t in trials)


def test_none_rng_runs():
    trials = random_successive_halving(parse_space({"x": [0.0, 1.0]}), _obj, budget=5)
    assert len(trials) == 5


def test_run_search_forwards_halving_kwargs():
    a = run_search(
        "random_successive_halving",
        _space(),
        _obj,
        budget=12,
        seed=1,
        eta=2,
        max_resource=8,
        min_resource=1,
    )
    b = run_search(
        "random_successive_halving",
        _space(),
        _obj,
        budget=12,
        seed=1,
        eta=3,
        max_resource=9,
        min_resource=1,
    )
    assert a.strategy == "random_successive_halving"
    assert len(a.trials) == 12
    assert [t.resource for t in a.trials] != [t.resource for t in b.trials]
    assert a.best_score == min(
        t.score
        for t in a.trials
        if t.resource >= max(tr.resource for tr in a.trials) - 1e-12
    )


def test_compare_strategies_can_select_random_successive_halving():
    results = compare_strategies(
        _space(),
        _obj,
        budget=8,
        strategies=("random", "random_successive_halving"),
        seed=2,
    )
    assert set(results) == {"random", "random_successive_halving"}
    assert len(results["random_successive_halving"].trials) == 8
    assert all(t.resource is not None for t in results["random_successive_halving"].trials)


def test_fidelity_alias_and_single_arg_objective():
    seen = []

    def fidelity_obj(params, fidelity=1.0):
        seen.append(fidelity)
        return float(params["x"])

    random_successive_halving(
        parse_space({"x": [0.0, 1.0]}),
        fidelity_obj,
        budget=6,
        rng=np.random.default_rng(0),
    )
    assert seen
    assert any(abs(r - 1.0) > 1e-9 for r in seen)

    trials = random_successive_halving(
        parse_space({"x": [0.0, 1.0]}),
        lambda p: (p["x"] - 0.25) ** 2,
        budget=8,
        rng=np.random.default_rng(1),
    )
    assert len(trials) == 8
    assert min(t.score for t in trials) < 0.2
