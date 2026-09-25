import numpy as np
import pytest

from hyperopt_kit.evaluate import best_trial, compare_strategies, run_search
from hyperopt_kit.searchers import (
    PBTHistory,
    Trial,
    pbt_best_trial,
    pbt_perturb,
    pbt_search,
    population_based_training,
    random_search,
)
from hyperopt_kit.spaces import parse_space


def _space():
    return parse_space({"x": [0.0, 1.0], "y": [0.0, 1.0]})


def _obj(params, resource=1.0):
    return float(params["x"]) + float(params.get("y", 0.0))


def _mixed_space():
    return parse_space(
        {
            "x": [0.0, 1.0],
            "y": [0, 10, "int"],
            "z": ["a", "b", "c"],
        }
    )


def test_population_based_training_aliases_pbt_search():
    assert population_based_training is pbt_search


def test_pbt_perturb_scales_float_and_int_and_resamples_categorical():
    space = parse_space(
        {"x": [0.1, 10.0], "n": [1, 100, "int"], "k": ["a", "b", "c"]}
    )
    params = {"x": 2.0, "n": 10, "k": "a"}
    out = pbt_perturb(
        space,
        params,
        np.random.default_rng(0),
        perturbation_factors=(0.8,),
        resample_probability=1.0,
    )
    assert out["x"] == pytest.approx(1.6)
    assert out["n"] == 8
    assert out["k"] in ("b", "c")
    assert params == {"x": 2.0, "n": 10, "k": "a"}


def test_pbt_perturb_clips_to_bounds_and_keeps_identity_factor():
    space = parse_space({"x": [0.1, 1.0], "n": [2, 8, "int"], "k": ["a", "b"]})
    interior = {"x": 0.4, "n": 4, "k": "a"}
    scaled = pbt_perturb(
        space,
        interior,
        np.random.default_rng(1),
        perturbation_factors=(0.5,),
        resample_probability=0.0,
    )
    assert scaled["x"] == pytest.approx(0.2)
    assert scaled["n"] == 2
    assert scaled["k"] == "a"

    on_bound = {"x": 0.1, "n": 4, "k": "a"}
    moved = pbt_perturb(
        space,
        on_bound,
        np.random.default_rng(1),
        perturbation_factors=(0.5,),
        resample_probability=0.0,
    )
    # 0.1 * 0.5 clips back to the bound, so explore takes one mutate step.
    assert 0.1 < moved["x"] <= 1.0
    assert moved["n"] == 2
    assert moved["k"] == "a"

    same = pbt_perturb(
        space,
        on_bound,
        np.random.default_rng(1),
        perturbation_factors=(1.0,),
        resample_probability=0.0,
    )
    assert same["x"] == pytest.approx(0.1)
    assert same["n"] == 4
    assert same["k"] == "a"


def test_pbt_perturb_log_scale_is_multiplicative():
    space = parse_space({"lr": [1e-4, 1.0, "log"]})
    out = pbt_perturb(
        space,
        {"lr": 0.01},
        np.random.default_rng(0),
        perturbation_factors=(1.2,),
        resample_probability=0.0,
    )
    assert out["lr"] == pytest.approx(0.012)
    assert 1e-4 <= out["lr"] <= 1.0


def test_pbt_perturb_rejects_bad_args():
    space = parse_space({"x": [0.0, 1.0]})
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        pbt_perturb(space, {"x": 0.2}, rng, perturbation_factors=())
    with pytest.raises(ValueError):
        pbt_perturb(space, {"x": 0.2}, rng, perturbation_factors=(0.0,))
    with pytest.raises(ValueError):
        pbt_perturb(space, {"x": 0.2}, rng, resample_probability=1.5)
    with pytest.raises(ValueError):
        pbt_perturb(space, {}, rng)
    with pytest.raises(ValueError):
        pbt_perturb({}, {"x": 0.2}, rng)


def test_pbt_search_respects_budget_and_records_rungs():
    trials = pbt_search(
        _space(),
        _obj,
        budget=12,
        rng=np.random.default_rng(0),
        population_size=4,
    )
    assert isinstance(trials, PBTHistory)
    assert isinstance(trials, list)
    assert len(trials) == 12
    assert trials.history == list(trials)
    assert trials.best == pbt_best_trial(trials)
    assert trials.best == best_trial(trials)
    assert trials.best in trials
    for i, trial in enumerate(trials):
        assert trial.iteration == i
        assert isinstance(trial.score, float)
        assert set(trial.params) == {"x", "y"}
        assert 0.0 <= trial.params["x"] <= 1.0
        assert 0.0 <= trial.params["y"] <= 1.0
    assert all(t.resource == pytest.approx(1.0 / 9.0) for t in trials[0:4])
    assert all(t.resource == pytest.approx(1.0 / 3.0) for t in trials[4:8])
    assert all(t.resource == pytest.approx(1.0) for t in trials[8:12])


def test_pbt_search_first_generation_matches_random_search():
    space = _space()
    pbt = pbt_search(space, _obj, 4, rng=np.random.default_rng(11), population_size=4)
    rnd = random_search(space, _obj, 4, rng=np.random.default_rng(11))
    assert [t.params for t in pbt] == [t.params for t in rnd]


def test_pbt_search_deterministic_with_seed():
    a = pbt_search(_space(), _obj, 16, rng=np.random.default_rng(11), population_size=4)
    b = pbt_search(_space(), _obj, 16, rng=np.random.default_rng(11), population_size=4)
    assert [t.params for t in a] == [t.params for t in b]
    assert [t.score for t in a] == [t.score for t in b]
    assert a.best == b.best


def test_pbt_search_differs_across_seeds():
    a = pbt_search(_space(), _obj, 16, rng=np.random.default_rng(11), population_size=4)
    b = pbt_search(_space(), _obj, 16, rng=np.random.default_rng(99), population_size=4)
    assert [t.params for t in a] != [t.params for t in b]


def test_pbt_exploit_copies_hparams_from_better_member():
    space = parse_space({"x": [0.0, 1.0]})
    trials = pbt_search(
        space,
        lambda p, resource=1.0: float(p["x"]),
        budget=8,
        rng=np.random.default_rng(3),
        population_size=4,
        exploit_interval=1,
        quantile=0.25,
        perturbation_factors=(1.0,),
        resample_probability=0.0,
    )
    first = trials[:4]
    second = trials[4:8]
    best_i = min(range(4), key=lambda i: first[i].score)
    worst_i = max(range(4), key=lambda i: (first[i].score, i))
    assert second[best_i].params["x"] == pytest.approx(first[best_i].params["x"])
    assert second[worst_i].params["x"] == pytest.approx(first[best_i].params["x"])
    assert worst_i != best_i


def test_pbt_explore_perturbs_copied_continuous_value():
    space = parse_space({"x": [0.0, 1.0]})
    trials = pbt_search(
        space,
        lambda p, resource=1.0: float(p["x"]),
        budget=8,
        rng=np.random.default_rng(5),
        population_size=4,
        perturbation_factors=(0.5,),
        resample_probability=0.0,
    )
    first = trials[:4]
    second = trials[4:8]
    best_i = min(range(4), key=lambda i: first[i].score)
    worst_i = max(range(4), key=lambda i: (first[i].score, i))
    expected = float(np.clip(first[best_i].params["x"] * 0.5, 0.0, 1.0))
    assert second[worst_i].params["x"] == pytest.approx(expected)
    assert second[best_i].params["x"] == pytest.approx(first[best_i].params["x"])


def test_pbt_explore_perturbs_integer_and_categorical():
    space = parse_space({"n": [0, 100, "int"], "k": ["a", "b", "c"]})

    def objective(params, resource=1.0):
        return float(params["n"])

    trials = pbt_search(
        space,
        objective,
        budget=8,
        rng=np.random.default_rng(4),
        population_size=4,
        perturbation_factors=(0.5,),
        resample_probability=1.0,
    )
    first = trials[:4]
    second = trials[4:8]
    best_i = min(range(4), key=lambda i: first[i].score)
    worst_i = max(range(4), key=lambda i: (first[i].score, i))
    donor_n = int(first[best_i].params["n"])
    half_n = int(np.clip(int(round(donor_n * 0.5)), 0, 100))
    child_n = second[worst_i].params["n"]
    assert isinstance(child_n, int)
    if half_n != donor_n:
        assert child_n == half_n
    else:
        assert abs(child_n - donor_n) == 1
    assert second[worst_i].params["k"] != first[best_i].params["k"]
    assert second[worst_i].params["k"] in ("a", "b", "c")
    assert second[best_i].params["k"] == first[best_i].params["k"]


def test_pbt_exploit_copies_weights_and_deepcopies_them():
    space = parse_space({"x": [0.0, 1.0]})
    received = []
    stored = []

    def objective(params, resource=1.0, weights=None):
        if weights is None:
            received.append(None)
            token = 0
        else:
            received.append((id(weights), weights["token"], weights["origin"]))
            token = int(weights["token"])
        new = {"token": token + 1, "origin": float(params["x"])}
        stored.append(new)
        return float(params["x"]), new

    trials = pbt_search(
        space,
        objective,
        budget=8,
        rng=np.random.default_rng(8),
        population_size=4,
        perturbation_factors=(1.0,),
        resample_probability=0.0,
    )
    first = trials[:4]
    best_i = min(range(4), key=lambda i: first[i].score)
    worst_i = max(range(4), key=lambda i: (first[i].score, i))
    assert received[:4] == [None, None, None, None]
    assert received[4 + best_i][0] == id(stored[best_i])
    assert received[4 + worst_i][0] != id(stored[best_i])
    assert received[4 + worst_i][1] == stored[best_i]["token"]
    assert received[4 + worst_i][2] == pytest.approx(first[best_i].params["x"])


def test_pbt_scalar_objective_still_copies_checkpoint():
    space = parse_space({"x": [0.0, 1.0]})
    received = []

    def objective(params, resource=1.0, weights=None):
        received.append(weights)
        return float(params["x"])

    trials = pbt_search(
        space,
        objective,
        budget=8,
        rng=np.random.default_rng(2),
        population_size=4,
        perturbation_factors=(1.0,),
        resample_probability=0.0,
    )
    first = trials[:4]
    best_i = min(range(4), key=lambda i: first[i].score)
    worst_i = max(range(4), key=lambda i: (first[i].score, i))
    assert all(item is None for item in received[:4])
    copied = received[4 + worst_i]
    assert copied["step"] == 1
    assert copied["params"]["x"] == pytest.approx(first[best_i].params["x"])
    assert copied["resource"] == pytest.approx(first[best_i].resource)


def test_pbt_exploit_interval_waits_for_completed_steps():
    space = parse_space({"x": [0.0, 1.0]})
    trials = pbt_search(
        space,
        lambda p, resource=1.0: float(p["x"]),
        budget=12,
        rng=np.random.default_rng(6),
        population_size=4,
        exploit_interval=2,
        perturbation_factors=(0.5,),
        resample_probability=0.0,
    )
    assert [t.params["x"] for t in trials[:4]] == pytest.approx(
        [t.params["x"] for t in trials[4:8]]
    )
    first = trials[:4]
    third = trials[8:12]
    worst_i = max(range(4), key=lambda i: (first[i].score, i))
    best_i = min(range(4), key=lambda i: first[i].score)
    expected = float(np.clip(first[best_i].params["x"] * 0.5, 0.0, 1.0))
    assert third[worst_i].params["x"] == pytest.approx(expected)


def test_pbt_search_budget_smaller_than_population():
    trials = pbt_search(
        _space(),
        _obj,
        budget=3,
        rng=np.random.default_rng(1),
        population_size=8,
    )
    assert len(trials) == 3
    assert all(t.resource == pytest.approx(1.0 / 9.0) for t in trials)


def test_pbt_search_rejects_bad_budget_and_space():
    with pytest.raises(ValueError):
        pbt_search(_space(), _obj, 0, rng=np.random.default_rng(0))
    with pytest.raises(ValueError):
        pbt_search({}, _obj, 5, rng=np.random.default_rng(0))
    with pytest.raises(ValueError):
        pbt_search(_space(), _obj, 5, population_size=1, rng=np.random.default_rng(0))
    with pytest.raises(ValueError):
        pbt_search(_space(), _obj, 5, exploit_interval=0, rng=np.random.default_rng(0))
    with pytest.raises(ValueError):
        pbt_search(_space(), _obj, 5, quantile=0.0, rng=np.random.default_rng(0))
    with pytest.raises(ValueError):
        pbt_search(_space(), _obj, 5, quantile=0.75, rng=np.random.default_rng(0))
    with pytest.raises(ValueError):
        pbt_search(_space(), _obj, 5, eta=1, rng=np.random.default_rng(0))
    with pytest.raises(ValueError):
        pbt_search(
            _space(), _obj, 5, perturbation_factors=(-1.0,), rng=np.random.default_rng(0)
        )
    with pytest.raises(TypeError):
        pbt_search(
            _space(),
            lambda p: (p, p),
            2,
            rng=np.random.default_rng(0),
            population_size=2,
        )


def test_pbt_search_mixed_space_stays_in_domain():
    trials = pbt_search(
        _mixed_space(), _obj, budget=12, rng=np.random.default_rng(3), population_size=4
    )
    assert all(0.0 <= t.params["x"] <= 1.0 for t in trials)
    assert all(isinstance(t.params["y"], int) and 0 <= t.params["y"] <= 10 for t in trials)
    assert all(t.params["z"] in ("a", "b", "c") for t in trials)


def test_pbt_search_finds_convex_minimum():
    space = parse_space({"x": [0.0, 1.0]})

    def objective(params, resource=1.0):
        return (params["x"] - 0.3) ** 2

    trials = pbt_search(
        space,
        objective,
        budget=24,
        rng=np.random.default_rng(7),
        population_size=4,
        perturbation_factors=(0.8, 1.2),
    )
    assert isinstance(trials.best, Trial)
    assert trials.best.score < 0.02
    assert min(t.score for t in trials) < 0.02


def test_pbt_search_early_stop_on_floor():
    space = parse_space({"x": [0.0, 1.0]})

    def stopper(best, n):
        return best <= 0.0 or n >= 1

    trials = pbt_search(
        space,
        lambda p, resource=1.0: 0.0,
        budget=20,
        rng=np.random.default_rng(0),
        population_size=4,
        stop_when=stopper,
    )
    assert len(trials) == 1


def test_pbt_search_with_none_rng():
    trials = pbt_search(parse_space({"x": [0.0, 1.0]}), _obj, 8)
    assert len(trials) == 8
    assert trials.best.score <= max(t.score for t in trials)


def test_pbt_best_trial_on_empty_raises():
    with pytest.raises(ValueError):
        pbt_best_trial([])


def test_pbt_passes_fidelity_alias():
    seen = []

    def objective(params, fidelity):
        seen.append(fidelity)
        return float(params["x"])

    pbt_search(
        parse_space({"x": [0.0, 1.0]}),
        objective,
        budget=2,
        rng=np.random.default_rng(0),
        population_size=2,
        min_resource=1,
        max_resource=1,
    )
    assert seen == [1.0, 1.0]


def test_run_search_pbt():
    result = run_search(
        "pbt",
        _space(),
        _obj,
        budget=12,
        seed=3,
        population_size=4,
        exploit_interval=1,
    )
    assert result.strategy == "pbt"
    assert len(result.trials) == 12
    assert result.best_score == best_trial(result.trials).score
    assert all(t.resource is not None for t in result.trials)


def test_run_search_forwarded_pbt_kwargs():
    a = run_search(
        "pbt",
        parse_space({"x": [0.0, 1.0]}),
        lambda p, resource=1.0: float(p["x"]),
        8,
        seed=1,
        population_size=4,
        perturbation_factors=(1.0,),
        resample_probability=0.0,
    )
    b = run_search(
        "pbt",
        parse_space({"x": [0.0, 1.0]}),
        lambda p, resource=1.0: float(p["x"]),
        8,
        seed=1,
        population_size=4,
        perturbation_factors=(0.5,),
        resample_probability=0.0,
    )
    assert [t.params for t in a.trials[:4]] == [t.params for t in b.trials[:4]]
    assert [t.params for t in a.trials[4:]] != [t.params for t in b.trials[4:]]


def test_compare_strategies_can_select_pbt():
    results = compare_strategies(
        _space(),
        _obj,
        8,
        strategies=("random", "pbt"),
        seed=2,
        strategy_kwargs={"pbt": {"population_size": 4, "exploit_interval": 1}},
    )
    assert set(results) == {"random", "pbt"}
    assert len(results["pbt"].trials) == 8
    assert all(t.resource is not None for t in results["pbt"].trials)
