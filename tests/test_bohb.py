import numpy as np
import pytest

from hyperopt_kit.evaluate import compare_strategies, run_search
from hyperopt_kit.searchers import (
    ProductKernelDensity,
    Trial,
    bohb_log_density_ratio,
    bohb_propose,
    bohb_search,
    bohb_select_trials,
    hyperband_search,
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


def test_scott_bandwidth_matches_normal_reference_rule():
    samples = np.array([[0.0], [0.25], [0.5], [1.0]])
    kde = ProductKernelDensity.fit(samples, "c", [0])
    std = float(np.std(samples[:, 0]))
    expected = 1.06 * std * (4.0 ** (-1.0 / 5.0))
    assert kde.bandwidth[0] == pytest.approx(expected)
    assert kde.kinds == "c"


def test_zero_variance_bandwidth_floors_at_minimum():
    kde = ProductKernelDensity.fit(np.full((5, 1), 0.3), "c", [0], min_bandwidth=1e-3)
    assert kde.bandwidth[0] == pytest.approx(1e-3)


def test_categorical_bandwidth_capped_at_uniform_kernel():
    samples = np.array([[0.0], [0.0], [3.0], [3.0]])
    kde = ProductKernelDensity.fit(samples, "u", [4], min_bandwidth=1e-3)
    assert kde.bandwidth[0] == pytest.approx(3.0 / 4.0)
    assert kde.bandwidth[0] <= 0.75 + 1e-12


def test_product_kernel_density_higher_near_observations():
    samples = np.full((12, 1), 0.2)
    kde = ProductKernelDensity.fit(samples, "c", [0])
    near = float(kde.logpdf(np.array([[0.2]]))[0])
    far = float(kde.logpdf(np.array([[0.9]]))[0])
    assert near > far
    assert np.isfinite(near) and np.isfinite(far)


def test_product_kernel_prefers_observed_joint_over_swapped_pair():
    """A product of kernels centered on joint rows captures correlation.

    TPE multiplies independent marginals, so (0.15, 0.85) and (0.15, 0.15)
    look alike when both coordinates are observed near both ends. The BOHB
    kernel is centered on the observed pairs and should prefer the pairs
    that were actually seen.
    """
    samples = np.array(
        [
            [0.15, 0.85],
            [0.18, 0.82],
            [0.12, 0.88],
            [0.85, 0.15],
            [0.82, 0.18],
            [0.88, 0.12],
        ]
    )
    kde = ProductKernelDensity.fit(samples, "cc", [0, 0])
    on_pair = float(kde.logpdf(np.array([[0.15, 0.85]]))[0])
    swapped = float(kde.logpdf(np.array([[0.15, 0.15]]))[0])
    assert on_pair > swapped


def test_categorical_kernel_prefers_majority_level():
    samples = np.array([[0.0], [0.0], [0.0], [0.0], [1.0]])
    kde = ProductKernelDensity.fit(samples, "u", [2])
    majority = float(kde.logpdf(np.array([[0.0]]))[0])
    minority = float(kde.logpdf(np.array([[1.0]]))[0])
    assert majority > minority


def test_product_kernel_sample_respects_bounds_and_widening():
    rng = np.random.default_rng(0)
    tight = np.full((8, 1), 0.5)
    kde = ProductKernelDensity.fit(tight, "c", [0])
    narrow = kde.sample(rng, 200, bandwidth_factor=1.0)
    wide = kde.sample(np.random.default_rng(0), 200, bandwidth_factor=5.0)
    assert narrow.shape == (200, 1)
    assert np.all((narrow >= 0.0) & (narrow <= 1.0))
    assert np.all((wide >= 0.0) & (wide <= 1.0))
    assert float(np.std(wide)) > float(np.std(narrow))


def test_categorical_sample_stays_inside_choice_set():
    rng = np.random.default_rng(1)
    samples = np.array([[0.0], [1.0], [1.0], [2.0]])
    kde = ProductKernelDensity.fit(samples, "u", [3])
    drawn = kde.sample(rng, 80, bandwidth_factor=3.0)
    assert set(np.rint(drawn[:, 0]).astype(int)).issubset({0, 1, 2})


def test_product_kernel_rejects_bad_inputs():
    with pytest.raises(ValueError):
        ProductKernelDensity.fit(np.zeros((0, 1)), "c", [0])
    with pytest.raises(ValueError):
        ProductKernelDensity.fit(np.zeros((3, 2)), "c", [0, 0])
    with pytest.raises(ValueError):
        ProductKernelDensity.fit(np.zeros((3, 1)), "u", [0])
    kde = ProductKernelDensity.fit(np.zeros((3, 1)) + 0.2, "c", [0])
    with pytest.raises(ValueError):
        kde.logpdf(np.zeros((2, 2)))
    with pytest.raises(ValueError):
        kde.sample(np.random.default_rng(0), 0)


def test_bohb_select_good_slice_is_the_best_observations():
    trials = [
        Trial(i, {"x": i / 10.0}, score=float(i), resource=1.0) for i in range(10)
    ]
    selected = bohb_select_trials(trials, 1, top_n_percent=15, min_points_in_model=2)
    assert selected is not None
    good, bad = selected
    assert [t.score for t in good] == [0.0, 1.0]
    assert [t.score for t in bad] == [float(i) for i in range(2, 10)]


def test_bohb_select_needs_more_bad_points_than_the_dimension():
    trials = [
        Trial(i, {"x": 0.1, "y": 0.2}, score=float(i), resource=1.0) for i in range(5)
    ]
    assert bohb_select_trials(trials, 2) is None
    trials.append(Trial(5, {"x": 0.2, "y": 0.3}, score=5.0, resource=1.0))
    assert bohb_select_trials(trials, 2) is not None


def test_bohb_select_uses_the_largest_fidelity_that_can_fit():
    xs = [0.1, 0.2, 0.3, 0.8, 0.9, 0.7]
    low = [
        Trial(i, {"x": x}, score=x, resource=1.0 / 9.0) for i, x in enumerate(xs)
    ]
    high_few = [
        Trial(10, {"x": 0.0}, score=0.0, resource=1.0),
        Trial(11, {"x": 0.4}, score=0.4, resource=1.0),
    ]
    selected = bohb_select_trials(low + high_few, 1, min_points_in_model=2)
    assert selected is not None
    good, bad = selected
    assert all(abs(t.resource - 1.0 / 9.0) < 1e-8 for t in good + bad)

    high = [
        Trial(20 + i, {"x": x}, score=x, resource=1.0) for i, x in enumerate(xs)
    ]
    selected = bohb_select_trials(low + high_few + high, 1, min_points_in_model=2)
    assert selected is not None
    good, bad = selected
    assert all(abs(t.resource - 1.0) < 1e-8 for t in good + bad)


def test_bohb_select_groups_missing_resource_as_full_fidelity():
    trials = [Trial(i, {"x": i / 10.0}, score=float(i)) for i in range(8)]
    selected = bohb_select_trials(trials, 1, min_points_in_model=2)
    assert selected is not None
    good, bad = selected
    assert good and bad


def test_bohb_log_density_ratio_prefers_good_region():
    space = parse_space({"x": [0.0, 1.0]})
    xs = [0.10, 0.12, 0.80, 0.85, 0.90, 0.70, 0.75, 0.88]
    trials = [Trial(i, {"x": x}, score=x, resource=1.0) for i, x in enumerate(xs)]
    good = bohb_log_density_ratio(
        {"x": 0.11}, space, trials, min_points_in_model=2
    )
    bad = bohb_log_density_ratio(
        {"x": 0.9}, space, trials, min_points_in_model=2
    )
    assert good > bad
    assert np.isfinite(good) and np.isfinite(bad)


def test_bohb_log_density_ratio_needs_a_fittable_model():
    space = parse_space({"x": [0.0, 1.0]})
    trials = [Trial(0, {"x": 0.2}, 0.2, resource=1.0)]
    with pytest.raises(ValueError):
        bohb_log_density_ratio({"x": 0.2}, space, trials, min_points_in_model=2)


def test_bohb_propose_concentrates_on_the_good_region():
    space = parse_space({"x": [0.0, 1.0]})
    xs = [0.10, 0.12, 0.80, 0.85, 0.90, 0.70, 0.75, 0.88]
    trials = [Trial(i, {"x": x}, score=x, resource=1.0) for i, x in enumerate(xs)]
    rng = np.random.default_rng(0)
    drawn = [
        bohb_propose(
            space,
            trials,
            rng,
            random_fraction=0.0,
            n_candidates=32,
            min_points_in_model=2,
            bandwidth_factor=1.0,
        )["x"]
        for _ in range(25)
    ]
    assert float(np.median(drawn)) < 0.4


def test_bohb_propose_random_fraction_ignores_the_model():
    space = parse_space({"x": [0.0, 1.0]})
    trials = [Trial(i, {"x": 0.1}, score=float(i), resource=1.0) for i in range(8)]
    rng = np.random.default_rng(4)
    drawn = [
        bohb_propose(space, trials, rng, random_fraction=1.0, min_points_in_model=2)["x"]
        for _ in range(30)
    ]
    assert float(np.mean(drawn)) > 0.3


def test_bohb_search_respects_budget():
    trials = bohb_search(_space(), _obj, budget=17, rng=np.random.default_rng(0))
    assert len(trials) == 17
    for i, trial in enumerate(trials):
        assert trial.iteration == i
        assert trial.resource is not None
        assert 0.0 < trial.resource <= 1.0 + 1e-12


def test_bohb_search_deterministic_with_seed():
    a = bohb_search(_space(), _obj, 18, rng=np.random.default_rng(11))
    b = bohb_search(_space(), _obj, 18, rng=np.random.default_rng(11))
    assert [t.params for t in a] == [t.params for t in b]
    assert [t.score for t in a] == [t.score for t in b]
    assert [t.resource for t in a] == [t.resource for t in b]


def test_bohb_search_differs_across_seeds():
    a = bohb_search(_space(), _obj, 18, rng=np.random.default_rng(11))
    b = bohb_search(_space(), _obj, 18, rng=np.random.default_rng(99))
    assert [t.params for t in a] != [t.params for t in b]


def test_bohb_search_departs_from_uniform_hyperband():
    space = parse_space({"x": [0.0, 1.0]})
    objective = lambda p, resource=1.0: float(p["x"])
    hyperband = hyperband_search(space, objective, 24, rng=np.random.default_rng(1))
    bohb = bohb_search(
        space,
        objective,
        24,
        rng=np.random.default_rng(1),
        random_fraction=0.0,
    )
    assert [t.params for t in hyperband] != [t.params for t in bohb]


def test_bohb_promotes_better_configs():
    space = parse_space({"x": [0.0, 1.0]})
    calls = []

    def obj(params, resource=1.0):
        calls.append((params["x"], resource))
        return float(params["x"])

    bohb_search(
        space,
        obj,
        budget=13,
        rng=np.random.default_rng(0),
        eta=3,
        min_resource=1,
        max_resource=9,
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


def test_bohb_uses_multiple_starting_fidelities():
    space = parse_space({"x": [0.0, 1.0]})
    first_resource = {}

    def obj(params, resource=1.0):
        key = round(float(params["x"]), 12)
        first_resource.setdefault(key, resource)
        return float(params["x"])

    bohb_search(
        space, obj, budget=22, rng=np.random.default_rng(2), eta=3, max_resource=9
    )
    starts = sorted({round(r, 10) for r in first_resource.values()})
    assert starts == pytest.approx([1.0 / 9.0, 1.0 / 3.0, 1.0])


def test_bohb_rejects_bad_budget_and_knobs():
    with pytest.raises(ValueError):
        bohb_search(_space(), _obj, 0, rng=np.random.default_rng(0))
    with pytest.raises(ValueError):
        bohb_search({}, _obj, 5, rng=np.random.default_rng(0))
    with pytest.raises(ValueError):
        bohb_search(_space(), _obj, 5, eta=1, rng=np.random.default_rng(0))
    with pytest.raises(ValueError):
        bohb_search(_space(), _obj, 5, top_n_percent=0, rng=np.random.default_rng(0))
    with pytest.raises(ValueError):
        bohb_search(_space(), _obj, 5, top_n_percent=100, rng=np.random.default_rng(0))
    with pytest.raises(ValueError):
        bohb_search(_space(), _obj, 5, min_points_in_model=2, rng=np.random.default_rng(0))
    with pytest.raises(ValueError):
        bohb_search(_space(), _obj, 5, n_candidates=0, rng=np.random.default_rng(0))
    with pytest.raises(ValueError):
        bohb_search(_space(), _obj, 5, random_fraction=1.5, rng=np.random.default_rng(0))
    with pytest.raises(ValueError):
        bohb_search(_space(), _obj, 5, bandwidth_factor=0.0, rng=np.random.default_rng(0))
    with pytest.raises(ValueError):
        bohb_search(_space(), _obj, 5, min_bandwidth=0.0, rng=np.random.default_rng(0))


def test_bohb_mixed_and_log_scale_spaces():
    trials = bohb_search(_mixed_space(), _obj, budget=12, rng=np.random.default_rng(3))
    assert all(t.params["z"] in ("a", "b", "c") for t in trials)
    assert all(isinstance(t.params["y"], int) for t in trials)
    space = parse_space({"lr": [1e-3, 1.0, "log"]})
    trials = bohb_search(
        space,
        lambda p, resource=1.0: (np.log10(p["lr"]) + 2.0) ** 2,
        budget=10,
        rng=np.random.default_rng(4),
    )
    assert all(1e-3 - 1e-12 <= t.params["lr"] <= 1.0 + 1e-12 for t in trials)


def test_bohb_single_arg_objective_and_early_stop():
    trials = bohb_search(
        parse_space({"x": [0.0, 1.0]}),
        lambda p: (p["x"] - 0.25) ** 2,
        budget=10,
        rng=np.random.default_rng(4),
    )
    assert len(trials) == 10

    def stopper(best, n):
        return best <= 0.0 or n >= 4

    stopped = bohb_search(
        parse_space({"x": [0.0, 1.0]}),
        lambda p, resource=1.0: 0.0,
        budget=20,
        rng=np.random.default_rng(0),
        stop_when=stopper,
    )
    assert len(stopped) <= 4


def test_bohb_search_with_none_rng():
    trials = bohb_search(parse_space({"x": [0.0, 1.0]}), _obj, 8)
    assert len(trials) == 8


def test_run_search_bohb_and_forwarded_kwargs():
    result = run_search("bohb", _space(), _obj, budget=12, seed=3)
    assert result.strategy == "bohb"
    assert len(result.trials) == 12
    assert result.best_score == min(
        t.score
        for t in result.trials
        if t.resource is not None
        and t.resource
        >= max(tr.resource for tr in result.trials if tr.resource is not None) - 1e-12
    )
    space = parse_space({"x": [0.0, 1.0]})
    guided = run_search(
        "bohb", space, _obj, 16, seed=1, random_fraction=0.0, top_n_percent=20
    )
    uniform = run_search(
        "bohb", space, _obj, 16, seed=1, random_fraction=1.0, top_n_percent=20
    )
    assert [t.params for t in guided.trials] != [t.params for t in uniform.trials]


def test_compare_strategies_includes_bohb():
    results = compare_strategies(
        _space(), _obj, 8, strategies=("random", "hyperband", "bohb"), seed=2
    )
    assert set(results) == {"random", "hyperband", "bohb"}
    for result in results.values():
        assert len(result.trials) == 8
