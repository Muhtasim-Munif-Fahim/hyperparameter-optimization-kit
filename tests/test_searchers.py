import numpy as np
import pytest

from hyperopt_kit.searchers import Trial, grid_search, random_search
from hyperopt_kit.spaces import Categorical, FloatRange, IntRange, parse_space


def _obj(params):
    return float(params.get("x", 0.0)) + float(params.get("y", 0.0))


def _categorical_obj(params):
    return float(ord(params["x"])) + float(ord(params["y"]))


def _mixed_space():
    return parse_space(
        {
            "x": [0.0, 1.0],
            "y": [0, 10, "int"],
            "z": ["a", "b", "c"],
        }
    )


def test_random_search_respects_budget():
    space = _mixed_space()
    trials = random_search(space, _obj, budget=25, rng=np.random.default_rng(0))
    assert len(trials) == 25
    for i, trial in enumerate(trials):
        assert trial.iteration == i
        assert isinstance(trial.score, float)
        assert set(trial.params) == {"x", "y", "z"}


def test_random_search_deterministic_with_seed():
    space = _mixed_space()
    a = random_search(space, _obj, 20, rng=np.random.default_rng(11))
    b = random_search(space, _obj, 20, rng=np.random.default_rng(11))
    assert [t.params for t in a] == [t.params for t in b]
    assert [t.score for t in a] == [t.score for t in b]


def test_random_search_differs_across_seeds():
    space = _mixed_space()
    a = random_search(space, _obj, 20, rng=np.random.default_rng(11))
    b = random_search(space, _obj, 20, rng=np.random.default_rng(99))
    assert [t.params for t in a] != [t.params for t in b]


def test_random_search_rejects_budget_zero():
    with pytest.raises(ValueError):
        random_search(_mixed_space(), _obj, 0, rng=np.random.default_rng(0))


def test_random_search_empty_space_rejected():
    with pytest.raises(ValueError):
        random_search({}, _obj, 5, rng=np.random.default_rng(0))
    with pytest.raises(ValueError):
        random_search({"x": "not-a-space"}, _obj, 5, rng=np.random.default_rng(0))


def test_grid_search_exhausts_small_grid():
    space = parse_space({"a": ["a1", "a2"], "b": ["b1", "b2", "b3"]})
    trials = grid_search(space, _obj, budget=100)
    assert len(trials) == 6
    assert {t.params["a"] for t in trials} == {"a1", "a2"}
    assert {t.params["b"] for t in trials} == {"b1", "b2", "b3"}


def test_grid_search_respects_budget_cap():
    space = parse_space({"a": [0.0, 1.0], "b": [0.0, 1.0]})
    trials = grid_search(space, _obj, budget=5)
    assert len(trials) == 5


def test_grid_search_deterministic():
    space = _mixed_space()
    a = grid_search(space, _obj, budget=100)
    b = grid_search(space, _obj, budget=100)
    assert [t.params for t in a] == [t.params for t in b]
    assert [t.iteration for t in a] == [t.iteration for t in b]


def test_grid_search_on_categorical_only_space():
    space = parse_space({"x": ["a", "b"], "y": ["c", "d", "e"]})
    trials = grid_search(space, _categorical_obj)
    assert len(trials) == 6
    assert len({(t.params["x"], t.params["y"]) for t in trials}) == 6


def test_grid_search_on_single_variable_space():
    space = parse_space({"x": [0.0, 1.0]})
    trials = grid_search(space, _obj, grid_points_per_dim=3)
    assert len(trials) == 3
    xs = [t.params["x"] for t in trials]
    assert xs == [0.0, 0.5, 1.0]


def test_grid_search_respects_log_scale_positions():
    space = parse_space({"lr": [1e-2, 1e2, "log"]})
    trials = grid_search(space, _obj, budget=4)
    values = sorted(t.params["lr"] for t in trials)
    assert values[0] == pytest.approx(1e-2)
    assert values[3] == pytest.approx(1e2)
    assert values[1] == pytest.approx(10 ** (-2.0 / 3.0))
    assert values[2] == pytest.approx(10 ** (2.0 / 3.0))


def test_grid_search_int_range_values():
    space = parse_space({"n": [0, 5, "int"]})
    trials = grid_search(space, _obj, grid_points_per_dim=6)
    assert sorted(t.params["n"] for t in trials) == [0, 1, 2, 3, 4, 5]


def test_grid_search_wide_int_range_is_capped():
    space = parse_space({"n": [0, 1000, "int"]})
    trials = grid_search(space, _obj, budget=1000)
    assert len(trials) == 4
    assert all(0 <= t.params["n"] <= 1000 for t in trials)


def test_grid_search_rejects_bad_budget():
    with pytest.raises(ValueError):
        grid_search(_mixed_space(), _obj, budget=0)


def test_grid_search_tracks_best_so_far():
    space = parse_space({"x": [0.0, 1.0]})
    trials = grid_search(space, lambda p: (p["x"] - 0.5) ** 2, grid_points_per_dim=3)
    scores = [t.score for t in trials]
    assert min(scores) == pytest.approx(0.0)


def test_grid_search_grid_points_per_dim_parameter():
    space = parse_space({"x": [0.0, 1.0]})
    trials = grid_search(space, _obj, grid_points_per_dim=2)
    assert len(trials) == 2


def test_random_search_categorical_values_valid():
    space = _mixed_space()
    trials = random_search(space, _obj, 50, rng=np.random.default_rng(3))
    assert all(t.params["z"] in ("a", "b", "c") for t in trials)
    assert all(isinstance(t.params["y"], int) for t in trials)


def test_random_search_with_none_rng():
    space = parse_space({"x": [0.0, 1.0]})
    trials = random_search(space, _obj, 10)
    assert len(trials) == 10


def test_trial_dataclass_equality():
    t1 = Trial(0, {"x": 1.0}, 2.0)
    t2 = Trial(0, {"x": 1.0}, 2.0)
    assert t1 == t2
    assert t1.params == {"x": 1.0}
