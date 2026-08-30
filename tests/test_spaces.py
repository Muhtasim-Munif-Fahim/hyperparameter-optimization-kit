import math

import numpy as np
import pytest

from hyperopt_kit.spaces import (
    Categorical,
    FloatRange,
    IntRange,
    Space,
    parse_space,
)


def test_categorical_sample_within_choices():
    rng = np.random.default_rng(0)
    sp = Categorical(["gini", "entropy", "log_loss"])
    for _ in range(300):
        assert sp.sample(rng) in sp.choices


def test_categorical_rejects_empty_choices():
    with pytest.raises(ValueError):
        Categorical([])


def test_categorical_single_choice_is_constant():
    rng = np.random.default_rng(0)
    sp = Categorical(["only"])
    for _ in range(20):
        assert sp.sample(rng) == "only"


def test_categorical_mutate_moves_and_stays_within_choices():
    rng = np.random.default_rng(1)
    sp = Categorical(["a", "b", "c", "d"])
    for _ in range(200):
        value = sp.sample(rng)
        neighbour = sp.mutate(value, rng)
        assert neighbour in sp.choices
        assert neighbour != value


def test_categorical_mutate_on_single_choice_returns_same():
    rng = np.random.default_rng(1)
    sp = Categorical(["only"])
    assert sp.mutate("only", rng) == "only"


def test_categorical_normalize_denormalize_round_trip():
    sp = Categorical(["a", "b", "c"])
    for choice in sp.choices:
        assert sp.denormalize(sp.normalize(choice)) == choice


def test_categorical_normalize_maps_edges():
    sp = Categorical(["a", "b", "c"])
    assert sp.normalize("a") == pytest.approx(0.0)
    assert sp.normalize("c") == pytest.approx(1.0)
    assert sp.denormalize(0.0) == "a"
    assert sp.denormalize(1.0) == "c"


def test_categorical_denormalize_clips_out_of_range():
    sp = Categorical(["a", "b"])
    assert sp.denormalize(-1.0) == "a"
    assert sp.denormalize(2.0) == "b"


def test_float_sample_within_bounds():
    rng = np.random.default_rng(2)
    sp = FloatRange(-3.0, 7.5)
    for _ in range(500):
        value = sp.sample(rng)
        assert -3.0 <= value <= 7.5


def test_float_rejects_high_below_low():
    with pytest.raises(ValueError):
        FloatRange(5.0, 1.0)


def test_float_degenerate_range_is_constant():
    rng = np.random.default_rng(2)
    sp = FloatRange(2.0, 2.0)
    for _ in range(20):
        assert sp.sample(rng) == 2.0
    assert sp.normalize(2.0) == pytest.approx(0.5)
    assert sp.denormalize(0.3) == 2.0


def test_float_log_scale_sample_within_bounds():
    rng = np.random.default_rng(3)
    sp = FloatRange(1e-3, 1e2, log_scale=True)
    for _ in range(500):
        value = sp.sample(rng)
        assert 1e-3 <= value <= 1e2


def test_float_log_scale_distribution_spans_orders_of_magnitude():
    rng = np.random.default_rng(3)
    sp = FloatRange(1e-3, 1e2, log_scale=True)
    values = [sp.sample(rng) for _ in range(1000)]
    assert min(values) < 0.1
    assert max(values) > 10.0


def test_float_log_scale_rejects_non_positive_low():
    with pytest.raises(ValueError):
        FloatRange(0.0, 10.0, log_scale=True)


def test_float_normalize_denormalize_round_trip_linear():
    rng = np.random.default_rng(4)
    sp = FloatRange(-2.0, 4.0)
    for _ in range(100):
        value = sp.sample(rng)
        assert sp.denormalize(sp.normalize(value)) == pytest.approx(value)


def test_float_normalize_denormalize_round_trip_log():
    rng = np.random.default_rng(4)
    sp = FloatRange(1e-2, 1e3, log_scale=True)
    for _ in range(100):
        value = sp.sample(rng)
        assert sp.denormalize(sp.normalize(value)) == pytest.approx(value)


def test_float_normalize_maps_bounds():
    sp = FloatRange(-2.0, 4.0)
    assert sp.normalize(-2.0) == pytest.approx(0.0)
    assert sp.normalize(4.0) == pytest.approx(1.0)
    sp_log = FloatRange(1.0, 100.0, log_scale=True)
    assert sp_log.normalize(1.0) == pytest.approx(0.0)
    assert sp_log.normalize(100.0) == pytest.approx(1.0)


def test_float_mutate_stays_within_bounds():
    rng = np.random.default_rng(5)
    sp = FloatRange(0.0, 1.0)
    for _ in range(300):
        value = sp.sample(rng)
        assert 0.0 <= sp.mutate(value, rng) <= 1.0


def test_float_mutate_stays_within_log_bounds():
    rng = np.random.default_rng(5)
    sp = FloatRange(1e-2, 1e2, log_scale=True)
    for _ in range(300):
        value = sp.sample(rng)
        assert 1e-2 <= sp.mutate(value, rng) <= 1e2


def test_int_sample_within_inclusive_range():
    rng = np.random.default_rng(6)
    sp = IntRange(1, 10)
    for _ in range(500):
        value = sp.sample(rng)
        assert 1 <= value <= 10
        assert isinstance(value, int)


def test_int_rejects_high_below_low():
    with pytest.raises(ValueError):
        IntRange(3, 1)


def test_int_sample_covers_degenerate_range():
    rng = np.random.default_rng(6)
    sp = IntRange(4, 4)
    assert sp.n_values == 1
    assert sp.sample(rng) == 4


def test_int_mutate_steps_by_one():
    rng = np.random.default_rng(7)
    sp = IntRange(0, 10)
    for _ in range(300):
        value = sp.sample(rng)
        neighbour = sp.mutate(value, rng)
        assert 0 <= neighbour <= 10
        assert abs(neighbour - value) == 1


def test_int_normalize_denormalize_round_trip():
    sp = IntRange(5, 15)
    for value in range(5, 16):
        assert sp.denormalize(sp.normalize(value)) == value


def test_int_log_scale_sample_within_bounds():
    rng = np.random.default_rng(8)
    sp = IntRange(1, 1000, log_scale=True)
    for _ in range(300):
        value = sp.sample(rng)
        assert 1 <= value <= 1000
    assert sp.denormalize(sp.normalize(1)) == 1
    assert sp.denormalize(sp.normalize(1000)) == 1000


def test_parse_space_compact_forms():
    space = parse_space(
        {
            "lr": [0.001, 0.1, "log"],
            "depth": [2, 8, "int"],
            "units": [16, 256, "int", "log"],
            "kernel": ["rbf", "linear"],
        }
    )
    assert isinstance(space["lr"], FloatRange)
    assert space["lr"].log_scale is True
    assert isinstance(space["depth"], IntRange)
    assert space["depth"].log_scale is False
    assert isinstance(space["units"], IntRange)
    assert space["units"].log_scale is True
    assert isinstance(space["kernel"], Categorical)
    assert space["kernel"].choices == ["rbf", "linear"]


def test_parse_space_plain_number_pairs_are_float_ranges():
    space = parse_space({"a": [0.0, 1.0], "b": [1, 10]})
    assert isinstance(space["a"], FloatRange)
    assert isinstance(space["b"], FloatRange)
    assert space["b"].low == 1.0
    assert space["b"].high == 10.0


def test_parse_space_dict_forms():
    space = parse_space(
        {
            "x": {"type": "float", "low": 1e-3, "high": 1.0, "log": True},
            "y": {"type": "int", "low": 1, "high": 5},
            "z": {"type": "categorical", "choices": ["a", "b"]},
        }
    )
    assert isinstance(space["x"], FloatRange)
    assert space["x"].log_scale is True
    assert isinstance(space["y"], IntRange)
    assert isinstance(space["z"], Categorical)


def test_parse_space_rejects_empty_and_invalid():
    with pytest.raises(ValueError):
        parse_space({})
    with pytest.raises(ValueError):
        parse_space("nope")
    with pytest.raises(ValueError):
        parse_space({"a": "oops"})
    with pytest.raises(ValueError):
        parse_space({"a": {"type": "unknown", "low": 0, "high": 1}})


def test_parse_space_log_range_with_non_positive_low_raises():
    with pytest.raises(ValueError):
        parse_space({"a": [0.0, 1.0, "log"]})
    with pytest.raises(ValueError):
        parse_space({"a": {"type": "float", "low": 0.0, "high": 1.0, "log": True}})


def test_parse_space_categorical_of_numbers():
    space = parse_space({"mode": [1, 2, 3]})
    assert isinstance(space["mode"], Categorical)
    assert space["mode"].choices == [1, 2, 3]


def test_parse_space_invalid_range_raises():
    with pytest.raises(ValueError):
        parse_space({"a": [5.0, 1.0]})


def test_sampling_is_reproducible_with_seed():
    space = parse_space(
        {"x": [0.0, 1.0], "y": [0, 10, "int"], "z": ["a", "b", "c"]}
    )
    a = [sp.sample(np.random.default_rng(42)) for sp in space.values()]
    b = [sp.sample(np.random.default_rng(42)) for sp in space.values()]
    assert a == b


def test_space_base_class_is_abstract():
    with pytest.raises(NotImplementedError):
        Space().sample(np.random.default_rng(0))
    with pytest.raises(NotImplementedError):
        Space().normalize(1.0)
