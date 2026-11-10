"""Tests for SPSA search."""

from __future__ import annotations

import numpy as np
import pytest

from hyperopt_kit import spsa_gains, spsa_search
from hyperopt_kit.evaluate import available_strategies, run_search
from hyperopt_kit.searchers import random_search
from hyperopt_kit.spaces import Categorical, FloatRange, IntRange


def _space(d=2):
    return {f"x{i}": FloatRange(0.0, 1.0) for i in range(d)}


def _sphere(params):
    return sum((float(v) - 0.3 - 0.1 * i) ** 2 for i, v in enumerate(params.values()))


def test_gains_follow_spall_schedule():
    a0, c0 = spsa_gains(0, 2.0, 0.1, 10.0)
    assert a0 == pytest.approx(2.0 / 11.0**0.602)
    assert c0 == pytest.approx(0.1)
    a5, c5 = spsa_gains(5, 2.0, 0.1, 10.0)
    assert a5 < a0 and c5 < c0


def test_budget_indices_and_bounds():
    trials = spsa_search(_space(), _sphere, 50, rng=np.random.default_rng(0))
    assert len(trials) == 50
    assert [t.iteration for t in trials] == list(range(50))
    assert all(0.0 <= v <= 1.0 for t in trials for v in t.params.values())


def test_deterministic_with_seed():
    a = spsa_search(_space(3), _sphere, 45, rng=np.random.default_rng(4))
    b = spsa_search(_space(3), _sphere, 45, rng=np.random.default_rng(4))
    assert [t.params for t in a] == [t.params for t in b]


def test_high_dimensional_sphere_beats_random():
    space = _space(10)
    sp, rs = [], []
    for seed in range(5):
        sp.append(min(t.score for t in spsa_search(space, _sphere, 150, rng=np.random.default_rng(seed))))
        rs.append(min(t.score for t in random_search(space, _sphere, 150, rng=np.random.default_rng(seed))))
    assert np.median(sp) < 0.5 * np.median(rs)


def test_noisy_objective_still_improves():
    rng_noise = np.random.default_rng(123)

    def noisy(p):
        return _sphere(p) + 0.01 * rng_noise.normal()

    trials = spsa_search(_space(4), noisy, 120, rng=np.random.default_rng(1))
    first = np.mean([t.score for t in trials[:10]])
    last = np.mean([t.score for t in trials[-15:]])
    assert last < first


def test_two_evaluations_per_iteration_mode():
    trials = spsa_search(_space(), _sphere, 20, rng=np.random.default_rng(0), evaluate_iterate=False)
    assert len(trials) == 20


def test_mixed_space_and_nonfinite():
    space = {"x": FloatRange(0.0, 1.0), "n": IntRange(1, 20), "kind": Categorical(["a", "b"])}

    def obj(p):
        if p["x"] > 0.95:
            return float("nan")
        return (p["x"] - 0.6) ** 2 + 0.01 * (p["n"] - 7) ** 2 + (0.0 if p["kind"] == "b" else 1.0)

    trials = spsa_search(space, obj, 90, rng=np.random.default_rng(0), patience=3)
    assert len(trials) == 90
    assert isinstance(trials[0].params["n"], (int, np.integer))
    assert {t.params["kind"] for t in trials} == {"a", "b"}  # restarts redraw categoricals


def test_categorical_only_falls_back_to_random():
    space = {"k": Categorical(["a", "b", "c"])}
    trials = spsa_search(space, lambda p: 0.0 if p["k"] == "a" else 1.0, 5, rng=np.random.default_rng(0))
    assert len(trials) == 5


def test_registered_strategy_and_stopper():
    assert "spsa" in available_strategies()
    res = run_search("spsa", _space(3), _sphere, 60, seed=0, c=0.05, evaluate_iterate=False)
    assert len(res.trials) == 60
    early = run_search("spsa", _space(3), _sphere, 200, seed=0, floor=10.0)
    assert len(early.trials) == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"c": 0.0},
        {"c": 1.5},
        {"initial_step": -1},
        {"alpha": 0.0},
        {"gamma": 0.7},
        {"stability": -1.0},
        {"patience": 0},
        {"tol": -1.0},
    ],
)
def test_invalid_params(kwargs):
    with pytest.raises(ValueError):
        spsa_search(_space(), _sphere, 10, rng=np.random.default_rng(0), **kwargs)
    with pytest.raises(ValueError):
        spsa_search(_space(), _sphere, 0)
