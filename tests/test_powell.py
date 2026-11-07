"""Tests for Powell's conjugate-direction search."""

from __future__ import annotations

import numpy as np
import pytest

from hyperopt_kit import powell, powell_search
from hyperopt_kit.evaluate import available_strategies, run_search
from hyperopt_kit.searchers import random_search
from hyperopt_kit.spaces import Categorical, FloatRange, IntRange


def _space(d=2):
    return {f"x{i}": FloatRange(0.0, 1.0) for i in range(d)}


def _sphere(params):
    return sum((float(v) - 0.3 - 0.1 * i) ** 2 for i, v in enumerate(params.values()))


def test_budget_indices_bounds_and_alias():
    assert powell is powell_search
    trials = powell_search(_space(), _sphere, 60, rng=np.random.default_rng(0))
    assert len(trials) == 60
    assert [t.iteration for t in trials] == list(range(60))
    assert all(0.0 <= v <= 1.0 for t in trials for v in t.params.values())


def test_deterministic_with_seed():
    a = powell_search(_space(3), _sphere, 40, rng=np.random.default_rng(7))
    b = powell_search(_space(3), _sphere, 40, rng=np.random.default_rng(7))
    assert [t.params for t in a] == [t.params for t in b]


def test_converges_on_sphere_and_beats_random():
    space = _space(3)
    pw_best, rs_best = [], []
    for seed in range(5):
        pw = powell_search(space, _sphere, 120, rng=np.random.default_rng(seed), n_initial=3)
        rs = random_search(space, _sphere, 120, rng=np.random.default_rng(seed))
        pw_best.append(min(t.score for t in pw))
        rs_best.append(min(t.score for t in rs))
    assert np.median(pw_best) < 1e-3
    assert np.median(pw_best) < np.median(rs_best)


def test_mixed_space_finds_integer_near_target():
    space = {"x": FloatRange(0.0, 1.0), "n": IntRange(1, 20), "kind": Categorical(["a", "b", "c"])}

    def obj(p):
        return (p["x"] - 0.6) ** 2 + 0.01 * (p["n"] - 7) ** 2 + (0.0 if p["kind"] == "b" else 1.0)

    trials = powell_search(space, obj, 180, rng=np.random.default_rng(2), n_initial=5)
    best = min(trials, key=lambda t: t.score)
    assert best.score < 0.5  # should find kind=b eventually via restarts
    assert isinstance(best.params["n"], (int, np.integer))


def test_purely_categorical_falls_back_to_random():
    space = {"k": Categorical(["a", "b", "c", "d"])}
    trials = powell_search(space, lambda p: 0.0 if p["k"] == "d" else 1.0, 20, rng=np.random.default_rng(0))
    assert len(trials) == 20
    assert min(t.score for t in trials) == 0.0


def test_validation_and_stop_when():
    with pytest.raises(ValueError):
        powell_search(_space(), _sphere, 0)
    with pytest.raises(ValueError):
        powell_search(_space(), _sphere, 10, line_evals=2)
    stopped = powell_search(
        _space(), _sphere, 100, rng=np.random.default_rng(0), stop_when=lambda best, n: n >= 12
    )
    assert len(stopped) == 12


def test_registered_and_run_search():
    assert {"powell", "powell_search"} <= set(available_strategies())
    result = run_search("powell", _space(), _sphere, budget=40, seed=0, line_evals=6)
    assert len(result.trials) == 40
