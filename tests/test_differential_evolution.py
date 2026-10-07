"""Tests for Differential Evolution search."""

from __future__ import annotations

import numpy as np
import pytest

from hyperopt_kit.evaluate import available_strategies, run_search
from hyperopt_kit.searchers import (
    de_crossover,
    de_mutant,
    de_search,
    differential_evolution_search,
    random_search,
)
from hyperopt_kit.spaces import parse_space


def _space():
    return parse_space({"x": [0.0, 1.0], "y": [0.0, 1.0]})


def _obj(params):
    return float(params["x"]) + float(params.get("y", 0.0))


def _sphere(params):
    return float(params["x"]) ** 2 + float(params["y"]) ** 2


def test_de_mutant_rand1_stays_in_unit_cube():
    rng = np.random.default_rng(0)
    pop = rng.random((8, 3))
    donor = de_mutant(pop, 0, rng, mutation=0.8, strategy="rand/1/bin")
    assert donor.shape == (3,)
    assert np.all(donor >= 0.0) and np.all(donor <= 1.0)


def test_de_mutant_best1_uses_best_index():
    rng = np.random.default_rng(1)
    pop = np.array(
        [
            [0.1, 0.1],
            [0.9, 0.9],
            [0.2, 0.8],
            [0.8, 0.2],
        ],
        dtype=float,
    )
    donor = de_mutant(
        pop, 1, rng, mutation=0.5, strategy="best/1/bin", best_index=0
    )
    # With F=0.5 and best at [0.1,0.1], donor is a convex/affine combo clipped to [0,1]
    assert donor.shape == (2,)
    assert np.all(donor >= 0.0) and np.all(donor <= 1.0)


def test_de_mutant_rejects_small_population():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="at least 4"):
        de_mutant(np.zeros((3, 2)), 0, rng)


def test_de_mutant_rand2_needs_six_members():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="at least 6"):
        de_mutant(np.zeros((5, 2)), 0, rng, strategy="rand/2/bin")


def test_de_crossover_forces_one_donor_coordinate():
    rng = np.random.default_rng(2)
    target = np.zeros(5)
    donor = np.ones(5)
    # With CR=0, only the forced index comes from the donor.
    child = de_crossover(target, donor, rng, crossover=0.0)
    assert int(np.count_nonzero(child)) == 1
    assert np.all((child == 0.0) | (child == 1.0))


def test_de_crossover_rejects_bad_cr():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="crossover"):
        de_crossover(np.zeros(2), np.ones(2), rng, crossover=1.5)


def test_de_search_respects_budget():
    trials = differential_evolution_search(
        _space(), _obj, budget=20, rng=np.random.default_rng(0), population_size=5
    )
    assert len(trials) == 20
    for i, trial in enumerate(trials):
        assert trial.iteration == i
        assert set(trial.params) == {"x", "y"}
        assert 0.0 <= trial.params["x"] <= 1.0
        assert 0.0 <= trial.params["y"] <= 1.0


def test_de_search_alias():
    a = de_search(_space(), _obj, 12, rng=np.random.default_rng(3), population_size=4)
    assert len(a) == 12


def test_de_search_deterministic_with_seed():
    a = differential_evolution_search(
        _space(), _obj, 24, rng=np.random.default_rng(11), population_size=6
    )
    b = differential_evolution_search(
        _space(), _obj, 24, rng=np.random.default_rng(11), population_size=6
    )
    assert [t.params for t in a] == [t.params for t in b]
    assert [t.score for t in a] == [t.score for t in b]


def test_de_search_differs_across_seeds():
    a = differential_evolution_search(
        _space(), _obj, 24, rng=np.random.default_rng(11), population_size=6
    )
    b = differential_evolution_search(
        _space(), _obj, 24, rng=np.random.default_rng(99), population_size=6
    )
    assert [t.params for t in a] != [t.params for t in b]


def test_de_search_beats_random_on_sphere():
    budget = 80
    de = differential_evolution_search(
        _space(),
        _sphere,
        budget,
        rng=np.random.default_rng(7),
        population_size=10,
        mutation=0.7,
        crossover=0.9,
    )
    rnd = random_search(_space(), _sphere, budget, rng=np.random.default_rng(7))
    assert min(t.score for t in de) <= min(t.score for t in rnd) + 1e-9


def test_de_search_concentrates_near_origin():
    trials = differential_evolution_search(
        _space(),
        _sphere,
        budget=120,
        rng=np.random.default_rng(5),
        population_size=12,
        strategy="best/1/bin",
    )
    best = min(trials, key=lambda t: t.score)
    assert best.score < 0.05
    assert best.params["x"] < 0.25
    assert best.params["y"] < 0.25


def test_de_search_rejects_bad_budget_and_strategy():
    with pytest.raises(ValueError, match="budget"):
        differential_evolution_search(_space(), _obj, 0)
    with pytest.raises(ValueError, match="strategy"):
        differential_evolution_search(
            _space(), _obj, 20, strategy="not-a-strategy", population_size=4
        )
    with pytest.raises(ValueError, match="population_size"):
        differential_evolution_search(
            _space(), _obj, 20, population_size=2
        )


def test_de_search_stop_when():
    trials = differential_evolution_search(
        _space(),
        _obj,
        budget=100,
        rng=np.random.default_rng(0),
        population_size=4,
        stop_when=lambda best, n: n >= 6,
    )
    assert len(trials) == 6


def test_de_registered_in_evaluate():
    assert "de" in available_strategies()
    assert "differential_evolution" in available_strategies()
    result = run_search(
        "de",
        _space(),
        _obj,
        budget=16,
        seed=0,
        population_size=4,
    )
    assert result.strategy == "de"
    assert len(result.trials) == 16
    assert result.best_score == min(t.score for t in result.trials)


def test_de_handles_int_and_categorical():
    space = parse_space({"x": [0.0, 1.0], "k": [0, 5, "int"], "c": ["a", "b", "c"]})
    trials = differential_evolution_search(
        space, lambda p: float(p["x"]) + float(p["k"]), 20,
        rng=np.random.default_rng(4), population_size=5,
    )
    assert len(trials) == 20
    for t in trials:
        assert t.params["c"] in ("a", "b", "c")
        assert isinstance(t.params["k"], (int, np.integer))
