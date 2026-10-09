"""Tests for Particle Swarm Optimization search."""

from __future__ import annotations

import numpy as np
import pytest

from hyperopt_kit.evaluate import available_strategies, run_search
from hyperopt_kit.searchers import particle_swarm_search, pso_search, random_search
from hyperopt_kit.spaces import parse_space


def _space():
    return parse_space({"x": [0.0, 1.0], "y": [0.0, 1.0]})


def _sphere(params):
    return float(params["x"]) ** 2 + float(params["y"]) ** 2


def test_alias():
    assert pso_search is particle_swarm_search


def test_budget_respected():
    trials = particle_swarm_search(
        _space(), _sphere, budget=25, rng=np.random.default_rng(0), swarm_size=5
    )
    assert len(trials) == 25


def test_finds_near_optimum():
    trials = particle_swarm_search(
        _space(), _sphere, budget=60, rng=np.random.default_rng(1), swarm_size=8
    )
    best = min(t.score for t in trials)
    # Random search baseline should usually be worse; PSO should get close to 0.
    assert best < 0.05


def test_beats_random_on_sphere():
    rng = np.random.default_rng(2)
    space = _space()
    pso = min(
        t.score
        for t in particle_swarm_search(space, _sphere, budget=40, rng=rng, swarm_size=8)
    )
    rnd = min(t.score for t in random_search(space, _sphere, budget=40, rng=rng))
    assert pso <= rnd + 1e-9 or pso < 0.1


def test_registered_strategy():
    assert "pso" in available_strategies()
    assert "particle_swarm" in available_strategies()
    result = run_search(
        "pso",
        _space(),
        _sphere,
        budget=20,
        seed=0,
        swarm_size=5,
    )
    assert len(result.trials) == 20


def test_invalid_params():
    with pytest.raises(ValueError, match="budget"):
        particle_swarm_search(_space(), _sphere, budget=0)
    with pytest.raises(ValueError, match="vmax"):
        particle_swarm_search(_space(), _sphere, budget=10, vmax=0.0)
    with pytest.raises(ValueError, match="swarm_size"):
        particle_swarm_search(_space(), _sphere, budget=10, swarm_size=1)


def test_stop_when():
    trials = particle_swarm_search(
        _space(),
        _sphere,
        budget=100,
        rng=np.random.default_rng(0),
        swarm_size=5,
        stop_when=lambda best, n: n >= 12,
    )
    assert len(trials) == 12
