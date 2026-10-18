"""Tests for cross-entropy method search."""

from __future__ import annotations

import json

import numpy as np
import pytest

from hyperopt_kit import cem_search, cross_entropy_search, cross_entropy_update
from hyperopt_kit.cli import main
from hyperopt_kit.evaluate import available_strategies, run_search
from hyperopt_kit.searchers import random_search
from hyperopt_kit.spaces import Categorical, FloatRange, IntRange


def _space(d=2):
    return {f"x{i}": FloatRange(0.0, 1.0) for i in range(d)}


def _sphere(params):
    return sum((float(v) - 0.3 - 0.1 * i) ** 2 for i, v in enumerate(params.values()))


def test_update_moves_toward_elites_and_floors_std():
    mean, std = np.array([0.5, 0.5]), np.array([0.3, 0.3])
    elites = np.array([[0.1, 0.9], [0.3, 0.9]])
    m, s = cross_entropy_update(mean, std, elites, smoothing=1.0, min_std=0.05)
    assert m == pytest.approx([0.2, 0.9])
    assert s == pytest.approx([0.1, 0.05])  # second column has zero spread -> floor
    m2, s2 = cross_entropy_update(mean, std, elites, smoothing=0.5, min_std=0.0)
    assert m2 == pytest.approx([0.35, 0.7])
    assert s2 == pytest.approx([0.2, 0.15])
    with pytest.raises(ValueError, match="smoothing"):
        cross_entropy_update(mean, std, elites, smoothing=0.0)
    with pytest.raises(ValueError, match="elites"):
        cross_entropy_update(mean, std, np.zeros((2, 3)))


def test_budget_indices_and_bounds():
    assert cem_search is cross_entropy_search
    trials = cross_entropy_search(_space(), _sphere, 53, rng=np.random.default_rng(0), population_size=10)
    assert len(trials) == 53  # last generation truncated
    assert [t.iteration for t in trials] == list(range(53))
    assert all(0.0 <= v <= 1.0 for t in trials for v in t.params.values())


def test_converges_and_beats_random():
    cem_best, rnd_best = [], []
    for seed in range(5):
        cem = cross_entropy_search(_space(3), _sphere, 300, rng=np.random.default_rng(seed))
        rnd = random_search(_space(3), _sphere, 300, rng=np.random.default_rng(seed))
        cem_best.append(min(t.score for t in cem))
        rnd_best.append(min(t.score for t in rnd))
    assert max(cem_best) < 1e-3
    assert np.median(cem_best) < np.median(rnd_best) / 5


def test_distribution_concentrates_late_generations():
    trials = cross_entropy_search(_space(2), _sphere, 400, rng=np.random.default_rng(1), population_size=20)
    first = np.array([[t.params["x0"], t.params["x1"]] for t in trials[:20]])
    last = np.array([[t.params["x0"], t.params["x1"]] for t in trials[-20:]])
    assert last.std(axis=0).max() < first.std(axis=0).min() / 3
    assert last.mean(axis=0) == pytest.approx([0.3, 0.4], abs=0.05)


def test_mixed_space_learns_categorical_and_integer():
    space = {
        "lr": FloatRange(1e-4, 1.0, log_scale=True),
        "n": IntRange(1, 20),
        "kind": Categorical(["a", "b", "c", "d"]),
    }

    def obj(p):
        penalty = 0.0 if p["kind"] == "c" else 1.0
        return (np.log10(p["lr"]) + 2.0) ** 2 + ((p["n"] - 7) / 20) ** 2 + penalty

    trials = cross_entropy_search(space, obj, 300, rng=np.random.default_rng(2), population_size=20)
    best = min(trials, key=lambda t: t.score)
    assert best.params["kind"] == "c"
    assert abs(best.params["n"] - 7) <= 1
    assert best.params["lr"] == pytest.approx(1e-2, rel=0.5)
    assert all(isinstance(t.params["n"], int) for t in trials)
    late = [t.params["kind"] for t in trials[-40:]]
    assert late.count("c") / len(late) > 0.7
    # min_probability keeps every choice reachable
    assert set(t.params["kind"] for t in trials) == {"a", "b", "c", "d"}


def test_non_finite_scores_and_stop_when():
    def obj(p):
        return float("nan") if p["x0"] > 0.8 else _sphere(p)

    trials = cross_entropy_search(_space(), obj, 60, rng=np.random.default_rng(3))
    assert len(trials) == 60
    assert min(t.score for t in trials if np.isfinite(t.score)) < 0.05
    stopped = cross_entropy_search(
        _space(), _sphere, 100, rng=np.random.default_rng(3), stop_when=lambda best, n: n >= 13
    )
    assert len(stopped) == 13


def test_registered_and_run_search():
    assert {"cem", "cross_entropy"} <= set(available_strategies())
    result = run_search("cem", _space(), _sphere, budget=40, seed=0, elite_fraction=0.3)
    assert len(result.trials) == 40
    assert result.best_score == min(t.score for t in result.trials)
    again = run_search("cross_entropy", _space(), _sphere, budget=40, seed=0, elite_fraction=0.3)
    assert [t.params for t in again.trials] == [t.params for t in result.trials]


def test_invalid_parameters():
    with pytest.raises(ValueError, match="budget"):
        cross_entropy_search(_space(), _sphere, 0)
    with pytest.raises(ValueError, match="population_size"):
        cross_entropy_search(_space(), _sphere, 10, population_size=1)
    with pytest.raises(ValueError, match="elite_fraction"):
        cross_entropy_search(_space(), _sphere, 10, elite_fraction=0.0)
    with pytest.raises(ValueError, match="initial_std"):
        cross_entropy_search(_space(), _sphere, 10, initial_std=0.0)
    with pytest.raises(ValueError, match="min_probability"):
        cross_entropy_search(_space(), _sphere, 10, min_probability=1.0)


def test_cli_search_cem(capsys):
    space = json.dumps({"a": [0.1, 5.0], "b": [0.1, 5.0]})
    code = main(["search", "--space", space, "--objective", "demo", "--budget", "30",
                 "--strategy", "cem", "--seed", "0"])
    assert code == 0
    out = capsys.readouterr().out
    assert "strategy: cem" in out
    assert "trials: 30" in out
