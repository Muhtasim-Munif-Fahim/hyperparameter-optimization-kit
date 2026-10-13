"""Tests for simulated annealing search."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from hyperopt_kit import (
    geometric_cooling_schedule,
    metropolis_acceptance,
    sa_search,
    simulated_annealing_search,
)
from hyperopt_kit.cli import main
from hyperopt_kit.evaluate import available_strategies, run_search
from hyperopt_kit.searchers import random_search
from hyperopt_kit.spaces import Categorical, FloatRange, IntRange, parse_space


def _space():
    return parse_space({"x": [0.0, 1.0], "y": [0.0, 1.0]})


def _shifted_sphere(params):
    return (float(params["x"]) - 0.3) ** 2 + (float(params["y"]) - 0.7) ** 2


def test_metropolis_acceptance():
    assert metropolis_acceptance(-1.0, 0.5) == 1.0
    assert metropolis_acceptance(0.0, 0.0) == 1.0
    assert metropolis_acceptance(1.0, 0.0) == 0.0
    assert metropolis_acceptance(1.0, 2.0) == pytest.approx(math.exp(-0.5))
    with pytest.raises(ValueError):
        metropolis_acceptance(1.0, -1.0)


def test_geometric_schedule_reaches_final_ratio():
    temps = geometric_cooling_schedule(2.0, 50, final_temperature_ratio=1e-2)
    assert temps[0] == pytest.approx(2.0)
    assert temps[-1] == pytest.approx(0.02)
    assert np.all(np.diff(temps) < 0)
    fixed = geometric_cooling_schedule(1.0, 4, cooling=0.5)
    assert np.allclose(fixed, [1.0, 0.5, 0.25, 0.125])
    with pytest.raises(ValueError):
        geometric_cooling_schedule(1.0, 4, cooling=1.0)
    with pytest.raises(ValueError):
        geometric_cooling_schedule(0.0, 4)


def test_alias_budget_and_trial_indices():
    assert sa_search is simulated_annealing_search
    trials = simulated_annealing_search(_space(), _shifted_sphere, 40, rng=np.random.default_rng(0))
    assert len(trials) == 40
    assert [t.iteration for t in trials] == list(range(40))
    assert all(0.0 <= t.params["x"] <= 1.0 and 0.0 <= t.params["y"] <= 1.0 for t in trials)


def test_converges_and_beats_random_on_average():
    sa_best, rnd_best = [], []
    for seed in range(8):
        sa = simulated_annealing_search(_space(), _shifted_sphere, 80, rng=np.random.default_rng(seed))
        rnd = random_search(_space(), _shifted_sphere, 80, rng=np.random.default_rng(seed))
        sa_best.append(min(t.score for t in sa))
        rnd_best.append(min(t.score for t in rnd))
    assert np.median(sa_best) < 0.005
    assert np.mean(sa_best) < np.mean(rnd_best)


def test_beats_random_on_multimodal_rastrigin():
    space = parse_space({k: [-5.12, 5.12] for k in "abcd"})

    def rastrigin(p):
        return 40.0 + sum(p[k] ** 2 - 10.0 * math.cos(2 * math.pi * p[k]) for k in "abcd")

    sa = [min(t.score for t in simulated_annealing_search(space, rastrigin, 300, rng=np.random.default_rng(s)))
          for s in range(6)]
    rnd = [min(t.score for t in random_search(space, rastrigin, 300, rng=np.random.default_rng(s)))
           for s in range(6)]
    assert np.mean(sa) < 0.75 * np.mean(rnd)


def test_reproducible_with_seed():
    a = simulated_annealing_search(_space(), _shifted_sphere, 30, rng=np.random.default_rng(5))
    b = simulated_annealing_search(_space(), _shifted_sphere, 30, rng=np.random.default_rng(5))
    assert [t.params for t in a] == [t.params for t in b]


def test_mixed_space_with_categorical_and_int():
    space = {
        "lr": FloatRange(1e-4, 1.0, log_scale=True),
        "depth": IntRange(1, 10),
        "act": Categorical(["relu", "tanh", "gelu"]),
    }

    def obj(p):
        penalty = 0.0 if p["act"] == "gelu" else 1.0
        return (math.log10(p["lr"]) + 2.0) ** 2 + (p["depth"] - 6) ** 2 * 0.1 + penalty

    trials = simulated_annealing_search(space, obj, 120, rng=np.random.default_rng(1))
    best = min(trials, key=lambda t: t.score)
    assert best.params["act"] == "gelu"
    assert isinstance(best.params["depth"], int)
    assert best.score < 0.5
    acts = {t.params["act"] for t in trials}
    assert acts == {"relu", "tanh", "gelu"}


def test_greedy_low_temperature_never_accepts_worse():
    # with a tiny fixed temperature the chain is greedy: each accepted state
    # is at least as good, so neighbour proposals come from the best state.
    space = {"x": FloatRange(0.0, 1.0)}
    seen = []

    def obj(p):
        seen.append(p["x"])
        return abs(p["x"] - 0.5)

    trials = simulated_annealing_search(
        space, obj, 60, rng=np.random.default_rng(2), initial_temperature=1e-12,
        n_initial=1, step_size=0.05, min_step=0.05,
    )
    best_so_far = trials[0].score
    current = trials[0].params["x"]
    for t in trials[1:]:
        # every proposal is within a few sigma of the current (=best) state
        assert abs(t.params["x"] - current) < 0.4
        if t.score <= best_so_far:
            best_so_far = t.score
            current = t.params["x"]


def test_restart_and_stop_when():
    trials = simulated_annealing_search(
        _space(), _shifted_sphere, 100, rng=np.random.default_rng(3), restart_after=5,
        stop_when=lambda best, n: n >= 17,
    )
    assert len(trials) == 17


def test_registered_and_run_search():
    names = available_strategies()
    assert "sa" in names and "simulated_annealing" in names
    result = run_search("sa", _space(), _shifted_sphere, budget=25, seed=0, step_size=0.2)
    assert len(result.trials) == 25
    assert result.best_score == min(t.score for t in result.trials)


def test_invalid_parameters():
    with pytest.raises(ValueError, match="budget"):
        simulated_annealing_search(_space(), _shifted_sphere, 0)
    with pytest.raises(ValueError, match="initial_temperature"):
        simulated_annealing_search(_space(), _shifted_sphere, 10, initial_temperature=0.0)
    with pytest.raises(ValueError, match="cooling"):
        simulated_annealing_search(_space(), _shifted_sphere, 10, cooling=1.5)
    with pytest.raises(ValueError, match="initial_acceptance"):
        simulated_annealing_search(_space(), _shifted_sphere, 10, initial_acceptance=1.0)
    with pytest.raises(ValueError, match="step_size"):
        simulated_annealing_search(_space(), _shifted_sphere, 10, step_size=0.0)
    with pytest.raises(ValueError, match="perturb_prob"):
        simulated_annealing_search(_space(), _shifted_sphere, 10, perturb_prob=0.0)
    with pytest.raises(ValueError, match="restart_after"):
        simulated_annealing_search(_space(), _shifted_sphere, 10, restart_after=0)


def test_cli_search_sa(capsys):
    space = json.dumps({"a": [0.1, 5.0], "b": [0.1, 5.0]})
    code = main(["search", "--space", space, "--objective", "demo", "--budget", "20",
                 "--strategy", "sa", "--seed", "0"])
    assert code == 0
    out = capsys.readouterr().out
    assert "strategy: sa" in out
    assert "trials: 20" in out
