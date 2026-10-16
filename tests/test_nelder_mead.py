"""Tests for Nelder-Mead simplex search."""

from __future__ import annotations

import json

import numpy as np
import pytest

from hyperopt_kit import nelder_mead_coefficients, nelder_mead_search, nm_search
from hyperopt_kit.cli import main
from hyperopt_kit.evaluate import available_strategies, run_search
from hyperopt_kit.searchers import random_search
from hyperopt_kit.spaces import Categorical, FloatRange, IntRange, parse_space


def _space(d=2):
    return {f"x{i}": FloatRange(0.0, 1.0) for i in range(d)}


def _sphere(params):
    return sum((float(v) - 0.3 - 0.1 * i) ** 2 for i, v in enumerate(params.values()))


def test_coefficients():
    assert nelder_mead_coefficients(2) == (1.0, 2.0, 0.5, 0.5)
    a, g, r, s = nelder_mead_coefficients(10, adaptive=True)
    assert (a, g, r, s) == pytest.approx((1.0, 1.2, 0.7, 0.9))
    with pytest.raises(ValueError):
        nelder_mead_coefficients(0)


def test_budget_indices_and_bounds():
    assert nm_search is nelder_mead_search
    trials = nelder_mead_search(_space(), _sphere, 60, rng=np.random.default_rng(0))
    assert len(trials) == 60
    assert [t.iteration for t in trials] == list(range(60))
    assert all(0.0 <= v <= 1.0 for t in trials for v in t.params.values())


def test_converges_on_smooth_objective_and_beats_random():
    nm_best, rnd_best = [], []
    for seed in range(5):
        nm = nelder_mead_search(_space(3), _sphere, 120, rng=np.random.default_rng(seed))
        rnd = random_search(_space(3), _sphere, 120, rng=np.random.default_rng(seed))
        nm_best.append(min(t.score for t in nm))
        rnd_best.append(min(t.score for t in rnd))
    assert max(nm_best) < 1e-4
    assert np.median(nm_best) < np.median(rnd_best) / 50


def test_rosenbrock_valley():
    space = {"a": FloatRange(-2.0, 2.0), "b": FloatRange(-1.0, 3.0)}

    def rosen(p):
        return (1 - p["a"]) ** 2 + 100 * (p["b"] - p["a"] ** 2) ** 2

    trials = nelder_mead_search(space, rosen, 400, rng=np.random.default_rng(1))
    best = min(trials, key=lambda t: t.score)
    assert best.score < 1e-3
    assert best.params["a"] == pytest.approx(1.0, abs=0.05)


def test_mixed_space_restarts_explore_categoricals():
    space = {
        "x": FloatRange(0.0, 1.0),
        "n": IntRange(1, 20),
        "kind": Categorical(["bad", "good"]),
    }

    def obj(p):
        penalty = 0.0 if p["kind"] == "good" else 1.0
        return (p["x"] - 0.6) ** 2 + ((p["n"] - 7) / 20) ** 2 + penalty

    trials = nelder_mead_search(space, obj, 200, rng=np.random.default_rng(2), n_initial=1, xtol=0.05)
    assert {t.params["kind"] for t in trials} == {"bad", "good"}
    best = min(trials, key=lambda t: t.score)
    assert best.params["kind"] == "good"
    assert best.params["n"] == 7
    assert all(isinstance(t.params["n"], int) for t in trials)


def test_categorical_only_falls_back_to_random():
    space = {"c": Categorical(["a", "b", "c"])}
    trials = nelder_mead_search(space, lambda p: 0.0 if p["c"] == "b" else 1.0, 12, rng=np.random.default_rng(0))
    assert len(trials) == 12


def test_non_finite_scores_and_stop_when():
    def obj(p):
        return float("nan") if p["x0"] > 0.8 else _sphere(p)

    trials = nelder_mead_search(_space(), obj, 50, rng=np.random.default_rng(3))
    assert len(trials) == 50
    stopped = nelder_mead_search(
        _space(), _sphere, 100, rng=np.random.default_rng(3), stop_when=lambda best, n: n >= 13
    )
    assert len(stopped) == 13


def test_registered_and_run_search():
    assert "nm" in available_strategies() and "nelder_mead" in available_strategies()
    result = run_search("nm", _space(), _sphere, budget=40, seed=0, initial_step=0.3)
    assert len(result.trials) == 40
    assert result.best_score == min(t.score for t in result.trials)


def test_invalid_parameters():
    with pytest.raises(ValueError, match="budget"):
        nelder_mead_search(_space(), _sphere, 0)
    with pytest.raises(ValueError, match="initial_step"):
        nelder_mead_search(_space(), _sphere, 10, initial_step=0.0)
    with pytest.raises(ValueError, match="n_initial"):
        nelder_mead_search(_space(), _sphere, 10, n_initial=0)
    with pytest.raises(ValueError, match="xtol"):
        nelder_mead_search(_space(), _sphere, 10, xtol=-1.0)


def test_cli_search_nm(capsys):
    space = json.dumps({"a": [0.1, 5.0], "b": [0.1, 5.0]})
    code = main(["search", "--space", space, "--objective", "demo", "--budget", "30",
                 "--strategy", "nm", "--seed", "0"])
    assert code == 0
    out = capsys.readouterr().out
    assert "strategy: nm" in out
    assert "trials: 30" in out
