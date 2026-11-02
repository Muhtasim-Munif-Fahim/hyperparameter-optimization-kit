"""Tests for Hooke-Jeeves pattern search."""

from __future__ import annotations

import json

import numpy as np
import pytest

from hyperopt_kit import hj_search, hooke_jeeves_search
from hyperopt_kit.cli import main
from hyperopt_kit.evaluate import available_strategies, run_search
from hyperopt_kit.searchers import random_search
from hyperopt_kit.spaces import Categorical, FloatRange, IntRange


def _space(d=2):
    return {f"x{i}": FloatRange(0.0, 1.0) for i in range(d)}


def _sphere(params):
    return sum((float(v) - 0.3 - 0.1 * i) ** 2 for i, v in enumerate(params.values()))


def test_budget_indices_bounds_and_alias():
    assert hj_search is hooke_jeeves_search
    trials = hooke_jeeves_search(_space(), _sphere, 60, rng=np.random.default_rng(0))
    assert len(trials) == 60
    assert [t.iteration for t in trials] == list(range(60))
    assert all(0.0 <= v <= 1.0 for t in trials for v in t.params.values())


def test_no_duplicate_evaluations():
    space = {"n": IntRange(1, 30), "m": IntRange(1, 30)}
    trials = hooke_jeeves_search(space, lambda p: (p["n"] - 11) ** 2 + (p["m"] - 4) ** 2, 80, rng=np.random.default_rng(1))
    keys = [(t.params["n"], t.params["m"]) for t in trials]
    assert len(keys) == len(set(keys))
    best = min(trials, key=lambda t: t.score)
    assert best.params == {"n": 11, "m": 4}


def test_converges_and_beats_random():
    hj_best, rnd_best = [], []
    for seed in range(5):
        hj = hooke_jeeves_search(_space(3), _sphere, 150, rng=np.random.default_rng(seed))
        rnd = random_search(_space(3), _sphere, 150, rng=np.random.default_rng(seed))
        hj_best.append(min(t.score for t in hj))
        rnd_best.append(min(t.score for t in rnd))
    assert max(hj_best) < 1e-4
    assert np.median(hj_best) < np.median(rnd_best) / 50


def test_pattern_move_accelerates_toward_far_optimum():
    space = {"x": FloatRange(0.0, 100.0)}
    for seed in (1, 3):  # starts at ~51 and ~8.6
        trials = hooke_jeeves_search(space, lambda p: abs(p["x"] - 97.0), 40, rng=np.random.default_rng(seed),
                                     n_initial=1, initial_step=0.01, min_step=1e-4)
        assert abs(97.0 - trials[0].params["x"]) > 40
        # Steps of 1 unit without pattern moves would need > 40 evaluations.
        assert min(t.score for t in trials) < 0.01


def test_categoricals_are_explored():
    space = {"x": FloatRange(0.0, 1.0), "kind": Categorical(["a", "b", "c", "d"])}

    def obj(p):
        return (p["x"] - 0.6) ** 2 + {"a": 3.0, "b": 2.0, "c": 0.0, "d": 1.0}[p["kind"]]

    trials = hooke_jeeves_search(space, obj, 80, rng=np.random.default_rng(4), n_initial=1)
    best = min(trials, key=lambda t: t.score)
    assert best.params["kind"] == "c"
    assert best.params["x"] == pytest.approx(0.6, abs=0.01)
    fixed = hooke_jeeves_search(space, obj, 30, rng=np.random.default_rng(4), n_initial=1, explore_categoricals=False)
    first = fixed[0].params["kind"]
    assert all(t.params["kind"] == first for t in fixed[:8])


def test_finite_space_terminates_early():
    space = {"c": Categorical(["a", "b", "c"]), "n": IntRange(0, 2)}
    trials = hooke_jeeves_search(space, lambda p: 0.0, 100, rng=np.random.default_rng(0))
    assert len(trials) <= 9
    only_cat = hooke_jeeves_search({"c": Categorical(["a", "b"])}, lambda p: 1.0, 10,
                                   rng=np.random.default_rng(0), explore_categoricals=False)
    assert len(only_cat) == 10  # random-search fallback


def test_non_finite_scores_and_stop_when():
    def obj(p):
        return float("nan") if p["x0"] > 0.8 else _sphere(p)

    trials = hooke_jeeves_search(_space(), obj, 50, rng=np.random.default_rng(3))
    assert len(trials) == 50
    assert min(t.score for t in trials if np.isfinite(t.score)) < 0.01
    stopped = hooke_jeeves_search(_space(), _sphere, 100, rng=np.random.default_rng(3),
                                  stop_when=lambda best, n: n >= 13)
    assert len(stopped) == 13


def test_registered_and_run_search():
    assert {"hj", "hooke_jeeves"} <= set(available_strategies())
    result = run_search("hooke_jeeves", _space(), _sphere, budget=40, seed=0, initial_step=0.3, step_reduction=0.4)
    assert len(result.trials) == 40
    assert result.best_score == min(t.score for t in result.trials)


def test_invalid_parameters():
    with pytest.raises(ValueError, match="budget"):
        hooke_jeeves_search(_space(), _sphere, 0)
    with pytest.raises(ValueError, match="initial_step"):
        hooke_jeeves_search(_space(), _sphere, 10, initial_step=1.5)
    with pytest.raises(ValueError, match="step_reduction"):
        hooke_jeeves_search(_space(), _sphere, 10, step_reduction=1.0)
    with pytest.raises(ValueError, match="min_step"):
        hooke_jeeves_search(_space(), _sphere, 10, min_step=0.5, initial_step=0.25)
    with pytest.raises(ValueError, match="n_initial"):
        hooke_jeeves_search(_space(), _sphere, 10, n_initial=0)


def test_cli_search_hj(capsys):
    space = json.dumps({"a": [0.1, 5.0], "b": [0.1, 5.0]})
    code = main(["search", "--space", space, "--objective", "demo", "--budget", "30",
                 "--strategy", "hj", "--seed", "0"])
    assert code == 0
    out = capsys.readouterr().out
    assert "strategy: hj" in out
    assert "trials: 30" in out
