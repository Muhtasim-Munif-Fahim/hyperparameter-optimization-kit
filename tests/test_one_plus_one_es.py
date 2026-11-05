"""Tests for the (1+1) evolution strategy with the 1/5th success rule."""

from __future__ import annotations

import json

import numpy as np
import pytest

from hyperopt_kit import es_search, one_plus_one_es_search
from hyperopt_kit.cli import main
from hyperopt_kit.evaluate import available_strategies, run_search
from hyperopt_kit.searchers import random_search
from hyperopt_kit.spaces import Categorical, FloatRange, IntRange


def _space(d=2):
    return {f"x{i}": FloatRange(0.0, 1.0) for i in range(d)}


def _sphere(params):
    return sum((float(v) - 0.3 - 0.1 * i) ** 2 for i, v in enumerate(params.values()))


def test_budget_indices_bounds_and_alias():
    assert es_search is one_plus_one_es_search
    trials = one_plus_one_es_search(_space(), _sphere, 70, rng=np.random.default_rng(0))
    assert len(trials) == 70
    assert [t.iteration for t in trials] == list(range(70))
    assert all(0.0 <= v <= 1.0 for t in trials for v in t.params.values())


def test_deterministic_with_seed():
    a = one_plus_one_es_search(_space(3), _sphere, 50, rng=np.random.default_rng(7))
    b = one_plus_one_es_search(_space(3), _sphere, 50, rng=np.random.default_rng(7))
    assert [t.params for t in a] == [t.params for t in b]


def test_converges_on_sphere_and_beats_random():
    space = _space(4)
    es_best, rs_best = [], []
    for seed in range(5):
        es = one_plus_one_es_search(space, _sphere, 200, rng=np.random.default_rng(seed))
        rs = random_search(space, _sphere, 200, rng=np.random.default_rng(seed))
        es_best.append(min(t.score for t in es))
        rs_best.append(min(t.score for t in rs))
    assert np.median(es_best) < 1e-3
    assert np.median(es_best) < np.median(rs_best)


def test_step_size_adapts_to_success_rate():
    # On a smooth problem the late-phase steps must be much smaller than early ones.
    trials = one_plus_one_es_search(_space(2), _sphere, 150, rng=np.random.default_rng(3),
                                    n_initial=1, min_sigma=1e-9)
    xs = np.array([[t.params["x0"], t.params["x1"]] for t in trials])
    early = np.mean(np.abs(np.diff(xs[1:30], axis=0)))
    late = np.mean(np.abs(np.diff(xs[-30:], axis=0)))
    assert late < 0.2 * early


def test_mixed_space_finds_categorical_and_integer():
    space = {"x": FloatRange(0.0, 1.0), "n": IntRange(1, 20), "kind": Categorical(["a", "b", "c"])}

    def obj(p):
        return (p["x"] - 0.6) ** 2 + 0.01 * (p["n"] - 7) ** 2 + (0.0 if p["kind"] == "b" else 1.0)

    trials = one_plus_one_es_search(space, obj, 200, rng=np.random.default_rng(1))
    best = min(trials, key=lambda t: t.score)
    assert best.params["kind"] == "b"
    assert abs(best.params["n"] - 7) <= 1
    assert isinstance(best.params["n"], (int, np.integer))


def test_purely_categorical_space_always_moves():
    space = {"k": Categorical(["a", "b", "c", "d"])}
    trials = one_plus_one_es_search(space, lambda p: 0.0 if p["k"] == "d" else 1.0, 12,
                                    rng=np.random.default_rng(0), n_initial=1)
    assert min(t.score for t in trials) == 0.0


def test_non_finite_scores_and_stop_when():
    trials = one_plus_one_es_search(_space(), lambda p: float("nan") if p["x0"] > 0.5 else _sphere(p),
                                    40, rng=np.random.default_rng(2))
    assert len(trials) == 40
    stopped = one_plus_one_es_search(_space(), _sphere, 100, rng=np.random.default_rng(0),
                                     stop_when=lambda best, n: n >= 15)
    assert len(stopped) == 15


def test_restart_when_sigma_collapses():
    # A tiny adaptation window forces repeated restarts; budget must still be honoured.
    trials = one_plus_one_es_search(_space(), lambda p: 1.0, 60, rng=np.random.default_rng(0),
                                    sigma0=0.01, min_sigma=0.009, n_initial=1)
    assert len(trials) == 60


def test_registered_and_run_search():
    assert {"es", "one_plus_one_es"} <= set(available_strategies())
    result = run_search("one_plus_one_es", _space(), _sphere, budget=40, seed=0, sigma0=0.3, success_target=0.25)
    assert len(result.trials) == 40
    assert result.best_score == min(t.score for t in result.trials)


def test_invalid_parameters():
    with pytest.raises(ValueError, match="budget"):
        one_plus_one_es_search(_space(), _sphere, 0)
    with pytest.raises(ValueError, match="sigma0"):
        one_plus_one_es_search(_space(), _sphere, 10, sigma0=0.9)
    with pytest.raises(ValueError, match="min_sigma"):
        one_plus_one_es_search(_space(), _sphere, 10, min_sigma=0.3)
    with pytest.raises(ValueError, match="success_target"):
        one_plus_one_es_search(_space(), _sphere, 10, success_target=1.0)
    with pytest.raises(ValueError, match="adaptation"):
        one_plus_one_es_search(_space(), _sphere, 10, adaptation=0.0)
    with pytest.raises(ValueError, match="categorical_rate"):
        one_plus_one_es_search(_space(), _sphere, 10, categorical_rate=1.5)
    with pytest.raises(ValueError, match="n_initial"):
        one_plus_one_es_search(_space(), _sphere, 10, n_initial=0)


def test_cli_search_es(capsys):
    space = json.dumps({"a": [0.1, 5.0], "b": [0.1, 5.0]})
    code = main(["search", "--space", space, "--objective", "demo", "--budget", "30",
                 "--strategy", "es", "--seed", "0"])
    assert code == 0
    out = capsys.readouterr().out
    assert "strategy: es" in out
    assert "trials: 30" in out
