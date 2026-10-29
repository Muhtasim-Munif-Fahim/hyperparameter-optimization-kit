"""Tests for TuRBO-1 trust-region Bayesian optimization."""

from __future__ import annotations

import json

import numpy as np
import pytest

from hyperopt_kit import (
    TurboState,
    trust_region_bounds,
    turbo_candidates,
    turbo_initial_state,
    turbo_search,
    update_turbo_state,
)
from hyperopt_kit.cli import main
from hyperopt_kit.evaluate import available_strategies, run_search
from hyperopt_kit.searchers import bayesian_search, random_search
from hyperopt_kit.spaces import Categorical, FloatRange, IntRange


def _space(d):
    return {f"x{i}": FloatRange(-5.0, 10.0) for i in range(d)}


def _sphere(params):
    x = np.array([float(v) for v in params.values()]) - 1.3
    return float(np.sum(x * x))


def test_state_expands_shrinks_and_flags_restart():
    state = turbo_initial_state(6)
    assert state.failure_tolerance == 6 and state.length == 0.8
    for _ in range(3):
        state = update_turbo_state(state, True)
    assert state.length == pytest.approx(1.6) and state.success_counter == 0
    state = update_turbo_state(update_turbo_state(update_turbo_state(state, True), True), True)
    assert state.length == pytest.approx(1.6)  # capped at length_max
    state = update_turbo_state(state, True)
    state = update_turbo_state(state, False)
    assert state.success_counter == 0 and state.failure_counter == 1
    small = TurboState(dims=2, length=0.01, length_min=0.008, failure_tolerance=2)
    small = update_turbo_state(small, False)
    assert not small.restart_triggered
    small = update_turbo_state(small, False)
    assert small.length == pytest.approx(0.005) and small.restart_triggered


def test_state_validation():
    with pytest.raises(ValueError):
        turbo_initial_state(0)
    with pytest.raises(ValueError):
        turbo_initial_state(2, length_init=2.0)
    with pytest.raises(ValueError):
        turbo_initial_state(2, success_tolerance=0)
    with pytest.raises(ValueError):
        turbo_initial_state(2, failure_tolerance=0)


def test_trust_region_box_and_candidates():
    lb, ub = trust_region_bounds(np.array([0.5, 0.02, 0.99]), 0.2)
    assert ub - lb == pytest.approx([0.2, 0.2, 0.2])
    assert lb == pytest.approx([0.4, 0.0, 0.8])
    lb, ub = trust_region_bounds(np.array([0.3, 0.7]), 1.6)
    assert lb == pytest.approx([0.0, 0.0]) and ub == pytest.approx([1.0, 1.0])

    rng = np.random.default_rng(0)
    center = np.full(40, 0.5)
    cand = turbo_candidates(center, 0.1, 300, rng)
    assert cand.shape == (300, 40)
    assert np.all(np.abs(cand - 0.5) <= 0.05 + 1e-12)
    changed = cand != 0.5
    assert changed.any(axis=1).all()  # every candidate moves somewhere
    # with d = 40 each coordinate is perturbed with probability 20 / 40
    assert changed.mean() == pytest.approx(0.5, abs=0.05)
    full = turbo_candidates(np.full(3, 0.5), 0.1, 50, rng)
    assert (full != 0.5).all()
    with pytest.raises(ValueError):
        turbo_candidates(center, 0.1, 10, rng, perturbation_probability=0.0)


def test_beats_global_bo_and_random_in_ten_dimensions():
    space = _space(10)
    turbo, gp, rnd = [], [], []
    for seed in range(2):
        turbo.append(min(t.score for t in turbo_search(space, _sphere, 120, rng=np.random.default_rng(seed))))
        gp.append(min(t.score for t in bayesian_search(space, _sphere, 120, rng=np.random.default_rng(seed))))
        rnd.append(min(t.score for t in random_search(space, _sphere, 120, rng=np.random.default_rng(seed))))
    assert max(turbo) < 1.0
    assert max(turbo) < min(gp) / 5
    assert max(turbo) < min(rnd) / 20


def test_budget_indices_history_and_restarts():
    history = []
    trials = turbo_search(
        _space(2), _sphere, 90, rng=np.random.default_rng(3), length_min=0.05, history=history
    )
    assert len(trials) == 90 and [t.iteration for t in trials] == list(range(90))
    assert all(-5.0 <= v <= 10.0 for t in trials for v in t.params.values())
    assert len(history) == 90
    assert history[0]["length"] is None and history[0]["run"] == 0
    restarts = [h for h in history if h["restart"]]
    assert restarts, "a tight length_min must trigger at least one restart"
    after = history[restarts[0]["iteration"] + 1]
    assert after["run"] == 1 and after["length"] is None  # new initial design
    lengths = [h["length"] for h in history if h["length"] is not None]
    assert min(lengths) < 0.8 and max(lengths) <= 1.6
    assert min(t.score for t in trials) < 1e-2


def test_mixed_space_non_finite_scores_and_stop_when():
    space = {
        "lr": FloatRange(1e-4, 1.0, log_scale=True),
        "n": IntRange(1, 30),
        "kind": Categorical(["a", "b", "c"]),
    }

    def obj(p):
        if p["n"] > 27:
            return float("nan")
        return (np.log10(p["lr"]) + 2.0) ** 2 + ((p["n"] - 7) / 10) ** 2 + (0.0 if p["kind"] == "b" else 0.5)

    trials = turbo_search(space, obj, 70, rng=np.random.default_rng(4))
    assert len(trials) == 70
    assert all(isinstance(t.params["n"], int) for t in trials)
    best = min((t for t in trials if np.isfinite(t.score)), key=lambda t: t.score)
    assert best.score < 0.1 and best.params["kind"] == "b"
    stopped = turbo_search(space, obj, 50, rng=np.random.default_rng(4), stop_when=lambda b, n: n >= 9)
    assert len(stopped) == 9
    nan_only = turbo_search(_space(2), lambda p: float("nan"), 25, rng=np.random.default_rng(0))
    assert len(nan_only) == 25


def test_reproducible_registered_and_run_search():
    assert "turbo" in available_strategies()
    a = run_search("turbo", _space(3), _sphere, budget=30, seed=5, n_candidates=64)
    b = run_search("turbo", _space(3), _sphere, budget=30, seed=5, n_candidates=64)
    assert [t.params for t in a.trials] == [t.params for t in b.trials]
    assert a.best_score == min(t.score for t in a.trials)


def test_invalid_parameters():
    with pytest.raises(ValueError, match="budget"):
        turbo_search(_space(2), _sphere, 0)
    with pytest.raises(ValueError, match="n_initial"):
        turbo_search(_space(2), _sphere, 10, n_initial=0)
    with pytest.raises(ValueError, match="n_candidates"):
        turbo_search(_space(2), _sphere, 10, n_candidates=0)
    with pytest.raises(ValueError, match="max_gp_points"):
        turbo_search(_space(2), _sphere, 10, max_gp_points=1)
    with pytest.raises(ValueError, match="length"):
        turbo_search(_space(2), _sphere, 10, length_min=0.9)


def test_cli_search_turbo(capsys):
    space = json.dumps({"a": [0.1, 5.0], "b": [0.1, 5.0]})
    code = main(["search", "--space", space, "--objective", "demo", "--budget", "25",
                 "--strategy", "turbo", "--seed", "0"])
    assert code == 0
    out = capsys.readouterr().out
    assert "strategy: turbo" in out and "trials: 25" in out
