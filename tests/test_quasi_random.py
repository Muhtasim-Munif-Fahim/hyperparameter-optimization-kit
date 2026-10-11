"""Tests for low-discrepancy designs and quasi-random search."""

from __future__ import annotations

import numpy as np
import pytest

from hyperopt_kit.cli import main
from hyperopt_kit.evaluate import available_strategies, compare_strategies, run_search
from hyperopt_kit.quasi_random import (
    centered_discrepancy,
    first_primes,
    halton_search,
    halton_sequence,
    latin_hypercube,
    lhs_search,
    quasi_random_search,
    unit_to_config,
)
from hyperopt_kit.searchers import random_search
from hyperopt_kit.spaces import Categorical, FloatRange, IntRange, parse_space


def test_first_primes():
    assert first_primes(8) == [2, 3, 5, 7, 11, 13, 17, 19]
    assert first_primes(0) == []
    with pytest.raises(ValueError):
        first_primes(-1)


def test_unscrambled_halton_hand_values():
    pts = halton_sequence(5, 2, scramble=False)
    assert np.allclose(pts[:, 0], [0.0, 0.5, 0.25, 0.75, 0.125])
    assert np.allclose(pts[:, 1], [0.0, 1 / 3, 2 / 3, 1 / 9, 4 / 9])
    skipped = halton_sequence(3, 2, scramble=False, start_index=2)
    assert np.allclose(skipped, pts[2:5])


def test_unscrambled_halton_matches_scipy():
    qmc = pytest.importorskip("scipy.stats.qmc")
    ours = halton_sequence(200, 6, scramble=False)
    ref = qmc.Halton(6, scramble=False).random(200)
    assert np.allclose(ours, ref, atol=1e-12)


def test_scrambled_halton_is_seeded_stratified_and_in_range():
    a = halton_sequence(81, 3, rng=np.random.default_rng(0))
    b = halton_sequence(81, 3, rng=np.random.default_rng(0))
    c = halton_sequence(81, 3, rng=np.random.default_rng(1))
    assert np.array_equal(a, b)
    assert not np.allclose(a, c)
    assert np.all((a >= 0.0) & (a < 1.0))
    # Base-3 coordinate: 81 = 3^4 points hit each of the 81 cells exactly once,
    # since the scrambling permutes digits without changing the stratification.
    cells = np.floor(a[:, 1] * 81).astype(int)
    assert sorted(cells.tolist()) == list(range(81))
    # Base-2 coordinate: each half has 40 or 41 of the 81 points.
    assert abs(np.sum(a[:, 0] < 0.5) - 40.5) <= 0.5


def test_latin_hypercube_one_point_per_stratum():
    n, d = 37, 4
    design = latin_hypercube(n, d, rng=np.random.default_rng(3))
    assert design.shape == (n, d)
    for j in range(d):
        assert sorted(np.floor(design[:, j] * n).astype(int).tolist()) == list(range(n))
    centered = latin_hypercube(5, 2, rng=np.random.default_rng(0), centered=True)
    assert sorted(centered[:, 0].tolist()) == pytest.approx([0.1, 0.3, 0.5, 0.7, 0.9])


def test_maximin_lhs_improves_min_distance():
    def min_dist(x):
        diff = x[:, None, :] - x[None, :, :]
        dist = np.sqrt((diff**2).sum(-1))
        return dist[np.triu_indices(len(x), 1)].min()

    single = [min_dist(latin_hypercube(20, 2, rng=np.random.default_rng(s))) for s in range(10)]
    maximin = [
        min_dist(latin_hypercube(20, 2, rng=np.random.default_rng(s), n_candidates=30))
        for s in range(10)
    ]
    assert np.mean(maximin) > np.mean(single)


def test_centered_discrepancy_matches_scipy_and_ranks_designs():
    qmc = pytest.importorskip("scipy.stats.qmc")
    rng = np.random.default_rng(5)
    for pts in (rng.random((50, 3)), halton_sequence(50, 3, rng=rng), latin_hypercube(50, 3, rng=rng)):
        assert centered_discrepancy(pts) == pytest.approx(qmc.discrepancy(pts, method="CD"), rel=1e-10)


def test_low_discrepancy_beats_iid_uniform():
    rng = np.random.default_rng(7)
    halton = np.mean([centered_discrepancy(halton_sequence(128, 4, rng=rng)) for _ in range(10)])
    lhs = np.mean([centered_discrepancy(latin_hypercube(128, 4, rng=rng)) for _ in range(10)])
    iid = np.mean([centered_discrepancy(rng.random((128, 4))) for _ in range(10)])
    assert halton < lhs < iid
    assert halton < iid / 5


def test_centered_discrepancy_validation_and_hand_value():
    # One point at the centre in 1-d: 13/12 - 2 * 1 + 1 = 1/12.
    assert centered_discrepancy([[0.5]]) == pytest.approx(1.0 / 12.0)
    with pytest.raises(ValueError):
        centered_discrepancy([[1.5]])
    with pytest.raises(ValueError):
        centered_discrepancy(np.zeros((0, 2)))


def test_unit_to_config_gives_equal_mass_to_ints_and_choices():
    space = {
        "c": Categorical(["a", "b", "c"]),
        "k": IntRange(1, 4),
        "lr": FloatRange(1e-4, 1e-1, log_scale=True),
    }
    names = list(space)
    u = (np.arange(1200) + 0.5) / 1200
    configs = [unit_to_config([x, x, x], names, space) for x in u]
    counts_c = {ch: sum(cfg["c"] == ch for cfg in configs) for ch in "abc"}
    counts_k = {k: sum(cfg["k"] == k for cfg in configs) for k in range(1, 5)}
    assert set(counts_c.values()) == {400}
    assert set(counts_k.values()) == {300}
    lrs = np.array([cfg["lr"] for cfg in configs])
    assert lrs.min() >= 1e-4 and lrs.max() <= 1e-1
    assert np.median(lrs) == pytest.approx(np.sqrt(1e-4 * 1e-1), rel=0.01)
    assert unit_to_config([1.0, 1.0, 1.0], names, space)["c"] == "c"


def _sphere(params):
    return (params["x"] - 0.3) ** 2 + (params["y"] - 0.7) ** 2


def _space():
    return parse_space({"x": [0.0, 1.0], "y": [0.0, 1.0]})


@pytest.mark.parametrize("method", ["halton", "lhs"])
def test_quasi_random_search_budget_and_reproducibility(method):
    a = quasi_random_search(_space(), _sphere, 30, rng=np.random.default_rng(0), method=method)
    b = quasi_random_search(_space(), _sphere, 30, rng=np.random.default_rng(0), method=method)
    assert len(a) == 30
    assert [t.params for t in a] == [t.params for t in b]
    assert [t.iteration for t in a] == list(range(30))


def test_quasi_random_beats_random_on_average():
    qr, rs = [], []
    for seed in range(30):
        qr.append(min(t.score for t in halton_search(_space(), _sphere, 16, rng=np.random.default_rng(seed))))
        rs.append(min(t.score for t in random_search(_space(), _sphere, 16, rng=np.random.default_rng(seed))))
    assert np.mean(qr) < np.mean(rs)


def test_stop_when_and_validation():
    trials = lhs_search(_space(), _sphere, 50, rng=np.random.default_rng(0), stop_when=lambda best, n: n >= 7)
    assert len(trials) == 7
    with pytest.raises(ValueError):
        quasi_random_search(_space(), _sphere, 0)
    with pytest.raises(ValueError):
        quasi_random_search(_space(), _sphere, 5, method="sobol")
    with pytest.raises(ValueError):
        latin_hypercube(5, 2, n_candidates=0)
    with pytest.raises(ValueError):
        halton_sequence(5, 0)


def test_registered_strategies_and_cli(capsys):
    for name in ("quasi_random", "halton", "lhs"):
        assert name in available_strategies()
    result = run_search("lhs", _space(), _sphere, budget=12, seed=0, n_candidates=5)
    assert len(result.trials) == 12
    result = run_search("quasi_random", _space(), _sphere, budget=8, seed=0, method="lhs")
    assert len(result.trials) == 8
    results = compare_strategies(_space(), _sphere, 10, strategies=("random", "halton"), seed=1)
    assert set(results) == {"random", "halton"}
    assert main([
        "search", "--strategy", "halton", "--space", '{"a": [0, 3], "b": [0, 4]}',
        "--objective", "demo", "--budget", "6", "--seed", "0",
    ]) in (0, None)
    assert "halton" in capsys.readouterr().out
