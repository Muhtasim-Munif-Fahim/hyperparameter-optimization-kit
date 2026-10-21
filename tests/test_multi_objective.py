import json

import numpy as np
import pytest

from hyperopt_kit import (
    Categorical,
    FloatRange,
    IntRange,
    crowding_distance,
    demo_multi_objective,
    dominates,
    generational_distance,
    hypervolume,
    igd,
    non_dominated_sort,
    nsga2_search,
    pareto_front_mask,
    pareto_trials,
    polynomial_mutation,
    sbx_crossover,
    zdt1,
)
from hyperopt_kit.cli import main


def test_dominance_and_sorting():
    assert dominates([1, 2], [1, 3]) and not dominates([1, 2], [1, 2]) and not dominates([0, 3], [1, 2])
    F = np.array([[1, 4], [2, 2], [4, 1], [3, 3], [4, 4], [2, 2]], dtype=float)
    fronts = [sorted(f.tolist()) for f in non_dominated_sort(F)]
    assert fronts == [[0, 1, 2, 5], [3], [4]]
    assert pareto_front_mask(F).tolist() == [True, True, True, False, False, True]


def test_crowding_distance_matches_deb_and_pymoo():
    F = np.array([[0.1, 0.9], [0.3, 0.5], [0.5, 0.45], [0.8, 0.1]])
    cd = crowding_distance(F)
    # pymoo's calc_crowding_distance * n_obj (pymoo averages over objectives)
    np.testing.assert_allclose(cd, [np.inf, 1.13392857, 1.21428571, np.inf], rtol=1e-7)
    assert np.all(np.isinf(crowding_distance(F[:2])))
    flat = crowding_distance(np.array([[0.0, 1.0], [0.5, 1.0], [1.0, 1.0]]))
    assert flat[1] == pytest.approx(1.0)


def test_hypervolume_exact_values():
    assert hypervolume([[1, 2], [2, 1]], [3, 3]) == pytest.approx(3.0)
    assert hypervolume([[0.5, 0.5, 0.5]], [1, 1, 1]) == pytest.approx(0.125)
    assert hypervolume([[2, 2]], [1, 1]) == 0.0  # does not dominate ref
    rng = np.random.default_rng(123)
    F3 = np.round(rng.random((12, 3)), 3)
    F4 = np.round(rng.random((10, 4)), 3)
    # reference values from moocore.hypervolume
    assert hypervolume(F3, [1.1] * 3) == pytest.approx(0.7427204340000003, abs=1e-12)
    assert hypervolume(F4, [1.2] * 4) == pytest.approx(1.434990996606, abs=1e-12)
    # dominated / duplicate points do not change the volume
    assert hypervolume(np.vstack([F3, F3, F3 + 0.05]), [1.1] * 3) == pytest.approx(0.7427204340000003, abs=1e-12)
    with pytest.raises(ValueError):
        hypervolume(F3, [1, 1])


def test_hypervolume_monte_carlo_sanity():
    rng = np.random.default_rng(0)
    F = rng.random((15, 3))
    U = rng.random((200000, 3))
    dominated = np.zeros(len(U), dtype=bool)
    for p in F:
        dominated |= np.all(U >= p, axis=1)
    assert hypervolume(F, [1, 1, 1]) == pytest.approx(dominated.mean(), abs=5e-3)


def test_igd_and_gd():
    R = np.array([[0.0, 1.0], [1.0, 0.0]])
    A = np.array([[0.0, 1.0]])
    assert igd(A, R) == pytest.approx((0 + np.sqrt(2)) / 2)
    assert generational_distance(A, R) == 0.0


def test_variation_operators_stay_in_unit_cube():
    rng = np.random.default_rng(1)
    for _ in range(300):
        p1, p2 = rng.random(4), rng.random(4)
        c1, c2 = sbx_crossover(p1, p2, rng, eta=2.0, prob_var=1.0)
        m = polynomial_mutation(c1, rng, eta=5.0, prob=1.0)
        for v in (c1, c2, m):
            assert np.all((v >= 0) & (v <= 1))
    same = sbx_crossover(np.full(3, 0.4), np.full(3, 0.4), rng)
    np.testing.assert_array_equal(same[0], np.full(3, 0.4))


def test_nsga2_converges_on_zdt1():
    space = {f"x{i}": FloatRange(0.0, 1.0) for i in range(5)}
    trials = nsga2_search(space, zdt1, 2000, np.random.default_rng(0), population_size=40)
    assert len(trials) == 2000 and [t.iteration for t in trials] == list(range(2000))
    P = np.array([t.objectives for t in pareto_trials(trials)])
    f = np.linspace(0, 1, 200)
    R = np.c_[f, 1 - np.sqrt(f)]
    init = np.array([t.objectives for t in pareto_trials(trials[:40])])
    assert igd(P, R) < 0.02 < igd(init, R)
    assert hypervolume(P, [1.1, 1.1]) > 0.86  # true front: ~0.8741
    again = nsga2_search(space, zdt1, 200, np.random.default_rng(0), population_size=40)
    assert [t.objectives for t in again] == [t.objectives for t in trials[:200]]


def test_nsga2_mixed_space_and_validation():
    space = {"lr": FloatRange(1e-4, 1e-1, log_scale=True), "units": IntRange(8, 64), "act": Categorical(["relu", "tanh"])}

    def obj(p):
        return (abs(np.log10(p["lr"]) + 2.0) + (0.1 if p["act"] == "tanh" else 0.0), p["units"] / 64.0)

    trials = nsga2_search(space, obj, 60, np.random.default_rng(2), population_size=10)
    assert len(trials) == 60
    assert all(8 <= t.params["units"] <= 64 and t.params["act"] in ("relu", "tanh") for t in trials)
    front = pareto_trials(trials)
    assert len({t.objectives for t in front}) == len(front)
    with pytest.raises(ValueError):
        nsga2_search(space, lambda p: (1.0,), 10, np.random.default_rng(0))
    with pytest.raises(ValueError):
        nsga2_search(space, obj, 1)
    with pytest.raises(ValueError):
        nsga2_search({}, obj, 10)


def test_demo_multi_objective_tradeoff():
    lo, cost = demo_multi_objective({"a": 1.0, "b": 2.0})
    assert lo == 0.0 and cost == 3.0
    assert demo_multi_objective({"a": 0.1, "b": 0.1})[1] < cost


def test_cli_pareto(capsys):
    space = '{"a": [0.1, 5.0], "b": [0.1, 5.0]}'
    assert main(["pareto", "--space", space, "--budget", "60", "--seed", "7"]) == 0
    out = capsys.readouterr().out
    assert "pareto front:" in out and "hypervolume:" in out and "| iter | f1 | f2 |" in out
    assert main(["pareto", "--space", space, "--budget", "40", "--seed", "1", "--json", "--ref", "200,11"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["trials"] == 40 and data["ref"] == [200.0, 11.0] and data["hypervolume"] > 0
    zspace = json.dumps({f"x{i}": [0.0, 1.0] for i in range(3)})
    assert main(["pareto", "--space", zspace, "--objective", "zdt1", "--budget", "50", "--seed", "0"]) == 0
    capsys.readouterr()
    with pytest.raises(SystemExit):
        main(["pareto", "--space", space, "--budget", "20", "--ref", "1"])
