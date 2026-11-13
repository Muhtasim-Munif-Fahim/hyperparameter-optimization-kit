import numpy as np
import pytest

from hyperopt_kit import (
    Categorical,
    FloatRange,
    IntRange,
    format_morris_table,
    morris_elementary_effects,
    morris_hyperparameter_screening,
    morris_screening,
    morris_trajectories,
)


def test_trajectory_structure():
    T = morris_trajectories(6, 5, p=4, rng=np.random.default_rng(0))
    assert T.shape == (6, 6, 5)
    delta = 4 / (2 * 3)
    levels = np.arange(4) / 3
    assert np.all(np.isin(np.round(T, 12), np.round(levels, 12)))
    for traj in T:
        step = np.diff(traj, axis=0)
        nonzero = np.abs(step) > 1e-12
        assert np.all(nonzero.sum(axis=1) == 1)
        assert sorted(np.argmax(nonzero, axis=1).tolist()) == list(range(5))
        np.testing.assert_allclose(np.abs(step[nonzero]), delta)


def test_linear_function_effects_are_exact():
    a = np.array([3.0, -2.0, 0.0, 0.5])
    res = morris_screening(lambda X: X @ a + 1.0, 4, r=8, rng=np.random.default_rng(1))
    np.testing.assert_allclose(res.mu, a, atol=1e-12)
    np.testing.assert_allclose(res.mu_star, np.abs(a), atol=1e-12)
    np.testing.assert_allclose(res.sigma, 0.0, atol=1e-12)
    assert res.n_evaluations == 8 * 5
    assert res.ranking()[0][0] == "x0"


def test_interaction_and_nonlinearity_raise_sigma():
    f = lambda X: X[:, 0] * X[:, 1] + 2.0 * X[:, 2] + X[:, 3] ** 3 * 0.0
    res = morris_screening(f, 4, r=30, rng=np.random.default_rng(2))
    assert res.sigma[0] > 0.1 and res.sigma[1] > 0.1
    assert res.sigma[2] < 1e-10
    assert res.mu_star[3] == 0.0
    names = [n for n, _ in res.ranking()]
    assert names[0] == "x2" and names[-1] == "x3"


def test_sign_cancellation_mu_vs_mu_star():
    # f = (x - 0.5)^2 has effects of both signs: |mu| small, mu* large.
    res = morris_screening(lambda X: 10 * (X[:, 0] - 0.5) ** 2, 1, r=40, rng=np.random.default_rng(3))
    assert abs(res.mu[0]) < 0.5 * res.mu_star[0]
    assert res.ranking("abs_mu")[0][1] == pytest.approx(abs(res.mu[0]))


def test_ishigami_screening_order():
    def ishigami(X):
        Z = -np.pi + 2 * np.pi * X
        return np.sin(Z[:, 0]) + 7 * np.sin(Z[:, 1]) ** 2 + 0.1 * Z[:, 2] ** 4 * np.sin(Z[:, 0])

    res = morris_screening(ishigami, 4, r=50, p=6, rng=np.random.default_rng(4), names=["a", "b", "c", "dummy"])
    assert res.mu_star[3] == 0.0
    assert min(res.mu_star[:3]) > 0.0
    assert res.as_dict()["dummy"]["mu_star"] == 0.0


def test_candidate_selection_improves_spread():
    rng = np.random.default_rng(5)
    plain = morris_trajectories(4, 3, rng=np.random.default_rng(5))
    chosen = morris_trajectories(4, 3, rng=rng, n_candidates=40)

    def spread(T):
        tot = 0.0
        for a in range(len(T)):
            for b in range(a + 1, len(T)):
                tot += np.sqrt(((T[a][:, None, :] - T[b][None, :, :]) ** 2).sum(-1)).sum()
        return tot

    assert chosen.shape == (4, 4, 3)
    assert spread(chosen) >= spread(plain)


def test_bootstrap_ci_and_table():
    res = morris_screening(lambda X: X[:, 0] + 0.1 * X[:, 1], 2, r=10, rng=np.random.default_rng(6), n_bootstrap=50)
    assert res.mu_star_ci.shape == (2, 2)
    assert np.all(res.mu_star_ci[:, 0] <= res.mu_star + 1e-12)
    table = format_morris_table(res)
    assert table.splitlines()[2].startswith("| x0 |")


def test_hyperparameter_screening_on_space():
    space = {
        "lr": FloatRange(1e-4, 1e-1, log_scale=True),
        "depth": IntRange(1, 10),
        "opt": Categorical(["sgd", "adam"]),
        "unused": FloatRange(0.0, 1.0),
    }

    def objective(p):
        return (np.log10(p["lr"]) + 2.0) ** 2 + 0.05 * p["depth"] + (0.3 if p["opt"] == "sgd" else 0.0)

    res = morris_hyperparameter_screening(space, objective, r=12, rng=np.random.default_rng(7))
    assert res.names == ["lr", "depth", "opt", "unused"]
    assert len(res.extra["trials"]) == res.n_evaluations == 12 * 5
    assert res.mu_star[3] == 0.0
    assert res.ranking()[0][0] == "lr"


def test_validation():
    with pytest.raises(ValueError):
        morris_trajectories(1, 3)
    with pytest.raises(ValueError):
        morris_trajectories(4, 3, p=5)
    with pytest.raises(ValueError):
        morris_trajectories(4, 3, n_candidates=2)
    with pytest.raises(ValueError):
        morris_screening(lambda X: X[:, 0], 2, names=["a"])
    with pytest.raises(ValueError):
        morris_screening(lambda X: np.full(len(X), np.nan), 2)
    with pytest.raises(ValueError):
        morris_screening(lambda X: X[:1, 0], 2)
    with pytest.raises(ValueError):
        morris_elementary_effects(np.zeros((2, 3, 2)), np.zeros((2, 3)))
    with pytest.raises(ValueError):
        morris_hyperparameter_screening({}, lambda p: 0.0)
