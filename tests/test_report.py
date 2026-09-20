import numpy as np
import pytest

from hyperopt_kit.evaluate import RunResult
from hyperopt_kit.report import (
    learning_curve_table,
    param_importance,
    render_report,
    write_report,
)
from hyperopt_kit.searchers import Trial
from hyperopt_kit.spaces import parse_space


def _result(strategy, trials):
    best = min(trials, key=lambda t: t.score)
    return RunResult(strategy=strategy, trials=trials, best_score=best.score, best_params=best.params)


def _trials(points):
    return [Trial(i, params, score) for i, (params, score) in enumerate(points)]


def _comparison(space, strategy_kwargs=None):
    from hyperopt_kit.evaluate import compare_strategies

    def objective(p):
        penalty = {"a": 0.0, "b": 0.5}.get(p.get("k"), 0.0)
        return (p.get("x", 0.0) - 0.5) ** 2 + penalty

    return compare_strategies(space, objective, budget=10, seed=3)


def test_render_report_contains_all_sections():
    space = parse_space({"x": [0.0, 1.0], "k": ["a", "b"]})
    comparison = _comparison(space)
    text = render_report(comparison, space, title="My report", budget=10)
    assert text.startswith("# My report")
    assert "## Search space" in text
    assert "## Results" in text
    assert "## Learning curves" in text
    assert "## Parameter importance" in text
    assert "## Caveats" in text
    assert "| `x` |" in text
    assert "| grid |" in text
    assert "| random |" in text
    assert "| bayesian |" in text
    assert "| tpe |" in text
    assert "| cmaes |" in text
    assert "| hyperband |" in text


def test_render_report_includes_objective_and_budget():
    comparison = _comparison(parse_space({"x": [0.0, 1.0]}))
    text = render_report(
        comparison,
        parse_space({"x": [0.0, 1.0]}),
        objective_name="my_objective",
        budget=25,
    )
    assert "`my_objective`" in text
    assert "25 trials per strategy" in text


def test_render_report_best_score_matches_result():
    space = parse_space({"x": [0.0, 1.0]})
    comparison = _comparison(space)
    text = render_report(comparison, space)
    for result in comparison.values():
        assert f"{result.best_score:.6g}" in text


def test_learning_curve_table_row_count():
    space = parse_space({"x": [0.0, 1.0]})
    comparison = _comparison(space, {"bayesian": {"n_initial": 2}})
    table = learning_curve_table(comparison)
    rows = [line for line in table.splitlines() if line.startswith("| ")]
    assert len(rows) == 1 + 1 + 10


def test_learning_curve_table_fills_forward_after_early_exhaustion():
    short = _trials([({"x": 0.0}, 0.9), ({"x": 0.5}, 0.4)])
    long = _trials([({"x": 0.0}, 0.8), ({"x": 0.3}, 0.5), ({"x": 0.7}, 0.6)])
    comparison = {"short": _result("short", short), "long": _result("long", long)}
    table = learning_curve_table(comparison)
    lines = [line for line in table.splitlines() if line.startswith("| ")]
    assert len(lines) == 2 + 3
    assert lines[2].split("|")[2].strip() == "0.9"
    assert lines[4].split("|")[2].strip() == "0.4"


def test_param_importance_categorical_groups():
    trials = _trials(
        [
            ({"k": "a"}, 0.1),
            ({"k": "a"}, 0.2),
            ({"k": "b"}, 0.9),
            ({"k": "b"}, 1.0),
        ]
    )
    comparison = {"r": _result("r", trials)}
    sections = param_importance(comparison, top_k=4)
    means = {label: mean for label, _, mean in sections["r"]["k"]}
    assert set(means) == {"a", "b"}
    assert means["a"] == pytest.approx(0.15)
    assert means["b"] == pytest.approx(0.95)


def test_param_importance_numeric_bins():
    trials = _trials(
        [( {"x": 0.0}, 0.9), ({"x": 0.1}, 0.8), ({"x": 0.9}, 0.1), ({"x": 1.0}, 0.2)]
    )
    comparison = {"r": _result("r", trials)}
    sections = param_importance(comparison, top_k=4, n_bins=2)
    groups = sections["r"]["x"]
    assert len(groups) == 2
    best_label, best_count, best_mean = groups[0]
    assert best_mean == pytest.approx(0.15)
    assert best_count == 2


def test_param_importance_single_value_numeric():
    trials = _trials([({"x": 0.5}, 1.0), ({"x": 0.5}, 0.5)])
    comparison = {"r": _result("r", trials)}
    sections = param_importance(comparison, top_k=2, n_bins=4)
    groups = sections["r"]["x"]
    assert len(groups) == 1
    assert groups[0][2] == pytest.approx(0.75)


def test_param_importance_respects_top_k():
    trials = _trials(
        [( {"x": 0.0}, 5.0), ({"x": 0.5}, 1.0), ({"x": 1.0}, 0.0)]
    )
    comparison = {"r": _result("r", trials)}
    sections = param_importance(comparison, top_k=1)
    assert sum(count for _, count, _ in sections["r"]["x"]) == 1


def test_write_report_creates_parent_directories(tmp_path):
    out = tmp_path / "nested" / "dir" / "report.md"
    write_report("# hi\n", str(out))
    assert out.exists()
    assert out.read_text(encoding="utf-8") == "# hi\n"


def test_render_report_mentions_tpe_cmaes_and_hyperband_caveats():
    space = parse_space({"x": [0.0, 1.0]})
    text = render_report(_comparison(space), space)
    assert "Hyperband" in text
    assert "TPE" in text
    assert "l(x)/g(x)" in text
    assert "CMA-ES" in text
    assert "covariance" in text


def test_render_report_custom_notes_appended():
    space = parse_space({"x": [0.0, 1.0]})
    text = render_report(
        _comparison(space), space, strategy_notes=["Grid exhausts below the budget."]
    )
    assert "- Grid exhausts below the budget." in text


def test_learning_curve_table_empty_comparison():
    assert learning_curve_table({}) == "| eval |  |\n| --- |  |"


def test_param_importance_empty_trials_skipped():
    comparison = {"r": RunResult("r", [], float("inf"), {})}
    assert param_importance(comparison) == {}
