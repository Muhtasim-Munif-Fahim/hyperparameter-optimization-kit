"""Markdown report rendering for search comparisons."""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple

import numpy as np

from .evaluate import RunResult, learning_curve


def _fmt(x: float) -> str:
    return f"{float(x):.6g}"


def learning_curve_table(comparison: Dict[str, RunResult]) -> str:
    """Best-so-far score per evaluation as a markdown table.

    A strategy that exhausts its grid or stops early keeps its final best
    value in later rows so the columns stay aligned.
    """
    strategies = list(comparison)
    curves = {s: learning_curve(comparison[s].trials) for s in strategies}
    max_len = max((len(c) for c in curves.values()), default=0)
    lines = [
        "| eval | " + " | ".join(strategies) + " |",
        "| --- | " + " | ".join("---" for _ in strategies) + " |",
    ]
    for i in range(max_len):
        cells = []
        for s in strategies:
            curve = curves[s]
            if curve:
                cells.append(_fmt(curve[min(i, len(curve) - 1)]))
            else:
                cells.append("")
        lines.append(f"| {i + 1} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _is_numeric(values: List) -> bool:
    return all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values)


def param_importance(
    comparison: Dict[str, RunResult], top_k: int = 10, n_bins: int = 4
) -> Dict[str, Dict[str, List[Tuple[str, int, float]]]]:
    """Aggregate the best ``top_k`` trials of each strategy by parameter value.

    Continuous parameters are split into quantile bins; categoricals are
    grouped by choice. Each group reports the trial count and the mean score
    of its members (lower is better), ordered from best to worst.
    """
    sections: Dict[str, Dict[str, List[Tuple[str, int, float]]]] = {}
    for strategy, result in comparison.items():
        trials = sorted(result.trials, key=lambda t: t.score)[:top_k]
        if not trials:
            continue
        names = list(trials[0].params)
        info: Dict[str, List[Tuple[str, int, float]]] = {}
        for name in names:
            values = [t.params[name] for t in trials]
            scores = [t.score for t in trials]
            if _is_numeric(values):
                quantiles = np.unique(
                    np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1))
                )
                if len(quantiles) == 1:
                    groups = [(_fmt(quantiles[0]), len(scores), float(np.mean(scores)))]
                else:
                    edges = quantiles[1:-1]
                    labels = [
                        f"{_fmt(a)}..{_fmt(b)}" for a, b in zip(quantiles, quantiles[1:])
                    ]
                    bins = np.digitize(values, edges, right=False)
                    groups = []
                    for i, label in enumerate(labels):
                        group_scores = [s for s, b in zip(scores, bins) if b == i]
                        if group_scores:
                            groups.append(
                                (label, len(group_scores), float(np.mean(group_scores)))
                            )
            else:
                seen: Dict[str, List[float]] = {}
                for value, score in zip(values, scores):
                    seen.setdefault(str(value), []).append(score)
                groups = [
                    (label, len(sc), float(np.mean(sc))) for label, sc in seen.items()
                ]
            groups.sort(key=lambda g: g[2])
            info[name] = groups
        sections[strategy] = info
    return sections


def _params_str(params: Dict) -> str:
    return json.dumps(params, default=str, sort_keys=True)


def render_report(
    comparison: Dict[str, RunResult],
    space,
    title: str = "Hyperparameter search comparison",
    objective_name: str = "objective",
    budget: Optional[int] = None,
    strategy_notes: Optional[List[str]] = None,
) -> str:
    """Render a full markdown report for a strategy comparison."""
    lines: List[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"Objective: `{objective_name}` (minimized).")
    if budget is not None:
        lines.append(f"Evaluation budget: {budget} trials per strategy.")
    lines.append("")

    lines.append("## Search space")
    lines.append("")
    lines.append("| parameter | type |")
    lines.append("| --- | --- |")
    for name, sp in space.items():
        lines.append(f"| `{name}` | {sp.describe()} |")
    lines.append("")

    lines.append("## Results")
    lines.append("")
    lines.append("| strategy | trials | best score | best parameters |")
    lines.append("| --- | ---: | ---: | --- |")
    for s, result in comparison.items():
        lines.append(
            f"| {s} | {len(result.trials)} | {_fmt(result.best_score)} | "
            f"`{_params_str(result.best_params)}` |"
        )
    lines.append("")

    lines.append("## Learning curves")
    lines.append("")
    lines.append("Best-so-far score after each evaluation (lower is better).")
    lines.append("")
    lines.append(learning_curve_table(comparison))
    lines.append("")

    lines.append("## Parameter importance")
    lines.append("")
    lines.append(
        "Best configurations per strategy, grouped by parameter value. Mean "
        "score is over the trials that fall into each group (lower is better)."
    )
    sections = param_importance(comparison)
    for strategy, info in sections.items():
        lines.append(f"### {strategy}")
        lines.append("")
        for name, groups in info.items():
            lines.append(f"**`{name}`**")
            lines.append("")
            lines.append("| value | count | mean score |")
            lines.append("| --- | ---: | ---: |")
            for label, count, mean in groups:
                lines.append(f"| {label} | {count} | {_fmt(mean)} |")
            lines.append("")

    lines.append("## Caveats")
    lines.append("")
    lines.append(
        "- The Gaussian-process surrogate treats categorical parameters as "
        "ordered, which can distort distances for nominal categories."
    )
    lines.append(
        "- Bayesian search starts from random points and only becomes "
        "sample-efficient once the surrogate has enough evaluations."
    )
    lines.append(
        "- Results on a noisy objective are stochastic; rerun with a "
        "different seed to gauge stability."
    )
    lines.append(
        "- Grid search cost grows with the product of per-dimension "
        "resolution; use it on low-dimensional spaces."
    )
    lines.append(
        "- Log-scale ranges are sampled and mutated on the log axis, so "
        "equal steps in normalized space are multiplicative on the raw scale."
    )
    if strategy_notes:
        for note in strategy_notes:
            lines.append(f"- {note}")
    return "\n".join(lines) + "\n"


def write_report(text: str, path) -> str:
    """Write the report to ``path``, creating parent directories."""
    from pathlib import Path

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return str(target)
