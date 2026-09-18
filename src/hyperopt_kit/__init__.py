"""hyperopt-kit: a lightweight hyperparameter optimization toolkit."""

from .evaluate import RunResult, available_strategies, compare_strategies, run_search
from .report import render_report, write_report
from .searchers import (
    GaussianProcess,
    Trial,
    adaptive_parzen,
    bayesian_search,
    expected_improvement,
    grid_search,
    hyperband_brackets,
    hyperband_search,
    random_search,
    successive_halving,
    tpe_log_density_ratio,
    tpe_search,
    tpe_split,
)
from .spaces import Categorical, FloatRange, IntRange, Space, parse_space

__version__ = "0.1.0"

__all__ = [
    "Categorical",
    "FloatRange",
    "GaussianProcess",
    "IntRange",
    "RunResult",
    "Space",
    "Trial",
    "adaptive_parzen",
    "available_strategies",
    "bayesian_search",
    "compare_strategies",
    "expected_improvement",
    "grid_search",
    "hyperband_brackets",
    "hyperband_search",
    "parse_space",
    "random_search",
    "render_report",
    "run_search",
    "successive_halving",
    "tpe_log_density_ratio",
    "tpe_search",
    "tpe_split",
    "write_report",
]
