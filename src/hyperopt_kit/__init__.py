"""hyperopt-kit: a lightweight hyperparameter optimization toolkit."""

from .searchers import (
    GaussianProcess,
    Trial,
    bayesian_search,
    expected_improvement,
    grid_search,
    random_search,
)
from .spaces import Categorical, FloatRange, IntRange, Space, parse_space

__version__ = "0.1.0"

__all__ = [
    "Categorical",
    "FloatRange",
    "GaussianProcess",
    "IntRange",
    "Space",
    "Trial",
    "bayesian_search",
    "expected_improvement",
    "grid_search",
    "parse_space",
    "random_search",
]
