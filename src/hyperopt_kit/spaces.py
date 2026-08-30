"""Search-space primitives for hyperparameter optimization.

Each :class:`Space` describes one hyperparameter: how to sample it, how to
mutate a value into a neighbouring value, and how to map between the raw
parameter scale and the unit interval used by surrogate models.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


class Space:
    """Base class for a single hyperparameter's search space."""

    def sample(self, rng: np.random.Generator) -> Any:
        """Draw a random value from the space."""
        raise NotImplementedError

    def mutate(self, value: Any, rng: np.random.Generator) -> Any:
        """Return a value one step away from ``value``."""
        raise NotImplementedError

    def normalize(self, value: Any) -> float:
        """Map a raw value onto [0, 1] for surrogate models."""
        raise NotImplementedError

    def denormalize(self, x: float) -> Any:
        """Map a value in [0, 1] back onto the raw parameter scale."""
        raise NotImplementedError

    def describe(self) -> str:
        """Human-readable one-line description used in reports."""
        raise NotImplementedError


class Categorical(Space):
    """A discrete space over an arbitrary list of choices."""

    def __init__(self, choices: Sequence[Any]):
        if len(choices) == 0:
            raise ValueError("categorical space needs at least one choice")
        self.choices = list(choices)

    def sample(self, rng: np.random.Generator) -> Any:
        return self.choices[int(rng.integers(len(self.choices)))]

    def mutate(self, value: Any, rng: np.random.Generator) -> Any:
        if len(self.choices) == 1:
            return value
        idx = self.choices.index(value)
        direction = 1 if rng.random() < 0.5 else -1
        new_idx = idx + direction
        if new_idx < 0 or new_idx >= len(self.choices):
            new_idx = idx - direction
        return self.choices[new_idx]

    def normalize(self, value: Any) -> float:
        idx = self.choices.index(value)
        if len(self.choices) == 1:
            return 0.5
        return idx / (len(self.choices) - 1)

    def denormalize(self, x: float) -> Any:
        if len(self.choices) == 1:
            return self.choices[0]
        idx = int(round(float(np.clip(x, 0.0, 1.0)) * (len(self.choices) - 1)))
        return self.choices[idx]

    def describe(self) -> str:
        return f"categorical{self.choices}"


class FloatRange(Space):
    """A continuous space on ``[low, high]``, optionally log-scaled."""

    def __init__(self, low: float, high: float, log_scale: bool = False):
        self.low = float(low)
        self.high = float(high)
        self.log_scale = bool(log_scale)
        if self.high < self.low:
            raise ValueError(f"FloatRange high {self.high} is below low {self.low}")
        if self.log_scale and self.low <= 0.0:
            raise ValueError("log_scale FloatRange requires low > 0")

    @property
    def degenerate(self) -> bool:
        return self.low == self.high

    def sample(self, rng: np.random.Generator) -> float:
        if self.degenerate:
            return self.low
        if self.log_scale:
            return float(np.exp(rng.uniform(np.log(self.low), np.log(self.high))))
        return float(rng.uniform(self.low, self.high))

    def mutate(self, value: Any, rng: np.random.Generator) -> float:
        x = self.normalize(value)
        step = float(rng.normal(0.0, 0.1))
        return self.denormalize(float(np.clip(x + step, 0.0, 1.0)))

    def normalize(self, value: Any) -> float:
        value = float(value)
        if self.degenerate:
            return 0.5
        if self.log_scale:
            return (math.log(value) - math.log(self.low)) / (
                math.log(self.high) - math.log(self.low)
            )
        return (value - self.low) / (self.high - self.low)

    def denormalize(self, x: float) -> float:
        x = float(np.clip(x, 0.0, 1.0))
        if self.degenerate:
            return self.low
        if self.log_scale:
            value = math.exp(math.log(self.low) + x * (math.log(self.high) - math.log(self.low)))
        else:
            value = self.low + x * (self.high - self.low)
        return float(np.clip(value, self.low, self.high))

    def describe(self) -> str:
        tag = " log" if self.log_scale else ""
        return f"float[{self.low:g}, {self.high:g}]{tag}"


class IntRange(Space):
    """A discrete space on the integers ``[low, high]`` (inclusive)."""

    def __init__(self, low: int, high: int, log_scale: bool = False):
        self.low = int(low)
        self.high = int(high)
        self.log_scale = bool(log_scale)
        if self.high < self.low:
            raise ValueError(f"IntRange high {self.high} is below low {self.low}")
        if self.log_scale and self.low <= 0:
            raise ValueError("log_scale IntRange requires low > 0")

    @property
    def n_values(self) -> int:
        return self.high - self.low + 1

    def sample(self, rng: np.random.Generator) -> int:
        if self.log_scale:
            v = np.exp(rng.uniform(np.log(self.low), np.log(self.high)))
            return int(np.clip(round(v), self.low, self.high))
        return int(rng.integers(self.low, self.high + 1))

    def mutate(self, value: Any, rng: np.random.Generator) -> int:
        if self.n_values == 1:
            return self.low
        v = int(value)
        delta = 1 if rng.random() < 0.5 else -1
        new = int(np.clip(v + delta, self.low, self.high))
        if new == v:
            new = int(np.clip(v - delta, self.low, self.high))
        return new

    def normalize(self, value: Any) -> float:
        if self.n_values == 1:
            return 0.5
        return (int(value) - self.low) / (self.high - self.low)

    def denormalize(self, x: float) -> int:
        if self.n_values == 1:
            return self.low
        return int(round(self.low + float(np.clip(x, 0.0, 1.0)) * (self.high - self.low)))

    def describe(self) -> str:
        tag = " log" if self.log_scale else ""
        return f"int[{self.low}, {self.high}]{tag}"


def _parse_param(entry: Any) -> Space:
    """Parse a single compact parameter spec into a :class:`Space`."""
    if isinstance(entry, dict):
        kind = entry.get("type")
        if kind == "float":
            return FloatRange(
                entry["low"], entry["high"], log_scale=bool(entry.get("log", False))
            )
        if kind == "int":
            return IntRange(
                entry["low"], entry["high"], log_scale=bool(entry.get("log", False))
            )
        if kind == "categorical":
            return Categorical(entry["choices"])
        raise ValueError(f"unknown space type {kind!r} in {entry!r}")

    if not isinstance(entry, (list, tuple)):
        raise ValueError(f"parameter spec must be a list or dict, got {entry!r}")

    entry = list(entry)
    log = False
    kind = None
    if entry and entry[-1] == "int":
        kind = "int"
        entry = entry[:-1]
    if entry and entry[-1] == "log":
        log = True
        entry = entry[:-1]
        if kind is None:
            kind = "float"
    if entry and entry[-1] == "int":
        kind = "int"
        entry = entry[:-1]

    if (
        len(entry) == 2
        and isinstance(entry[0], (int, float))
        and isinstance(entry[1], (int, float))
        and not isinstance(entry[0], bool)
        and not isinstance(entry[1], bool)
    ):
        if kind == "int":
            return IntRange(entry[0], entry[1], log_scale=log)
        return FloatRange(entry[0], entry[1], log_scale=log)

    if log or kind == "int":
        raise ValueError(f"cannot interpret {entry!r} as a bounded range spec")

    return Categorical(entry)


def parse_space(spec: Dict[str, Any]) -> Dict[str, Space]:
    """Convert a compact dict spec into a ``{name: Space}`` mapping.

    Supported compact forms per parameter:

    ``[low, high]``
        :class:`FloatRange`
    ``[low, high, "log"]``
        log-scaled :class:`FloatRange`
    ``[low, high, "int"]``
        :class:`IntRange`
    ``[low, high, "int", "log"]``
        log-scaled :class:`IntRange`
    ``[choice1, choice2, ...]``
        :class:`Categorical`

    The same spaces can be expressed as dicts: ``{"type": "float"|"int",
    "low": ..., "high": ..., "log": bool}`` and
    ``{"type": "categorical", "choices": [...]}``.
    """
    if not isinstance(spec, dict) or not spec:
        raise ValueError("space spec must be a non-empty mapping of name -> spec")
    return {str(name): _parse_param(value) for name, value in spec.items()}
