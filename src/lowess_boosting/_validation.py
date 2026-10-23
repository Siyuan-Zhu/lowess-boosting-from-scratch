"""Hyper-parameter checks shared by the estimators."""

from __future__ import annotations

import math
import numbers
from collections.abc import Iterable
from typing import Any

import numpy as np


def check_fraction(name: str, value: Any) -> None:
    """Require a real number in (0, 1]."""
    if not _is_real(value) or not 0 < value <= 1:
        raise ValueError(f"{name} must be a float in (0, 1], got {value!r}.")


def check_non_negative(name: str, value: Any) -> None:
    """Require a finite real number >= 0."""
    if not _is_real(value) or not 0 <= value < math.inf:
        raise ValueError(f"{name} must be a finite float >= 0, got {value!r}.")


def check_positive_or_none(name: str, value: Any) -> None:
    """Require None or a finite real number > 0."""
    if value is not None and (not _is_real(value) or not 0 < value < math.inf):
        raise ValueError(f"{name} must be None or a finite float > 0, got {value!r}.")


def check_choice(name: str, value: Any, choices: Iterable[str]) -> None:
    """Require one of the given strings."""
    options = sorted(choices)
    if not isinstance(value, str) or value not in options:
        raise ValueError(f"{name} must be one of {options}, got {value!r}.")


def as_numeric_target(y: np.ndarray) -> np.ndarray:
    """Return ``y`` as float64, or raise a ValueError naming the problem."""
    try:
        return np.asarray(y, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"y must be numeric for regression; got values of dtype {y.dtype}."
        ) from exc


def check_int(name: str, value: Any, *, low: int) -> None:
    """Require an integer >= ``low`` (booleans are rejected)."""
    if (
        not isinstance(value, numbers.Integral)
        or isinstance(value, bool)
        or value < low
    ):
        raise ValueError(f"{name} must be an int >= {low}, got {value!r}.")


def count_of(fraction: float, n: int) -> int:
    """``fraction * n`` rounded up, ignoring floating-point noise.

    ``0.07 * 100`` evaluates to ``7.000000000000001``; a plain ``ceil`` would
    give 8.
    """
    return max(1, math.ceil(round(fraction * n, 9)))


def _is_real(value: Any) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool)
