"""Signal model and input normalization."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Sequence

import numpy as np

from .exceptions import ValidationError

REL_TOL = 1e-6
ABS_TOL = 1e-12


@dataclass
class Signal:
    y: np.ndarray
    x: np.ndarray | None = None
    sampling_rate: float | None = None
    name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.y = np.asarray(self.y, dtype=float)
        self.x = None if self.x is None else np.asarray(self.x, dtype=float)
        self.sampling_rate = None if self.sampling_rate is None else float(self.sampling_rate)
        validate_signal(self)


def validate_signal(signal: Signal) -> Signal:
    if signal.y.ndim != 1 or signal.y.size == 0:
        raise ValidationError("Signal y must be non-empty 1D.")
    if not np.isfinite(signal.y).all():
        raise ValidationError("Signal y must contain only finite values.")

    inferred: float | None = None
    if signal.x is not None:
        if signal.x.ndim != 1 or signal.x.shape != signal.y.shape:
            raise ValidationError("Signal x must be 1D and match y shape.")
        if not np.isfinite(signal.x).all():
            raise ValidationError("Signal x must contain only finite values.")
        if signal.x.size < 2:
            raise ValidationError("Signal x must have at least two samples.")
        dx = np.diff(signal.x)
        if not np.allclose(dx, dx[0], rtol=REL_TOL, atol=ABS_TOL):
            raise ValidationError("Signal x must be uniformly sampled.")
        if dx[0] <= 0:
            raise ValidationError("Signal x spacing must be positive.")
        inferred = 1.0 / float(dx[0])

    if signal.sampling_rate is None:
        if inferred is None:
            raise ValidationError("sampling_rate is required when x is not provided.")
        signal.sampling_rate = inferred
    else:
        if not math.isfinite(signal.sampling_rate) or signal.sampling_rate <= 0:
            raise ValidationError("sampling_rate must be positive and finite.")
        if inferred is not None and not math.isclose(signal.sampling_rate, inferred, rel_tol=REL_TOL, abs_tol=ABS_TOL):
            raise ValidationError("x and sampling_rate disagree.")
    return signal


def prepare_signals(
    signals: Signal | np.ndarray | Sequence[Any],
    *,
    x: np.ndarray | Sequence[float] | None = None,
    sampling_rate: float | None = None,
    names: Sequence[str | None] | None = None,
) -> list[Signal]:
    if isinstance(signals, Signal):
        return [validate_signal(signals)]

    if isinstance(signals, np.ndarray):
        if signals.ndim == 1:
            return [Signal(signals, x=x, sampling_rate=sampling_rate, name=names[0] if names else None)]
        if signals.ndim != 2:
            raise ValidationError("signals ndarray must be 1D or 2D.")
        if names is not None and len(names) != signals.shape[0]:
            raise ValidationError("names length must match number of signals.")
        return [Signal(signals[i], x=x, sampling_rate=sampling_rate, name=None if names is None else names[i]) for i in range(signals.shape[0])]

    if not isinstance(signals, Sequence) or len(signals) == 0:
        raise ValidationError("signals must contain at least one signal.")

    if isinstance(signals[0], (float, int, np.floating, np.integer)):
        return [Signal(np.asarray(signals, dtype=float), x=x, sampling_rate=sampling_rate, name=names[0] if names else None)]

    if names is not None and len(names) != len(signals):
        raise ValidationError("names length must match number of signals.")

    out: list[Signal] = []
    for i, item in enumerate(signals):
        name = None if names is None else names[i]
        if isinstance(item, Signal):
            if name is not None:
                item.name = name
            out.append(validate_signal(item))
        else:
            arr = np.asarray(item, dtype=float)
            if arr.ndim != 1:
                raise ValidationError("Each signal array must be 1D.")
            out.append(Signal(arr, x=x, sampling_rate=sampling_rate, name=name))
    return out


__all__ = ["Signal", "validate_signal", "prepare_signals"]
