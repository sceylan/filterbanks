import numpy as np
import pytest

from filterbanks.exceptions import ValidationError
from filterbanks.signals import Signal, prepare_signals


def test_signal_valid_with_x_and_sampling_rate() -> None:
    x = np.arange(0.0, 1.0, 0.1)
    y = np.sin(2.0 * np.pi * x)
    sig = Signal(y=y, x=x, sampling_rate=10.0, name="a")
    assert sig.name == "a"
    assert sig.sampling_rate == pytest.approx(10.0)


def test_signal_x_only_infers_sampling_rate() -> None:
    x = np.array([0.0, 0.1, 0.2, 0.3])
    y = np.ones_like(x)
    sig = Signal(y=y, x=x, sampling_rate=None)
    assert sig.sampling_rate == pytest.approx(10.0)


def test_signal_irregular_x_rejected() -> None:
    x = np.array([0.0, 0.1, 0.23, 0.3])
    y = np.ones_like(x)
    with pytest.raises(ValidationError, match="uniformly"):
        Signal(y=y, x=x, sampling_rate=None)


def test_signal_sampling_rate_conflict_rejected() -> None:
    x = np.array([0.0, 0.1, 0.2, 0.3])
    y = np.ones_like(x)
    with pytest.raises(ValidationError, match="disagree"):
        Signal(y=y, x=x, sampling_rate=20.0)


def test_signal_non_finite_y_rejected() -> None:
    with pytest.raises(ValidationError, match="finite"):
        Signal(y=np.array([0.0, np.inf, 1.0]), sampling_rate=10.0)


def test_prepare_signals_from_2d_array() -> None:
    ys = np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
    signals = prepare_signals(ys, sampling_rate=5.0, names=["s1", "s2"])
    assert len(signals) == 2
    assert signals[0].name == "s1"
    assert signals[1].name == "s2"
