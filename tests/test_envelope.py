from __future__ import annotations

import inspect

import numpy as np
import pytest

from filterbanks.bands import FilterBand
from filterbanks.envelope import add_envelopes_to_results, compute_envelope, smooth_envelope
from filterbanks.exceptions import ValidationError
from filterbanks.filtering import BandResult, SignalResult


def test_default_smoothing_window_seconds_is_5s() -> None:
    smooth_default = inspect.signature(smooth_envelope).parameters["window_seconds"].default
    add_default = inspect.signature(add_envelopes_to_results).parameters["smooth_window_seconds"].default
    assert float(smooth_default) == pytest.approx(5.0)
    assert float(add_default) == pytest.approx(5.0)


def test_hilbert_envelope_length_and_sine_behavior() -> None:
    sr = 200.0
    t = np.arange(0.0, 10.0, 1.0 / sr)
    y = 2.0 * np.sin(2.0 * np.pi * 5.0 * t)

    env = compute_envelope(y, sampling_rate=sr, method="hilbert")

    assert env.size == y.size
    assert np.all(env >= 0.0)
    central = env[200:-200]
    assert float(np.mean(central)) == pytest.approx(2.0, rel=0.03)


def test_hilbert_fft_returns_same_length() -> None:
    y = np.sin(np.linspace(0.0, 20.0, 1234))
    env = compute_envelope(y, method="hilbert_fft")
    assert env.size == y.size


def test_moving_average_same_mode_returns_same_length() -> None:
    y = np.linspace(0.0, 1.0, 100)
    out = smooth_envelope(y, sampling_rate=10.0, method="moving_average", window_seconds=1.2, mode="same")
    assert out.size == y.size


def test_moving_average_valid_mode_returns_shorter_length() -> None:
    y = np.linspace(0.0, 1.0, 100)
    sr = 10.0
    window_seconds = 2.0
    out = smooth_envelope(y, sampling_rate=sr, method="moving_average", window_seconds=window_seconds, mode="valid")
    window = max(1, int(window_seconds * sr))
    window = min(window, y.size)
    expected = y.size - window + 1
    assert out.size == expected


def test_edge_policy_nan_mask_introduces_nans() -> None:
    y = np.sin(np.linspace(0.0, 10.0, 200))
    out = smooth_envelope(y, sampling_rate=20.0, window_seconds=1.0, mode="same", edge_policy="nan_mask")
    assert np.isnan(out).any()


def test_edge_policy_keep_has_no_nans() -> None:
    y = np.sin(np.linspace(0.0, 10.0, 200))
    out = smooth_envelope(y, sampling_rate=20.0, window_seconds=1.0, mode="same", edge_policy="keep")
    assert np.isfinite(out).all()


def test_callable_envelope_method_is_accepted() -> None:
    y = np.array([0.0, -1.0, 2.0, -3.0])

    def _env(arr: np.ndarray, sampling_rate: float | None, **kwargs: object) -> np.ndarray:
        assert sampling_rate == 50.0
        return np.abs(arr)

    out = compute_envelope(y, sampling_rate=50.0, method=_env)
    assert np.allclose(out, np.abs(y))


def test_callable_smoothing_method_is_accepted() -> None:
    y = np.array([1.0, 2.0, 3.0])

    def _smooth(arr: np.ndarray, sampling_rate: float, **kwargs: object) -> np.ndarray:
        assert sampling_rate == 40.0
        return arr + 1.0

    out = smooth_envelope(y, sampling_rate=40.0, method=_smooth)
    assert np.allclose(out, np.array([2.0, 3.0, 4.0]))


def test_invalid_method_names_raise_validation_error() -> None:
    y = np.ones(10)
    with pytest.raises(ValidationError, match="envelope method"):
        compute_envelope(y, method="bad")
    with pytest.raises(ValidationError, match="smoothing method"):
        smooth_envelope(y, sampling_rate=10.0, method="bad")


def test_invalid_mode_raises_validation_error() -> None:
    with pytest.raises(ValidationError, match="mode"):
        smooth_envelope(np.ones(10), sampling_rate=10.0, mode="full")


def test_invalid_edge_policy_raises_validation_error() -> None:
    with pytest.raises(ValidationError, match="edge_policy"):
        smooth_envelope(np.ones(10), sampling_rate=10.0, edge_policy="bad")


@pytest.mark.parametrize(
    "sampling_rate,window_seconds",
    [
        (0.0, 1.0),
        (-1.0, 1.0),
        (10.0, 0.0),
        (10.0, -2.0),
        (float("inf"), 1.0),
        (10.0, float("nan")),
    ],
)
def test_invalid_sampling_rate_or_window_seconds_raises(
    sampling_rate: float,
    window_seconds: float,
) -> None:
    with pytest.raises(ValidationError, match="positive and finite"):
        smooth_envelope(np.ones(10), sampling_rate=sampling_rate, window_seconds=window_seconds)


def test_add_envelopes_to_results_processed_only() -> None:
    band = FilterBand(
        name="b1",
        center_frequency=2.0,
        fmin=1.0,
        fmax=3.0,
        original_fmin=1.0,
        original_fmax=3.0,
    )
    processed = BandResult(band=band, status="processed", filtered_y=np.sin(np.linspace(0.0, 20.0, 200)))
    skipped = BandResult(band=band, status="skipped", filtered_y=None, reason="signal_too_short")
    invalid = BandResult(band=band, status="invalid", filtered_y=None, reason="invalid_band_edges")
    results = [SignalResult(signal_name="s", sampling_rate=20.0, sample_count=200, band_results=[processed, skipped, invalid])]

    add_envelopes_to_results(
        results,
        envelope_method="hilbert",
        smooth_method="moving_average",
        smooth_window_seconds=1.0,
        smooth_mode="same",
        smooth_edge_policy="keep",
    )

    assert processed.envelope is not None
    assert processed.smoothed_envelope is not None
    assert processed.envelope.size == processed.filtered_y.size
    assert processed.smoothed_envelope.size == processed.filtered_y.size
    assert skipped.envelope is None and skipped.smoothed_envelope is None
    assert invalid.envelope is None and invalid.smoothed_envelope is None
