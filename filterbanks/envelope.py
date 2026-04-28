"""Envelope and smoothing helpers."""

from __future__ import annotations

from collections.abc import Callable
import math
from typing import Any

import numpy as np

from .exceptions import ValidationError
from .filtering import SignalResult


def compute_envelope(
    y: np.ndarray,
    sampling_rate: float | None = None,
    method: str | Callable[..., np.ndarray] = "hilbert",
    **kwargs: Any,
) -> np.ndarray:
    signal = _as_finite_1d(y, "y")

    if callable(method):
        out = method(signal, sampling_rate, **kwargs)
        return _validate_output(out, signal.size, "envelope")

    method_name = str(method).lower()
    if method_name not in {"hilbert", "hilbert_fft"}:
        raise ValidationError("envelope method must be one of: hilbert, hilbert_fft, or callable.")

    try:
        from scipy.fft import next_fast_len
        from scipy.signal import hilbert
    except Exception as exc:
        raise ValidationError("SciPy is required for Hilbert envelope methods.") from exc

    if method_name == "hilbert":
        out = np.abs(hilbert(signal))
    else:
        n = signal.size
        n_fast = int(next_fast_len(n))
        padded = np.pad(signal, (0, n_fast - n), mode="constant") if n_fast > n else signal
        out = np.abs(hilbert(padded))[:n]

    return _validate_output(out, signal.size, "envelope")


def smooth_envelope(
    y: np.ndarray,
    sampling_rate: float,
    method: str | Callable[..., np.ndarray] = "moving_average",
    window_seconds: float = 5.0,
    mode: str = "same",
    edge_policy: str = "keep",
    **kwargs: Any,
) -> np.ndarray:
    signal = _as_finite_1d(y, "y")
    _validate_positive_finite(sampling_rate, "sampling_rate")
    _validate_positive_finite(window_seconds, "window_seconds")

    conv_mode = str(mode).lower()
    if conv_mode not in {"same", "valid"}:
        raise ValidationError("mode must be one of: same, valid.")

    policy = str(edge_policy).lower()
    if policy not in {"keep", "nan_mask"}:
        raise ValidationError("edge_policy must be one of: keep, nan_mask.")

    if callable(method):
        out = np.asarray(method(signal, sampling_rate, **kwargs), dtype=float)
    else:
        method_name = str(method).lower()
        if method_name != "moving_average":
            raise ValidationError("smoothing method must be moving_average or callable.")

        n_window = max(1, int(float(window_seconds) * float(sampling_rate)))
        n_window = min(n_window, signal.size)
        kernel = np.ones(n_window, dtype=float) / float(n_window)
        out = np.convolve(signal, kernel, mode=conv_mode)

    if out.ndim != 1:
        raise ValidationError("smoothed output must be 1D.")

    if policy == "nan_mask":
        taper = int(2.0 * float(window_seconds) * float(sampling_rate))
        if taper > 0 and out.size > 0:
            out = out.copy()
            left = min(taper, out.size)
            right_start = max(out.size - taper, 0)
            out[:left] = np.nan
            out[right_start:] = np.nan
        return out

    if not np.isfinite(out).all():
        raise ValidationError("smoothed output must be finite for edge_policy='keep'.")
    return out


def add_envelopes_to_results(
    results: list[SignalResult],
    *,
    envelope_method: str | Callable[..., np.ndarray] = "hilbert",
    smooth_method: str | Callable[..., np.ndarray] = "moving_average",
    smooth_window_seconds: float = 5.0,
    smooth_mode: str = "same",
    smooth_edge_policy: str = "keep",
    **kwargs: Any,
) -> list[SignalResult]:
    """Compute envelope fields in-place for processed band results only."""
    for signal_result in results:
        for band_result in signal_result.band_results:
            if band_result.status != "processed" or band_result.filtered_y is None:
                continue

            env = compute_envelope(
                band_result.filtered_y,
                sampling_rate=signal_result.sampling_rate,
                method=envelope_method,
                **kwargs,
            )
            smooth = smooth_envelope(
                env,
                sampling_rate=signal_result.sampling_rate,
                method=smooth_method,
                window_seconds=smooth_window_seconds,
                mode=smooth_mode,
                edge_policy=smooth_edge_policy,
                **kwargs,
            )
            band_result.envelope = env
            band_result.smoothed_envelope = smooth

    return results


def _as_finite_1d(y: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(y, dtype=float)
    if arr.ndim != 1:
        raise ValidationError(f"{name} must be 1D.")
    if not np.isfinite(arr).all():
        raise ValidationError(f"{name} must contain only finite values.")
    return arr


def _validate_output(y: np.ndarray, expected_len: int, label: str) -> np.ndarray:
    out = np.asarray(y, dtype=float)
    if out.ndim != 1:
        raise ValidationError(f"{label} output must be 1D.")
    if out.size != expected_len:
        raise ValidationError(f"{label} output length must match input length.")
    if not np.isfinite(out).all():
        raise ValidationError(f"{label} output must be finite.")
    return out


def _validate_positive_finite(value: float, name: str) -> None:
    if not math.isfinite(float(value)) or float(value) <= 0.0:
        raise ValidationError(f"{name} must be positive and finite.")


__all__ = ["compute_envelope", "smooth_envelope", "add_envelopes_to_results"]
