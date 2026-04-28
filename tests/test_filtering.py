from __future__ import annotations

from importlib.util import find_spec

import numpy as np
import pytest

from filterbanks.bands import create_bands
from filterbanks.exceptions import ValidationError
from filterbanks.filtering import apply_filterbank, preprocess_signal
from filterbanks.signals import Signal

HAS_SCIPY = find_spec("scipy") is not None
HAS_OBSPY = find_spec("obspy") is not None


def _sine_signal(*, freq: float, sampling_rate: float = 100.0, seconds: float = 8.0) -> Signal:
    t = np.arange(0.0, seconds, 1.0 / sampling_rate)
    y = np.sin(2.0 * np.pi * freq * t)
    return Signal(y=y, sampling_rate=sampling_rate, name=f"sine_{freq:g}")


def _single_band(*, fmin: float, fmax: float, center: float) -> list:
    return create_bands(
        spacing_type="explicit",
        explicit_bands=[{"name": "band_001", "center_frequency": center}],
        bandwidth_type="hz",
        bandwidth_value=fmax - fmin,
        sampling_rate=None,
    )


def test_preprocess_demean() -> None:
    y = np.sin(np.linspace(0.0, 5.0, 200)) + 5.0
    processed, meta = preprocess_signal(y, detrend="demean", taper=False)
    assert abs(float(np.mean(processed))) < 1e-12
    assert meta["detrend"] == "demean"


def test_apply_filterbank_rejects_empty_signals() -> None:
    band = create_bands(
        spacing_type="linear",
        spacing_step_hz=1.0,
        bandwidth_type="hz",
        bandwidth_value=1.0,
        freq_min=4.0,
        freq_max=4.0,
    )
    with pytest.raises(ValidationError, match="at least one"):
        apply_filterbank([], band)


@pytest.mark.skipif(not HAS_SCIPY, reason="SciPy is required")
def test_scipy_passband_and_stopband_behavior() -> None:
    band = create_bands(
        spacing_type="linear",
        spacing_step_hz=1.0,
        bandwidth_type="hz",
        bandwidth_value=2.0,
        freq_min=5.0,
        freq_max=5.0,
    )

    pass_sig = _sine_signal(freq=5.0)
    stop_sig = _sine_signal(freq=15.0)

    pass_result = apply_filterbank(pass_sig, band, backend="scipy", detrend="none", taper=False)
    stop_result = apply_filterbank(stop_sig, band, backend="scipy", detrend="none", taper=False)

    y_pass = pass_result[0].band_results[0].filtered_y
    y_stop = stop_result[0].band_results[0].filtered_y
    assert y_pass is not None
    assert y_stop is not None

    pass_ratio = np.sqrt(np.mean(y_pass**2)) / np.sqrt(np.mean(pass_sig.y**2))
    stop_ratio = np.sqrt(np.mean(y_stop**2)) / np.sqrt(np.mean(stop_sig.y**2))

    assert pass_ratio > 0.7
    assert stop_ratio < 0.2


@pytest.mark.skipif(not (HAS_SCIPY and HAS_OBSPY), reason="Requires ObsPy and SciPy")
def test_backend_parity_obspy_vs_scipy() -> None:
    sampling_rate = 100.0
    t = np.arange(0.0, 10.0, 1.0 / sampling_rate)
    y = np.sin(2.0 * np.pi * 5.0 * t) + 0.2 * np.sin(2.0 * np.pi * 12.0 * t)
    sig = Signal(y=y, sampling_rate=sampling_rate)

    band = create_bands(
        spacing_type="linear",
        spacing_step_hz=1.0,
        bandwidth_type="hz",
        bandwidth_value=2.0,
        freq_min=5.0,
        freq_max=5.0,
    )

    ob = apply_filterbank(sig, band, backend="obspy", detrend="none", taper=False)
    sc = apply_filterbank(sig, band, backend="scipy", detrend="none", taper=False)

    y_ob = ob[0].band_results[0].filtered_y
    y_sc = sc[0].band_results[0].filtered_y
    assert y_ob is not None and y_sc is not None

    central = slice(200, -200)
    rel_rms = np.sqrt(np.mean((y_sc[central] - y_ob[central]) ** 2)) / np.sqrt(np.mean(y_ob[central] ** 2))
    assert rel_rms < 0.02


def test_short_signal_is_skipped() -> None:
    backend = "scipy" if HAS_SCIPY else "obspy"
    if backend == "obspy" and not HAS_OBSPY:
        pytest.skip("No backend available")

    sig = Signal(y=np.linspace(0.0, 1.0, 10), sampling_rate=100.0)
    band = create_bands(
        spacing_type="linear",
        spacing_step_hz=1.0,
        bandwidth_type="hz",
        bandwidth_value=2.0,
        freq_min=5.0,
        freq_max=5.0,
    )
    result = apply_filterbank(sig, band, backend=backend, detrend="none", taper=False)
    br = result[0].band_results[0]

    assert br.status == "skipped"
    assert br.reason == "signal_too_short"
    assert br.filtered_y is None
    assert "signal_too_short" in br.warnings


def test_low_cycles_warning_is_runtime_only() -> None:
    backend = "scipy" if HAS_SCIPY else "obspy"
    if backend == "obspy" and not HAS_OBSPY:
        pytest.skip("No backend available")

    sig = _sine_signal(freq=0.5, sampling_rate=100.0, seconds=2.0)
    band = create_bands(
        spacing_type="linear",
        spacing_step_hz=1.0,
        bandwidth_type="hz",
        bandwidth_value=0.6,
        freq_min=0.5,
        freq_max=0.5,
    )

    assert "low_cycles_per_record" not in band[0].warnings

    result = apply_filterbank(sig, band, backend=backend, detrend="none", taper=False)
    warnings = result[0].band_results[0].warnings
    assert "low_cycles_per_record" in warnings


def test_default_phase_is_causal() -> None:
    backend = "scipy" if HAS_SCIPY else "obspy"
    if backend == "obspy" and not HAS_OBSPY:
        pytest.skip("No backend available")

    sig = _sine_signal(freq=5.0, sampling_rate=100.0, seconds=4.0)
    band = create_bands(
        spacing_type="linear",
        spacing_step_hz=1.0,
        bandwidth_type="hz",
        bandwidth_value=2.0,
        freq_min=5.0,
        freq_max=5.0,
    )
    result = apply_filterbank(sig, band, backend=backend, detrend="none", taper=False)
    assert result[0].diagnostics["phase"] == "causal"


def test_zero_phase_remains_accepted_explicitly() -> None:
    backend = "scipy" if HAS_SCIPY else "obspy"
    if backend == "obspy" and not HAS_OBSPY:
        pytest.skip("No backend available")

    sig = _sine_signal(freq=5.0, sampling_rate=100.0, seconds=4.0)
    band = create_bands(
        spacing_type="linear",
        spacing_step_hz=1.0,
        bandwidth_type="hz",
        bandwidth_value=2.0,
        freq_min=5.0,
        freq_max=5.0,
    )
    result = apply_filterbank(sig, band, backend=backend, phase="zero", detrend="none", taper=False)
    assert result[0].diagnostics["phase"] == "zero"
