"""Filterbank application with minimal runtime diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.util import find_spec
import math
from typing import Any, Iterable, Literal

import numpy as np

from .bands import FilterBand
from .exceptions import ValidationError
from .signals import Signal, validate_signal

LOW_CYCLES_THRESHOLD = 3.0


@dataclass
class BandResult:
    band: FilterBand
    status: Literal["processed", "skipped", "invalid"]
    filtered_y: np.ndarray | None
    envelope: np.ndarray | None = None
    smoothed_envelope: np.ndarray | None = None
    warnings: list[str] = field(default_factory=list)
    reason: str | None = None
    backend: str | None = None


@dataclass
class SignalResult:
    signal_name: str | None
    sampling_rate: float
    sample_count: int
    x: np.ndarray | None = None
    waveform_y: np.ndarray | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    band_results: list[BandResult] = field(default_factory=list)


def preprocess_signal(
    y: np.ndarray,
    *,
    detrend: str = "demean",
    taper: bool = True,
    taper_fraction: float = 0.05,
) -> tuple[np.ndarray, dict[str, Any]]:
    mode = detrend.lower()
    if mode not in {"demean", "none"}:
        raise ValidationError("detrend must be one of: demean, none.")
    if not (0.0 <= taper_fraction <= 0.5):
        raise ValidationError("taper_fraction must be in [0.0, 0.5].")

    out = np.asarray(y, dtype=float).copy()
    meta: dict[str, Any] = {"detrend": mode, "taper": bool(taper), "taper_fraction": float(taper_fraction), "sample_count": int(out.size)}
    if mode == "demean":
        m = float(np.mean(out))
        out = out - m
        meta["mean_before"], meta["mean_after"] = m, float(np.mean(out))

    if taper and out.size >= 2 and taper_fraction > 0.0:
        n = min(out.size // 2, int(out.size * taper_fraction))
        if n > 0:
            edge = np.hanning(2 * n)
            window = np.ones(out.size, dtype=float)
            window[:n], window[-n:] = edge[:n], edge[n:]
            out = out * window
            meta["taper_samples_per_side"] = int(n)

    return out, meta


def apply_filterbank(
    signals: Signal | Iterable[Signal],
    bands: list[FilterBand],
    *,
    backend: str = "auto",
    phase: str = "causal",
    order: int = 4,
    detrend: str = "demean",
    taper: bool = True,
    taper_fraction: float = 0.05,
    nyquist_epsilon: float = 1e-6,
) -> list[SignalResult]:
    sigs = [signals] if isinstance(signals, Signal) else list(signals)
    if not sigs:
        raise ValidationError("signals must contain at least one Signal.")
    if any(not isinstance(s, Signal) for s in sigs):
        raise ValidationError("signals must be Signal or iterable[Signal].")
    if not bands or any(not isinstance(b, FilterBand) for b in bands):
        raise ValidationError("bands must contain at least one FilterBand.")
    if order <= 0:
        raise ValidationError("order must be > 0.")

    choice = _resolve_backend(backend)
    phase_mode = _resolve_phase(phase)
    min_samples = 3 * (2 * int(order) + 1)
    out: list[SignalResult] = []

    for s in sigs:
        validate_signal(s)
        assert s.sampling_rate is not None
        y_pre, pre = preprocess_signal(s.y, detrend=detrend, taper=taper, taper_fraction=taper_fraction)
        duration = float(s.y.size / s.sampling_rate)

        band_results: list[BandResult] = []
        for b in bands:
            run_band, warnings, reason = _runtime_band(b, float(s.sampling_rate), nyquist_epsilon)
            if b.center_frequency * duration < LOW_CYCLES_THRESHOLD:
                warnings = _warn(warnings, "low_cycles_per_record")

            if reason is not None:
                band_results.append(
                    BandResult(
                        band=run_band,
                        status="invalid",
                        filtered_y=None,
                        warnings=warnings,
                        reason=reason,
                        backend=None,
                    )
                )
                continue

            if y_pre.size < min_samples:
                # Heuristic: 3*(2*order+1) avoids outputs dominated by transient response.
                band_results.append(
                    BandResult(
                        band=run_band,
                        status="skipped",
                        filtered_y=None,
                        warnings=_warn(warnings, "signal_too_short"),
                        reason="signal_too_short",
                        backend=None,
                    )
                )
                continue

            band_results.append(
                BandResult(
                    band=run_band,
                    status="processed",
                    filtered_y=_filter(y_pre, float(s.sampling_rate), run_band, int(order), choice, phase_mode),
                    warnings=warnings,
                    reason=None,
                    backend=choice,
                )
            )

        out.append(
            SignalResult(
                signal_name=s.name,
                sampling_rate=float(s.sampling_rate),
                sample_count=int(s.y.size),
                x=None if s.x is None else np.asarray(s.x, dtype=float).copy(),
                waveform_y=np.asarray(s.y, dtype=float).copy(),
                diagnostics={
                    "backend": choice,
                    "phase": phase_mode,
                    "duration_seconds": duration,
                    "minimum_recommended_samples": min_samples,
                    "preprocessing": pre,
                },
                band_results=band_results,
            )
        )
    return out


def _resolve_backend(backend: str) -> str:
    choice = backend.lower()
    if choice not in {"auto", "obspy", "scipy"}:
        raise ValidationError("backend must be one of: auto, obspy, scipy.")
    has_obspy, has_scipy = find_spec("obspy") is not None, find_spec("scipy") is not None
    if choice == "auto":
        if has_obspy:
            return "obspy"
        if has_scipy:
            return "scipy"
        raise ValidationError("No filtering backend available. Install ObsPy or SciPy.")
    if choice == "obspy" and not has_obspy:
        raise ValidationError("backend='obspy' requested but ObsPy is not installed.")
    if choice == "scipy" and not has_scipy:
        raise ValidationError("backend='scipy' requested but SciPy is not installed.")
    return choice


def _resolve_phase(phase: str) -> str:
    mode = str(phase).lower()
    if mode not in {"causal", "zero"}:
        raise ValidationError("phase must be one of: causal, zero.")
    return mode


def _runtime_band(band: FilterBand, sampling_rate: float, nyquist_epsilon: float) -> tuple[FilterBand, list[str], str | None]:
    warnings = list(band.warnings)
    if not math.isfinite(band.center_frequency) or band.center_frequency <= 0:
        return band, warnings, "invalid_center_frequency"
    if not (math.isfinite(band.fmin) and math.isfinite(band.fmax)):
        return band, warnings, "non_finite_band_edges"

    fmin, fmax = float(band.fmin), float(band.fmax)
    nyq = 0.5 * sampling_rate
    clipped = False
    if fmax >= nyq:
        fmax = nyq * (1.0 - nyquist_epsilon)
        clipped = True
        warnings = _warn(warnings, "clipped_high_nyquist")

    rb = FilterBand(
        name=band.name,
        center_frequency=band.center_frequency,
        fmin=fmin,
        fmax=fmax,
        original_fmin=band.original_fmin,
        original_fmax=band.original_fmax,
        clipped_high_nyquist=clipped,
        warnings=warnings,
        overlap_with_next=band.overlap_with_next,
    )

    if fmin <= 0:
        return rb, warnings, "invalid_band_edges"
    if fmin >= fmax:
        return rb, warnings, "band_outside_nyquist"
    return rb, warnings, None


def _filter(y: np.ndarray, sampling_rate: float, band: FilterBand, order: int, backend: str, phase: str) -> np.ndarray:
    zerophase = phase == "zero"
    if backend == "obspy":
        from obspy.signal.filter import bandpass

        out = bandpass(
            np.asarray(y, dtype=float),
            freqmin=float(band.fmin),
            freqmax=float(band.fmax),
            df=float(sampling_rate),
            corners=int(order),
            zerophase=zerophase,
        )
        return np.asarray(out, dtype=float)

    from scipy.signal import iirfilter, sosfilt

    wn = [float(band.fmin) / (sampling_rate / 2.0), float(band.fmax) / (sampling_rate / 2.0)]
    # SciPy path mirrors ObsPy's design; zero-phase uses two-pass sosfilt (no sosfiltfilt padding).
    sos = iirfilter(order, wn, btype="band", ftype="butter", output="sos")
    if not zerophase:
        return np.asarray(sosfilt(sos, y), dtype=float)
    first = np.flip(sosfilt(sos, y))
    return np.asarray(np.flip(sosfilt(sos, first)), dtype=float)


def _warn(warnings: list[str], code: str) -> list[str]:
    out = list(warnings)
    if code not in out:
        out.append(code)
    return out


__all__ = ["BandResult", "SignalResult", "preprocess_signal", "apply_filterbank"]
