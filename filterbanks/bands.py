"""Band generation and inspection helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Iterable, Mapping

from .exceptions import ValidationError

DEFAULT_SPACING_TYPE = "log"
DEFAULT_BANDWIDTH_TYPE = "octave"
DEFAULT_SPACING_STEP_OCTAVES = 0.25
DEFAULT_BANDWIDTH_VALUE = 0.25
DEFAULT_FREQ_MIN = 0.01
DEFAULT_FREQ_MAX = 5.0
DEFAULT_NYQUIST_EPSILON = 1e-6


@dataclass
class FilterBand:
    name: str
    center_frequency: float
    fmin: float
    fmax: float
    original_fmin: float
    original_fmax: float
    clipped_high_nyquist: bool = False
    warnings: list[str] = field(default_factory=list)
    overlap_with_next: float | None = None


def create_bands(
    *,
    spacing_type: str = DEFAULT_SPACING_TYPE,
    spacing_step_octaves: float | None = DEFAULT_SPACING_STEP_OCTAVES,
    spacing_step_hz: float | None = None,
    bandwidth_type: str = DEFAULT_BANDWIDTH_TYPE,
    bandwidth_value: float | None = DEFAULT_BANDWIDTH_VALUE,
    overlap: float | None = None,
    freq_min: float = DEFAULT_FREQ_MIN,
    freq_max: float = DEFAULT_FREQ_MAX,
    sampling_rate: float | None = None,
    explicit_bands: Iterable[Mapping[str, Any]] | None = None,
    nyquist_epsilon: float = DEFAULT_NYQUIST_EPSILON,
) -> list[FilterBand]:
    bw_type = bandwidth_type.lower()
    if bw_type not in {"octave", "hz"}:
        raise ValidationError("bandwidth_type must be one of: octave, hz.")

    centers, names = _build_centers(
        spacing_type=spacing_type.lower(),
        spacing_step_octaves=spacing_step_octaves,
        spacing_step_hz=spacing_step_hz,
        freq_min=freq_min,
        freq_max=freq_max,
        explicit_bands=explicit_bands,
    )
    bw = _resolve_bandwidth(centers, bw_type, bandwidth_value, overlap)

    bands = []
    for i, c in enumerate(centers):
        fmin, fmax = _edges(c, bw_type, bw)
        bands.append(
            FilterBand(
                name=names[i] if i < len(names) else f"band_{i + 1:03d}",
                center_frequency=c,
                fmin=fmin,
                fmax=fmax,
                original_fmin=fmin,
                original_fmax=fmax,
            )
        )

    if sampling_rate is not None:
        _clip_to_nyquist(bands, sampling_rate, nyquist_epsilon)

    for i in range(len(bands) - 1):
        w = bands[i].original_fmax - bands[i].original_fmin
        bands[i].overlap_with_next = None if w <= 0 else (bands[i].original_fmax - bands[i + 1].original_fmin) / w
    if bands:
        bands[-1].overlap_with_next = None
    return bands


def inspect_bands(**kwargs: Any) -> dict[str, Any]:
    bands = create_bands(**kwargs)
    return _json_safe(
        {
            "spacing_type": kwargs.get("spacing_type", DEFAULT_SPACING_TYPE),
            "bandwidth_type": kwargs.get("bandwidth_type", DEFAULT_BANDWIDTH_TYPE),
            "bandwidth_value": kwargs.get("bandwidth_value", DEFAULT_BANDWIDTH_VALUE),
            "overlap": kwargs.get("overlap"),
            "freq_min": kwargs.get("freq_min", DEFAULT_FREQ_MIN),
            "freq_max": kwargs.get("freq_max", DEFAULT_FREQ_MAX),
            "sampling_rate": kwargs.get("sampling_rate"),
            "band_count": len(bands),
            "bands": [
                {
                    "name": b.name,
                    "center_frequency": b.center_frequency,
                    "fmin": b.fmin,
                    "fmax": b.fmax,
                    "original_fmin": b.original_fmin,
                    "original_fmax": b.original_fmax,
                    "clipped_high_nyquist": b.clipped_high_nyquist,
                    "warnings": list(b.warnings),
                    "overlap_with_next": b.overlap_with_next,
                }
                for b in bands
            ],
        }
    )


def _build_centers(
    *,
    spacing_type: str,
    spacing_step_octaves: float | None,
    spacing_step_hz: float | None,
    freq_min: float,
    freq_max: float,
    explicit_bands: Iterable[Mapping[str, Any]] | None,
) -> tuple[list[float], list[str]]:
    if spacing_type == "explicit":
        if explicit_bands is None:
            raise ValidationError("explicit_bands is required for spacing_type='explicit'.")
        centers, names = [], []
        for i, item in enumerate(explicit_bands, start=1):
            if not isinstance(item, Mapping):
                raise ValidationError("each explicit band entry must be an object.")
            unknown = set(item) - {"name", "center_frequency"}
            if unknown:
                raise ValidationError(f"unknown keys for explicit band: {sorted(unknown)}")
            centers.append(_pos(item.get("center_frequency"), "center_frequency"))
            names.append(str(item.get("name", f"band_{i:03d}")))
        if not centers:
            raise ValidationError("explicit_bands must contain at least one entry.")
        return centers, names

    _finite(freq_min, "freq_min")
    _finite(freq_max, "freq_max")
    if freq_min <= 0 or freq_max < freq_min:
        raise ValidationError("Require freq_min > 0 and freq_max >= freq_min.")

    out = []
    if spacing_type == "log":
        _finite(spacing_step_octaves, "spacing_step_octaves")
        assert spacing_step_octaves is not None
        if spacing_step_octaves <= 0:
            raise ValidationError("spacing_step_octaves must be > 0.")
        # Log spacing: center[n+1] = center[n] * 2**step.
        ratio, c = 2.0 ** spacing_step_octaves, float(freq_min)
        while c <= float(freq_max) * (1.0 + 1e-12):
            out.append(c)
            c *= ratio
        return out, []

    if spacing_type != "linear":
        raise ValidationError("spacing_type must be one of: log, linear, explicit.")
    _finite(spacing_step_hz, "spacing_step_hz")
    assert spacing_step_hz is not None
    if spacing_step_hz <= 0:
        raise ValidationError("spacing_step_hz must be > 0.")
    c = float(freq_min)
    while c <= float(freq_max) + 1e-12:
        out.append(c)
        c += spacing_step_hz
    return out, []


def _resolve_bandwidth(centers: list[float], bw_type: str, bw: float | None, overlap: float | None) -> float:
    if bw is not None:
        _finite(bw, "bandwidth_value")
        if bw <= 0:
            raise ValidationError("bandwidth_value must be > 0.")
    if overlap is None:
        if bw is None:
            raise ValidationError("Provide either bandwidth_value or overlap.")
        return float(bw)

    _finite(overlap, "overlap")
    if overlap < 0 or overlap >= 1:
        raise ValidationError("overlap must be in [0, 1).")
    if bw is not None:
        d = _mean_overlap(centers, bw_type, float(bw))
        if not math.isclose(overlap, d, rel_tol=1e-6, abs_tol=1e-9):
            raise ValidationError(f"overlap ({overlap}) does not match derived overlap ({d:.6g}).")
        return float(bw)
    if len(centers) < 2:
        raise ValidationError("overlap-driven mode requires at least two center frequencies.")

    lo, hi = 1e-12, 1.0
    while _mean_overlap(centers, bw_type, hi) < overlap:
        hi *= 2.0
        if hi > 1e12:
            raise ValidationError("Could not solve bandwidth for requested overlap.")
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if _mean_overlap(centers, bw_type, mid) < overlap:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _edges(center: float, bw_type: str, bw: float) -> tuple[float, float]:
    if bw_type == "octave":
        # Octave bandwidth is symmetric in log-frequency about the center.
        r = 2.0 ** (bw / 2.0)
        return center / r, center * r
    half = bw / 2.0
    return center - half, center + half


def _mean_overlap(centers: list[float], bw_type: str, bw: float) -> float:
    vals = []
    for i in range(len(centers) - 1):
        lfmin, lfmax = _edges(centers[i], bw_type, bw)
        rfmin, _ = _edges(centers[i + 1], bw_type, bw)
        w = lfmax - lfmin
        if w <= 0:
            raise ValidationError("bandwidth produced non-positive band width.")
        vals.append((lfmax - rfmin) / w)
    return sum(vals) / len(vals)


def _clip_to_nyquist(bands: list[FilterBand], sampling_rate: float, eps: float) -> None:
    _finite(sampling_rate, "sampling_rate")
    _finite(eps, "nyquist_epsilon")
    if sampling_rate <= 0:
        raise ValidationError("sampling_rate must be > 0.")
    if eps <= 0 or eps >= 1:
        raise ValidationError("nyquist_epsilon must be between 0 and 1.")
    # Keep high cutoff strictly below Nyquist; exact-Nyquist cutoff is disallowed.
    nyq, limit = 0.5 * sampling_rate, 0.5 * sampling_rate * (1.0 - eps)
    for b in bands:
        if b.fmax >= nyq:
            b.fmax, b.clipped_high_nyquist = limit, True
            if "clipped_high_nyquist" not in b.warnings:
                b.warnings.append("clipped_high_nyquist")


def _finite(v: float | None, name: str) -> None:
    if v is None or not math.isfinite(float(v)):
        raise ValidationError(f"{name} must be finite.")


def _pos(v: Any, name: str) -> float:
    try:
        n = float(v)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{name} must be a number.") from exc
    if not math.isfinite(n) or n <= 0:
        raise ValidationError(f"{name} must be > 0.")
    return n


def _json_safe(x: Any) -> Any:
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, list):
        return [_json_safe(v) for v in x]
    if isinstance(x, dict):
        return {k: _json_safe(v) for k, v in x.items()}
    return x


__all__ = ["FilterBand", "create_bands", "inspect_bands"]
