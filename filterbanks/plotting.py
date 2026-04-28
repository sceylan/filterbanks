"""Analysis-focused plotting for filterbank results."""

from __future__ import annotations

import math
from typing import Any
import warnings

import numpy as np

from .exceptions import ValidationError
from .filtering import SignalResult


def plot_results(
    results: Any,
    normalization: str = "per_signal",
    plot_scale: float = 0.45,
    show_waveforms: bool = False,
    tref: float | None = None,
    ax: Any = None,
):
    try:
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
    except Exception as exc:
        raise ValidationError("matplotlib is required for plotting.") from exc

    signals = _as_signal_results(results)
    _validate_plot_args(normalization, plot_scale)

    if not signals:
        warnings.warn("No signal results to plot.", stacklevel=2)
        fig, axis = plt.subplots(1, 1)
        return fig, [axis]

    n_signals = len(signals)
    if ax is None:
        fig = plt.figure(figsize=(4.0 * n_signals, 4.0))
        grid = GridSpec(1, n_signals, figure=fig)
        axes: list[Any] = []
        for i in range(n_signals):
            share = None if i == 0 else axes[0]
            axes.append(fig.add_subplot(grid[0, i], sharex=share))
    else:
        axes = _coerce_axes(ax, n_signals)
        fig = axes[0].figure

    prepared = [_prepare_signal_data(sig) for sig in signals]
    global_max = _global_envelope_max(prepared)
    if global_max is None:
        warnings.warn("No envelope data found in results; returning empty figure.", stacklevel=2)
        for i, axis in enumerate(axes):
            if i == 0:
                axis.set_ylabel("Band")
            axis.set_xlabel("Time [s]")
        return fig, axes

    max_duration = max(p["duration"] for p in prepared)
    use_minutes = max_duration > 3600.0
    xlabel = "Time [min]" if use_minutes else "Time [s]"

    for col, (axis, sig, pdata) in enumerate(zip(axes, signals, prepared)):
        band_rows = pdata["bands"]
        signal_max = _signal_envelope_max(band_rows)

        for row, band_row in enumerate(band_rows):
            env = band_row["envelope"]
            if env is None:
                warnings.warn(
                    f"Missing envelope for signal '{sig.signal_name}' band '{band_row['label']}'; skipping.",
                    stacklevel=2,
                )
                continue

            if show_waveforms:
                wav = band_row["filtered_y"]
                if wav is None:
                    warnings.warn(
                        f"Missing filtered_y for signal '{sig.signal_name}' band '{band_row['label']}'; skipping.",
                        stacklevel=2,
                    )
                    continue
                scale = _shared_band_scale(env, wav)
            else:
                scale = _normalization_scale(env, normalization, signal_max, global_max)

            if scale is None or not np.isfinite(scale) or scale <= 0.0:
                warnings.warn(
                    f"Non-finite or zero scale for signal '{sig.signal_name}' band '{band_row['label']}'; skipping.",
                    stacklevel=2,
                )
                continue

            y_plot = row + float(plot_scale) * (env / scale)

            t = _time_for_length(pdata["time"], env.size, sig.sampling_rate)
            if tref is not None:
                t = t - float(tref)
            if use_minutes:
                t = t / 60.0
            axis.plot(t, y_plot, color=_tab10(row), linewidth=1.0)

            if show_waveforms:
                wav = band_row["filtered_y"]
                assert wav is not None
                y_wave = row + float(plot_scale) * (wav / scale)
                t_wave = _time_for_length(pdata["time"], wav.size, sig.sampling_rate)
                if tref is not None:
                    t_wave = t_wave - float(tref)
                if use_minutes:
                    t_wave = t_wave / 60.0
                axis.plot(t_wave, y_wave, color="gray", linewidth=0.5)

        axis.set_title(sig.signal_name or f"Signal {col + 1}")
        axis.set_xlabel(xlabel)
        axis.set_yticks(np.arange(len(band_rows), dtype=float))
        if col == 0:
            axis.set_yticklabels([b["label"] for b in band_rows])
            axis.set_ylabel("Band")
        else:
            axis.set_yticklabels([])

    return fig, axes


def _as_signal_results(results: Any) -> list[SignalResult]:
    if hasattr(results, "signal_results"):
        items = list(getattr(results, "signal_results"))
    else:
        items = list(results)
    for item in items:
        if not isinstance(item, SignalResult):
            raise ValidationError("results must be list[SignalResult] or expose .signal_results.")
    return items


def _validate_plot_args(normalization: str, plot_scale: float) -> None:
    if str(normalization).lower() not in {"per_envelope", "per_signal", "global"}:
        raise ValidationError("normalization must be one of: per_envelope, per_signal, global.")
    if not math.isfinite(float(plot_scale)):
        raise ValidationError("plot_scale must be finite.")


def _coerce_axes(ax: Any, n_signals: int) -> list[Any]:
    if n_signals == 1 and hasattr(ax, "plot"):
        return [ax]
    try:
        axes = list(ax)
    except Exception as exc:
        raise ValidationError("ax must be an Axes or a sequence of Axes.") from exc
    if len(axes) != n_signals:
        raise ValidationError("ax length must match number of signals.")
    return axes


def _prepare_signal_data(sig: SignalResult) -> dict[str, Any]:
    _validate_sampling_rate(sig.sampling_rate)
    if sig.x is not None:
        time = np.asarray(sig.x, dtype=float).copy()
    else:
        time = np.arange(int(sig.sample_count), dtype=float) / float(sig.sampling_rate)

    processed = [br for br in sig.band_results if br.status == "processed"]
    processed.sort(key=lambda br: (float(br.band.center_frequency), float(br.band.fmin), br.band.name))

    bands = []
    for br in processed:
        env = br.smoothed_envelope if br.smoothed_envelope is not None else br.envelope
        env_arr = None if env is None else np.asarray(env, dtype=float).copy()
        if env_arr is not None and env_arr.ndim != 1:
            raise ValidationError("Envelope arrays must be 1D.")
        filtered = None if br.filtered_y is None else np.asarray(br.filtered_y, dtype=float).copy()
        if filtered is not None and filtered.ndim != 1:
            raise ValidationError("BandResult.filtered_y must be 1D when provided.")
        bands.append(
            {
                "envelope": env_arr,
                "filtered_y": filtered,
                "label": f"{float(br.band.center_frequency):.3g} Hz",
            }
        )

    return {
        "time": time,
        "duration": float(sig.sample_count) / float(sig.sampling_rate),
        "bands": bands,
    }


def _validate_sampling_rate(sampling_rate: float) -> None:
    if not math.isfinite(float(sampling_rate)) or float(sampling_rate) <= 0.0:
        raise ValidationError("SignalResult.sampling_rate must be positive and finite.")


def _global_envelope_max(prepared: list[dict[str, Any]]) -> float | None:
    vals: list[float] = []
    for pdata in prepared:
        for band in pdata["bands"]:
            env = band["envelope"]
            if env is None:
                continue
            m = _finite_abs_max(env)
            if m is not None:
                vals.append(m)
    if not vals:
        return None
    out = max(vals)
    return out if out > 0.0 else 1.0


def _signal_envelope_max(bands: list[dict[str, Any]]) -> float:
    vals: list[float] = []
    for band in bands:
        env = band["envelope"]
        if env is None:
            continue
        m = _finite_abs_max(env)
        if m is not None:
            vals.append(m)
    if not vals:
        return 1.0
    out = max(vals)
    return out if out > 0.0 else 1.0


def _normalization_scale(
    env: np.ndarray,
    normalization: str,
    signal_max: float,
    global_max: float,
) -> float:
    mode = str(normalization).lower()
    if mode == "per_envelope":
        v = _finite_abs_max(env)
        return float(v) if v is not None and np.isfinite(v) and v > 0.0 else 1.0
    if mode == "per_signal":
        return float(signal_max)
    if mode == "global":
        return float(global_max)
    raise ValidationError("normalization must be one of: per_envelope, per_signal, global.")


def _shared_band_scale(env: np.ndarray, wav: np.ndarray) -> float | None:
    emax = _finite_abs_max(env)
    wmax = _finite_abs_max(wav)
    vals = [v for v in (emax, wmax) if v is not None and np.isfinite(v)]
    if not vals:
        return None
    return float(max(vals))


def _finite_abs_max(values: np.ndarray) -> float | None:
    if values.size == 0:
        return 0.0
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    return float(np.max(np.abs(finite)))


def _time_for_length(base_time: np.ndarray, n: int, sampling_rate: float) -> np.ndarray:
    if n <= base_time.size:
        return base_time[:n]
    return np.arange(n, dtype=float) / float(sampling_rate)


def _tab10(i: int) -> Any:
    import matplotlib.pyplot as plt

    return plt.get_cmap("tab10")(i % 10)


__all__ = ["plot_results"]
