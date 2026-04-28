"""Synthetic signal demo orchestration."""

from __future__ import annotations

from typing import Any

import numpy as np

from .bands import create_bands
from .envelope import add_envelopes_to_results
from .exceptions import ValidationError
from .filtering import apply_filterbank
from .plotting import plot_results
from .signals import Signal

DEFAULT_COMPONENTS = "0.02,0.05,0.2,1,3,8,20,40,70"
DEFAULT_AMPLITUDES = "1.0,1.0,0.8,0.6,0.45,0.35,0.25,0.18,0.12"


def make_synthetic_signal(
    sampling_rate: float,
    duration: float,
    components: list[float],
    amplitudes: list[float],
    noise: float,
    seed: int | None = None,
) -> tuple[Signal, dict[str, Any]]:
    if len(components) != len(amplitudes):
        raise ValidationError("components and amplitudes must have the same length.")
    if sampling_rate <= 0.0 or duration <= 0.0:
        raise ValidationError("sampling_rate and duration must be > 0.")
    if noise < 0.0:
        raise ValidationError("noise must be >= 0.")

    n_samples = int(round(float(sampling_rate) * float(duration)))
    if n_samples <= 0:
        raise ValidationError("sampling_rate * duration must produce at least one sample.")

    t = np.arange(n_samples, dtype=float) / float(sampling_rate)
    y = np.zeros(n_samples, dtype=float)
    dt = 1.0 / float(sampling_rate)
    nyquist = float(sampling_rate) / 2.0
    used: list[float] = []
    skipped: list[float] = []
    burst_components: list[dict[str, float]] = []

    centers = np.linspace(0.1 * float(duration), 0.9 * float(duration), num=len(components), dtype=float)

    for idx, (freq, amp) in enumerate(zip(components, amplitudes)):
        f = float(freq)
        if f >= nyquist:
            skipped.append(f)
            continue
        used.append(f)
        center = float(centers[idx]) if centers.size else 0.5 * float(duration)
        burst_duration = _burst_duration_seconds(
            frequency=f,
            duration=float(duration),
            sampling_rate=float(sampling_rate),
        )
        sigma = max(burst_duration / 6.0, dt)
        window = np.exp(-0.5 * ((t - center) / sigma) ** 2)
        y += float(amp) * np.sin(2.0 * np.pi * f * t) * window
        burst_components.append(
            {
                "frequency": f,
                "amplitude": float(amp),
                "center_time_sec": center,
                "width_sec": burst_duration,
            }
        )

    y += np.random.default_rng(seed).normal(loc=0.0, scale=float(noise), size=n_samples)
    signal = Signal(y=y, x=t, sampling_rate=float(sampling_rate), name="synthetic_demo")
    return signal, {
        "signal_model": "localized_gaussian_sine_bursts",
        "sampling_rate": float(sampling_rate),
        "duration": float(duration),
        "frequencies_used": used,
        "frequencies_skipped": skipped,
        "amplitudes": [float(a) for a in amplitudes],
        "burst_components": burst_components,
        "noise": float(noise),
        "seed": seed,
    }


def add_demo_arguments(parser: Any) -> None:
    parser.add_argument("--sampling-rate", type=float, default=200.0)
    parser.add_argument("--duration", type=float, default=300.0)
    parser.add_argument("--components", type=str, default=DEFAULT_COMPONENTS)
    parser.add_argument("--amplitudes", type=str, default=DEFAULT_AMPLITUDES)
    parser.add_argument("--noise", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--no-show", action="store_true")

    parser.add_argument("--freq-min", type=float, default=0.01)
    parser.add_argument("--freq-max", type=float, default=80.0)
    parser.add_argument("--spacing-type", choices=["log", "linear", "explicit"], default="log")
    parser.add_argument("--spacing-step-octaves", type=float, default=0.25)
    parser.add_argument("--spacing-step-hz", type=float, default=None)
    parser.add_argument("--bandwidth-type", choices=["octave", "hz"], default="octave")
    parser.add_argument("--bandwidth-value", type=float, default=0.25)
    parser.add_argument("--overlap", type=float, default=None)
    parser.add_argument("--phase", choices=["causal", "zero"], default="causal")

    parser.add_argument("--envelope-method", type=str, default="hilbert_fft")
    parser.add_argument("--smooth-window-sec", type=float, default=5.0)
    parser.add_argument("--smooth-mode", choices=["same", "valid"], default="same")
    parser.add_argument("--edge-policy", choices=["keep", "nan_mask"], default="nan_mask")

    parser.add_argument("--show-waveforms", action="store_true")
    parser.add_argument("--normalization", choices=["per_envelope", "per_signal", "global"], default="per_signal")
    parser.add_argument("--plot-scale", type=float, default=0.45)
    parser.add_argument("--tref", type=float, default=None)


def run_demo(args: Any) -> int:
    signal, meta = make_synthetic_signal(
        sampling_rate=float(args.sampling_rate),
        duration=float(args.duration),
        components=_parse_float_csv(args.components, "components"),
        amplitudes=_parse_float_csv(args.amplitudes, "amplitudes"),
        noise=float(args.noise),
        seed=args.seed,
    )

    requested_freq_max = float(args.freq_max)
    effective_freq_max, demo_freqmax_clipped = _clip_demo_freq_max(requested_freq_max, signal.sampling_rate)
    meta["requested_freq_max"] = requested_freq_max
    meta["effective_freq_max"] = effective_freq_max
    meta["demo_freqmax_clipped"] = demo_freqmax_clipped

    bands = create_bands(
        spacing_type=str(args.spacing_type),
        spacing_step_octaves=args.spacing_step_octaves,
        spacing_step_hz=args.spacing_step_hz,
        bandwidth_type=str(args.bandwidth_type),
        bandwidth_value=args.bandwidth_value,
        overlap=args.overlap,
        freq_min=args.freq_min,
        freq_max=effective_freq_max,
        sampling_rate=signal.sampling_rate,
    )
    results = apply_filterbank(signal, bands, backend="auto", phase=str(args.phase))
    add_envelopes_to_results(
        results,
        envelope_method=str(args.envelope_method),
        smooth_method="moving_average",
        smooth_window_seconds=float(args.smooth_window_sec),
        smooth_mode=str(args.smooth_mode),
        smooth_edge_policy=str(args.edge_policy),
    )

    fig, _axes = plot_results(
        results,
        normalization=str(args.normalization),
        plot_scale=float(args.plot_scale),
        show_waveforms=bool(args.show_waveforms),
        tref=args.tref,
    )
    if args.save:
        fig.savefig(args.save)
        print(f"Wrote {args.save}")

    if not bool(args.no_show):
        import matplotlib.pyplot as plt

        plt.show()

    _print_chain_summary(signal, meta, bands, results, args.save)

    import matplotlib.pyplot as plt

    plt.close(fig)

    return 0


def _print_chain_summary(
    signal: Signal,
    meta: dict[str, Any],
    bands: list[Any],
    results: list[Any],
    saved_plot: str | None,
) -> None:
    processed = 0
    skipped = 0
    invalid = 0
    backend = None
    for sig_result in results:
        backend = backend or sig_result.diagnostics.get("backend")
        for br in sig_result.band_results:
            if br.status == "processed":
                processed += 1
            elif br.status == "skipped":
                skipped += 1
            elif br.status == "invalid":
                invalid += 1

    print(f"sampling_rate: {meta['sampling_rate']}")
    print(f"duration: {meta['duration']}")
    print(f"samples: {signal.y.size}")
    print(f"signal_model: {meta.get('signal_model')}")
    print(f"frequencies_used: {meta['frequencies_used']}")
    print(f"frequencies_skipped: {meta['frequencies_skipped']}")
    print("signal_note: localized Gaussian-windowed sine bursts")
    if bool(meta.get("demo_freqmax_clipped")):
        print(f"requested_freq_max: {meta.get('requested_freq_max')}")
        print(f"effective_freq_max: {meta.get('effective_freq_max')}")
    print(f"generated_bands: {len(bands)}")
    print(f"band_results_processed: {processed}")
    print(f"band_results_skipped: {skipped}")
    print(f"band_results_invalid: {invalid}")
    print(f"backend: {backend}")
    if saved_plot:
        print(f"plot_saved: {saved_plot}")


def _parse_float_csv(text: str, name: str) -> list[float]:
    chunks = [x.strip() for x in str(text).split(",")]
    if not chunks or any(x == "" for x in chunks):
        raise ValidationError(f"{name} must be a non-empty comma-separated list of floats.")
    try:
        return [float(x) for x in chunks]
    except ValueError as exc:
        raise ValidationError(f"{name} must be a comma-separated list of floats.") from exc


def _burst_duration_seconds(*, frequency: float, duration: float, sampling_rate: float) -> float:
    if frequency <= 0.0:
        return max(6.0 / sampling_rate, 0.1 * duration)

    raw = 6.0 / frequency
    min_width = max(6.0 / sampling_rate, 0.03 * duration)
    max_width = max(min_width, 0.35 * duration)
    return min(max(raw, min_width), max_width)


def _clip_demo_freq_max(requested_freq_max: float, sampling_rate: float) -> tuple[float, bool]:
    nyquist = 0.5 * float(sampling_rate)
    safe_max = 0.9 * nyquist
    effective = min(float(requested_freq_max), safe_max)
    return effective, bool(effective < float(requested_freq_max))
