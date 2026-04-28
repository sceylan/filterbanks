from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

mpl = pytest.importorskip("matplotlib")
mpl.use("Agg")

from filterbanks.bands import FilterBand
from filterbanks.filtering import BandResult, SignalResult
from filterbanks.plotting import plot_results


@dataclass
class _ResultWrapper:
    signal_results: list[SignalResult]


def _make_band(name: str, center: float) -> FilterBand:
    return FilterBand(
        name=name,
        center_frequency=center,
        fmin=max(center - 0.2, 0.01),
        fmax=center + 0.2,
        original_fmin=max(center - 0.2, 0.01),
        original_fmax=center + 0.2,
    )


def _make_signal_result(
    *,
    name: str,
    sampling_rate: float,
    n: int,
    env_scales: tuple[float, float],
    with_waveform: bool = True,
) -> SignalResult:
    b1 = _make_band("b1", 1.0)
    b2 = _make_band("b2", 2.0)

    env1 = np.ones(n, dtype=float) * env_scales[0]
    env2 = np.ones(n, dtype=float) * env_scales[1]

    y1 = np.sin(np.linspace(0.0, 8.0, n, dtype=float))
    y2 = np.cos(np.linspace(0.0, 10.0, n, dtype=float))
    br1 = BandResult(band=b1, status="processed", filtered_y=y1, envelope=env1)
    br2 = BandResult(band=b2, status="processed", filtered_y=y2, envelope=env2)

    waveform = np.sin(np.linspace(0.0, 20.0, n)) if with_waveform else None
    return SignalResult(
        signal_name=name,
        sampling_rate=sampling_rate,
        sample_count=n,
        waveform_y=waveform,
        band_results=[br1, br2],
    )


def test_plotting_runs_without_error() -> None:
    results = [_make_signal_result(name="s1", sampling_rate=10.0, n=200, env_scales=(1.0, 2.0))]
    fig, axes = plot_results(results)
    assert fig is not None
    assert len(axes) == 1


def test_per_signal_normalization_works() -> None:
    results = [_make_signal_result(name="s1", sampling_rate=10.0, n=100, env_scales=(1.0, 2.0))]
    fig, axes = plot_results(results, normalization="per_signal", plot_scale=1.0)
    lines = axes[0].lines
    y0 = lines[0].get_ydata()
    y1 = lines[1].get_ydata()
    assert float(np.max(y0)) == pytest.approx(0.5)
    assert float(np.max(y1)) == pytest.approx(2.0)


def test_global_normalization_works() -> None:
    s1 = _make_signal_result(name="s1", sampling_rate=10.0, n=100, env_scales=(1.0, 2.0))
    s2 = _make_signal_result(name="s2", sampling_rate=10.0, n=100, env_scales=(2.0, 4.0))
    fig, axes = plot_results([s1, s2], normalization="global", plot_scale=1.0)

    y0 = axes[0].lines[0].get_ydata()
    y1 = axes[0].lines[1].get_ydata()
    assert float(np.max(y0)) == pytest.approx(0.25)
    assert float(np.max(y1)) == pytest.approx(1.5)


def test_per_envelope_normalization_works() -> None:
    results = [_make_signal_result(name="s1", sampling_rate=10.0, n=100, env_scales=(1.0, 10.0))]
    fig, axes = plot_results(results, normalization="per_envelope", plot_scale=1.0)
    y0 = axes[0].lines[0].get_ydata()
    y1 = axes[0].lines[1].get_ydata()
    assert float(np.max(y0)) == pytest.approx(1.0)
    assert float(np.max(y1)) == pytest.approx(2.0)


def test_missing_envelopes_triggers_warning() -> None:
    band = _make_band("b1", 1.0)
    sig = SignalResult(
        signal_name="s1",
        sampling_rate=10.0,
        sample_count=100,
        band_results=[BandResult(band=band, status="processed", filtered_y=np.zeros(100), envelope=None)],
    )
    with pytest.warns(UserWarning, match="No envelope data found|Missing envelope"):
        fig, axes = plot_results([sig])


def test_waveform_plotting_works_without_error() -> None:
    results = [_make_signal_result(name="s1", sampling_rate=10.0, n=120, env_scales=(1.0, 2.0), with_waveform=True)]
    fig, axes = plot_results(results, show_waveforms=True)
    assert len(axes[0].lines) == 4


def test_waveforms_are_plotted_per_processed_band_at_band_rows() -> None:
    n = 120
    b1 = _make_band("b1", 1.0)
    b2 = _make_band("b2", 2.0)
    b3 = _make_band("b3", 3.0)
    b4 = _make_band("b4", 4.0)
    env = np.ones(n, dtype=float)

    sig = SignalResult(
        signal_name="s1",
        sampling_rate=10.0,
        sample_count=n,
        waveform_y=np.sin(np.linspace(0.0, 20.0, n, dtype=float)),
        band_results=[
            BandResult(
                band=b1,
                status="processed",
                filtered_y=np.sin(np.linspace(0.0, 6.0, n, dtype=float)),
                envelope=env,
            ),
            BandResult(
                band=b2,
                status="processed",
                filtered_y=0.25 * np.cos(np.linspace(0.0, 9.0, n, dtype=float)),
                envelope=2.0 * env,
            ),
            BandResult(band=b3, status="skipped", filtered_y=None, envelope=None, reason="signal_too_short"),
            BandResult(band=b4, status="invalid", filtered_y=None, envelope=None, reason="invalid_band_edges"),
        ],
    )

    plot_scale = 0.45
    fig, axes = plot_results([sig], show_waveforms=True, plot_scale=plot_scale)
    lines = axes[0].lines
    waveform_lines = [line for line in lines if line.get_color() == "gray"]
    assert len(waveform_lines) == 2

    envelope_lines = [line for line in lines if line.get_color() != "gray"]
    assert len(envelope_lines) == 2

    # row 0: env max=1, wav max~1 -> shared scale=1
    env0 = np.asarray(envelope_lines[0].get_ydata(), dtype=float)
    wav0 = np.asarray(waveform_lines[0].get_ydata(), dtype=float)
    assert float(np.max(env0)) == pytest.approx(0.0 + plot_scale)
    assert float(np.max(wav0)) == pytest.approx(0.0 + plot_scale, rel=1e-3)
    assert float(np.min(wav0)) == pytest.approx(0.0 - plot_scale, rel=1e-3)

    # row 1: env max=2, wav max=0.25 -> shared scale=2
    env1 = np.asarray(envelope_lines[1].get_ydata(), dtype=float)
    wav1 = np.asarray(waveform_lines[1].get_ydata(), dtype=float)
    assert float(np.max(env1)) == pytest.approx(1.0 + plot_scale)
    assert float(np.max(wav1)) == pytest.approx(1.0 + plot_scale * 0.125, rel=1e-3)
    assert float(np.min(wav1)) == pytest.approx(1.0 - plot_scale * 0.125, rel=1e-3)


def test_waveform_and_envelope_ticks_align_to_band_rows() -> None:
    sig = _make_signal_result(name="s1", sampling_rate=20.0, n=80, env_scales=(0.5, 1.5))
    fig, axes = plot_results([sig], show_waveforms=True, plot_scale=0.4)
    yticks = np.asarray(axes[0].get_yticks(), dtype=float)
    assert np.array_equal(yticks, np.array([0.0, 1.0], dtype=float))
    labels = [tick.get_text() for tick in axes[0].get_yticklabels()]
    assert labels == ["1 Hz", "2 Hz"]


def test_plot_scale_applies_to_both_envelope_and_waveform() -> None:
    sig = _make_signal_result(name="s1", sampling_rate=10.0, n=100, env_scales=(1.0, 2.0))

    fig_small, axes_small = plot_results([sig], show_waveforms=True, plot_scale=0.2)
    fig_large, axes_large = plot_results([sig], show_waveforms=True, plot_scale=0.6)

    env_small = np.asarray(axes_small[0].lines[0].get_ydata(), dtype=float)
    wav_small = np.asarray(axes_small[0].lines[1].get_ydata(), dtype=float)
    env_large = np.asarray(axes_large[0].lines[0].get_ydata(), dtype=float)
    wav_large = np.asarray(axes_large[0].lines[1].get_ydata(), dtype=float)

    env_small_amp = float(np.max(np.abs(env_small - 0.0)))
    wav_small_amp = float(np.max(np.abs(wav_small - 0.0)))
    env_large_amp = float(np.max(np.abs(env_large - 0.0)))
    wav_large_amp = float(np.max(np.abs(wav_large - 0.0)))

    assert env_large_amp / env_small_amp == pytest.approx(3.0, rel=1e-3)
    assert wav_large_amp / wav_small_amp == pytest.approx(3.0, rel=1e-3)


def test_tref_shifts_axis_correctly() -> None:
    results = [_make_signal_result(name="s1", sampling_rate=10.0, n=100, env_scales=(1.0, 2.0))]
    fig, axes = plot_results(results, tref=5.0)
    x0 = axes[0].lines[0].get_xdata()
    assert float(x0[0]) == pytest.approx(-5.0)


def test_time_unit_switches_to_minutes_when_duration_long() -> None:
    long_sig = _make_signal_result(name="long", sampling_rate=1.0, n=7200, env_scales=(1.0, 2.0))
    fig, axes = plot_results(_ResultWrapper([long_sig]))
    assert axes[0].get_xlabel() == "Time [min]"
