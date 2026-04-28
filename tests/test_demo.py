from __future__ import annotations

import argparse
import inspect

import numpy as np
import pytest

from filterbanks.cli import main
from filterbanks.demo import make_synthetic_signal
import filterbanks.demo as demo_module


def _demo_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    demo_module.add_demo_arguments(p)
    return p


def test_make_synthetic_signal_length_matches_sampling_rate_times_duration() -> None:
    signal, _meta = make_synthetic_signal(
        sampling_rate=100.0,
        duration=10.0,
        components=[1.0, 2.0],
        amplitudes=[1.0, 0.5],
        noise=0.0,
        seed=1,
    )
    assert signal.y.size == 1000


def test_make_synthetic_signal_skips_components_at_or_above_nyquist() -> None:
    _signal, meta = make_synthetic_signal(
        sampling_rate=100.0,
        duration=5.0,
        components=[10.0, 49.0, 50.0, 70.0],
        amplitudes=[1.0, 1.0, 1.0, 1.0],
        noise=0.0,
        seed=1,
    )
    assert meta["frequencies_used"] == [10.0, 49.0]
    assert meta["frequencies_skipped"] == [50.0, 70.0]
    assert all(f >= 50.0 for f in meta["frequencies_skipped"])


def test_make_synthetic_signal_seed_reproducibility() -> None:
    s1, m1 = make_synthetic_signal(
        sampling_rate=80.0,
        duration=4.0,
        components=[1.0, 5.0],
        amplitudes=[1.0, 0.4],
        noise=0.2,
        seed=123,
    )
    s2, m2 = make_synthetic_signal(
        sampling_rate=80.0,
        duration=4.0,
        components=[1.0, 5.0],
        amplitudes=[1.0, 0.4],
        noise=0.2,
        seed=123,
    )
    assert np.array_equal(s1.y, s2.y)
    assert m1 == m2


def test_make_synthetic_signal_uses_localized_gaussian_bursts() -> None:
    signal, meta = make_synthetic_signal(
        sampling_rate=100.0,
        duration=20.0,
        components=[0.2, 1.0, 8.0],
        amplitudes=[1.0, 0.8, 0.3],
        noise=0.0,
        seed=123,
    )
    assert meta["signal_model"] == "localized_gaussian_sine_bursts"
    burst_components = meta["burst_components"]
    assert len(burst_components) == len(meta["frequencies_used"])
    assert all("center_time_sec" in comp and "width_sec" in comp for comp in burst_components)

    windows = np.array_split(signal.y, 20)
    rms = np.array([float(np.sqrt(np.mean(w**2))) for w in windows if w.size > 0], dtype=float)
    assert float(np.max(rms)) > (3.0 * float(np.median(rms)) + 1e-12)


def test_cli_demo_runs_full_chain_by_default(capsys, monkeypatch) -> None:
    calls = {"show": 0}

    def _fake_show() -> None:
        calls["show"] += 1

    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", _fake_show)

    rc = main(
        [
            "demo",
            "--sampling-rate",
            "80",
            "--duration",
            "10",
            "--components",
            "1,3,7,15",
            "--amplitudes",
            "1,0.6,0.3,0.1",
            "--noise",
            "0.0",
            "--edge-policy",
            "keep",
            "--freq-min",
            "0.5",
            "--freq-max",
            "20",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "generated_bands:" in out
    assert "band_results_processed:" in out
    assert "backend:" in out
    assert calls["show"] == 1


def test_demo_default_filterbank_values() -> None:
    args = _demo_parser().parse_args([])
    assert float(args.freq_min) == pytest.approx(0.01)
    assert float(args.freq_max) == pytest.approx(80.0)
    assert str(args.spacing_type) == "log"
    assert float(args.spacing_step_octaves) == pytest.approx(0.25)
    assert str(args.bandwidth_type) == "octave"
    assert float(args.bandwidth_value) == pytest.approx(0.25)
    assert str(args.phase) == "causal"
    assert float(args.smooth_window_sec) == pytest.approx(5.0)


def test_demo_clips_freq_max_to_90pct_nyquist(capsys) -> None:
    rc = main(
        [
            "demo",
            "--no-show",
            "--sampling-rate",
            "100",
            "--duration",
            "10",
            "--freq-max",
            "80",
            "--freq-min",
            "0.5",
            "--components",
            "1,3,7",
            "--amplitudes",
            "1,0.6,0.3",
            "--noise",
            "0.0",
            "--edge-policy",
            "keep",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "requested_freq_max: 80.0" in out
    assert "effective_freq_max: 45.0" in out


def test_cli_demo_save_saves_figure(tmp_path) -> None:
    out = tmp_path / "demo_plot.png"
    rc = main(
        [
            "demo",
            "--save",
            str(out),
            "--no-show",
            "--sampling-rate",
            "80",
            "--duration",
            "10",
            "--components",
            "1,3,7,15",
            "--amplitudes",
            "1,0.6,0.3,0.1",
            "--noise",
            "0.0",
            "--edge-policy",
            "keep",
            "--freq-min",
            "0.5",
            "--freq-max",
            "20",
        ]
    )
    assert rc == 0
    assert out.exists()
    assert out.stat().st_size > 0


def test_cli_demo_save_and_no_show_works(tmp_path, monkeypatch) -> None:
    calls = {"show": 0}

    def _fake_show() -> None:
        calls["show"] += 1

    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", _fake_show)

    out = tmp_path / "demo_plot_no_show.png"
    rc = main(
        [
            "demo",
            "--save",
            str(out),
            "--no-show",
            "--sampling-rate",
            "60",
            "--duration",
            "12",
            "--components",
            "1,2,4,8",
            "--amplitudes",
            "1,0.7,0.4,0.2",
            "--noise",
            "0.0",
            "--edge-policy",
            "keep",
            "--freq-min",
            "0.5",
            "--freq-max",
            "20",
        ]
    )
    assert rc == 0
    assert out.exists()
    assert calls["show"] == 0


def test_demo_module_stays_orchestration_only() -> None:
    src = inspect.getsource(demo_module)
    assert "create_bands(" in src
    assert "apply_filterbank(" in src
    assert "add_envelopes_to_results(" in src
    assert "plot_results(" in src
    assert "sosfilt" not in src
    assert "iirfilter" not in src
    assert "hilbert(" not in src


def test_cli_demo_no_longer_accepts_waveform_scale() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["demo", "--waveform-scale", "0.3", "--no-show"])
    assert exc.value.code == 2
