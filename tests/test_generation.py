import json

import pytest

from filterbanks.bands import create_bands, inspect_bands


def test_default_log_octave_generation() -> None:
    bands = create_bands()
    assert len(bands) > 1
    assert bands[0].center_frequency == pytest.approx(0.01)
    ratio = bands[1].center_frequency / bands[0].center_frequency
    assert ratio == pytest.approx(2**0.25)


def test_default_bandwidth_value_is_quarter_octave() -> None:
    payload = inspect_bands()
    assert payload["bandwidth_value"] == pytest.approx(0.25)


def test_linear_generation() -> None:
    bands = create_bands(
        spacing_type="linear",
        spacing_step_hz=1.0,
        bandwidth_type="hz",
        bandwidth_value=1.0,
        freq_min=1.0,
        freq_max=3.0,
    )
    centers = [b.center_frequency for b in bands]
    assert centers == pytest.approx([1.0, 2.0, 3.0])


def test_overlap_driven_bandwidth_solution() -> None:
    bands = create_bands(
        spacing_type="linear",
        spacing_step_hz=1.0,
        bandwidth_type="hz",
        bandwidth_value=None,
        overlap=0.5,
        freq_min=1.0,
        freq_max=4.0,
    )
    assert bands[0].fmax - bands[0].fmin == pytest.approx(2.0, rel=1e-6)
    assert bands[0].overlap_with_next == pytest.approx(0.5, rel=1e-6)


def test_freq_min_freq_max_constrain_centers_only() -> None:
    bands = create_bands(
        spacing_type="linear",
        spacing_step_hz=1.0,
        bandwidth_type="octave",
        bandwidth_value=1.0,
        freq_min=1.0,
        freq_max=1.0,
    )
    assert len(bands) == 1
    assert bands[0].center_frequency == pytest.approx(1.0)
    assert bands[0].fmax > 1.0


def test_nyquist_clipping_uses_epsilon() -> None:
    bands = create_bands(
        spacing_type="linear",
        spacing_step_hz=1.0,
        bandwidth_type="hz",
        bandwidth_value=3.0,
        freq_min=4.0,
        freq_max=4.0,
        sampling_rate=10.0,
    )
    assert bands[0].clipped_high_nyquist is True
    assert bands[0].fmax == pytest.approx(5.0 * (1.0 - 1e-6))
    assert bands[0].fmax < 5.0


def test_inspect_json_has_no_nan() -> None:
    payload = inspect_bands(
        spacing_type="linear",
        spacing_step_hz=1.0,
        bandwidth_type="hz",
        bandwidth_value=2.0,
        freq_min=1.0,
        freq_max=3.0,
    )
    text = json.dumps(payload, allow_nan=False)
    assert "NaN" not in text
    assert "Infinity" not in text
    json.loads(text)


def test_create_bands_does_not_accept_signal_duration() -> None:
    with pytest.raises(TypeError):
        create_bands(signal_duration=10.0)  # type: ignore[call-arg]
