import numpy as np
import matplotlib.pyplot as plt

from filterbanks import (
    Signal,
    create_bands,
    inspect_bands,
    apply_filterbank,
    add_envelopes_to_results,
    plot_results,
)

# 1) Build an informative synthetic signal.
sampling_rate = 200.0
duration = 120.0
t = np.arange(0.0, duration, 1.0 / sampling_rate)

rng = np.random.default_rng(42)
y = np.zeros_like(t)

def burst(freq, center, width, amplitude):
    window = np.exp(-0.5 * ((t - center) / width) ** 2)
    return amplitude * np.sin(2.0 * np.pi * freq * t) * window

# Time-localized wave packets at different frequencies.
y += burst(0.2, 90.0, 12.0, 1.0)
y += burst(1.0, 75.0, 6.0, 0.8)
y += burst(3.0, 62.0, 4.0, 0.6)
y += burst(8.0, 48.0, 2.5, 0.5)
y += burst(20.0, 35.0, 1.5, 0.35)
y += burst(40.0, 25.0, 1.0, 0.25)

# Small noise floor.
y += 0.02 * rng.normal(size=t.size)

signal = Signal(y=y, x=t, sampling_rate=sampling_rate, name="synthetic_bursts")

# 2) Create quarter-octave filter bands.
bands = create_bands(
    spacing_type="log",
    spacing_step_octaves=0.25,
    bandwidth_type="octave",
    bandwidth_value=0.25,
    freq_min=0.05,
    freq_max=50.0,
    sampling_rate=sampling_rate,
)

inspection = inspect_bands(
    spacing_type="log",
    spacing_step_octaves=0.25,
    bandwidth_type="octave",
    bandwidth_value=0.25,
    freq_min=0.05,
    freq_max=50.0,
    sampling_rate=sampling_rate,
)

print("band_count:", inspection["band_count"])
for b in bands[:5]:
    print(f"{b.name}: center={b.center_frequency:.3f} Hz, range=[{b.fmin:.3f}, {b.fmax:.3f}] Hz")

# 3) Apply the filterbank.
results = apply_filterbank(signal, bands, backend="auto", phase="causal")

# 4) Add Hilbert envelopes and 5 s smoothing.
add_envelopes_to_results(
    results,
    envelope_method="hilbert_fft",
    smooth_method="moving_average",
    smooth_window_seconds=5.0,
    smooth_mode="same",
    smooth_edge_policy="keep",
)

# 5) Print final state.
signal_result = results[0]
processed = [br for br in signal_result.band_results if br.status == "processed"]
skipped = [br for br in signal_result.band_results if br.status == "skipped"]
invalid = [br for br in signal_result.band_results if br.status == "invalid"]

print("processed:", len(processed))
print("skipped:", len(skipped))
print("invalid:", len(invalid))
print("backend:", signal_result.diagnostics.get("backend"))

# 6) Plot full filterbank result using the package plotter.
fig, axes = plot_results(
    results,
    normalization="per_signal",
    show_waveforms=True,
    plot_scale=0.45,
)
plt.show()


import numpy as np
import matplotlib.pyplot as plt

from filterbanks import (
    Signal,
    create_bands,
    apply_filterbank,
    add_envelopes_to_results,
    plot_results,
)

# -------------------------------
# 1) Build a good demo signal
# -------------------------------
fs = 200.0
T = 120.0
t = np.arange(0.0, T, 1.0 / fs)

rng = np.random.default_rng(42)
y = np.zeros_like(t)

def burst(freq, center, width, amp):
    w = np.exp(-0.5 * ((t - center) / width) ** 2)
    return amp * np.sin(2 * np.pi * freq * t) * w

y += burst(0.2, 90, 12, 1.0)
y += burst(1.0, 75, 6, 0.8)
y += burst(3.0, 62, 4, 0.6)
y += burst(8.0, 48, 2.5, 0.5)
y += burst(20.0, 35, 1.5, 0.35)
y += burst(40.0, 25, 1.0, 0.25)

y += 0.02 * rng.normal(size=t.size)

signal = Signal(y=y, x=t, sampling_rate=fs)

# -------------------------------
# 2) Filter bank
# -------------------------------
bands = create_bands(
    spacing_type="log",
    spacing_step_octaves=0.25,
    bandwidth_type="octave",
    bandwidth_value=0.25,
    freq_min=0.05,
    freq_max=50.0,
)

results = apply_filterbank(signal, bands, phase="causal")

# -------------------------------
# 3) Envelope + smoothing
# -------------------------------
add_envelopes_to_results(
    results,
    envelope_method="hilbert_fft",
    smooth_method="moving_average",
    smooth_window_seconds=5.0,
)

# -------------------------------
# 4) Quick plot (recommended)
# -------------------------------
fig, axes = plot_results(
    results,
    normalization="per_signal",
    show_waveforms=True,
    plot_scale=0.45,
)


# -------------------------------
# 5) Manual plotting example: how to plot 3 selected bands yourself
# -------------------------------
plt.close("all")
sr = results[0]
processed = [br for br in sr.band_results if br.status == "processed"]

# pick a few representative bands
selected = processed[5:8]

fig, ax = plt.subplots(figsize=(10, 5))

for i, br in enumerate(selected):
    env = br.smoothed_envelope
    wav = br.filtered_y

    if env is None or wav is None:
        continue

    env = np.asarray(env)
    wav = np.asarray(wav)

    # Shared scaling (IMPORTANT)
    scale = np.nanmax(np.abs(np.concatenate([env, wav])))
    if scale <= 0:
        continue

    env_plot = i + 0.45 * env / scale
    wav_plot = i + 0.45 * wav / scale

    ax.plot(t[:env.size], env_plot, label=f"{br.band.center_frequency:.2f} Hz")
    ax.plot(t[:wav.size], wav_plot, color="gray", linewidth=0.5)

# Y-axis = band index → label with frequency
ax.set_yticks(range(len(selected)))
ax.set_yticklabels([f"{br.band.center_frequency:.2f} Hz" for br in selected])

ax.set_xlabel("Time [s]")
ax.set_ylabel("Filter band (center frequency)")
ax.set_title("Manual filter-bank plot (3 bands)")

ax.legend()
plt.tight_layout()
plt.show()