# filterbanks

# filterbanks

`filterbanks` is a lightweight toolkit for seismic waveform analysis using **filter banks**. 
It splits a waveform into multiple frequency bands for the purpose of analyzing energy content per band.
The package supports NumPy arrays and ObsPy objects.

---

## Installation

Local editable install:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Dev/test install:

```bash
pip install -e ".[dev]"
```

Optional dependencies:

```bash
pip install obspy scipy matplotlib
```

Notes:
- NumPy is required
- Filtering requires ObsPy or SciPy
- ObsPy is the primary backend
- SciPy is used as fallback and for envelope computation
- matplotlib is required for plotting

---

## Python API

## Full example

A complete example is available in:

`readme_example.py`

Run it with:

```bash
python readme_example.py
```

---

## Workflow

The typical usage consists of:

1. Prepare a signal
2. Create frequency bands
3. Apply the filter bank
4. (Optional) compute envelopes
5. Plot results

---

### 1. Prepare a signal

#### From NumPy

```python
from filterbanks import Signal

signal = Signal(y=y, x=t, sampling_rate=fs)
```

#### From ObsPy Trace

```python
from obspy import Trace
from filterbanks import prepare_signals

tr = Trace(data=y.astype(float))
tr.stats.sampling_rate = fs

signals = prepare_signals(tr)
```

#### From ObsPy Stream

```python
signals = prepare_signals(stream)
```

`prepare_signals(...)` converts input data into internal Signal objects.

---

### 2. Create frequency bands

```python
from filterbanks import create_bands

bands = create_bands(
    spacing_type="log",
    spacing_step_octaves=0.25,
    bandwidth_type="octave",
    bandwidth_value=0.25,
)
```

This defines the set of band-pass filters that will be applied.

---

### 3. Apply the filter bank

```python
from filterbanks import apply_filterbank

results = apply_filterbank(signal, bands)
```

This applies each band-pass filter to the signal and stores the outputs in `results`.

---

### 4. Add envelopes

```python
from filterbanks import add_envelopes_to_results

add_envelopes_to_results(results)
```

`add_envelopes_to_results` computes an envelope for each filtered signal and adds it to the existing results.

---

### 5. Plot results

```python
from filterbanks import plot_results

fig, axes = plot_results(results)
```

`plot_results` method creates quick plots for the filtered signals (and envelopes if present) for quick analysis.

---

## CLI usage

### Inspect bands

```bash
filterbanks inspect \
  --spacing-type log \
  --spacing-step-octaves 0.25 \
  --bandwidth-type octave \
  --bandwidth-value 0.25 \
  --freq-min 0.01 \
  --freq-max 5 \
  --sampling-rate 20
```

JSON output:

```bash
filterbanks inspect --format json
```

`inspect` only generates and prints band definitions. Sample output:

```json
{
  "band_count": 36,
  "bands": [
    {
      "center_frequency": 0.01,
      "clipped_high_nyquist": false,
      "fmax": 0.010905077326652577,
      "fmin": 0.009170040432046712,
      "name": "band_001",
      "original_fmax": 0.010905077326652577,
      "original_fmin": 0.009170040432046712,
      "overlap_with_next": 9.99819359098338e-16,
      "warnings": []
    },
    ...
```

---

### Run demo

The package comes with a demo run. Follow the code in `filterbanks/demo.py` for more information on the package utilities.

```bash
filterbanks demo
```

This runs:
- synthetic signal generation
- band creation
- filtering
- envelope computation
- plotting

Save output:

```bash
filterbanks demo --save output.png
```

Disable plotting window:

```bash
filterbanks demo --save output.png --no-show
```

---

## Defaults

Package defaults:
- spacing: log
- spacing_step_octaves: 0.25
- bandwidth_type: octave
- bandwidth_value: 0.25
- frequency range: 0.01–5 Hz
- phase: causal
- envelope smoothing window: 5 s

Demo defaults:
- sampling rate: 200 Hz
- duration: 300 s
- components: 0.02, 0.05, 0.2, 1, 3, 8, 20, 40, 70 Hz
- frequency range: 0.01–80 Hz (clipped to 0.9 × Nyquist)
- envelope method: hilbert_fft
- edge policy: nan_mask
