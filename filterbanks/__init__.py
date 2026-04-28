"""Small, function-first filterbank toolkit."""

from .bands import FilterBand, create_bands, inspect_bands
from .envelope import add_envelopes_to_results, compute_envelope, smooth_envelope
from .filtering import BandResult, SignalResult, apply_filterbank, preprocess_signal
from .plotting import plot_results
from .signals import Signal, prepare_signals, validate_signal

__all__ = [
    "Signal",
    "FilterBand",
    "BandResult",
    "SignalResult",
    "create_bands",
    "inspect_bands",
    "compute_envelope",
    "smooth_envelope",
    "add_envelopes_to_results",
    "plot_results",
    "prepare_signals",
    "validate_signal",
    "preprocess_signal",
    "apply_filterbank",
]
