"""CLI entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from .bands import inspect_bands
from . import demo
from .exceptions import ValidationError

ALLOWED = {
    "spacing_type",
    "spacing_step_octaves",
    "spacing_step_hz",
    "bandwidth_type",
    "bandwidth_value",
    "overlap",
    "freq_min",
    "freq_max",
    "sampling_rate",
    "explicit_bands",
    "format",
}
REJECT_SIGNAL_KEYS = {"signal_duration", "signals", "signal", "x", "y", "detrend", "taper", "backend"}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="filterbanks")
    sub = p.add_subparsers(dest="command")
    i = sub.add_parser("inspect", help="Inspect band generation")
    i.add_argument("--config", type=str, default=None)
    i.add_argument("--spacing-type", choices=["log", "linear", "explicit"], default=None)
    i.add_argument("--spacing-step-octaves", type=float, default=None)
    i.add_argument("--spacing-step-hz", type=float, default=None)
    i.add_argument("--bandwidth-type", choices=["octave", "hz"], default=None)
    i.add_argument("--bandwidth-value", type=float, default=None)
    i.add_argument("--overlap", type=float, default=None)
    i.add_argument("--freq-min", type=float, default=None)
    i.add_argument("--freq-max", type=float, default=None)
    i.add_argument("--sampling-rate", type=float, default=None)
    i.add_argument("--format", choices=["text", "json"], default=None)
    d = sub.add_parser("demo", help="Generate a synthetic demo signal")
    demo.add_demo_arguments(d)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "inspect":
        return _run_inspect(args)
    if args.command == "demo":
        return _run_demo(args)
    build_parser().print_help(sys.stderr)
    return 1


def _run_inspect(args: argparse.Namespace) -> int:
    try:
        cfg = _load(args.config) if args.config else {}
        over = {
            "spacing_type": args.spacing_type,
            "spacing_step_octaves": args.spacing_step_octaves,
            "spacing_step_hz": args.spacing_step_hz,
            "bandwidth_type": args.bandwidth_type,
            "bandwidth_value": args.bandwidth_value,
            "overlap": args.overlap,
            "freq_min": args.freq_min,
            "freq_max": args.freq_max,
            "sampling_rate": args.sampling_rate,
            "format": args.format,
        }
        merged = dict(cfg)
        merged.update({k: v for k, v in over.items() if v is not None})
        _validate(merged)

        fmt = merged.get("format", "text")
        payload = inspect_bands(**{k: v for k, v in merged.items() if k in ALLOWED and k != "format"})
        if fmt == "json":
            print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
        else:
            print(_to_text(payload))
        return 0
    except ValidationError as exc:
        print(f"ValidationError: {exc}", file=sys.stderr)
        return 2


def _run_demo(args: argparse.Namespace) -> int:
    try:
        return demo.run_demo(args)
    except ValidationError as exc:
        print(f"ValidationError: {exc}", file=sys.stderr)
        return 2


def _to_text(payload: dict[str, Any]) -> str:
    lines = [
        "Filterbands Inspect",
        f"spacing_type: {payload.get('spacing_type')}",
        f"bandwidth_type: {payload.get('bandwidth_type')}",
        f"bandwidth_value: {payload.get('bandwidth_value')}",
        f"overlap: {payload.get('overlap')}",
        f"freq_range: {payload.get('freq_min')} .. {payload.get('freq_max')}",
        f"sampling_rate: {payload.get('sampling_rate')}",
        f"band_count: {payload.get('band_count')}",
    ]
    for b in payload.get("bands", []):
        lines.append(f"{b['name']} | center={b['center_frequency']:.6g} | fmin={b['fmin']:.6g} | fmax={b['fmax']:.6g} | warnings={b['warnings']}")
    return "\n".join(lines)


def _load(path: str) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValidationError("Config JSON must be an object.")
    return data


def _validate(values: dict[str, Any]) -> None:
    for key in values:
        if key in REJECT_SIGNAL_KEYS:
            raise ValidationError(f"inspect does not accept signal-related key: {key}")
        if key not in ALLOWED:
            raise ValidationError(f"Unknown inspect config key: {key}")
    if values.get("format", "text") not in {"text", "json"}:
        raise ValidationError("format must be one of: text, json.")


if __name__ == "__main__":
    raise SystemExit(main())
