import json

from filterbanks.cli import main


def test_cli_inspect_inline_args(capsys) -> None:
    rc = main(
        [
            "inspect",
            "--spacing-type",
            "linear",
            "--spacing-step-hz",
            "1.0",
            "--bandwidth-type",
            "hz",
            "--bandwidth-value",
            "2.0",
            "--freq-min",
            "1.0",
            "--freq-max",
            "3.0",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "Filterbands Inspect" in out
    assert "spacing_type: linear" in out


def test_cli_inspect_json_config(tmp_path, capsys) -> None:
    config = {
        "spacing_type": "linear",
        "spacing_step_hz": 1.0,
        "bandwidth_type": "hz",
        "bandwidth_value": 2.0,
        "freq_min": 1.0,
        "freq_max": 3.0,
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    rc = main(["inspect", "--config", str(path), "--format", "json"])
    out = capsys.readouterr().out
    payload = json.loads(out)

    assert rc == 0
    assert payload["spacing_type"] == "linear"
    assert payload["band_count"] == 3


def test_cli_rejects_signal_related_keys(tmp_path, capsys) -> None:
    config = {
        "spacing_type": "linear",
        "spacing_step_hz": 1.0,
        "bandwidth_type": "hz",
        "bandwidth_value": 2.0,
        "freq_min": 1.0,
        "freq_max": 3.0,
        "signal_duration": 10.0,
    }
    path = tmp_path / "config_bad_key.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    rc = main(["inspect", "--config", str(path)])
    err = capsys.readouterr().err

    assert rc == 2
    assert "signal-related key" in err
    assert "signal_duration" in err
