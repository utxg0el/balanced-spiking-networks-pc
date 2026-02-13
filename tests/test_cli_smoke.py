"""CLI smoke tests."""

from __future__ import annotations

from pathlib import Path

import yaml
from typer.testing import CliRunner

from bsn_pc.cli import app


runner = CliRunner()


def test_cli_run_figure3_quick(tmp_path: Path) -> None:
    result = runner.invoke(app, ["run", "figure3", "--output-dir", str(tmp_path), "--quick"])
    assert result.exit_code == 0, result.stdout
    run_dir = Path(result.stdout.strip())
    assert run_dir.exists()
    assert (run_dir / "figures").exists()


def test_cli_run_figure6_quick(tmp_path: Path) -> None:
    result = runner.invoke(app, ["run", "figure6", "--output-dir", str(tmp_path), "--quick"])
    assert result.exit_code == 0, result.stdout
    run_dir = Path(result.stdout.strip())
    assert run_dir.exists()
    assert (run_dir / "figures").exists()


def test_cli_sweep_quick(tmp_path: Path) -> None:
    cfg = {
        "experiment": "figure6",
        "repeats": 1,
        "parameters": {"sigma_V": [0.0005], "m": [1e-6]},
    }
    cfg_path = tmp_path / "sweep.yaml"
    with cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    result = runner.invoke(
        app,
        [
            "sweep",
            "--config",
            str(cfg_path),
            "--output-dir",
            str(tmp_path / "sweeps"),
            "--quick",
        ],
    )
    assert result.exit_code == 0, result.stdout
    out_csv = Path(result.stdout.strip())
    assert out_csv.exists()

