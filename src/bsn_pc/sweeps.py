"""Hyperparameter sweep utilities."""

from __future__ import annotations

import itertools
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, MutableMapping, Tuple

import pandas as pd
import yaml

from bsn_pc.experiments.figure3 import Figure3Config, Figure3Experiment
from bsn_pc.experiments.figure6 import Figure6Config, Figure6Experiment


MetricCallback = Callable[[Dict[str, float]], Dict[str, float]]


class SweepRunner:
    """Generic Cartesian hyperparameter sweep runner."""

    def __init__(
        self,
        *,
        parameter_grid: Mapping[str, Iterable],
        run_fn: Callable[[Dict[str, object], int], Dict[str, float]],
        repeats: int = 1,
    ) -> None:
        self.parameter_grid = {k: list(v) for k, v in parameter_grid.items()}
        self.run_fn = run_fn
        self.repeats = int(repeats)
        if self.repeats <= 0:
            raise ValueError("repeats must be positive.")

    def _combinations(self) -> Iterable[Dict[str, object]]:
        keys = list(self.parameter_grid.keys())
        values = [self.parameter_grid[k] for k in keys]
        for combo in itertools.product(*values):
            yield {k: combo[i] for i, k in enumerate(keys)}

    def run(self) -> pd.DataFrame:
        """Run sweep and return a flat results table."""
        rows: List[Dict[str, object]] = []
        for params in self._combinations():
            for repeat in range(self.repeats):
                metrics = self.run_fn(params, repeat)
                row: Dict[str, object] = {"repeat": repeat, **params, **metrics}
                rows.append(row)
        return pd.DataFrame(rows)


def _update_dataclass_fields(cfg, params: Mapping[str, object]):
    """Return a dataclass clone updated with sweep parameter values."""
    for key in params:
        if not hasattr(cfg, key):
            raise ValueError(f"Unknown sweep parameter '{key}' for config {type(cfg).__name__}.")
    return replace(cfg, **params)


def run_sweep_from_yaml(
    *,
    config_path: Path,
    output_dir: Path,
    quick: bool = True,
) -> Path:
    """Run a sweep defined in YAML and save CSV/JSON artifacts.

    Expected YAML schema:

    ```yaml
    experiment: figure3  # or figure6
    repeats: 1
    parameters:
      sigma_V: [0.0, 0.001]
      m: [1e-6, 1e-5]
    ```
    """
    with config_path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)

    experiment_name = str(payload.get("experiment", "figure6"))
    repeats = int(payload.get("repeats", 1))
    param_grid = payload.get("parameters", {})
    if not param_grid:
        raise ValueError("Sweep config must include a non-empty 'parameters' mapping.")

    output_dir.mkdir(parents=True, exist_ok=True)

    if experiment_name == "figure3":
        base_cfg = Figure3Config.quick() if quick else Figure3Config()

        def run_fn(params: Dict[str, object], repeat: int) -> Dict[str, float]:
            cfg = _update_dataclass_fields(base_cfg, params)
            cfg = replace(cfg, seed=cfg.seed + repeat)
            exp = Figure3Experiment(output_root=output_dir / "runs", config=cfg)
            outputs = exp.run(quick=quick)
            summary_path = Path(outputs["run_dir"]) / "perturbed" / "summary.json"
            with summary_path.open("r", encoding="utf-8") as sf:
                summary = json.load(sf)
            return {
                "rmse": float(summary["summary"]["rmse"]),
                "mean_rate_hz": float(summary["summary"]["mean_rate_hz"]),
            }

    elif experiment_name == "figure6":
        base_cfg = Figure6Config.quick() if quick else Figure6Config()

        def run_fn(params: Dict[str, object], repeat: int) -> Dict[str, float]:
            cfg = _update_dataclass_fields(base_cfg, params)
            cfg = replace(cfg, seed=cfg.seed + repeat)
            exp = Figure6Experiment(output_root=output_dir / "runs", config=cfg)
            outputs = exp.run(quick=quick)
            summary_path = Path(outputs["run_dir"]) / "summary.json"
            with summary_path.open("r", encoding="utf-8") as sf:
                summary = json.load(sf)
            return {
                "rmse_diff": float(summary["differentiator_summary"]["rmse"]),
                "rmse_osc": float(summary["oscillator_summary"]["rmse"]),
                "mean_corr_diff": float(summary["differentiator_summary"]["mean_corr"]),
                "mean_corr_osc": float(summary["oscillator_summary"]["mean_corr"]),
            }

    else:
        raise ValueError("experiment must be one of: 'figure3', 'figure6'.")

    runner = SweepRunner(parameter_grid=param_grid, run_fn=run_fn, repeats=repeats)
    df = runner.run()

    csv_path = output_dir / "sweep_results.csv"
    json_path = output_dir / "sweep_results.json"
    df.to_csv(csv_path, index=False)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(df.to_dict(orient="records"), f, indent=2)

    return csv_path
