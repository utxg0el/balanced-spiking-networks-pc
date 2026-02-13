"""Simulation orchestration and artifact persistence."""

from __future__ import annotations

import json
import subprocess
import time
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import yaml

from bsn_pc.analysis import basic_summary
from bsn_pc.model import BalancedSpikingNetwork
from bsn_pc.system import LinearDynamicalSystem
from bsn_pc.types import ExperimentArtifacts, RunMetadata, SimulationConfig, SimulationResult, SpikeDelaySpec


def _get_git_commit(cwd: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(cwd), stderr=subprocess.DEVNULL
        )
        return out.decode("utf-8").strip()
    except Exception:
        return "unknown"


class SimulationRunner:
    """Runs fixed-step simulations for a system/network pair."""

    def __init__(
        self,
        *,
        system: LinearDynamicalSystem,
        network: BalancedSpikingNetwork,
        config: SimulationConfig,
        align_estimate_to_system: bool = True,
    ) -> None:
        self.system = system
        self.network = network
        self.config = config
        self.align_estimate_to_system = align_estimate_to_system

    def _warn_if_potentially_stiff(self) -> None:
        eigs = np.linalg.eigvals(self.system.A)
        max_real = float(np.max(np.abs(np.real(eigs))))
        stiffness = max_real * self.config.dt
        if stiffness > 0.2:
            warnings.warn(
                (
                    "dt may be too large for stable/accurate Euler integration: "
                    f"max|Re(eig(A))|*dt={stiffness:.3f}."
                ),
                RuntimeWarning,
            )

    def run(
        self,
        *,
        command_values: Optional[np.ndarray] = None,
        spike_delay_spec: Optional[SpikeDelaySpec] = None,
    ) -> SimulationResult:
        """Execute simulation and return a full ``SimulationResult``."""
        self.config.validate()
        self._warn_if_potentially_stiff()

        t = np.linspace(self.config.t_start, self.config.t_stop, self.config.num_steps)

        if command_values is not None:
            c_values = np.asarray(command_values, dtype=float)
            if c_values.shape != (t.size, self.system.dim):
                raise ValueError(
                    "command_values must have shape "
                    f"({t.size}, {self.system.dim}), got {c_values.shape}."
                )
        else:
            c_values = np.vstack([self.system.command(tt) for tt in t])

        x = np.zeros((t.size, self.system.dim), dtype=float)
        x_hat = np.zeros((t.size, self.system.dim), dtype=float)
        V = np.zeros((t.size, self.network.N), dtype=float) if self.config.record_voltage else None
        r = np.zeros((t.size, self.network.N), dtype=float) if self.config.record_rate else None

        self.system.reset()
        self.network.reset(x_hat0=self.system.x if self.align_estimate_to_system else None)

        if spike_delay_spec is not None:
            self.network.inject_spike_delay(spike_delay_spec)
        else:
            self.network.clear_spike_delay()

        x[0] = self.system.x
        x_hat[0] = self.network.x_hat
        if V is not None:
            V[0] = self.network.V
        if r is not None:
            r[0] = self.network.r

        t_start_wall = time.perf_counter()

        for k in range(t.size - 1):
            c_t = c_values[k]
            self.network.step(
                c_t=c_t,
                dt=self.config.dt,
                t=float(t[k]),
                max_spikes_per_step=self.config.max_spikes_per_step,
            )
            self.system.step_with_command(c_t, self.config.dt)

            x[k + 1] = self.system.x
            x_hat[k + 1] = self.network.x_hat
            if V is not None:
                V[k + 1] = self.network.V
            if r is not None:
                r[k + 1] = self.network.r

        runtime = time.perf_counter() - t_start_wall
        spike_times, spike_neurons = self.network.spike_history

        summary = basic_summary(
            x=x,
            x_hat=x_hat,
            spike_times=spike_times,
            spike_neurons=spike_neurons,
            N=self.network.N,
            duration=self.config.t_stop - self.config.t_start,
        )

        metadata = RunMetadata(
            experiment_name=self.config.name,
            seed=self.config.seed,
            runtime_seconds=runtime,
            git_commit=_get_git_commit(Path.cwd()),
        )

        return SimulationResult(
            config=self.config,
            metadata=metadata,
            t=t,
            x=x,
            x_hat=x_hat,
            c=c_values,
            spike_times=spike_times,
            spike_neurons=spike_neurons,
            V=V,
            r=r,
            summary=summary,
        )


def save_simulation_result(
    result: SimulationResult,
    *,
    run_dir: Path,
    figures: Optional[Dict[str, Path]] = None,
    extra_config: Optional[Dict[str, Any]] = None,
) -> ExperimentArtifacts:
    """Persist simulation arrays and metadata to a run directory."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)

    config_path = run_dir / "config.yaml"
    summary_path = run_dir / "summary.json"
    timeseries_path = run_dir / "timeseries.npz"
    metrics_path = run_dir / "metrics.csv"

    config_payload: Dict[str, Any] = {
        "simulation_config": asdict(result.config),
        "metadata": result.metadata.to_dict(),
    }
    if extra_config:
        config_payload["extra_config"] = extra_config
    with config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config_payload, f, sort_keys=False)

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(result.to_summary_dict(), f, indent=2)

    npz_payload = {
        "t": result.t,
        "x": result.x,
        "x_hat": result.x_hat,
        "c": result.c,
        "spike_times": result.spike_times,
        "spike_neurons": result.spike_neurons,
    }
    if result.V is not None:
        npz_payload["V"] = result.V
    if result.r is not None:
        npz_payload["r"] = result.r
    np.savez_compressed(timeseries_path, **npz_payload)

    pd.DataFrame([result.summary]).to_csv(metrics_path, index=False)

    figure_paths: Dict[str, Path] = {}
    if figures:
        for key, path in figures.items():
            figure_paths[key] = path

    return ExperimentArtifacts(
        run_dir=run_dir,
        config_path=config_path,
        summary_path=summary_path,
        timeseries_path=timeseries_path,
        metrics_path=metrics_path,
        figures=figure_paths,
    )


def load_simulation_timeseries(run_dir: Path) -> Dict[str, np.ndarray]:
    """Load arrays from ``timeseries.npz`` for plotting/regeneration."""
    path = run_dir / "timeseries.npz"
    with np.load(path) as data:
        return {k: data[k] for k in data.files}
