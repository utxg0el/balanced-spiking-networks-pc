"""CLI entrypoints for simulations, sweeps, and figure regeneration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import typer

from bsn_pc.experiments.figure3 import Figure3Experiment
from bsn_pc.experiments.figure6 import Figure6Experiment
from bsn_pc.plotting.common import save_figure
from bsn_pc.plotting.figure3 import make_figure3_plot
from bsn_pc.plotting.figure6 import make_figure6_plot
from bsn_pc.sweeps import run_sweep_from_yaml
from bsn_pc.types import RunMetadata, SimulationConfig, SimulationResult

app = typer.Typer(add_completion=False, no_args_is_help=True)


def _load_result_from_npz(npz_path: Path, name: str) -> SimulationResult:
    with np.load(npz_path) as data:
        t = data["t"]
        x = data["x"]
        x_hat = data["x_hat"]
        c = data["c"]
        spike_times = data["spike_times"]
        spike_neurons = data["spike_neurons"]
        V = data["V"] if "V" in data.files else None
        r = data["r"] if "r" in data.files else None

    cfg = SimulationConfig(name=name, t_start=float(t[0]), t_stop=float(t[-1]), dt=float(t[1] - t[0]))
    meta = RunMetadata(experiment_name=name, seed=0)
    return SimulationResult(
        config=cfg,
        metadata=meta,
        t=t,
        x=x,
        x_hat=x_hat,
        c=c,
        spike_times=spike_times,
        spike_neurons=spike_neurons,
        V=V,
        r=r,
        summary={},
    )


@app.command("run")
def run_command(
    experiment: str = typer.Argument(..., help="Experiment name: figure3 or figure6"),
    output_dir: Path = typer.Option(Path("outputs"), help="Root output directory."),
    quick: bool = typer.Option(False, help="Use reduced runtime configuration."),
) -> None:
    """Run one experiment and write artifacts."""
    if experiment == "figure3":
        exp = Figure3Experiment(output_root=output_dir)
    elif experiment == "figure6":
        exp = Figure6Experiment(output_root=output_dir)
    else:
        raise typer.BadParameter("experiment must be one of: figure3, figure6")

    outputs = exp.run(quick=quick)
    typer.echo(str(outputs["run_dir"]))


@app.command("sweep")
def sweep_command(
    config: Path = typer.Option(..., exists=True, dir_okay=False, help="Sweep YAML config."),
    output_dir: Path = typer.Option(Path("outputs/sweeps"), help="Sweep output directory."),
    quick: bool = typer.Option(True, help="Use reduced runtime during sweep."),
) -> None:
    """Run a parameter sweep from YAML specification."""
    csv_path = run_sweep_from_yaml(config_path=config, output_dir=output_dir, quick=quick)
    typer.echo(str(csv_path))


@app.command("plot")
def plot_command(
    run_dir: Path = typer.Option(..., exists=True, file_okay=False, help="Existing run directory."),
) -> None:
    """Regenerate figure files from saved run artifacts."""
    name = run_dir.parent.name

    if name == "figure3":
        baseline = _load_result_from_npz(run_dir / "baseline" / "timeseries.npz", "figure3_baseline")
        perturbed = _load_result_from_npz(run_dir / "perturbed" / "timeseries.npz", "figure3_perturbed")
        sensory_signal = perturbed.c[:, 0]
        if perturbed.V is not None:
            N = int(perturbed.V.shape[1])
        elif baseline.V is not None:
            N = int(baseline.V.shape[1])
        elif perturbed.spike_neurons.size or baseline.spike_neurons.size:
            N = int(np.max(np.append(perturbed.spike_neurons, baseline.spike_neurons))) + 1
        else:
            N = 2
        pos_group = np.arange(0, N // 2)
        neg_group = np.arange(N // 2, N)
        fig = make_figure3_plot(
            baseline=baseline,
            perturbed=perturbed,
            sensory_signal=sensory_signal,
            positive_group=pos_group,
            negative_group=neg_group,
        )
        paths = save_figure(fig, run_dir / "figures", "figure3_regenerated")
        typer.echo(str(paths["png"]))

    elif name == "figure6":
        diff = _load_result_from_npz(run_dir / "differentiator" / "timeseries.npz", "figure6_differentiator")
        osc = _load_result_from_npz(run_dir / "oscillator" / "timeseries.npz", "figure6_oscillator")
        fig = make_figure6_plot(differentiator=diff, oscillator=osc)
        paths = save_figure(fig, run_dir / "figures", "figure6_regenerated")
        typer.echo(str(paths["png"]))

    else:
        raise typer.BadParameter(
            "Could not infer experiment type from run_dir. Expected .../figure3/<timestamp> or .../figure6/<timestamp>."
        )


def main() -> None:
    """Console script entrypoint."""
    app()


if __name__ == "__main__":
    main()
