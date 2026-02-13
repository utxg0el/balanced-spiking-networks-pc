"""Simulation-level tests for repeatability and qualitative behavior."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from bsn_pc.experiments.figure3 import Figure3Config, Figure3Experiment
from bsn_pc.experiments.figure6 import Figure6Config, Figure6Experiment
from bsn_pc.inputs import figure6a_command
from bsn_pc.kernels import normalized_gaussian_kernels
from bsn_pc.model import BalancedSpikingNetwork
from bsn_pc.simulator import SimulationRunner
from bsn_pc.system import LinearDynamicalSystem
from bsn_pc.types import SimulationConfig


def _run_small_deterministic(seed: int = 0):
    J, N = 2, 24
    dt = 1e-3
    t = np.linspace(0.0, 0.2, 201)

    A = np.array([[-4.0, -1.5], [0.8, -2.0]], dtype=float)
    C = normalized_gaussian_kernels(J, N, target_norm=0.08, rng=np.random.default_rng(seed))
    cmd = figure6a_command(t) * 0.2

    def c_fn(tt: float) -> np.ndarray:
        idx = int(np.searchsorted(t, tt, side="right") - 1)
        idx = int(np.clip(idx, 0, t.size - 1))
        return cmd[idx]

    system = LinearDynamicalSystem(A=A, c_fn=c_fn, x0=np.zeros(J))
    network = BalancedSpikingNetwork(
        C=C,
        A=A,
        lambda_d=10.0,
        lambda_V=20.0,
        m=1e-6,
        n=1e-5,
        sigma_V=0.0,
        seed=seed,
    )
    cfg = SimulationConfig(name="deterministic_repeat", t_start=0.0, t_stop=0.2, dt=dt, seed=seed)
    result = SimulationRunner(system=system, network=network, config=cfg).run(command_values=cmd)
    return result


def test_deterministic_repeatability() -> None:
    r1 = _run_small_deterministic(seed=9)
    r2 = _run_small_deterministic(seed=9)

    assert np.array_equal(r1.spike_times, r2.spike_times)
    assert np.array_equal(r1.spike_neurons, r2.spike_neurons)
    assert np.allclose(r1.x_hat, r2.x_hat)


def test_figure3_perturbation_diverges_after_trigger(tmp_path: Path) -> None:
    cfg = Figure3Config(
        N=60,
        J=6,
        dt=1e-4,
        t_stop=0.25,
        input_cutoff=0.18,
        perturb_time=0.12,
        seed=7,
    )
    exp = Figure3Experiment(output_root=tmp_path, config=cfg)
    outputs = exp.run(quick=False)

    with np.load(outputs["baseline_dir"] / "timeseries.npz") as bdata:
        bt = bdata["spike_times"]
        bn = bdata["spike_neurons"]
    with np.load(outputs["perturbed_dir"] / "timeseries.npz") as pdata:
        pt = pdata["spike_times"]
        pn = pdata["spike_neurons"]

    pre_b = bt < cfg.perturb_time
    pre_p = pt < cfg.perturb_time

    assert np.array_equal(bt[pre_b], pt[pre_p])
    assert np.array_equal(bn[pre_b], pn[pre_p])

    # After perturbation, sequence should no longer be identical.
    assert not (np.array_equal(bt, pt) and np.array_equal(bn, pn))


def test_figure6_tracks_dynamics_with_positive_correlation(tmp_path: Path) -> None:
    cfg = Figure6Config.quick(seed=13)
    exp = Figure6Experiment(output_root=tmp_path, config=cfg)
    outputs = exp.run(quick=True)

    with (outputs["run_dir"] / "summary.json").open("r", encoding="utf-8") as f:
        summary = json.load(f)

    assert summary["differentiator_summary"]["mean_corr"] > 0.4
    assert summary["oscillator_summary"]["mean_corr"] > 0.2
