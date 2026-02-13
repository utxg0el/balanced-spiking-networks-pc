"""Regression-style metric checks against a frozen baseline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from bsn_pc.analysis import basic_summary
from bsn_pc.inputs import figure6b_command
from bsn_pc.kernels import normalized_gaussian_kernels
from bsn_pc.model import BalancedSpikingNetwork
from bsn_pc.simulator import SimulationRunner
from bsn_pc.system import LinearDynamicalSystem
from bsn_pc.types import SimulationConfig


BASELINE_PATH = Path("tests/data/regression_baseline.json")


def _run_regression_case(seed: int = 31) -> dict:
    J, N = 2, 30
    dt = 1e-3
    t_start, t_stop = 0.0, 0.3
    t = np.linspace(t_start, t_stop, int((t_stop - t_start) / dt) + 1)

    A = np.array([[-4.8, -22.4], [40.0, 0.0]], dtype=float)
    C = normalized_gaussian_kernels(J, N, target_norm=0.04, rng=np.random.default_rng(seed))
    cmd = figure6b_command(t)

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
        n=0.0,
        sigma_V=0.0,
        seed=seed,
    )
    cfg = SimulationConfig(name="regression_case", t_start=t_start, t_stop=t_stop, dt=dt, seed=seed)
    result = SimulationRunner(system=system, network=network, config=cfg).run(command_values=cmd)
    return result.summary


def test_regression_metrics_within_tolerance() -> None:
    with BASELINE_PATH.open("r", encoding="utf-8") as f:
        baseline = json.load(f)

    summary = _run_regression_case(seed=int(baseline["seed"]))

    for metric, spec in baseline["metrics"].items():
        expected = float(spec["value"])
        tolerance = float(spec["tolerance"])
        actual = float(summary[metric])
        assert abs(actual - expected) <= tolerance, (
            f"Regression drift for {metric}: actual={actual}, expected={expected}, tol={tolerance}"
        )
