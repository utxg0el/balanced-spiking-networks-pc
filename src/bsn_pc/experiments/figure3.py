"""Figure 3 experiment: inhomogeneous integrator with spike-delay perturbation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from bsn_pc.experiments.base import Experiment
from bsn_pc.inputs import figure3_sensory_command
from bsn_pc.kernels import inhomogeneous_sparse_signed_kernels, split_neuron_groups_by_mean_sign
from bsn_pc.model import BalancedSpikingNetwork
from bsn_pc.plotting.common import save_figure
from bsn_pc.plotting.figure3 import make_figure3_plot
from bsn_pc.simulator import SimulationRunner, save_simulation_result
from bsn_pc.system import LinearDynamicalSystem
from bsn_pc.types import SimulationConfig, SpikeDelaySpec


@dataclass
class Figure3Config:
    """Configuration defaults for Figure 3 style simulation."""

    N: int = 400
    J: int = 30
    lambda_s: float = 0.0
    lambda_d: float = 10.0
    lambda_V: float = 20.0
    m: float = 1e-6
    n: float = 1e-5
    sigma_V: float = 0.0
    sigma_s: float = 0.01
    dt: float = 1e-4
    t_start: float = 0.0
    t_stop: float = 2.0
    input_cutoff: float = 1.6
    perturb_time: float = 1.65
    perturb_delay: float = 1e-3
    seed: int = 0

    @classmethod
    def quick(cls, seed: int = 0) -> "Figure3Config":
        """Fast config used for smoke tests and notebook quick mode."""
        return cls(
            N=80,
            J=8,
            t_stop=0.35,
            input_cutoff=0.22,
            perturb_time=0.24,
            seed=seed,
        )


class Figure3Experiment(Experiment):
    """Runs baseline + perturbed simulations and plots a Figure-3-style summary."""

    def __init__(self, *, output_root: Path, config: Figure3Config | None = None) -> None:
        super().__init__(output_root=output_root)
        self.config = config or Figure3Config()

    @property
    def name(self) -> str:
        return "figure3"

    def _build_network(self, C: np.ndarray, A: np.ndarray, seed: int) -> BalancedSpikingNetwork:
        cfg = self.config
        return BalancedSpikingNetwork(
            C=C,
            A=A,
            lambda_d=cfg.lambda_d,
            lambda_V=cfg.lambda_V,
            m=cfg.m,
            n=cfg.n,
            sigma_V=cfg.sigma_V,
            seed=seed,
        )

    @staticmethod
    def _first_divergence(
        base_times: np.ndarray,
        base_neurons: np.ndarray,
        pert_times: np.ndarray,
        pert_neurons: np.ndarray,
    ) -> float:
        """Return first divergence time between two spike sequences."""
        m = min(base_times.size, pert_times.size)
        for i in range(m):
            if base_neurons[i] != pert_neurons[i] or abs(base_times[i] - pert_times[i]) > 1e-12:
                return float(min(base_times[i], pert_times[i]))
        if base_times.size != pert_times.size:
            return float(base_times[m - 1] if m > 0 else 0.0)
        return float("nan")

    def run(self, *, quick: bool = False) -> Dict[str, Path]:
        cfg = self.config if not quick else Figure3Config.quick(seed=self.config.seed)
        run_dir = self.make_run_dir()

        t = np.linspace(cfg.t_start, cfg.t_stop, int(np.floor((cfg.t_stop - cfg.t_start) / cfg.dt)) + 1)

        rng_kernel = np.random.default_rng(cfg.seed)
        C = inhomogeneous_sparse_signed_kernels(
            J=cfg.J,
            N=cfg.N,
            rng=rng_kernel,
        )
        A = -cfg.lambda_s * np.eye(cfg.J)

        rng_input = np.random.default_rng(cfg.seed + 17)
        sensory_signal, command_values = figure3_sensory_command(
            t,
            J=cfg.J,
            sigma_s=cfg.sigma_s,
            cutoff_time=cfg.input_cutoff,
            rng=rng_input,
        )

        def c_fn(tt: float) -> np.ndarray:
            idx = int(np.searchsorted(t, tt, side="right") - 1)
            idx = int(np.clip(idx, 0, t.size - 1))
            return command_values[idx]

        system_base = LinearDynamicalSystem(A=A, c_fn=c_fn, x0=np.zeros(cfg.J))
        net_base = self._build_network(C, A, seed=cfg.seed)
        sim_cfg_base = SimulationConfig(
            name="figure3_baseline",
            t_start=cfg.t_start,
            t_stop=cfg.t_stop,
            dt=cfg.dt,
            seed=cfg.seed,
            max_spikes_per_step=2048,
        )
        baseline = SimulationRunner(system=system_base, network=net_base, config=sim_cfg_base).run(
            command_values=command_values
        )

        system_pert = LinearDynamicalSystem(A=A, c_fn=c_fn, x0=np.zeros(cfg.J))
        net_pert = self._build_network(C, A, seed=cfg.seed)
        sim_cfg_pert = SimulationConfig(
            name="figure3_perturbed",
            t_start=cfg.t_start,
            t_stop=cfg.t_stop,
            dt=cfg.dt,
            seed=cfg.seed,
            max_spikes_per_step=2048,
        )
        perturbed = SimulationRunner(system=system_pert, network=net_pert, config=sim_cfg_pert).run(
            command_values=command_values,
            spike_delay_spec=SpikeDelaySpec(
                trigger_time=cfg.perturb_time,
                delay=cfg.perturb_delay,
                neuron_index=None,
                occurrence=1,
            ),
        )

        pos_group, neg_group = split_neuron_groups_by_mean_sign(C)
        fig = make_figure3_plot(
            baseline=baseline,
            perturbed=perturbed,
            sensory_signal=sensory_signal,
            positive_group=pos_group,
            negative_group=neg_group,
            tau_rate=0.1,
        )
        figure_paths = save_figure(fig, run_dir / "figures", "figure3")

        save_simulation_result(
            baseline,
            run_dir=run_dir / "baseline",
            extra_config={"figure3_config": asdict(cfg), "kernel_shape": list(C.shape)},
        )
        save_simulation_result(
            perturbed,
            run_dir=run_dir / "perturbed",
            extra_config={"figure3_config": asdict(cfg), "kernel_shape": list(C.shape)},
        )

        first_divergence = self._first_divergence(
            baseline.spike_times,
            baseline.spike_neurons,
            perturbed.spike_times,
            perturbed.spike_neurons,
        )

        with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "config": asdict(cfg),
                    "first_divergence_time": first_divergence,
                    "num_spikes_baseline": int(baseline.spike_times.size),
                    "num_spikes_perturbed": int(perturbed.spike_times.size),
                    "figure_paths": {k: str(v) for k, v in figure_paths.items()},
                },
                f,
                indent=2,
            )

        return {
            "run_dir": run_dir,
            "figure_png": figure_paths["png"],
            "figure_svg": figure_paths["svg"],
            "baseline_dir": run_dir / "baseline",
            "perturbed_dir": run_dir / "perturbed",
        }
