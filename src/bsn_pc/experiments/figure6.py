"""Figure 6 experiment: leaky differentiator and damped oscillator."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from bsn_pc.experiments.base import Experiment
from bsn_pc.inputs import figure6a_command, figure6b_command
from bsn_pc.kernels import normalized_gaussian_kernels
from bsn_pc.model import BalancedSpikingNetwork
from bsn_pc.plotting.common import save_figure
from bsn_pc.plotting.figure6 import make_figure6_plot
from bsn_pc.simulator import SimulationRunner, save_simulation_result
from bsn_pc.system import LinearDynamicalSystem
from bsn_pc.types import SimulationConfig


@dataclass
class Figure6Config:
    """Configuration for the two-system Figure 6 example."""

    N: int = 100
    J: int = 2
    kernel_norm: float = 0.03
    lambda_d: float = 10.0
    lambda_V: float = 20.0
    m: float = 1e-6
    n: float = 0.0
    sigma_V: float = 1e-3
    dt: float = 1e-4
    t_start: float = 0.0
    t_stop: float = 1.6
    seed: int = 1

    @classmethod
    def quick(cls, seed: int = 1) -> "Figure6Config":
        """Fast config used for smoke tests and notebook quick mode."""
        return cls(N=40, t_stop=0.35, seed=seed)


class Figure6Experiment(Experiment):
    """Runs Figure-6-style dynamical system examples."""

    def __init__(self, *, output_root: Path, config: Figure6Config | None = None) -> None:
        super().__init__(output_root=output_root)
        self.config = config or Figure6Config()

    @property
    def name(self) -> str:
        return "figure6"

    def _run_case(
        self,
        *,
        label: str,
        A: np.ndarray,
        command_values: np.ndarray,
        C: np.ndarray,
        cfg: Figure6Config,
    ):
        t = np.linspace(cfg.t_start, cfg.t_stop, int(np.floor((cfg.t_stop - cfg.t_start) / cfg.dt)) + 1)

        def c_fn(tt: float) -> np.ndarray:
            idx = int(np.searchsorted(t, tt, side="right") - 1)
            idx = int(np.clip(idx, 0, t.size - 1))
            return command_values[idx]

        system = LinearDynamicalSystem(A=A, c_fn=c_fn, x0=np.zeros(cfg.J))
        network = BalancedSpikingNetwork(
            C=C,
            A=A,
            lambda_d=cfg.lambda_d,
            lambda_V=cfg.lambda_V,
            m=cfg.m,
            n=cfg.n,
            sigma_V=cfg.sigma_V,
            seed=cfg.seed,
        )
        sim_cfg = SimulationConfig(
            name=f"figure6_{label}",
            t_start=cfg.t_start,
            t_stop=cfg.t_stop,
            dt=cfg.dt,
            seed=cfg.seed,
            max_spikes_per_step=2048,
        )
        result = SimulationRunner(system=system, network=network, config=sim_cfg).run(
            command_values=command_values
        )
        return result

    def run(self, *, quick: bool = False) -> Dict[str, Path]:
        cfg = self.config if not quick else Figure6Config.quick(seed=self.config.seed)
        run_dir = self.make_run_dir()

        t = np.linspace(cfg.t_start, cfg.t_stop, int(np.floor((cfg.t_stop - cfg.t_start) / cfg.dt)) + 1)
        rng = np.random.default_rng(cfg.seed)
        C = normalized_gaussian_kernels(cfg.J, cfg.N, target_norm=cfg.kernel_norm, rng=rng)

        A_diff = np.array([[-400.0, -800.0], [50.0, 0.0]], dtype=float)
        A_osc = np.array([[-4.8, -22.4], [40.0, 0.0]], dtype=float)

        cmd_diff = figure6a_command(t)
        cmd_osc = figure6b_command(t)

        differentiator = self._run_case(
            label="differentiator",
            A=A_diff,
            command_values=cmd_diff,
            C=C,
            cfg=cfg,
        )
        oscillator = self._run_case(
            label="oscillator",
            A=A_osc,
            command_values=cmd_osc,
            C=C,
            cfg=cfg,
        )

        fig = make_figure6_plot(differentiator=differentiator, oscillator=oscillator)
        figure_paths = save_figure(fig, run_dir / "figures", "figure6")

        save_simulation_result(
            differentiator,
            run_dir=run_dir / "differentiator",
            extra_config={"figure6_config": asdict(cfg), "A": A_diff.tolist(), "kernel_norm": cfg.kernel_norm},
        )
        save_simulation_result(
            oscillator,
            run_dir=run_dir / "oscillator",
            extra_config={"figure6_config": asdict(cfg), "A": A_osc.tolist(), "kernel_norm": cfg.kernel_norm},
        )

        with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "config": asdict(cfg),
                    "differentiator_summary": differentiator.summary,
                    "oscillator_summary": oscillator.summary,
                    "figure_paths": {k: str(v) for k, v in figure_paths.items()},
                },
                f,
                indent=2,
            )

        return {
            "run_dir": run_dir,
            "figure_png": figure_paths["png"],
            "figure_svg": figure_paths["svg"],
            "differentiator_dir": run_dir / "differentiator",
            "oscillator_dir": run_dir / "oscillator",
        }
