"""Figure 6 style plotting routines."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

from bsn_pc.types import SimulationResult


def _plot_panel(ax_cmd: plt.Axes, ax_state: plt.Axes, result: SimulationResult, title: str) -> None:
    t = result.t
    c1 = result.c[:, 0]

    ax_cmd.plot(t, c1, color="#1f77b4", linewidth=1.4)
    ax_cmd.set_title(title)
    ax_cmd.set_ylabel("c1(t)")

    ax_state.plot(t, result.x[:, 0], color="#6baed6", linewidth=1.0, label="x1")
    ax_state.plot(t, result.x[:, 1], color="#9e9ac8", linewidth=1.0, label="x2")
    ax_state.plot(t, result.x_hat[:, 0], color="#d62728", linewidth=2.0, label="x_hat1")
    ax_state.plot(t, result.x_hat[:, 1], color="#9467bd", linewidth=2.0, label="x_hat2")
    ax_state.set_ylabel("state")

    ax_raster = ax_state.twinx()
    ax_raster.scatter(
        result.spike_times,
        result.spike_neurons,
        s=5,
        c="black",
        marker=".",
        alpha=0.35,
    )
    ax_raster.set_ylabel("neuron")
    ax_raster.set_ylim(-2, max(10, np.max(result.spike_neurons) + 2 if result.spike_neurons.size else 10))


def make_figure6_plot(*, differentiator: SimulationResult, oscillator: SimulationResult) -> plt.Figure:
    """Build combined Figure-6-style visualization for both example systems."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex="col")
    _plot_panel(axes[0, 0], axes[1, 0], differentiator, "(A) Leaky Differentiator")
    _plot_panel(axes[0, 1], axes[1, 1], oscillator, "(B) Damped Oscillator")

    axes[1, 0].set_xlabel("time (s)")
    axes[1, 1].set_xlabel("time (s)")
    axes[1, 0].legend(loc="upper left", fontsize=8)

    fig.suptitle("Figure 6-style network examples", fontsize=13)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.98])
    return fig
