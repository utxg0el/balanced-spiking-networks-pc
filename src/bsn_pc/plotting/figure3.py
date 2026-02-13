"""Figure 3 style visualization routines."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

from bsn_pc.analysis import exponential_population_rate
from bsn_pc.types import SimulationResult


def make_figure3_plot(
    *,
    baseline: SimulationResult,
    perturbed: SimulationResult,
    sensory_signal: np.ndarray,
    positive_group: np.ndarray,
    negative_group: np.ndarray,
    tau_rate: float = 0.1,
) -> plt.Figure:
    """Build a qualitative Figure 3 reproduction plot."""
    t = baseline.t
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    ax0, ax1, ax2 = axes

    ax0.plot(t, sensory_signal, color="#1f77b4", linewidth=1.5)
    ax0.set_ylabel("sensory input")
    ax0.set_title("Figure 3-style inhomogeneous integrator response")

    ax1.scatter(
        baseline.spike_times,
        baseline.spike_neurons,
        s=7,
        c="black",
        marker=".",
        alpha=0.6,
        label="baseline",
    )
    ax1.scatter(
        perturbed.spike_times,
        perturbed.spike_neurons,
        s=10,
        facecolors="none",
        edgecolors="#d62728",
        marker="o",
        alpha=0.8,
        label="perturbed",
    )
    ax1.set_ylabel("neuron index")
    ax1.legend(loc="upper right", fontsize=9)

    pos_rate = exponential_population_rate(
        t=t,
        spike_times=perturbed.spike_times,
        spike_neurons=perturbed.spike_neurons,
        neuron_group=positive_group,
        tau=tau_rate,
    )
    neg_rate = exponential_population_rate(
        t=t,
        spike_times=perturbed.spike_times,
        spike_neurons=perturbed.spike_neurons,
        neuron_group=negative_group,
        tau=tau_rate,
    )
    ax2.plot(t, pos_rate, color="#e377c2", linewidth=1.5, label="positive kernels")
    ax2.plot(t, neg_rate, color="#2ca02c", linewidth=1.5, label="negative kernels")
    ax2.set_ylabel("rate (Hz)")
    ax2.set_xlabel("time (s)")
    ax2.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    return fig
