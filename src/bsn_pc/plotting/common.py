"""Shared plotting utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np


def save_figure(fig: plt.Figure, out_dir: Path, stem: str, dpi: int = 180) -> Dict[str, Path]:
    """Save a Matplotlib figure as PNG and SVG."""
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{stem}.png"
    svg_path = out_dir / f"{stem}.svg"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    return {"png": png_path, "svg": svg_path}


def plot_raster(ax: plt.Axes, spike_times: np.ndarray, spike_neurons: np.ndarray, *, color: str, alpha: float = 1.0, marker: str = ".") -> None:
    """Scatter-style spike raster."""
    if spike_times.size == 0:
        return
    ax.scatter(
        spike_times,
        spike_neurons,
        s=6,
        c=color,
        marker=marker,
        linewidths=0.0,
        alpha=alpha,
    )
