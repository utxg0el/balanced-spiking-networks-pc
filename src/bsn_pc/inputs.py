"""Reusable command/sensory input signal generators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


@dataclass
class DiscreteCommandSignal:
    """Time-indexed command function backed by precomputed samples."""

    t: np.ndarray
    values: np.ndarray

    def __post_init__(self) -> None:
        self.t = np.asarray(self.t, dtype=float)
        self.values = np.asarray(self.values, dtype=float)
        if self.t.ndim != 1:
            raise ValueError("t must be a 1D array.")
        if self.values.ndim != 2:
            raise ValueError("values must be a 2D array with shape (T, J).")
        if self.values.shape[0] != self.t.size:
            raise ValueError("values first dimension must match t length.")

    def __call__(self, t_query: float) -> np.ndarray:
        idx = int(np.searchsorted(self.t, t_query, side="right") - 1)
        idx = int(np.clip(idx, 0, self.t.size - 1))
        return self.values[idx]


CommandArrayBuilder = Callable[[np.ndarray], np.ndarray]


def piecewise_constant_1d(t: np.ndarray, breakpoints: Sequence[float], values: Sequence[float]) -> np.ndarray:
    """Create a scalar piecewise constant profile.

    ``len(values)`` must be ``len(breakpoints) + 1``.
    """
    t = np.asarray(t, dtype=float)
    if len(values) != len(breakpoints) + 1:
        raise ValueError("values must have one more element than breakpoints.")
    out = np.full_like(t, fill_value=float(values[0]), dtype=float)
    for i, bp in enumerate(breakpoints):
        out[t >= bp] = float(values[i + 1])
    return out


def figure3_sensory_command(
    t: np.ndarray,
    *,
    J: int,
    sigma_s: float,
    cutoff_time: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate Figure-3-style scalar sensory input and replicated J-D command.

    Returns:
        scalar_signal: Shape ``(T,)`` sensory input shown in top panel.
        command: Shape ``(T, J)`` command fed to the dynamical system.
    """
    if J <= 0:
        raise ValueError("J must be positive.")
    if sigma_s < 0.0:
        raise ValueError("sigma_s must be non-negative.")

    t_end = float(t[-1]) if t.size else 1.0
    breakpoints = [
        0.08 * t_end,
        0.22 * t_end,
        0.38 * t_end,
        0.54 * t_end,
        0.68 * t_end,
    ]
    base = piecewise_constant_1d(
        t,
        breakpoints=breakpoints,
        values=[0.0, 3.0, -2.0, 4.0, 1.5, 0.0],
    )

    noise = rng.normal(0.0, sigma_s, size=t.size)
    active = (t < cutoff_time).astype(float)
    scalar_signal = base + active * noise
    scalar_signal = np.where(t < cutoff_time, scalar_signal, 0.0)

    command = np.repeat(scalar_signal[:, None], repeats=J, axis=1)
    return scalar_signal, command


def figure6a_command(t: np.ndarray) -> np.ndarray:
    """Command signal for the leaky differentiator example (J=2, only c1 active)."""
    c1 = piecewise_constant_1d(
        t,
        breakpoints=[0.25, 0.45, 0.75, 0.95, 1.25],
        values=[0.0, 40.0, -20.0, 30.0, -10.0, 0.0],
    )
    c2 = np.zeros_like(c1)
    return np.column_stack([c1, c2])


def figure6b_command(t: np.ndarray) -> np.ndarray:
    """Initial kick command for damped harmonic oscillator example (J=2)."""
    c1 = np.zeros_like(t)
    kick = (t >= 0.06) & (t <= 0.12)
    c1[kick] = 1.2
    c1 += 0.4 * np.exp(-((t - 0.14) / 0.04) ** 2)
    c2 = np.zeros_like(c1)
    return np.column_stack([c1, c2])
