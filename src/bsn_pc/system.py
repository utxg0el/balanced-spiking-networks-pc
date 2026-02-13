"""Linear dynamical systems used as target processes for network tracking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from bsn_pc.types import ensure_numpy_array


CommandFunction = Callable[[float], np.ndarray]


@dataclass
class LinearDynamicalSystem:
    """State-space system implementing ``dot(x) = A x + c(t)``.

    The system is deterministic given ``A``, ``x0``, and command function ``c_fn``.
    """

    A: np.ndarray
    c_fn: CommandFunction
    x0: np.ndarray

    def __post_init__(self) -> None:
        self.A = ensure_numpy_array(self.A, ndim=2, name="A")
        self.x0 = ensure_numpy_array(self.x0, ndim=1, name="x0")
        if self.A.shape[0] != self.A.shape[1]:
            raise ValueError(f"A must be square, got shape {self.A.shape}.")
        if self.A.shape[0] != self.x0.size:
            raise ValueError(
                "A and x0 dimensionality mismatch: "
                f"A is {self.A.shape}, x0 has length {self.x0.size}."
            )
        self._x = self.x0.copy()

    @property
    def dim(self) -> int:
        """State dimensionality J."""
        return self.x0.size

    @property
    def x(self) -> np.ndarray:
        """Current state vector."""
        return self._x

    def reset(self, x0: np.ndarray | None = None) -> np.ndarray:
        """Reset state to provided ``x0`` or the initial value from construction."""
        if x0 is None:
            self._x = self.x0.copy()
        else:
            x0_arr = ensure_numpy_array(x0, ndim=1, name="x0")
            if x0_arr.size != self.dim:
                raise ValueError(f"x0 has length {x0_arr.size}, expected {self.dim}.")
            self._x = x0_arr.copy()
        return self._x

    def command(self, t: float) -> np.ndarray:
        """Return command vector ``c(t)``."""
        c_t = ensure_numpy_array(self.c_fn(t), ndim=1, name="c(t)")
        if c_t.size != self.dim:
            raise ValueError(f"c(t) has length {c_t.size}, expected {self.dim}.")
        return c_t

    def step(self, t: float, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """Advance system by one Euler step and return ``(x, c_t)``.

        Args:
            t: Current simulation time in seconds.
            dt: Step size in seconds.
        """
        if dt <= 0.0:
            raise ValueError("dt must be strictly positive.")
        c_t = self.command(t)
        self._x = self._x + dt * (self.A @ self._x + c_t)
        return self._x, c_t

    def step_with_command(self, c_t: np.ndarray, dt: float) -> np.ndarray:
        """Advance system by one Euler step using an externally provided command."""
        if dt <= 0.0:
            raise ValueError("dt must be strictly positive.")
        c_t_arr = ensure_numpy_array(c_t, ndim=1, name="c_t")
        if c_t_arr.size != self.dim:
            raise ValueError(f"c_t has length {c_t_arr.size}, expected {self.dim}.")
        self._x = self._x + dt * (self.A @ self._x + c_t_arr)
        return self._x
