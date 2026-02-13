"""Balanced spiking predictive-coding network model.

This module implements the classic deterministic/stochastic LIF network derived in:
Boerlin, Machens, Denève (2013).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from bsn_pc.types import SpikeDelaySpec, ensure_numpy_array


@dataclass
class _PendingDelayedSpike:
    """Internal storage for scheduled delayed spike events."""

    due_time: float
    neuron: int


class BalancedSpikingNetwork:
    """Predictive-coding network with balanced fast/slow recurrent connections.

    Mathematical notation follows the paper:
    ``C, A, x, x_hat, o, r, V, T, lambda_d, lambda_V, m, n, sigma_V``.
    """

    def __init__(
        self,
        C: np.ndarray,
        A: np.ndarray,
        *,
        lambda_d: float,
        lambda_V: float,
        m: float,
        n: float,
        sigma_V: float,
        seed: int = 0,
    ) -> None:
        self.C = ensure_numpy_array(C, ndim=2, name="C")
        self.A = ensure_numpy_array(A, ndim=2, name="A")

        if self.A.shape[0] != self.A.shape[1]:
            raise ValueError(f"A must be square, got shape {self.A.shape}.")
        if self.C.shape[0] != self.A.shape[0]:
            raise ValueError(
                f"C has J={self.C.shape[0]} rows, but A has shape {self.A.shape}."
            )

        self.J, self.N = self.C.shape
        self.lambda_d = float(lambda_d)
        self.lambda_V = float(lambda_V)
        self.m = float(m)
        self.n = float(n)
        self.sigma_V = float(sigma_V)

        self._validate_parameters()

        eye_J = np.eye(self.J, dtype=float)
        eye_N = np.eye(self.N, dtype=float)

        # Eqns. (10)-(11) in paper notation.
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            self.Omega_f = self.C.T @ self.C + self.m * (self.lambda_d**2) * eye_N
            self.Omega_s = self.C.T @ (self.A + self.lambda_d * eye_J) @ self.C
        if not np.isfinite(self.Omega_f).all() or not np.isfinite(self.Omega_s).all():
            raise FloatingPointError(
                "Connectivity matrices contain non-finite values. "
                "Check kernel magnitudes and model parameters."
            )

        # Eqn. (7).
        self.T = (
            self.n * self.lambda_d
            + 0.5 * self.m * (self.lambda_d**2)
            + 0.5 * np.sum(self.C**2, axis=0)
        )

        self.r = np.zeros(self.N, dtype=float)
        self.V = np.zeros(self.N, dtype=float)
        self.x_hat = np.zeros(self.J, dtype=float)

        self._rng = np.random.default_rng(seed)
        self.seed = int(seed)

        self._spike_times: List[float] = []
        self._spike_neurons: List[int] = []

        self._delay_spec: Optional[SpikeDelaySpec] = None
        self._delay_eligibility_count: int = 0
        self._delay_applied_count: int = 0
        self._pending_delayed_spikes: List[_PendingDelayedSpike] = []
        self._suppressed_until = np.full(self.N, -np.inf, dtype=float)

    def _validate_parameters(self) -> None:
        if self.lambda_d <= 0.0:
            raise ValueError("lambda_d must be positive.")
        if self.lambda_V <= 0.0:
            raise ValueError("lambda_V must be positive.")
        if self.m < 0.0:
            raise ValueError("m must be non-negative.")
        if self.n < 0.0:
            raise ValueError("n must be non-negative.")
        if self.sigma_V < 0.0:
            raise ValueError("sigma_V must be non-negative.")

    @property
    def spike_history(self) -> tuple[np.ndarray, np.ndarray]:
        """Return spike times and neuron ids as arrays."""
        return np.asarray(self._spike_times), np.asarray(self._spike_neurons, dtype=int)

    def decode(self) -> np.ndarray:
        """Return current network estimate ``x_hat``."""
        return self.x_hat.copy()

    def reset(
        self,
        *,
        x_hat0: Optional[np.ndarray] = None,
        r0: Optional[np.ndarray] = None,
        V0: Optional[np.ndarray] = None,
        clear_spike_history: bool = True,
    ) -> None:
        """Reset network dynamical state."""
        self.x_hat = (
            np.zeros(self.J, dtype=float)
            if x_hat0 is None
            else ensure_numpy_array(x_hat0, ndim=1, name="x_hat0").copy()
        )
        self.r = (
            np.zeros(self.N, dtype=float)
            if r0 is None
            else ensure_numpy_array(r0, ndim=1, name="r0").copy()
        )
        self.V = (
            np.zeros(self.N, dtype=float)
            if V0 is None
            else ensure_numpy_array(V0, ndim=1, name="V0").copy()
        )

        if self.x_hat.size != self.J:
            raise ValueError(f"x_hat0 has length {self.x_hat.size}, expected {self.J}.")
        if self.r.size != self.N:
            raise ValueError(f"r0 has length {self.r.size}, expected {self.N}.")
        if self.V.size != self.N:
            raise ValueError(f"V0 has length {self.V.size}, expected {self.N}.")

        self._pending_delayed_spikes.clear()
        self._suppressed_until.fill(-np.inf)
        self._delay_eligibility_count = 0
        self._delay_applied_count = 0

        if clear_spike_history:
            self._spike_times.clear()
            self._spike_neurons.clear()

    def inject_spike_delay(self, spec: SpikeDelaySpec) -> None:
        """Activate a single-spike delay perturbation protocol.

        The network will delay the Nth eligible spike (``occurrence``) after
        ``trigger_time`` by ``delay`` seconds.
        """
        if spec.delay <= 0.0:
            raise ValueError("Spike delay must be strictly positive.")
        if spec.occurrence <= 0:
            raise ValueError("occurrence must be >= 1.")
        if spec.neuron_index is not None and not (0 <= spec.neuron_index < self.N):
            raise ValueError(f"neuron_index must be in [0, {self.N}), got {spec.neuron_index}.")

        self._delay_spec = spec
        self._delay_eligibility_count = 0
        self._delay_applied_count = 0
        self._pending_delayed_spikes.clear()
        self._suppressed_until.fill(-np.inf)

    def clear_spike_delay(self) -> None:
        """Disable the spike delay perturbation."""
        self._delay_spec = None
        self._delay_eligibility_count = 0
        self._delay_applied_count = 0
        self._pending_delayed_spikes.clear()
        self._suppressed_until.fill(-np.inf)

    def _record_spike(self, t: float, neuron: int) -> None:
        self._spike_times.append(float(t))
        self._spike_neurons.append(int(neuron))

    def _apply_spike(self, t: float, neuron: int, step_spikes: List[int]) -> None:
        """Apply instantaneous spike update: fast reset + updates to ``r`` and ``x_hat``."""
        self.r[neuron] += 1.0
        self.x_hat += self.C[:, neuron]
        self.V -= self.Omega_f[:, neuron]
        self._record_spike(t, neuron)
        step_spikes.append(neuron)

    def _maybe_schedule_delayed_spike(self, t: float, neuron: int) -> bool:
        """Check whether the current spike candidate should be delayed.

        Returns ``True`` when the spike was delayed and should not be applied now.
        """
        spec = self._delay_spec
        if spec is None:
            return False
        if self._delay_applied_count >= 1:
            return False
        if t < spec.trigger_time:
            return False
        if spec.neuron_index is not None and neuron != spec.neuron_index:
            return False

        self._delay_eligibility_count += 1
        if self._delay_eligibility_count != spec.occurrence:
            return False

        due = t + spec.delay
        self._pending_delayed_spikes.append(_PendingDelayedSpike(due_time=due, neuron=neuron))
        self._suppressed_until[neuron] = due
        self._delay_applied_count += 1
        return True

    def _apply_due_delayed_spikes(self, t: float, step_spikes: List[int]) -> None:
        """Inject delayed spikes whose due time has been reached."""
        if not self._pending_delayed_spikes:
            return

        still_pending: List[_PendingDelayedSpike] = []
        for pending in self._pending_delayed_spikes:
            if pending.due_time <= t + 1e-12:
                self._suppressed_until[pending.neuron] = -np.inf
                self._apply_spike(t, pending.neuron, step_spikes)
            else:
                still_pending.append(pending)
        self._pending_delayed_spikes = still_pending

    def step(
        self,
        *,
        c_t: np.ndarray,
        dt: float,
        t: float,
        max_spikes_per_step: int,
    ) -> np.ndarray:
        """Advance the network state by one Euler step.

        Args:
            c_t: Command vector ``c(t)`` with shape ``(J,)``.
            dt: Integration step size in seconds.
            t: Current simulation time.
            max_spikes_per_step: Safety cap for runaway spike cascades.

        Returns:
            Array of neuron indices that spiked at this step.
        """
        if dt <= 0.0:
            raise ValueError("dt must be strictly positive.")
        if max_spikes_per_step <= 0:
            raise ValueError("max_spikes_per_step must be positive.")

        c_t_arr = ensure_numpy_array(c_t, ndim=1, name="c_t")
        if c_t_arr.size != self.J:
            raise ValueError(f"c_t has length {c_t_arr.size}, expected {self.J}.")

        step_spikes: List[int] = []

        # Delayed spikes are injected before continuous dynamics for this step.
        self._apply_due_delayed_spikes(t, step_spikes)

        # Continuous dynamics (Euler) for filtered rates and decoder state.
        self.r += dt * (-self.lambda_d * self.r)
        self.x_hat += dt * (-self.lambda_d * self.x_hat)

        # Continuous membrane dynamics (Eqn. 8 without delta term).
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            drive = -self.lambda_V * self.V + self.Omega_s @ self.r + self.C.T @ c_t_arr
        if not np.isfinite(drive).all():
            raise FloatingPointError(
                "Non-finite membrane drive encountered. "
                "Consider reducing dt or adjusting network parameters."
            )
        self.V += dt * drive
        if self.sigma_V > 0.0:
            # Euler-Maruyama increment for white-noise drive.
            self.V += self.sigma_V * np.sqrt(dt) * self._rng.normal(size=self.N)

        # Greedy within-step spike loop.
        for _ in range(max_spikes_per_step):
            margins = self.V - self.T
            suppressed = t < self._suppressed_until
            if np.any(suppressed):
                margins = margins.copy()
                margins[suppressed] = -np.inf

            neuron = int(np.argmax(margins))
            if margins[neuron] < 0.0:
                break

            if self._maybe_schedule_delayed_spike(t, neuron):
                # Skip emission now; neuron is held silent until delayed release.
                continue

            self._apply_spike(t, neuron, step_spikes)

        return np.asarray(step_spikes, dtype=int)
