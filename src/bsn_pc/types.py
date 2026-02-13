"""Type definitions and shared dataclasses for simulations and experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np


@dataclass(frozen=True)
class SpikeDelaySpec:
    """Specification for delaying a single spike event.

    Attributes:
        trigger_time: Time (seconds) after which the first eligible spike can be delayed.
        delay: Delay amount (seconds) applied to the selected spike.
        neuron_index: Optional neuron index to target. If ``None``, any neuron is eligible.
        occurrence: Delay the Nth eligible spike (1-indexed) after ``trigger_time``.
    """

    trigger_time: float
    delay: float
    neuron_index: Optional[int] = None
    occurrence: int = 1


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration for simulating a network/system pair."""

    name: str
    t_start: float
    t_stop: float
    dt: float
    seed: int = 0
    max_spikes_per_step: int = 1024
    record_voltage: bool = True
    record_rate: bool = True
    record_command: bool = True

    def validate(self) -> None:
        """Validate scalar ranges used by the simulator."""
        if self.t_stop <= self.t_start:
            raise ValueError("t_stop must be larger than t_start.")
        if self.dt <= 0.0:
            raise ValueError("dt must be strictly positive.")
        if self.max_spikes_per_step <= 0:
            raise ValueError("max_spikes_per_step must be positive.")

    @property
    def num_steps(self) -> int:
        """Number of fixed-step simulation points."""
        return int(np.floor((self.t_stop - self.t_start) / self.dt)) + 1


@dataclass
class RunMetadata:
    """Run-level metadata for reproducibility and provenance."""

    experiment_name: str
    seed: int
    created_at_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    runtime_seconds: float = 0.0
    git_commit: str = "unknown"
    package_version: str = "0.1.0"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize metadata to plain Python objects."""
        return asdict(self)


@dataclass
class SimulationResult:
    """Container for all simulation artifacts produced by a run."""

    config: SimulationConfig
    metadata: RunMetadata
    t: np.ndarray
    x: np.ndarray
    x_hat: np.ndarray
    c: np.ndarray
    spike_times: np.ndarray
    spike_neurons: np.ndarray
    V: Optional[np.ndarray] = None
    r: Optional[np.ndarray] = None
    summary: Dict[str, float] = field(default_factory=dict)
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_summary_dict(self) -> Dict[str, Any]:
        """Build a dictionary suitable for summary JSON export."""
        payload: Dict[str, Any] = {
            "config": asdict(self.config),
            "metadata": self.metadata.to_dict(),
            "summary": dict(self.summary),
            "num_spikes": int(self.spike_times.size),
            "num_steps": int(self.t.size),
        }
        return payload


@dataclass
class SweepConfig:
    """Configuration for hyperparameter sweeps."""

    name: str
    output_dir: Path
    parameters: Mapping[str, list]
    repeats: int = 1


@dataclass
class ExperimentArtifacts:
    """Paths of artifacts produced by an experiment execution."""

    run_dir: Path
    config_path: Path
    summary_path: Path
    timeseries_path: Path
    metrics_path: Path
    figures: Dict[str, Path]


def ensure_numpy_array(x: np.ndarray, *, ndim: Optional[int] = None, name: str = "array") -> np.ndarray:
    """Return ``x`` as float array and optionally validate dimensionality."""
    arr = np.asarray(x, dtype=float)
    if ndim is not None and arr.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}D, got shape {arr.shape}.")
    return arr
