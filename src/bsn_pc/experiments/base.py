"""Experiment base classes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path


class Experiment(ABC):
    """Base class for reproducible experiment execution."""

    def __init__(self, *, output_root: Path) -> None:
        self.output_root = output_root

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique experiment name used in artifacts and CLI."""

    def make_run_dir(self) -> Path:
        """Create a timestamped run directory."""
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        run_dir = self.output_root / self.name / ts
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    @abstractmethod
    def run(self, *, quick: bool = False):
        """Execute experiment and return run artifact info."""
