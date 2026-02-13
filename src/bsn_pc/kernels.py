"""Decoding-kernel generation utilities for balanced spiking networks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class InhomogeneousKernelSpec:
    """Sampling spec used in the inhomogeneous integrator network (Fig. 3 style)."""

    p_active: float = 0.7
    positive_range: Tuple[float, float] = (0.06, 0.1)
    negative_range: Tuple[float, float] = (-0.1, -0.06)


def homogeneous_signed_kernels(J: int, N: int, magnitude: float = 0.1) -> np.ndarray:
    """Generate a homogeneous matrix with half positive and half negative kernels."""
    if J <= 0 or N <= 0:
        raise ValueError("J and N must be positive integers.")
    if magnitude <= 0.0:
        raise ValueError("magnitude must be positive.")

    C = np.full((J, N), magnitude, dtype=float)
    C[:, N // 2 :] = -magnitude
    return C


def inhomogeneous_sparse_signed_kernels(
    J: int,
    N: int,
    *,
    rng: np.random.Generator,
    spec: InhomogeneousKernelSpec | None = None,
) -> np.ndarray:
    """Generate sparse random kernels matching the paper's inhomogeneous recipe.

    The first half of neurons are sampled from positive support, the second half from
    negative support, each masked by an independent Bernoulli ``B(1, p_active)``.
    """
    if J <= 0 or N <= 0:
        raise ValueError("J and N must be positive integers.")
    spec = spec or InhomogeneousKernelSpec()
    if not (0.0 < spec.p_active <= 1.0):
        raise ValueError("p_active must lie in (0, 1].")

    C = np.zeros((J, N), dtype=float)
    half = N // 2

    pos_low, pos_high = spec.positive_range
    neg_low, neg_high = spec.negative_range
    if not (pos_low < pos_high):
        raise ValueError("positive_range must be ordered.")
    if not (neg_low < neg_high):
        raise ValueError("negative_range must be ordered.")

    mask_pos = rng.binomial(1, spec.p_active, size=(J, half))
    values_pos = rng.uniform(pos_low, pos_high, size=(J, half))
    C[:, :half] = mask_pos * values_pos

    neg_count = N - half
    mask_neg = rng.binomial(1, spec.p_active, size=(J, neg_count))
    values_neg = rng.uniform(neg_low, neg_high, size=(J, neg_count))
    C[:, half:] = mask_neg * values_neg
    return C


def normalized_gaussian_kernels(
    J: int,
    N: int,
    *,
    target_norm: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate kernels by Gaussian draw and normalize each neuron kernel norm."""
    if J <= 0 or N <= 0:
        raise ValueError("J and N must be positive integers.")
    if target_norm <= 0.0:
        raise ValueError("target_norm must be positive.")

    C = rng.normal(0.0, 1.0, size=(J, N))
    norms = np.linalg.norm(C, axis=0)
    zero_mask = norms <= 1e-15
    if np.any(zero_mask):
        C[:, zero_mask] = 1.0
        norms = np.linalg.norm(C, axis=0)
    C = C * (target_norm / norms)
    return C


def split_neuron_groups_by_mean_sign(C: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split neurons into positive/negative groups using mean kernel sign."""
    means = np.mean(C, axis=0)
    pos = np.where(means >= 0.0)[0]
    neg = np.where(means < 0.0)[0]
    return pos, neg
