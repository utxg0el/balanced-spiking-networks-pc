"""Unit tests for core model math and spike rule invariants."""

from __future__ import annotations

import numpy as np

from bsn_pc.model import BalancedSpikingNetwork


def _penalized_energy(e: np.ndarray, r: np.ndarray, C: np.ndarray, lambda_d: float, m: float, n: float) -> float:
    """Instantaneous surrogate from paper derivation (up to constants)."""
    return float(
        np.sum(e**2)
        + 2.0 * n * lambda_d * np.sum(r)
        + m * (lambda_d**2) * np.sum(r**2)
    )


def test_threshold_and_connectivity_matrices_match_formula() -> None:
    C = np.array([[0.1, -0.2, 0.05], [0.03, 0.04, -0.01]], dtype=float)
    A = np.array([[-2.0, 0.0], [0.0, -1.0]], dtype=float)

    lambda_d = 10.0
    m = 1e-6
    n = 1e-5

    net = BalancedSpikingNetwork(
        C=C,
        A=A,
        lambda_d=lambda_d,
        lambda_V=20.0,
        m=m,
        n=n,
        sigma_V=0.0,
        seed=0,
    )

    expected_Omega_f = C.T @ C + m * (lambda_d**2) * np.eye(C.shape[1])
    expected_Omega_s = C.T @ (A + lambda_d * np.eye(C.shape[0])) @ C
    expected_T = n * lambda_d + 0.5 * m * (lambda_d**2) + 0.5 * np.sum(C**2, axis=0)

    assert np.allclose(net.Omega_f, expected_Omega_f)
    assert np.allclose(net.Omega_s, expected_Omega_s)
    assert np.allclose(net.T, expected_T)


def test_spike_rule_decreases_penalized_surrogate_when_margin_positive() -> None:
    C = np.array([[0.12, -0.08, 0.03], [0.01, 0.07, -0.05]], dtype=float)
    A = np.array([[-0.5, 0.0], [0.0, -0.5]], dtype=float)

    lambda_d = 10.0
    m = 1e-6
    n = 1e-5

    net = BalancedSpikingNetwork(
        C=C,
        A=A,
        lambda_d=lambda_d,
        lambda_V=20.0,
        m=m,
        n=n,
        sigma_V=0.0,
        seed=1,
    )

    # Construct an error vector that strongly favors neuron k.
    e = np.array([0.9, 0.15], dtype=float)
    r = np.array([0.0, 0.0, 0.0], dtype=float)
    V = C.T @ e - m * lambda_d * r
    margins = V - net.T
    k = int(np.argmax(margins))

    assert margins[k] > 0.0

    E_before = _penalized_energy(e, r, C, lambda_d, m, n)

    e_after = e - C[:, k]
    r_after = r.copy()
    r_after[k] += 1.0
    E_after = _penalized_energy(e_after, r_after, C, lambda_d, m, n)

    assert E_after < E_before
