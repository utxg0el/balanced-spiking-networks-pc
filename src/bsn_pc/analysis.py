"""Analysis utilities for simulation quality and spike-train statistics."""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np


def rmse(x: np.ndarray, y: np.ndarray) -> float:
    """Root-mean-square error between two arrays."""
    diff = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    return float(np.sqrt(np.mean(diff**2)))


def per_dimension_correlation(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pearson correlation for each state dimension."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError(f"Shape mismatch: {x.shape} vs {y.shape}")
    if x.ndim != 2:
        raise ValueError("x and y must be 2D arrays of shape (T, J).")

    corrs = np.zeros(x.shape[1], dtype=float)
    for j in range(x.shape[1]):
        xj = x[:, j]
        yj = y[:, j]
        xstd = float(np.std(xj))
        ystd = float(np.std(yj))
        if xstd < 1e-12 or ystd < 1e-12:
            corrs[j] = np.nan
        else:
            corrs[j] = np.corrcoef(xj, yj)[0, 1]
    return corrs


def cv2_from_spike_times(spike_times: np.ndarray) -> float:
    """Compute CV2 from a spike-time vector for one neuron.

    CV2 = mean(2 |ISI_i+1 - ISI_i| / (ISI_i+1 + ISI_i)).
    """
    st = np.asarray(spike_times, dtype=float)
    if st.size < 3:
        return float("nan")
    isi = np.diff(st)
    if isi.size < 2:
        return float("nan")

    numer = 2.0 * np.abs(isi[1:] - isi[:-1])
    denom = isi[1:] + isi[:-1]
    mask = denom > 1e-12
    if not np.any(mask):
        return float("nan")
    return float(np.mean(numer[mask] / denom[mask]))


def cv2_per_neuron(
    spike_times: np.ndarray,
    spike_neurons: np.ndarray,
    N: int,
) -> np.ndarray:
    """Compute CV2 for each neuron independently."""
    out = np.full(N, np.nan, dtype=float)
    for neuron in range(N):
        st = spike_times[spike_neurons == neuron]
        out[neuron] = cv2_from_spike_times(st)
    return out


def firing_rates_hz(
    spike_neurons: np.ndarray,
    N: int,
    duration: float,
) -> np.ndarray:
    """Return per-neuron firing rates in Hz."""
    counts = np.bincount(spike_neurons.astype(int), minlength=N)
    if duration <= 0:
        raise ValueError("duration must be positive.")
    return counts / duration


def exponential_population_rate(
    t: np.ndarray,
    spike_times: np.ndarray,
    spike_neurons: np.ndarray,
    neuron_group: np.ndarray,
    tau: float,
) -> np.ndarray:
    """Exponential moving average firing-rate estimate for a neuron group.

    Uses a simple recursive discretization of a spike-count low-pass filter and returns
    average Hz per neuron in the provided group.
    """
    t = np.asarray(t, dtype=float)
    if t.ndim != 1 or t.size < 2:
        raise ValueError("t must be 1D with at least 2 time points.")
    if tau <= 0.0:
        raise ValueError("tau must be positive.")

    dt = float(t[1] - t[0])
    alpha = np.exp(-dt / tau)

    group = np.asarray(neuron_group, dtype=int)
    if group.size == 0:
        return np.zeros_like(t)
    group_mask = np.zeros(int(np.max(np.append(spike_neurons, group))) + 1, dtype=bool)
    group_mask[group] = True

    out = np.zeros_like(t)
    state = 0.0

    # Bin spikes by nearest lower index.
    idx = np.searchsorted(t, spike_times, side="right") - 1
    valid = (idx >= 0) & (idx < t.size)
    idx = idx[valid]
    sn = spike_neurons[valid].astype(int)
    in_group = group_mask[sn]

    spike_counts = np.zeros(t.size, dtype=float)
    np.add.at(spike_counts, idx[in_group], 1.0)

    for k in range(t.size):
        state = alpha * state + spike_counts[k]
        out[k] = state / tau / float(group.size)
    return out


def basic_summary(
    *,
    x: np.ndarray,
    x_hat: np.ndarray,
    spike_times: np.ndarray,
    spike_neurons: np.ndarray,
    N: int,
    duration: float,
) -> Dict[str, float]:
    """Compute default summary metrics for a simulation run."""
    rates = firing_rates_hz(spike_neurons, N=N, duration=duration)
    cv2 = cv2_per_neuron(spike_times, spike_neurons, N=N)
    corr = per_dimension_correlation(x, x_hat)
    valid_cv2 = cv2[~np.isnan(cv2)]
    valid_corr = corr[~np.isnan(corr)]
    mean_cv2 = float(np.mean(valid_cv2)) if valid_cv2.size else float("nan")
    median_cv2 = float(np.median(valid_cv2)) if valid_cv2.size else float("nan")
    mean_corr = float(np.mean(valid_corr)) if valid_corr.size else float("nan")

    summary = {
        "rmse": rmse(x, x_hat),
        "mean_rate_hz": float(np.mean(rates)),
        "median_rate_hz": float(np.median(rates)),
        "mean_cv2": mean_cv2,
        "median_cv2": median_cv2,
        "num_spikes": float(spike_times.size),
        "mean_corr": mean_corr,
    }
    return summary
