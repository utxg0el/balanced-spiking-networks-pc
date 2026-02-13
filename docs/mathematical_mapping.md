# Mathematical Mapping

This document maps implemented equations to Boerlin et al. (2013) notation.

## Target Dynamics

Implemented in `LinearDynamicalSystem`:

- `dot(x) = A x + c(t)`

## Decoder Dynamics

Implemented in `BalancedSpikingNetwork` state updates:

- `dot(x_hat) = -lambda_d x_hat + C o(t)`

Discrete event update when neuron `k` spikes:

- `x_hat <- x_hat + C[:, k]`

## Membrane Potential and Threshold

Model state variable and threshold:

- `V_i = C_i^T (x - x_hat) - m lambda_d r_i`
- `T_i = n lambda_d + (m lambda_d^2)/2 + ||C_i||^2/2`

Implementation uses the equivalent differential form (Eqn. 8 style):

- `dot(V) = -lambda_V V + Omega_s r + C^T c(t) + sigma_V xi(t)`
- Fast reset at spikes via `-Omega_f`

with

- `Omega_f = C^T C + m lambda_d^2 I`
- `Omega_s = C^T (A + lambda_d I) C`

## Spiking Rule

Greedy within-step rule:

1. Compute margins `V - T`.
2. Select the neuron with maximum margin.
3. If margin >= 0: emit spike and apply instantaneous updates.
4. Repeat until all margins are negative or `max_spikes_per_step` is reached.

## Noise Convention

`sigma_V` is treated as per-step membrane noise standard deviation, consistent with the paper caption convention where noise parameters correspond to injected noise at each `dt`.

## Numerical Integration

- Euler integration with configurable `dt` (default `0.1 ms` in experiments).
- Simulator warns if `max|Re(eig(A))| * dt` suggests potentially stiff dynamics.
