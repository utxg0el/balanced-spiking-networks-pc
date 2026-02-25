# Codebase Explainer: Balanced Spiking Predictive-Coding Networks (Boerlin et al., 2013)

This repository is a from-scratch implementation of the classic model from Boerlin, Machens, and Deneve (2013). The goal is:

- You define a *target* linear dynamical system that evolves a state vector `x(t)` via `dot(x) = A x + c(t)`.
- A spiking neural network produces spikes `o(t)`.
- A simple linear decoder reads those spikes out into an estimate `x_hat(t)`.
- The network uses recurrent connectivity derived from `(A, C)` so that it spikes only when it improves the representation (a predictive-coding / error-correcting spiking code).

This document explains:

- The math and the shapes of every variable.
- The class structure (what each file does).
- The simulation loop (what happens at each `dt`).
- How the Figure 3 and Figure 6 experiments are assembled.
- How to run, change, and extend the code.

## 0) Quick Orientation (Reading Order)

If you want to understand the code with the least context switching:

1. `src/bsn_pc/types.py` (configs and results)
2. `src/bsn_pc/system.py` (the target system `dot(x)=Ax+c(t)`)
3. `src/bsn_pc/model.py` (the spiking network model)
4. `src/bsn_pc/simulator.py` (the time loop that runs system+network)
5. `src/bsn_pc/kernels.py` and `src/bsn_pc/inputs.py` (how `C` and `c(t)` are generated)
6. `src/bsn_pc/experiments/figure3.py` and `src/bsn_pc/experiments/figure6.py` (how the paper-like runs are assembled)
7. `src/bsn_pc/plotting/figure3.py` and `src/bsn_pc/plotting/figure6.py` (how figures are produced)
8. `src/bsn_pc/cli.py` (how everything is exposed as commands)

## 1) Variable Glossary (with Shapes)

Throughout the code, we follow the paper's symbol names as closely as practical.

Core dimensions:

- `J`: dimension of the dynamical variable (size of the state `x`).
- `N`: number of neurons.

Core matrices/vectors:

- `A` shape `(J, J)`
  - State transition matrix of the target system.
- `c(t)` shape `(J,)`
  - Time-varying command/input signal.
- `x(t)` shape `(J,)`
  - True state of the target system.
- `C` shape `(J, N)`
  - Decoder / output kernel matrix. Column `C[:, i]` is neuron `i`'s kernel (called `C_i` in the paper).
- `o(t)` shape `(N,)` conceptually
  - Spike train vector. In code we store spikes as a list of `(time, neuron_id)` events.

Network internal state:

- `x_hat(t)` shape `(J,)`
  - Readout estimate of `x`, produced by filtering spikes through `C`.
- `r(t)` shape `(N,)`
  - A filtered version of spikes (think: each neuron has a low-pass filtered spike count).
- `V(t)` shape `(N,)`
  - Membrane potential vector.
- `T` shape `(N,)`
  - Per-neuron thresholds.

Derived recurrent connectivity:

- `Omega_f` shape `(N, N)`
  - "Fast" lateral matrix (instantaneous reset/competition term).
- `Omega_s` shape `(N, N)`
  - "Slow" lateral matrix (predictive/cooperative term).

Rates / time constants:

- `lambda_d` (Hz)
  - Decoder decay rate and also the decay for filtered rates `r`.
- `lambda_V` (Hz)
  - Membrane leak rate.

Cost terms:

- `n` (>= 0)
  - Linear spike cost (shifts thresholds).
- `m` (>= 0)
  - Quadratic spike cost (affects thresholds and `Omega_f`).

Noise:

- `sigma_V` (>= 0)
  - Membrane noise scale. Implemented as Euler-Maruyama: `sigma_V * sqrt(dt) * Normal(0,1)`.

## 2) The Math Implemented

### 2.1 Target system (ground truth)

Implemented in `src/bsn_pc/system.py` as `LinearDynamicalSystem`.

- `dot(x) = A x + c(t)`

The simulator uses a simple Euler step:

- `x <- x + dt * (A x + c_t)`

### 2.2 Decoder / estimate

The network produces an estimate `x_hat(t)` that is a leaky integration of spikes:

- `dot(x_hat) = -lambda_d * x_hat + C o(t)`

Discrete-time implementation splits into:

1. Continuous decay each step:
   - `x_hat <- x_hat + dt * (-lambda_d * x_hat)`
2. Instantaneous update for each spike from neuron `k`:
   - `x_hat <- x_hat + C[:, k]`

This is implemented in `src/bsn_pc/model.py`.

### 2.3 Filtered rates `r`

Similarly:

- `dot(r) = -lambda_d * r + o(t)`

Implemented as:

- `r <- r + dt * (-lambda_d * r)`
- spike from neuron `k`: `r[k] += 1`

### 2.4 Membrane potentials and spiking rule

The key idea of the paper: each neuron monitors a projection of the *prediction error* and spikes when it helps.

In the derivation, you get:

- Thresholds:
  - `T_i = n * lambda_d + (m * lambda_d^2)/2 + ||C_i||^2 / 2`

- Connectivity matrices:
  - `Omega_f = C^T C + m * lambda_d^2 * I`
  - `Omega_s = C^T (A + lambda_d * I) C`

- Membrane dynamics (informal):
  - `dot(V) = -lambda_V V + Omega_s r + C^T c(t) + sigma_V * white_noise - Omega_f o(t)`

In the code, this is implemented as:

1. Continuous Euler step (no spikes yet):
   - `drive = -lambda_V * V + Omega_s @ r + C^T @ c_t`
   - `V <- V + dt * drive`
   - optional noise: `V <- V + sigma_V * sqrt(dt) * Normal(0,1)`
2. Greedy spiking loop *within the same dt*:
   - compute `margin = V - T`
   - pick neuron with largest margin
   - if max margin < 0: stop
   - else emit spike and apply instantaneous reset:
     - `V <- V - Omega_f[:, k]`
     - `r[k] += 1`
     - `x_hat += C[:, k]`

All of that is in `src/bsn_pc/model.py`.

## 3) Architecture: Classes and Responsibilities

### 3.1 Shared types and configs

File: `src/bsn_pc/types.py`

Key dataclasses:

- `SimulationConfig`
  - Time span (`t_start`, `t_stop`), step size `dt`, `max_spikes_per_step` safety cap.
  - Recording toggles for `V` and `r`.
- `RunMetadata`
  - Timestamp, runtime seconds, git commit hash, etc.
- `SimulationResult`
  - Stores time series arrays, spike event arrays, summary metrics.
- `SpikeDelaySpec`
  - A small protocol used by Figure 3: delay one spike by 1ms after some trigger time.

### 3.2 Target dynamical system

File: `src/bsn_pc/system.py`

Class: `LinearDynamicalSystem`

- Holds `A`, a command function `c_fn(t)`, and initial state `x0`.
- Provides:
  - `command(t)` -> `c(t)`
  - `step(t, dt)` -> Euler step using internally computed `c(t)`
  - `step_with_command(c_t, dt)` -> Euler step using a provided `c_t` (this is what the simulator uses)

### 3.3 Spiking network model

File: `src/bsn_pc/model.py`

Class: `BalancedSpikingNetwork`

State:

- `x_hat` (J)
- `r` (N)
- `V` (N)
- `T` (N)

Derived matrices:

- `Omega_f`, `Omega_s`

Main method:

- `step(c_t, dt, t, max_spikes_per_step)`

Spike recording:

- Internally stores spike times and spike neuron ids.
- `spike_history` returns those as arrays.

#### Spike delay perturbation (Figure 3)

The paper's Figure 3 includes a chaos-style demonstration: two identical runs, then a single spike is delayed by 1ms in one run, after which the spike sequence diverges.

Mechanism in code:

- `inject_spike_delay(SpikeDelaySpec(...))` activates the protocol.
- When the first eligible spike occurs after `trigger_time`, instead of applying it immediately, the network schedules it for `t + delay` and suppresses that neuron's spikes until that time.
- At each step start, `_apply_due_delayed_spikes()` injects the spike if due.

This is intentionally a minimal perturbation mechanism, not a general spike-time editing framework.

### 3.4 The simulation loop

File: `src/bsn_pc/simulator.py`

Class: `SimulationRunner`

What it does:

1. Build the time grid `t`.
2. Precompute commands `c_values[t_k]` (either passed in or from the system's `command()` function).
3. Allocate arrays:
   - `x[t_k, :]`, `x_hat[t_k, :]`, optionally `V[t_k, :]` and `r[t_k, :]`.
4. Reset the system and network.
5. For each time step `k`:
   - Feed `c_t` to the network.
   - Step the system using the *same* `c_t`.
   - Record everything.
6. After the loop, pull spike events from the network and compute summary metrics.

Artifact saving:

- `save_simulation_result(...)` writes:
  - `config.yaml`
  - `summary.json`
  - `timeseries.npz`
  - `metrics.csv`

## 4) How `C` (kernels) are generated

File: `src/bsn_pc/kernels.py`

You should think of `C[:, i]` as "what a spike from neuron i *means* in state space".

Functions:

- `inhomogeneous_sparse_signed_kernels(J, N, rng, spec)`
  - Figure 3 style.
  - First half neurons: sparse positive entries.
  - Second half neurons: sparse negative entries.
  - Sparsity mask is Bernoulli with probability `p_active=0.7`.
  - Nonzero magnitudes are uniform in `(0.06, 0.1)` or `(-0.1, -0.06)`.

- `normalized_gaussian_kernels(J, N, target_norm, rng)`
  - Figure 6 style.
  - Sample each `C[:, i]` from Normal(0,1) and then scale so `||C[:, i]|| = target_norm`.

- `split_neuron_groups_by_mean_sign(C)`
  - Returns indices split by mean sign, used for "positive kernel" vs "negative kernel" population rates.

## 5) How inputs `c(t)` are generated

File: `src/bsn_pc/inputs.py`

Key helpers:

- `piecewise_constant_1d(t, breakpoints, values)`
  - Makes a stepwise-constant scalar signal.

Figure-specific generators:

- `figure3_sensory_command(t, J, sigma_s, cutoff_time, rng)`
  - Builds a scalar base signal with breakpoints defined as fractions of the simulation duration.
  - Adds Gaussian sensory noise (only before `cutoff_time`).
  - Replicates the scalar across `J` dimensions to get `c(t)` of shape `(T, J)`.

- `figure6a_command(t)`
  - The differentiator's `c1(t)` is stepwise constant with relatively large amplitudes.
  - `c2(t) = 0`.

- `figure6b_command(t)`
  - The oscillator's `c1(t)` provides a short kick + a small Gaussian bump.
  - `c2(t) = 0`.

## 6) Experiments: assembling paper-like runs

### 6.1 Figure 3 experiment

File: `src/bsn_pc/experiments/figure3.py`

What it builds:

- `N=400`, `J=30` (full) or smaller (quick).
- `A = -lambda_s * I` where `lambda_s=0` means perfect integrator.
- Two simulations:
  - baseline
  - perturbed (delays exactly one spike by `perturb_delay` after `perturb_time`)

What it saves:

- `outputs/figure3/<timestamp>/baseline/*`
- `outputs/figure3/<timestamp>/perturbed/*`
- `outputs/figure3/<timestamp>/figures/figure3.png` and `.svg`

### 6.2 Figure 6 experiment

File: `src/bsn_pc/experiments/figure6.py`

Two systems:

- Differentiator:
  - `A = [[-400, -800], [50, 0]]`
  - `c(t)` from `figure6a_command()`

- Oscillator:
  - `A = [[-4.8, -22.4], [40, 0]]`
  - `c(t)` from `figure6b_command()`

Shared network hyperparameters:

- `N=100` full, `N=40` quick
- `||C_i|| = 0.03`
- `lambda_d=10`, `lambda_V=20`, `m=1e-6`, `n=0`
- `sigma_V=1e-3` (note: noise is correctly scaled by `sqrt(dt)` in the model)

What it saves:

- `outputs/figure6/<timestamp>/differentiator/*`
- `outputs/figure6/<timestamp>/oscillator/*`
- `outputs/figure6/<timestamp>/figures/figure6.png` and `.svg`

## 7) Plotting

Files:

- `src/bsn_pc/plotting/figure3.py`
- `src/bsn_pc/plotting/figure6.py`
- `src/bsn_pc/plotting/common.py`

Important implementation detail:

- These modules force the Matplotlib backend to `Agg` so plots work in headless environments.

Figure 3 plot layout:

- Top: sensory input signal.
- Middle: raster plot overlaying baseline spikes (dots) and perturbed spikes (circles).
- Bottom: exponential-window population firing rate (positive vs negative kernel groups).

Figure 6 plot layout:

- 2x2 grid:
  - Top row: command `c1(t)` for differentiator and oscillator.
  - Bottom row: true `x` and estimate `x_hat` for both dimensions.
  - Raster is drawn on a twinx axis on the bottom plots.

## 8) Metrics and summaries

File: `src/bsn_pc/analysis.py`

- `rmse(x, x_hat)`
- `per_dimension_correlation(x, x_hat)`
- `cv2_*` spike-train irregularity statistics
- `exponential_population_rate(...)` used for Figure 3 population firing rates
- `basic_summary(...)` produces the default `result.summary` dict

## 9) CLI

File: `src/bsn_pc/cli.py`

Commands:

- Run experiments:
  - `bsn-pc run figure3 --output-dir outputs`
  - `bsn-pc run figure6 --output-dir outputs`
  - add `--quick` for smaller faster runs

- Run sweeps:
  - `bsn-pc sweep --config sweep.yaml --output-dir outputs/sweeps --quick`

- Regenerate figures from a run directory:
  - `bsn-pc plot --run-dir outputs/figure3/<timestamp>`

Under the hood:

- The CLI mostly instantiates `Figure3Experiment` / `Figure6Experiment`.
- `plot` loads `timeseries.npz` via a small helper and re-runs the figure builder.

## 10) How to Run Locally (Installation vs. PYTHONPATH)

Two common ways:

1. Editable install (recommended):

```bash
pip install -e .
bsn-pc run figure3 --output-dir outputs
```

2. Without installing (for quick hacking):

```bash
PYTHONPATH=src python3 -m bsn_pc run figure3 --output-dir outputs
```

## 11) How to Modify / Extend

### Change hyperparameters

Look at:

- `Figure3Config` in `src/bsn_pc/experiments/figure3.py`
- `Figure6Config` in `src/bsn_pc/experiments/figure6.py`

You can also directly create your own experiment script that:

- generates `A`, `C`, `command_values`
- builds `LinearDynamicalSystem` + `BalancedSpikingNetwork`
- runs `SimulationRunner(...).run(...)`

### Add a new dynamical system

1. Define your `A` and a `c(t)` generator in `src/bsn_pc/inputs.py` (or in a new module).
2. Create a new experiment module under `src/bsn_pc/experiments/`.
3. Add a new plot module under `src/bsn_pc/plotting/`.
4. Optionally add a CLI hook in `src/bsn_pc/cli.py`.

### Add a sweep

- Provide a YAML file matching the schema described in `src/bsn_pc/sweeps.py`.
- Run `bsn-pc sweep ...`.

## 12) Practical Debugging Tips

- If you get huge firing rates, check:
  - `sigma_V` scaling (already `sqrt(dt)` scaled in `src/bsn_pc/model.py`).
  - `dt` (Euler stability).
  - command magnitudes in `src/bsn_pc/inputs.py`.

- If you get no spikes, check:
  - command magnitudes.
  - kernel norms `||C_i||` and threshold `T_i` (bigger kernels lower relative threshold crossing).

- If Euler seems unstable, reduce `dt`.
  - `SimulationRunner` warns when `max|Re(eig(A))| * dt` is large.

## 13) What to Do Next (Suggested Learning Path)

1. Run Figure 6 quick:
   - Understand how `A`, `C`, and `c(t)` affect tracking.
2. Open `src/bsn_pc/model.py` and step through `BalancedSpikingNetwork.step()`.
3. Modify `sigma_V` and see how spike timing changes.
4. Add your own `A` and `c(t)` and plot `x` vs `x_hat`.
