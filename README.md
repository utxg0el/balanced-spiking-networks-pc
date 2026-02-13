# balanced-spiking-networks-pc

Reimplementation of the balanced spiking predictive-coding network from:

- Boerlin, Machens, Denève (2013), *Predictive Coding of Dynamical Variables in Balanced Spiking Networks*.

The package implements the classic model using the paper's notation (`A, C, x, x_hat, o, r, V, T, lambda_d, lambda_V, m, n, sigma_V`), plus reproducible experiment pipelines for Figure 3 and Figure 6 style simulations.

## Features

- Object-oriented core model (`BalancedSpikingNetwork`, `LinearDynamicalSystem`)
- Euler simulation runner with within-step greedy spiking
- Figure 3 inhomogeneous integrator perturbation experiment (single-spike delay)
- Figure 6 leaky differentiator and damped oscillator experiments
- Artifact-oriented outputs per run:
  - `config.yaml`
  - `summary.json`
  - `timeseries.npz`
  - `metrics.csv`
  - `figures/*.png`, `figures/*.svg`
- Sweep framework for hyperparameter studies
- CLI and notebook workflows

## Installation

```bash
pip install -e .
```

## Quick Start

Run Figure 3 (full config):

```bash
bsn-pc run figure3 --output-dir outputs
```

Run Figure 6 in quick mode:

```bash
bsn-pc run figure6 --output-dir outputs --quick
```

Regenerate figure files from a prior run directory:

```bash
bsn-pc plot --run-dir outputs/figure3/<timestamp>
```

Run a sweep from YAML:

```bash
bsn-pc sweep --config sweep.yaml --output-dir outputs/sweeps --quick
```

## Project Structure

- `src/bsn_pc/model.py`: Core predictive-coding spiking network
- `src/bsn_pc/system.py`: Target linear dynamical systems
- `src/bsn_pc/simulator.py`: Simulation runner + artifact persistence
- `src/bsn_pc/experiments/`: Figure-specific experiments
- `src/bsn_pc/plotting/`: Figure builders
- `src/bsn_pc/sweeps.py`: Hyperparameter sweep runner
- `docs/`: Math mapping and extension docs
- `notebooks/`: Notebook workflows
- `tests/`: Unit/integration/regression smoke tests

## Notes

- The implementation intentionally does **not** enforce Dale's law in the base model, matching the main paper formulation.
- Figure reproductions are model-faithful and qualitative, not pixel-exact trace reconstructions.
