# Experiments

## Figure 3

`Figure3Experiment` reproduces a Figure-3-style inhomogeneous integrator scenario.

- Inhomogeneous sparse signed kernels (`N=400, J=30` by default)
- Deterministic network (`sigma_V = 0`)
- Shared sensory noise before cutoff (`sigma_s = 0.01`)
- Two runs with identical inputs and initial conditions:
  - Baseline run
  - Perturbed run with one delayed spike (`~1 ms`) around `t ~ 1.65 s`

Outputs:

- `outputs/figure3/<timestamp>/baseline/*`
- `outputs/figure3/<timestamp>/perturbed/*`
- `outputs/figure3/<timestamp>/figures/figure3.(png|svg)`

## Figure 6

`Figure6Experiment` runs two 2D systems with shared network hyperparameters.

### A) Leaky Differentiator

- `A = [[-400, -800], [50, 0]]`
- `c2(t) = 0`

### B) Damped Oscillator

- `A = [[-4.8, -22.4], [40, 0]]`
- `c2(t) = 0`
- `c1(t)` provides an initial kick

Outputs:

- `outputs/figure6/<timestamp>/differentiator/*`
- `outputs/figure6/<timestamp>/oscillator/*`
- `outputs/figure6/<timestamp>/figures/figure6.(png|svg)`

## CLI

```bash
bsn-pc run figure3 --output-dir outputs
bsn-pc run figure6 --output-dir outputs
bsn-pc run figure3 --output-dir outputs --quick
```

## Sweep Example

```yaml
experiment: figure6
repeats: 1
parameters:
  sigma_V: [0.0005, 0.001, 0.002]
  m: [1e-6, 5e-6]
```

```bash
bsn-pc sweep --config sweep.yaml --output-dir outputs/sweeps --quick
```
