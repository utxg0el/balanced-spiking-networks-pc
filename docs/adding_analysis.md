# Adding New Analyses

## Principles

- Keep model code and analysis code decoupled.
- Store every run as immutable artifacts (`config`, `summary`, `timeseries`, `metrics`, figures).
- Add metrics in `analysis.py` and expose them in experiment summaries.

## Workflow

1. Add analysis function in `src/bsn_pc/analysis.py`.
2. Consume it in an experiment module (`figure3.py`, `figure6.py`, or new module).
3. Add plot utility in `src/bsn_pc/plotting/`.
4. Save generated figures in the run's `figures/` directory.
5. Add tests in `tests/` for metric correctness and expected ranges.

## Suggested Extensions

- Error-vs-parameter sweep dashboards
- Robustness analyses under spike dropout and noise perturbations
- Alternative command generators and task families
- Additional statistics (Fano factor, cross-correlograms, GLM-inspired filters)

## Reproducibility Checklist

- Include seed in config
- Include git commit in metadata
- Keep command traces serialized in `timeseries.npz`
- Store summaries in machine-readable JSON and CSV
