# sparsetune 0.1.0

The first release of `sparsetune` provides a reproducible way to inspect sparse
matrices, compare native SciPy and CuPy conjugate-gradient solvers in isolated
workers, and reuse the selected backend through a tuning profile.

## Highlights

- Canonical Matrix Market loading, stable fingerprints, and conservative SPD
  screening.
- Native SciPy CPU and optional CuPy CUDA backends with independent CPU residual
  validation.
- Structured `timeout`, `oom`, `process_crash`, `unsupported`, and numerical
  solver outcomes that do not terminate the parent process.
- Separate end-to-end and steady-state measurements, median timing, recommendation
  reasons, speedup, and GPU break-even estimates.
- `inspect`, `bench`, `tune`, `solve`, and `doctor` CLI commands plus the Python
  API.
- Versioned profile validation for matrix, solver configuration, environment,
  and GPU identity.
- Python 3.10-3.14 CPU CI, 80% coverage gate, wheel smoke tests, and Trusted
  Publishing workflows.

## Validation

Release-candidate validation used three public SuiteSparse stiffness matrices
from 48 to 11,948 rows. All converged with independently checked relative
residuals below `1e-6`; operational timeout and unavailable-GPU paths remained
structured. See [docs/VALIDATION.md](docs/VALIDATION.md) for sources, checksums,
environment, timings, residuals, limitations, and reproduction commands.

## Known limitations

- v0.1 supports real/integer, coordinate Matrix Market input and one right-hand
  side.
- CG requires a matrix that passes the structural SPD screen, or an explicit
  `assume_spd` override for `unknown`; the screen is not a proof of positive
  definiteness.
- GPU support requires a compatible NVIDIA CUDA environment and the matching
  `cuda12` or `cuda13` extra.
- GPU timings and physical GPU OOM behavior were not measured in the release
  validation environment.
