# sparsetune 0.1.11

This maintenance release keeps the v0.1 solver and profile contracts while
strengthening compatibility, input validation, reproducibility, release
automation, and documented GPU evidence.

## Compatibility and correctness

- Support the declared SciPy 1.10 minimum by selecting its `tol` CG calling
  convention while retaining `rtol` support on newer SciPy versions (#25).
- Allow the public inspection API to diagnose non-square inputs while keeping
  `bench`, `tune`, and `solve` square-only (#26).
- Reject complex right-hand sides before real-valued canonicalization can lose
  information (#27).
- Define break-even semantics, validate recommendations, and use a single
  versioned JSON schema for benchmark and profile output (#28, #30, #31).
- Capture complete CPU, BLAS, package, CUDA, and GPU identity in profiles and
  validate solver options and malformed profiles at the public boundary
  (#29, #33).
- Keep CUDA discovery out of the parent process and reject explicit empty
  backend lists instead of silently selecting defaults (#42, #43).
- Make `solve` CLI output compact and unambiguous while keeping the solution in
  the requested output file (#32).

## Security and CI

- Maintain Python 3.10-3.14 CPU CI, exercise the declared minimum NumPy/SciPy
  dependencies, derive TestPyPI smoke-test versions from built artifacts, and
  validate wheels in a clean environment (#34).
- Pin third-party GitHub Actions by full commit SHA, minimize workflow
  permissions, add dependency and CodeQL scanning, and configure Dependabot
  for actions and Python dependencies (#35).
- Synchronize README and SPEC claims with the shipped public API, schemas, and
  runtime behavior (#36).

## Validation

CPU release-candidate results, public matrix sources, checksums, failure-path
checks, and reproduction commands are in
[docs/VALIDATION.md](docs/VALIDATION.md).

A Google Colab run confirmed SciPy and CuPy convergence with independently
checked relative residuals on a Tesla T4. Its environment, parameters,
per-matrix outcomes, recommendations, reproduction path, and evidence limits
are recorded in
[docs/GPU_VALIDATION.md](docs/GPU_VALIDATION.md) (#37).

## Known limitations

- v0.1.11 supports real/integer coordinate Matrix Market input and one
  right-hand side.
- CG requires a matrix that passes the structural SPD screen, or an explicit
  `assume_spd` override for `unknown`; the screen is not proof of positive
  definiteness.
- GPU support requires a compatible NVIDIA CUDA environment and the matching
  `cuda12` or `cuda13` extra.
- The tracked GPU evidence covers one Tesla T4 and a CUDA 12 CuPy
  installation. Physical GPU OOM, multi-GPU execution, CUDA 13 installation,
  other devices, and stable cross-device performance were not validated.
