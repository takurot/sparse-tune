# v0.1.11 release-candidate validation report

**Cutoff:** 2026-07-31 (Asia/Tokyo)

**Version:** `0.1.11`

**Release commit:** `356062dac4035c79ac7d74a678111d034e19f22e`

**Overall result:** passed

This report records release-candidate checks against artifacts built from the
stated commit. It distinguishes checks run locally, checks provided by GitHub
Actions on that commit, historical GPU evidence, and the one remaining gate.
Timing in this report is not a stable performance baseline or a cross-machine
comparison.

## Source and built artifacts

The source tree was exported from the exact release commit with `git archive`,
then built in an isolated Python 3.12.13 environment:

```bash
python -m build
python -m twine check dist/*
python scripts/artifact_version.py dist/*
shasum -a 256 dist/*
```

`twine check` passed and the artifact metadata reported `0.1.11`.

| Artifact | SHA-256 |
| --- | --- |
| `sparsetune-0.1.11-py3-none-any.whl` | `9ebea9f1e3824e5d57302c3a53cbd2606781b8b819cf9a3319cd1d1765735cfc` |
| `sparsetune-0.1.11.tar.gz` | `cfd551a0c13fe8a3f545e8c70031ea6150dff6d59dc048dc2bf0bbd9731234a6` |

These are release-candidate validation artifacts, not published artifacts.
The final TestPyPI and PyPI workflow runs, published hashes, and exact-version
smoke checks are recorded in
[docs/RELEASE_EVIDENCE.md](RELEASE_EVIDENCE.md).

## Supported Python and dependency gates

GitHub Actions run
[`30584853023`](https://github.com/takurot/sparse-tune/actions/runs/30584853023)
completed successfully on the release commit. Its successful jobs covered:

| Gate | Result |
| --- | --- |
| Python 3.10 | passed |
| Python 3.11 | passed |
| Python 3.12 | passed |
| Python 3.13 | passed |
| Python 3.14 | passed |
| Distribution build and clean-wheel smoke | passed |

The minimum environment used Python 3.10.19 with
**NumPy 1.24.0 / SciPy 1.10.0**. It installed the built wheel without resolving
newer dependencies, completed a real 2 x 2 SciPy CG solve with status
`converged` and relative residual `0.0`, and ran the CPU suite:

```bash
python -m pip install "numpy==1.24.0" "scipy==1.10.0" "pytest==7.0.0"
python -m pip install --no-deps sparsetune-0.1.11-py3-none-any.whl
python -m pytest -m "not gpu" -q
```

Result: 196 passed, 1 GPU test deselected. The latest-dependency CI matrix and
the local Python 3.12 environment also completed the full CPU suite.

## Clean-wheel CLI and Python API smoke

The built wheel was installed in a clean Python 3.12.13 environment with
NumPy 2.5.1 and SciPy 1.18.0. `python -m pip check` reported no broken
requirements. The following flows passed on
`tests/fixtures/small_symmetric.mtx`:

```bash
sparsetune --version
sparsetune doctor --format json
sparsetune inspect MATRIX --format json --quiet
sparsetune bench MATRIX --backends scipy --runs 1 --format json --quiet
sparsetune tune MATRIX --backends scipy --runs 1 --output profile.json --quiet
sparsetune solve MATRIX --profile profile.json --output solution.mtx --quiet
```

`sparsetune --version` printed `0.1.11`; `sparsetune doctor` reported the
isolated package and `scipy:cpu`; inspect reported shape 3 x 3; bench converged
with relative residual `9.36e-17`; tune wrote schema `1.0`; and solve wrote a
Matrix Market solution. Direct `sparsetune.inspect()` and `sparsetune.solve()`
Python API calls against the installed wheel also passed.

## Representative SuiteSparse CPU matrices

The public archives and checksums were revalidated before extraction:

| Matrix | Archive URL | Archive SHA-256 |
| --- | --- | --- |
| `bcsstk01` | `https://sparse.tamu.edu/MM/HB/bcsstk01.tar.gz` | `fbc7d883fbb1048ae7e98b12fd5b7c49fa68e0ec070b60f45047db44f63ba984` |
| `bcsstk13` | `https://sparse.tamu.edu/MM/HB/bcsstk13.tar.gz` | `8a100114d7a05226b82fe5107018ebe021c4f3fe5dd600022d744d00dab3572b` |
| `bcsstk18` | `https://sparse.tamu.edu/MM/HB/bcsstk18.tar.gz` | `ebd14190b838f623f7a3ab9c73ff88badc9f9c084b8f12862891929a7bbb8fc8` |

Each built-wheel benchmark used `b = A @ ones`, `float64`, five runs per
measurement mode, `rtol=1e-6`, `atol=0`, and `max_iter=50000`. sparsetune
copied each solution back and independently recomputed its residual on the
CPU before accepting convergence.

| Matrix | Shape / nnz | Iterations | Absolute / relative residual | Threshold | End-to-end | Steady-state | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `bcsstk01` | 48 x 48 / 400 | 77 | `9.71e3` / `9.51e-7` | `1.02e4` | `0.000882 s` | `0.000831 s` | `converged` |
| `bcsstk13` | 2003 x 2003 / 83,883 | 12,402 | `2.18e6` / `9.19e-7` | `2.37e6` | `0.829726 s` | `0.833690 s` | `converged` |
| `bcsstk18` | 11,948 x 11,948 / 149,090 | 9,524 | `2.66e5` / `9.52e-7` | `2.79e5` | `1.320215 s` | `1.321580 s` | `converged` |

The large absolute residuals reflect the matrices' coefficient scale. The
dimensionless relative residual is below `rtol` in every accepted result.
Times are session-specific observations and are not a stable performance
baseline.

## Structured failure paths

The built wheel and targeted tests verified that failures remain structured
and do not terminate the parent process:

- A `1e-6` second CLI benchmark timeout returned `status="timeout"`, no
  recommendation, and exit code 0.
- Requesting unavailable `cupy:cuda:0` returned `status="unsupported"`, no
  recommendation, and exit code 0.
- Targeted tests cover timeout, malformed worker output, inconsistent solve
  payloads, unsupported worker failures, and bounded/redacted diagnostics.
- A profile mismatch in identity/config and malformed profile fields raise
  `ProfileMismatchError` before matrix processing or worker launch.
- The full suite covers worker crash and simulated CPU/CuPy OOM mappings.
  A physical GPU OOM was not attempted.

The focused failure-path selection passed 21 tests.

## Security and repository gates

GitHub Actions run
[`30584853041`](https://github.com/takurot/sparse-tune/actions/runs/30584853041)
completed successfully on the release commit:

- `pip-audit` dependency audit: passed.
- CodeQL Python analysis: passed.

Local release checks also run `ruff check`, `ruff format --check`, `mypy`,
`compileall`, `actionlint`, `pip-audit`, the full CPU suite, build, twine, and
clean-wheel smoke. Generated validation artifacts and downloaded matrices are
not committed.

## Real-GPU gate

The 2026-07-31 [Tesla T4 report](GPU_VALIDATION.md) validates the exact reviewed
release-candidate wheel. The generated JSON is retained as
`temp/colab_gpu_validation_results_20260731.json` with SHA-256
`7a111ddc30aee0d7a2b846e0eeb4e10db0546dd2115689baf52a6a836464ace3`.

Machine validation confirmed:

- `package_source="release-candidate-wheel"` and sparsetune `0.1.11`;
- wheel SHA-256 `9ebea9f1e3824e5d57302c3a53cbd2606781b8b819cf9a3319cd1d1765735cfc`;
- Tesla T4 with CUDA 12.8 compatibility and `cupy-cuda12x` 14.1.1;
- three SciPy and three CuPy results, all `converged` after 6 iterations;
- every independently checked relative residual between `5.873e-7` and
  `6.496e-7`, below `rtol=1e-6`.

The session did not attempt physical GPU OOM, multi-GPU, CUDA 13, or another
GPU model. Those limitations are recorded rather than reported as passes.
