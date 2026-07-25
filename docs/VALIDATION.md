# v0.1.0 validation report

**Cutoff:** 2026-07-26 (Asia/Tokyo)

**Version:** `0.1.0`

This report records release-candidate validation on public sparse matrices. It
separates measured results from checks that require hardware or credentials not
available in the validation environment.

## Method

The matrices came from the
[SuiteSparse Matrix Collection](https://sparse.tamu.edu/) Matrix Market
downloads:

| Matrix | Archive URL | Archive SHA-256 |
| --- | --- | --- |
| `bcsstk01` | `https://sparse.tamu.edu/MM/HB/bcsstk01.tar.gz` | `fbc7d883fbb1048ae7e98b12fd5b7c49fa68e0ec070b60f45047db44f63ba984` |
| `bcsstk13` | `https://sparse.tamu.edu/MM/HB/bcsstk13.tar.gz` | `8a100114d7a05226b82fe5107018ebe021c4f3fe5dd600022d744d00dab3572b` |
| `bcsstk18` | `https://sparse.tamu.edu/MM/HB/bcsstk18.tar.gz` | `ebd14190b838f623f7a3ab9c73ff88badc9f9c084b8f12862891929a7bbb8fc8` |

Each benchmark used the generated right-hand side `b = A @ ones`, `float64`,
five trials per measurement mode, `rtol=1e-6`, `atol=0`, and
`max_iter=50000`. Reported times are medians. `sparsetune` independently
recomputed `||Ax-b||` on the CPU before accepting a converged result.

```bash
sparsetune bench MATRIX \
  --backends scipy \
  --runs 5 \
  --rtol 1e-6 \
  --atol 0 \
  --max-iter 50000 \
  --format json
```

## Environment

| Field | Value |
| --- | --- |
| OS | `macOS-26.5.2-arm64-arm-64bit` |
| Python | `3.10.19` |
| CPU | Apple Silicon, 8 logical cores (reported as `arm`) |
| NumPy | `2.2.6` |
| SciPy | `1.15.3` |
| GPU backends | none |

## Results

| Matrix | Shape / nnz | Iterations | Absolute / relative residual | Threshold | End-to-end | Steady-state | Status / recommendation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `bcsstk01` | 48 x 48 / 400 | 77 | `9.71e3` / `9.51e-7` | `1.02e4` | `0.000740 s` | `0.000698 s` | `converged`; `scipy:cpu` |
| `bcsstk13` | 2003 x 2003 / 83,883 | 12,402 | `2.18e6` / `9.19e-7` | `2.37e6` | `0.808392 s` | `0.814875 s` | `converged`; `scipy:cpu` |
| `bcsstk18` | 11,948 x 11,948 / 149,090 | 9,524 | `2.66e5` / `9.52e-7` | `2.79e5` | `1.339040 s` | `1.334818 s` | `converged`; `scipy:cpu` |

The recommendation reason for every measured mode was the fastest converged
result. `speedup` and `break_even_solves` were `null` because this environment
had only one available backend. The absolute residuals and thresholds are
large because these stiffness matrices have large-magnitude coefficients; the
dimensionless relative residual is below `rtol` in every accepted result.

With the default `max_iter=10000`, `bcsstk13` returned `max_iter` with relative
residual `3.72e-6` and no recommendation. Raising the explicit limit to 50,000
produced the converged result above at iteration 12,402. This confirms that
non-convergence remains a structured result rather than a process failure.

## Failure-path checks

- Running `bcsstk18` with `--timeout 0.001` returned `status="timeout"` and
  `error="Timed out after 0.001 seconds"`; the parent process completed
  normally.
- Requesting `cupy:cuda:0` returned `status="unsupported"` and
  `error="Backend is unavailable"` without importing CuPy in the parent.
- The CPU test suite covers `MemoryError` and CuPy-style `OutOfMemoryError`
  mapping to `oom`. A physical GPU OOM was not attempted because no CUDA device
  was available.
- GPU integration tests were skipped for the same reason. No GPU performance or
  break-even claim is made by this report.

## Release gates

The release candidate passed these local gates:

```bash
pytest -m "not gpu"
coverage run -m pytest -m "not gpu"
coverage combine
coverage report
python -m compileall src tests
ruff check .
ruff format --check .
mypy src
python -m build
twine check dist/*
```

- CPU tests: 84 passed.
- GPU tests: 1 skipped because CUDA was unavailable.
- Combined line coverage, including subprocess workers: 88% (target: 80%).
- Ruff, format, mypy, compileall, actionlint, build, and twine: passed.
- The `0.1.0` wheel installed in a clean Python 3.10 environment;
  `sparsetune --version` returned `0.1.0` and `pip check` passed.
- After updating validation-environment packaging tools, `pip-audit` found no
  known vulnerability in the installed runtime environment. The unpublished
  local `sparsetune` distribution was not present in the audit index.

The TestPyPI workflow builds the artifacts again, publishes with OIDC Trusted
Publishing, installs `sparsetune==0.1.0` back from TestPyPI without dependency
fallback, runs `sparsetune --version`, and runs `pip check`.

TestPyPI and production PyPI Trusted Publisher records are external release
configuration. They must name this repository, their matching GitHub
environment (`testpypi` or `pypi`), and the corresponding workflow before the
first publish.
