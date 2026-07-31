# Tesla T4 GPU functional validation

**Cutoff:** 2026-07-31 (Google Colab session)

This report records a single-session functional validation of `sparsetune 0.1.11`
from the reviewed release-candidate wheel. It confirms that SciPy and CuPy
produced independently checked converged solutions on one Tesla T4. It is not
a stable performance baseline: Colab CPU allocation, GPU allocation, clocks,
and session load vary.

## Artifact provenance

| Artifact | SHA-256 |
| --- | --- |
| `sparsetune-0.1.11-py3-none-any.whl` | `9ebea9f1e3824e5d57302c3a53cbd2606781b8b819cf9a3319cd1d1765735cfc` |
| `colab_gpu_validation_results_20260731.json` | `7a111ddc30aee0d7a2b846e0eeb4e10db0546dd2115689baf52a6a836464ace3` |

The JSON records `package_source="release-candidate-wheel"`, the same wheel
hash, and `sparsetune="0.1.11"`. It remains gitignored under
`temp/colab_gpu_validation_results_20260731.json` because it is a generated,
session-specific result artifact.

The session used Python 3.12.13, SciPy 1.16.3, and CuPy 14.1.1.

## Environment and method

| Field | Value |
| --- | --- |
| GPU | Tesla T4, 15,360 MiB |
| NVIDIA driver / reported CUDA | 580.82.07 / 13.0 |
| CUDA toolkit used by CuPy | 12.8 |
| Python | 3.12.13 |
| NumPy / SciPy | 2.0.2 / 1.16.3 |
| CuPy | 14.1.1 (`cupy-cuda12x`) |
| sparsetune | 0.1.11 |

The validation used `float64`, three measured runs per mode, `rtol=1e-6`,
`atol=0`, `max_iter=10000`, and a 300-second worker timeout. Each matrix was a
seeded, symmetric tridiagonal SPD matrix. With seed 23, off-diagonal entries
were sampled uniformly from `[-0.25, 0.25]`; each diagonal entry was 2 plus the
absolute values of its adjacent off-diagonal entries. The exact construction
is in the
[Colab validation notebook](../notebooks/colab_gpu_validation.ipynb).

`sparsetune` ran `scipy:cpu` and `cupy:cuda:0` in isolated subprocesses in both
end-to-end and steady-state modes. It copied each solution back and recomputed
the relative residual on the CPU. A result was accepted only when its status
was `converged` and its relative residual was at most `rtol`.

## Results

All six results had status `converged` after 6 iterations.

| Case | Shape / nnz | Backend | Relative residual | End-to-end | Steady-state |
| --- | ---: | --- | ---: | ---: | ---: |
| small | 10,000 x 10,000 / 29,998 | SciPy CPU | `6.496e-7` | `0.003666 s` | `0.003327 s` |
| small | 10,000 x 10,000 / 29,998 | CuPy T4 | `6.496e-7` | `0.004561 s` | `0.002903 s` |
| medium | 100,000 x 100,000 / 299,998 | SciPy CPU | `5.873e-7` | `0.010589 s` | `0.007027 s` |
| medium | 100,000 x 100,000 / 299,998 | CuPy T4 | `5.873e-7` | `0.009416 s` | `0.003161 s` |
| large | 1,000,000 x 1,000,000 / 2,999,998 | SciPy CPU | `6.167e-7` | `0.115643 s` | `0.077576 s` |
| large | 1,000,000 x 1,000,000 / 2,999,998 | CuPy T4 | `6.167e-7` | `0.056252 s` | `0.012090 s` |

SciPy was recommended for the small end-to-end case. CuPy was recommended for
small steady-state and both modes in the medium and large cases. These
recommendations apply only to this matrix construction and session; the
timings must not be used as cross-device or cross-machine performance claims.

## Reproduction and evidence limits

The tracked
[`notebooks/colab_gpu_validation.ipynb`](../notebooks/colab_gpu_validation.ipynb)
reproduces this method. Before publication, keep
`PACKAGE_SOURCE="release-candidate-wheel"` and upload the reviewed wheel when
prompted. After publication, set `PACKAGE_SOURCE="pypi"` to validate the exact
production package. The notebook records runtime and package identity,
validates every result, and writes `colab_gpu_validation_results.json`.

This session did not test physical GPU OOM, multi-GPU execution, a CuPy CUDA 13
installation, or other GPU models. It therefore makes no claim about those
paths, CUDA 13 compatibility, or stable cross-device performance.
