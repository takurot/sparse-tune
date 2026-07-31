# Tesla T4 GPU functional validation

**Cutoff:** 2026-07-31 (Google Colab session)

This report records two single-session functional validations of
`sparsetune 0.1.11`: one from the reviewed release-candidate wheel and one from
the production PyPI package. Both confirm that SciPy and CuPy produced
independently checked converged solutions on a Tesla T4. They are not stable
performance baselines: Colab CPU allocation, GPU allocation, clocks, and
session load vary.

## Artifact provenance

| Artifact | SHA-256 |
| --- | --- |
| `sparsetune-0.1.11-py3-none-any.whl` | `9ebea9f1e3824e5d57302c3a53cbd2606781b8b819cf9a3319cd1d1765735cfc` |
| `colab_gpu_validation_results_20260731.json` | `7a111ddc30aee0d7a2b846e0eeb4e10db0546dd2115689baf52a6a836464ace3` |
| `colab_gpu_validation_results_20260730-3.json` | `de0479e4ca23b2cdd125a601a4c18ea2cf654c04e7444b280b9b24e0aa91d478` |

The JSON records `package_source="release-candidate-wheel"`, the same wheel
hash, and `sparsetune="0.1.11"`. It remains gitignored under
`temp/colab_gpu_validation_results_20260731.json` because it is a generated,
session-specific result artifact.

The post-publication JSON records `package_source="pypi"`,
`sparsetune="0.1.11"`, and a null `wheel_sha256`, as expected when pip resolves
the production PyPI package instead of an uploaded local wheel. It remains
gitignored under `temp/colab_gpu_validation_results_20260730-3.json`. Its
embedded runtime date is 2026-07-31; the filename is retained as supplied.

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

## Production PyPI package confirmation

The post-publication run installed `sparsetune==0.1.11` from PyPI with
`PACKAGE_SOURCE="pypi"`. Its environment matched the release-candidate run:
Python 3.12.13, NumPy 2.0.2, SciPy 1.16.3, CuPy 14.1.1 from `cupy-cuda12x`, and
a Tesla T4 using the CUDA 12.8 toolkit. `sparsetune doctor` and the subprocess
probe both reported `scipy:cpu` and `cupy:cuda:0`.

All six aggregate results and all 36 measured samples had status `converged`.
Every independently checked relative residual was at most `rtol=1e-6`; the
aggregate residuals were unchanged from the reviewed-wheel run.

| Case | Backend | Relative residual | End-to-end | Steady-state |
| --- | --- | ---: | ---: | ---: |
| small | SciPy CPU | `6.496e-7` | `0.001746 s` | `0.001254 s` |
| small | CuPy T4 | `6.496e-7` | `0.010264 s` | `0.007197 s` |
| medium | SciPy CPU | `5.873e-7` | `0.027960 s` | `0.007700 s` |
| medium | CuPy T4 | `5.873e-7` | `0.026341 s` | `0.010048 s` |
| large | SciPy CPU | `6.167e-7` | `0.132563 s` | `0.085368 s` |
| large | CuPy T4 | `6.167e-7` | `0.056759 s` | `0.007868 s` |

This confirms the production package's CUDA 12 functional path. The different
recommendations and timings between Colab sessions also reinforce that these
measurements are session-specific and not performance baselines.

## Reproduction and evidence limits

The tracked
[`notebooks/colab_gpu_validation.ipynb`](../notebooks/colab_gpu_validation.ipynb)
reproduces this method. Before publication, keep
`PACKAGE_SOURCE="release-candidate-wheel"` and upload the reviewed wheel when
prompted. After publication, set `PACKAGE_SOURCE="pypi"` to validate the exact
production package. The notebook records runtime and package identity,
validates every result, and writes `colab_gpu_validation_results.json`.

Neither session tested physical GPU OOM, multi-GPU execution, a CuPy CUDA 13
installation, or other GPU models. This report therefore makes no claim about
those paths, CUDA 13 compatibility, or stable cross-device performance.
