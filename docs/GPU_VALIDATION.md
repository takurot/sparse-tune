# Tesla T4 GPU functional validation

**Cutoff:** 2026-07-26 (Google Colab session)

This report records a single-session functional validation of `sparsetune
0.1.0`. It confirms that SciPy and CuPy produced independently checked
converged solutions on one Tesla T4. It is not a stable performance baseline:
Colab CPU allocation, GPU allocation, clocks, and session load vary.

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
| sparsetune | 0.1.0 |

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

All results below had status `converged` after 6 iterations.

| Case | Shape / nnz | Backend | Relative residual | End-to-end | Steady-state |
| --- | ---: | --- | ---: | ---: | ---: |
| small | 10,000 x 10,000 / 29,998 | SciPy CPU | `6.496e-7` | `0.003080 s` | `0.002264 s` |
| small | 10,000 x 10,000 / 29,998 | CuPy T4 | `6.496e-7` | `0.005955 s` | `0.003648 s` |
| medium | 100,000 x 100,000 / 299,998 | SciPy CPU | `5.873e-7` | `0.021156 s` | `0.010146 s` |
| medium | 100,000 x 100,000 / 299,998 | CuPy T4 | `5.873e-7` | `0.011355 s` | `0.003700 s` |
| large | 1,000,000 x 1,000,000 / 2,999,998 | SciPy CPU | `6.167e-7` | `0.139013 s` | `0.104827 s` |
| large | 1,000,000 x 1,000,000 / 2,999,998 | CuPy T4 | `6.167e-7` | `0.063935 s` | `0.009162 s` |

SciPy was recommended for both modes in the small case. CuPy was recommended
for both modes in the medium and large cases. These recommendations apply only
to this matrix construction and session; the timings must not be used as
cross-device or cross-machine performance claims.

## Reproduction and evidence

Open the
[tracked notebook](../notebooks/colab_gpu_validation.ipynb) in a Colab GPU
runtime and run all cells. The notebook installs the production package and
the single CUDA-matched CuPy distribution, records runtime and package
identity, validates every result, and writes
`colab_gpu_validation_results.json`. Download that file when prompted.

The raw JSON remains intentionally omitted from version control because it is
a generated, session-specific artifact and `temp/` is gitignored. The notebook
is the authoritative regeneration path; this tracked report summarizes the
2026-07-26 artifact retained locally under
`temp/colab_gpu_validation_results.json`.

This session did not test physical GPU OOM, multi-GPU execution, a CuPy CUDA 13
installation, or other GPU models. It therefore makes no claim about those
paths, CUDA 13 compatibility, or stable cross-device performance.
