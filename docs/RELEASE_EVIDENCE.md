# v0.1.11 publication evidence

**Published:** 2026-07-31

**Release commit:** `a73f75b4c49f984f7a2e344a29fc647f288498fe`

**Result:** passed

This report records the immutable publication and post-publication checks for
v0.1.11. The tag, both package indexes, and both publishing workflows resolve
to the release commit above.

## Release and publishing workflows

- GitHub Release: [sparsetune 0.1.11](https://github.com/takurot/sparse-tune/releases/tag/v0.1.11)
- TestPyPI OIDC workflow: [run 30633795456](https://github.com/takurot/sparse-tune/actions/runs/30633795456)
- PyPI OIDC workflow: [run 30633927610](https://github.com/takurot/sparse-tune/actions/runs/30633927610)

The `v0.1.11` tag resolves to the release commit. Both workflows checked out
that commit, ran `python -m build` and `twine check`, and published successfully.
The TestPyPI workflow also installed the derived exact version in a clean
environment, ran `sparsetune --version`, and passed `pip check`.

## Published artifacts

The hashes below come from the package-index JSON APIs after publication.
TestPyPI and PyPI build independently from the same commit, so their archive
hashes are recorded separately.

| Index | Artifact | SHA-256 |
| --- | --- | --- |
| TestPyPI | `sparsetune-0.1.11-py3-none-any.whl` | `170eeb839180b666e56da3b40cee8077956a2c2307e863a1a303442582120bd2` |
| TestPyPI | `sparsetune-0.1.11.tar.gz` | `4829164f3c57128969b577e2095d0b4bd284115180c583922d57ac786cc03dd8` |
| PyPI | `sparsetune-0.1.11-py3-none-any.whl` | `1f0a5fe179d698adfc91e1b28cc626d0739a4fec217b6791693066b73510c72b` |
| PyPI | `sparsetune-0.1.11.tar.gz` | `ecb4aed86b98482966a467b83873b9d74366305775d260e5c70149a2fb608988` |

Package pages:

- [TestPyPI v0.1.11](https://test.pypi.org/project/sparsetune/0.1.11/)
- [PyPI v0.1.11](https://pypi.org/project/sparsetune/0.1.11/)

## Production CPU smoke

A clean Python 3.14.6 environment on an Apple M1 installed
`sparsetune==0.1.11` using only the production index. The following checks
passed with NumPy 2.5.1 and SciPy 1.18.0:

- `sparsetune --version` reported `0.1.11`;
- `python -m pip check` reported no broken requirements;
- `sparsetune doctor` reported the expected CPU environment;
- `inspect`, one-run SciPy `bench`, and direct SciPy `solve` completed on the
  small symmetric fixture; and
- both solver operations returned `converged` with relative residual
  `9.362222582871203e-17`, below `rtol=1e-6`.

## Trusted publisher configuration

Publication uses OIDC Trusted Publishing and no repository upload token:

| Index | GitHub environment | Workflow | Event |
| --- | --- | --- | --- |
| TestPyPI | `testpypi` | `.github/workflows/testpypi.yml` | manual dispatch from `main` |
| PyPI | `pypi` | `.github/workflows/publish.yml` | published GitHub Release |

Each publishing job grants `id-token: write`; the repository-level default
remains read-only. The successful uploads confirm that the external TestPyPI
and PyPI trusted publishers accept this repository, workflow, and environment
combination.

## GPU limitation

The publication host was an Apple M1 without NVIDIA CUDA, so the exact
production wheel was not reinstalled in a GPU environment. The reviewed
v0.1.11 release-candidate wheel had already passed SciPy and CuPy convergence
on a Tesla T4 with CUDA 12.8 compatibility and CuPy 14.1.1; that evidence and
its artifact hash are recorded in [GPU_VALIDATION.md](GPU_VALIDATION.md).
CUDA 13 and a post-publication production-index GPU reinstall were not tested.
