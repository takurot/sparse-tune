# sparsetune v0.1 機能仕様書 (SPEC v2.1)

> **ステータス:** レビュー済み（v2.1 — ChatGPT 再レビュー反映）
> **作成日:** 2026-07-25
> **最終改訂:** 2026-07-26
>
> **プロジェクト定義:**
> ユーザー提供の疎行列について、利用可能な CPU/GPU 線形ソルバを隔離環境で比較し、
> 再現可能な性能・精度レポートと推奨構成を生成するツール。
> **クロスバックエンド疎行列ソルバ・オートチューナー**であり、「ユニバーサルソルバ」ではない。
>
> **v0.1 の焦点:** CLI ベンチマーク + tune/solve の分離 + プロファイルキャッシュ。
> SciPy (CPU) + CuPy (CUDA) のみ。各バックエンドのネイティブ CG を使用。
> **開発期間:** 個人開発者 2 週間。

---

## 1. プロジェクトメタデータ

| 項目 | 値 |
|------|-----|
| 名称 | `sparsetune`（旧 `sparselab` — PyPI 既存パッケージとの競合回避） |
| パッケージ名 | `sparsetune` (PyPI) |
| ライセンス | MIT |
| Python | >= 3.10 |
| リポジトリ | https://github.com/takurot/sparse-tune |

---

## 2. 競合との差別化

### 2.1 真の競合

| 競合 | 何か | sparsetune との違い |
|------|------|-------------------|
| **pyGinkgo** (2025) | Python 向け疎線形代数フレームワーク。SciPy/CuPy/PyTorch SpMV・反復ソルバ比較機能あり | pyGinkgo はソルバライブラリそのもの。sparsetune はソルバ選定のオーケストレーター |
| **Ginkgo** C++ | CUDA/HIP/SYCL/OpenMP マルチバックエンド。ベンチマークスイート内蔵 | C++ API のみ。pip install 不可。ユーザ行列の自動比較 UX なし |
| **PETSc** | Krylov 法 + プリコンディショナの統一インターフェース | インストール重い。MPI 前提。「どのソルバが速いか教えて」に答えない |
| **NVIDIA AMGX** | GPU AMG + Krylov。Python バインディングあり | CUDA 限定。pip install 非対応。自動比較 UX なし |
| **torch-sla** | PyTorch 向け疎行列ソルバラッパー | PyTorch 限定。バックエンド横断比較できない |
| **JAX-AMG** | JAX 向け AMGX ラッパー | JAX + AmgX 限定 |

### 2.2 sparsetune の差別化ポイント

**「バックエンド非依存」は差別化にならない**——Ginkgo が既に実現している。

本当の差別化:

1. **ユーザー行列中心**: `.mtx` を渡すだけ。pip install 一発
2. **サブプロセス隔離**: 各バックエンドを別プロセスで実行。タイムアウト・OOM・セグフォを安全に捕捉
3. **end-to-end / steady-state 分離**: 一回解く場合と繰り返し解く場合で推奨が変わる
4. **再現可能レポート**: ハードウェア・ドライバ・ライブラリバージョンを自動記録
5. **プロファイルキャッシュ**: tune 結果を再利用。アプリ組み込み向け
6. **数値精度＋性能の同時評価**: CPU 上で独立残差検証。収束理由を詳細列挙
7. **ソルバ選定のオーケストレーター**: ソルバそのものを作らず、既存ソルバを賢く選ぶ

---

## 3. アーキテクチャ

```
CLI / Python API
        │
        ▼
Matrix Inspector
  - mmread → COO → CSR canonicalization
  - validate shape, nnz
  - structural statistics
  - matrix fingerprint (sha256)
        │
        ▼
Experiment Planner
  - backend capability probe
  - dtype compatibility check
  - run ordering
        │
        ▼
Subprocess Runner
  ├─ SciPy Worker  (separate process)
  └─ CuPy Worker   (separate process)
        │
        ▼
Independent Validator (CPU)
  - residual ||Ax - b|| / ||b||
  - solution error (when x_true known)
  - finite check
        │
        ▼
Recommender
  - end-to-end recommendation
  - steady-state recommendation
        │
        ▼
Profile Cache / JSON output
```

**サブプロセス分離の利点:**
- タイムアウト時に worker を kill できる
- CUDA コンテキストを確実に破棄できる
- CuPy と他ライブラリのメモリアロケータ干渉なし
- セグメンテーションフォルトから親プロセスを保護
- バックエンドごとに独立した環境変数

---

## 4. CLI 仕様

### 4.1 エントリポイント

```bash
sparsetune <command> [options]
```

v0.1 コマンド:

| コマンド | 説明 |
|----------|------|
| `inspect <matrix>` | 行列の統計情報表示 |
| `bench <matrix>` | 全バックエンドでベンチマーク実行 |
| `tune <matrix>` | ベンチマーク実行＋プロファイル保存 |
| `solve <matrix>` | 保存済みプロファイルで求解 |
| `doctor` | 環境診断（利用可能バックエンド・GPU 情報） |
| `--version` | バージョン表示 |

### 4.2 `bench` コマンド

```bash
sparsetune bench <matrix_file> [options]
```

**オプション:**

| オプション | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `--backends` | str | `scipy,cupy` | 使用バックエンド |
| `--dtype` | str | `float64` | `float32`, `float64` |
| `--measure` | str | `end-to-end,steady-state` | 計測モード |
| `--runs` | int | 5 | 各バックエンド試行回数（中央値採用） |
| `--rtol` | float | 1e-6 | 相対収束許容値 |
| `--atol` | float | 0.0 | 絶対収束許容値 |
| `--tol` | float | — | `--rtol` の後方互換エイリアス（非推奨） |
| `--max-iter` | int | 10000 | 最大反復回数 |
| `--timeout` | float | 300.0 | バックエンドごとのタイムアウト（秒） |
| `--assume-spd` | flag | false | SPD と仮定して CG を実行 |
| `--output` | path | stdout | 結果出力先（JSON） |
| `--format` | str | `json` | `json`, `table`, `csv` |

**出力例 (JSON):**

```json
{
  "schema_version": "1.0",
  "matrix": {
    "path": "model.mtx",
    "shape": [150000, 150000],
    "nnz": 4712366,
    "fingerprint": "sha256:abc123...",
    "dtype": "float64"
  },
  "environment": {
    "os": "Linux",
    "python": "3.12.4",
    "cuda_device": "NVIDIA RTX 4090",
    "cuda_driver": "550.90.07",
    "cupy_version": "14.1.0",
    "scipy_version": "1.17.0"
  },
  "results": [
    {
      "backend": "cupy:cuda:0",
      "solver_impl": "cupyx.scipy.sparse.linalg.cg",
      "dtype": "float64",
      "transfer_seconds": 0.31,
      "setup_seconds": 0.04,
      "solve_seconds": 1.23,
      "total_seconds": 1.61,
      "iterations": 142,
      "residual_norm": 3.5e-09,
      "relative_residual": 1.2e-10,
      "convergence_threshold": 3.0e-05,
      "status": "converged"
    },
    {
      "backend": "scipy:cpu",
      "solver_impl": "scipy.sparse.linalg.cg",
      "dtype": "float64",
      "transfer_seconds": 0.0,
      "setup_seconds": 0.01,
      "solve_seconds": 47.20,
      "total_seconds": 47.21,
      "iterations": 142,
      "residual_norm": 1.1e-06,
      "relative_residual": 3.5e-08,
      "convergence_threshold": 3.0e-05,
      "status": "converged"
    }
  ],
  "recommendations": {
    "end_to_end": "cupy:cuda:0",
    "steady_state": "cupy:cuda:0"
  }
}
```

**計測モード定義:**

| モード | 計測対象 |
|--------|---------|
| `end-to-end` | 変換＋転送＋前処理＋solve＋結果転送。一回解く場合の実時間 |
| `steady-state` | solve のみ。行列と前処理はデバイス上に保持済みの前提 |

### 4.3 `tune` コマンド

```bash
sparsetune tune <matrix_file> [options]
```

`bench` と同じオプション＋プロファイル保存。

```bash
sparsetune tune model.mtx --output model.profile.json
```

プロファイルには以下を含む:
- 行列フィンガープリント (sha256)
- ベンチマーク結果
- 環境情報
- 推奨バックエンド

### 4.4 `solve` コマンド

```bash
sparsetune solve <matrix_file> --profile <profile.json> [options]
```

| オプション | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `--profile` | path | — | tune で生成したプロファイル（--backend と排他） |
| `--backend` | str | — | バックエンド直接指定（--profile と排他） |
| `--rhs` | path | 自動生成 | 右辺ベクトル（Matrix Market N×1）。省略時は ones 解で生成 |
| `--selection-mode` | str | `end-to-end` | プロファイル内の推奨選択: `end-to-end`, `steady-state` |
| `--output` | path | stdout | 解ベクトル出力（Matrix Market） |
| `--rtol` | float | 1e-6 | 相対収束許容値 |
| `--atol` | float | 0.0 | 絶対収束許容値 |
| `--max-iter` | int | 10000 | 最大反復回数 |
| `--assume-spd` | flag | false | SPD と仮定して CG を実行 |

### 4.5 `inspect` コマンド

```bash
sparsetune inspect <matrix_file> [--format json|table]
```

### 4.6 `doctor` コマンド

```bash
sparsetune doctor
```

環境情報を出力。`import sparsetune` では何も出力しない（副作用回避）。

```
sparsetune doctor v0.1.0
─────────────────────────
Python:     3.12.4
scipy:      1.17.0
cupy:       14.1.0
CUDA:       550.90.07
GPU 0:      NVIDIA RTX 4090 (24.0 GB)
```

---

## 5. Python API

### 5.1 公開 API

```python
# バックエンド検出（import 時に自動実行しない）
sparsetune.list_backends() -> list[str]
# → ['scipy:cpu', 'cupy:cuda:0']

# 行列読み込み
sparsetune.load_matrix(path: str) -> SparseMatrix

# 行列情報
sparsetune.inspect(A: SparseMatrix) -> MatrixInfo

# ベンチマーク → BenchmarkResult
sparsetune.benchmark(A, *, backends=None, dtype='float64',
    measure=('end-to-end','steady-state'),
    runs=5, rtol=1e-6, atol=0.0, max_iter=10000, timeout=300.0,
    assume_spd=False) -> BenchmarkResult

# チューニング → プロファイル dict
sparsetune.tune(A, **bench_kwargs) -> Profile

# プロファイルまたは直接指定で求解
sparsetune.solve(A, b=None, *, profile=None, backend=None,
    selection_mode='end-to-end', rtol=1e-6, atol=0.0, max_iter=10000,
    assume_spd=False) -> SolveResult
```

### 5.2 `solve()` — SolveResult を返す

生の NumPy 配列ではなく、結果オブジェクトを返す（反復回数や精度が失われるのを防ぐ）:

```python
@dataclass
class SolveResult:
    x: ndarray | None
    backend: str           # 'cupy:cuda:0'
    dtype: str             # 'float64'
    status: SolveStatus
    iterations: int
    residual_norm: float       # ||Ax - b||
    relative_residual: float | None  # null when b_norm == 0
    convergence_threshold: float
    setup_seconds: float
    solve_seconds: float
    total_seconds: float
    error: str | None

class SolveStatus(str, Enum):
    CONVERGED = "converged"
    ACCURACY_FAILED = "accuracy_failed"   # info=0 but residual > threshold
    MAX_ITER = "max_iter"
    BREAKDOWN = "breakdown"
    NAN_INF = "nan_inf"
    OOM = "oom"
    TIMEOUT = "timeout"
    PROCESS_CRASH = "process_crash"
    UNSUPPORTED = "unsupported"
    INTERNAL_ERROR = "internal_error"
```

**ステータス判定順序（重要）:**

```python
def classify(info, x, residual_norm, threshold):
    # 有限値チェックを最初に行う（全 info 値より優先）
    if not np.all(np.isfinite(x)):
        return SolveStatus.NAN_INF

    if info == 0:
        if residual_norm <= threshold:
            return SolveStatus.CONVERGED
        return SolveStatus.ACCURACY_FAILED

    if info > 0:
        return SolveStatus.MAX_ITER

    return SolveStatus.BREAKDOWN
```

使用例:

```python
result = sparsetune.solve(A, b, backend="cupy:cuda:0")
if result.status == SolveStatus.CONVERGED:
    x = result.x
```

### 5.3 データ型

```python
@dataclass
class SparseMatrix:
    """Sparse matrix in canonical CSR format (CPU-resident)."""
    data: ndarray          # float64 or float32
    indices: ndarray       # int32 column indices
    indptr: ndarray        # int32 row pointers
    shape: tuple[int, int]

@dataclass
class MatrixInfo:
    path: str
    shape: tuple[int, int]
    nnz: int
    density: float
    is_square: bool
    symmetry_ratio: float
    diagonal_sign: str     # 'all_positive', 'all_negative', 'mixed', 'zeros'
    spd_status: str        # 'screen_passed', 'failed', 'unknown'
    fingerprint: str       # sha256 of canonical CSR

@dataclass
class BenchmarkResult:
    matrix: MatrixInfo
    environment: dict
    results: list[SolverResult]
    recommendations: dict  # {'end_to_end': 'cupy:cuda:0', 'steady_state': '...'}

    def best(self, mode='end-to-end') -> SolverResult: ...
    def to_json(self) -> str: ...
    def to_dict(self) -> dict: ...
    def to_table(self) -> str: ...

@dataclass
class SolverResult:
    backend: str           # 'cupy:cuda:0'
    solver_impl: str       # 'cupyx.scipy.sparse.linalg.cg'
    dtype: str
    transfer_seconds: float
    setup_seconds: float
    solve_seconds: float
    total_seconds: float
    iterations: int
    residual_norm: float
    relative_residual: float | None
    convergence_threshold: float
    pool_used_gb: float | None       # CuPy pool used_bytes() delta (reference only)
    status: str            # SolveStatus value
    error: str | None

Profile = dict  # JSON-serializable benchmark result
```

---

## 6. バックエンド仕様

### 6.1 v0.1 バックエンド

| バックエンド | ID | 必須パッケージ | デバイス |
|------------|-----|--------------|---------|
| SciPy | `scipy:cpu` | `scipy` | CPU |
| CuPy | `cupy:cuda:N` | `cupy-cuda12x` | NVIDIA CUDA |

**v0.1 では PyTorch と MPS を含まない。** v0.2 以降で検討。

### 6.2 バックエンド能力プローブ

起動時ではなく、ベンチマーク実行時に実行列で検証:

```python
def probe_backend(backend_id: str, dtype: str) -> CapabilityResult:
    """
    Create a small CSR matrix, run matvec + solve,
    verify residual computation works.

    Returns CapabilityResult with:
    - supported: bool
    - error: str | None
    - experimental: bool
    """
```

### 6.3 SciPy バックエンド

```python
class SciPyBackend:
    id = "scipy:cpu"

    def solve_cg(self, A_csr, b, rtol, atol, max_iter):
        """
        Use scipy.sparse.linalg.cg.

        Returns:
            x, info (0=success, >0=iter limit, <0=illegal input/breakdown)
        """
        iterations = 0
        def callback(_xk):
            nonlocal iterations
            iterations += 1

        x, info = scipy.sparse.linalg.cg(
            A_csr, b, rtol=rtol, atol=atol,
            maxiter=max_iter, callback=callback
        )
        return x, info, iterations
```

### 6.4 CuPy バックエンド

```python
class CuPyBackend:
    id = "cupy:cuda:0"  # GPU index from CUDA_VISIBLE_DEVICES

    def solve_cg(self, A_csr_cpu, b_cpu, dtype_str, rtol, atol, max_iter):
        """
        Transfer CSR to GPU, run cupyx.scipy.sparse.linalg.cg.

        dtype_str: 'float64' or 'float32'

        Returns:
            x (CPU ndarray), info (0=success, >0=iter limit), iterations
        """
        cp_dtype = cp.float32 if dtype_str == 'float32' else cp.float64

        A_gpu = cp.sparse.csr_matrix(
            (cp.asarray(A_csr_cpu.data, dtype=cp_dtype),
             cp.asarray(A_csr_cpu.indices),
             cp.asarray(A_csr_cpu.indptr)),
            shape=A_csr_cpu.shape
        )
        b_gpu = cp.asarray(b_cpu, dtype=cp_dtype)

        iterations = 0
        def callback(_xk):
            nonlocal iterations
            iterations += 1

        x_gpu, info = cupyx.scipy.sparse.linalg.cg(
            A_gpu, b_gpu, tol=rtol, atol=atol,
            maxiter=max_iter, callback=callback
        )
        cp.cuda.Stream.null.synchronize()

        return cp.asnumpy(x_gpu), info, iterations
```

---

## 7. サブプロセス実行

### 7.1 設計

各バックエンド = 1 worker プロセス。親プロセスは CSR 正規化を一度だけ行い、
NPZ で共有。worker 内部で end-to-end 全試行 + steady-state 全試行を実行。

```python
import subprocess, json, tempfile, time
from pathlib import Path

def run_backend_in_subprocess(
    backend_id: str,
    canonical_npz_path: str,    # 親で正規化済みの CSR (NPZ)
    b_path: str | None,
    config: dict,
    timeout: float,
) -> SolverResult:
    """
    Spawn subprocess, pass canonical matrix via NPZ file, collect result.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        config_path = temp_dir / "config.json"
        result_path = temp_dir / "result.json"

        config_path.write_text(json.dumps(config), encoding="utf-8")

        try:
            proc = subprocess.run(
                [sys.executable, '-m', 'sparsetune._worker',
                 '--backend', backend_id,
                 '--matrix-npz', canonical_npz_path,
                 '--config', str(config_path),
                 '--result', str(result_path),
                 *(['--rhs', b_path] if b_path else [])],
                capture_output=True, text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return SolverResult(
                backend=backend_id,
                status=SolveStatus.TIMEOUT,
                error=f"Timed out after {timeout}s",
            )

        if proc.returncode != 0:
            return SolverResult(
                backend=backend_id,
                status=SolveStatus.PROCESS_CRASH,
                error=proc.stderr[:500],
            )

        return SolverResult(**json.loads(result_path.read_text()))
```

### 7.2 行列正規化（親プロセス、一度だけ）

```python
def canonicalize_matrix(matrix_path: str, dtype_str: str) -> tuple[Path, MatrixInfo]:
    """
    mmread → CSR canonicalize → save NPZ → return info.
    Done once in parent process, shared across all workers.
    """
    A = scipy.io.mmread(matrix_path)
    A = A.tocsr()
    A.sum_duplicates()
    A.sort_indices()

    np_dtype = np.dtype(dtype_str)
    A.data = A.data.astype(np_dtype)

    info = diagnose_matrix(A)

    npz_path = Path(tempfile.gettempdir()) / f"sparsetune_{info.fingerprint[:16]}.npz"
    scipy.sparse.save_npz(str(npz_path), A)

    return npz_path, info
```

### 7.3 Worker プロセス (`sparsetune._worker`)

```python
# sparsetune/_worker.py
"""Isolated process for running one backend across all measurement modes."""

def main():
    args = parse_args()
    config = json.loads(Path(args.config).read_text())

    # Load canonicalized matrix (parent already did mmread + CSR normalize)
    A = scipy.sparse.load_npz(args.matrix_npz)
    n = A.shape[0]

    # Load or generate RHS
    if args.rhs:
        b_raw = scipy.io.mmread(args.rhs)
        if scipy.sparse.issparse(b_raw):
            b_raw = b_raw.toarray()
        b = np.asarray(b_raw, dtype=A.data.dtype).reshape(-1)
        if len(b) != n:
            raise ValueError(f"RHS size {len(b)} != matrix size {n}")
    else:
        x_true = np.ones(n, dtype=A.data.dtype)
        b = A @ x_true

    backend = get_backend(args.backend)
    results = []

    # --- Warmup (outside timing) ---
    _ = backend.solve_cg(A, b, **config)  # warmup, not measured

    # --- end-to-end runs ---
    for run in range(config['runs']):
        t0 = time.perf_counter()
        x, info, iterations = backend.solve_cg(A, b, **config)
        elapsed = time.perf_counter() - t0
        results.append(_make_result('end-to-end', elapsed, x, info, iterations, A, b, config))

    # --- steady-state runs (matrix already on device via warmup) ---
    for run in range(config['runs']):
        t0 = time.perf_counter()
        x, info, iterations = backend.solve_cg(A, b, **config)
        elapsed = time.perf_counter() - t0
        results.append(_make_result('steady-state', elapsed, x, info, iterations, A, b, config))

    Path(args.result).write_text(json.dumps({
        'backend': args.backend,
        'results': [asdict(r) for r in results],
    }))

def _make_result(mode, elapsed, x, info, iterations, A_csr, b, config):
    residual_norm = np.linalg.norm(A_csr @ x - b)
    b_norm = np.linalg.norm(b)
    threshold = max(config['rtol'] * b_norm, config['atol'])
    status = classify(info, x, residual_norm, threshold)

    return SolverResult(
        status=status,
        solve_seconds=elapsed,
        iterations=iterations,
        residual_norm=residual_norm,
        relative_residual=(residual_norm / b_norm) if b_norm > 0 else None,
        convergence_threshold=threshold,
        ...
    )
```

---

## 8. 行列読み込み仕様

### 8.1 読み込みパイプライン（修正版）

**重要:** SciPy の `mmread()` は symmetric Matrix Market の反対側三角要素を**自動復元する**。
独自の対称展開は不要（二重展開のバグになる）。

```python
def load_matrix(path: str) -> SparseMatrix:
    """
    Load Matrix Market coordinate file. Canonicalize to sorted CSR.
    """
    A = scipy.io.mmread(path)         # COO or CSR, symmetric already expanded
    A = A.tocsr()                      # Convert to CSR
    A.sum_duplicates()                 # Merge duplicate entries
    A.sort_indices()                   # Sort column indices per row
    return SparseMatrix(
        data=A.data.astype(np.float64),
        indices=A.indices.astype(np.int32),
        indptr=A.indptr.astype(np.int32),
        shape=A.shape,
    )
```

**サポートフォーマット:**
- Matrix Market coordinate (real, integer)
- square 行列のみ
- `symmetric` / `general` / `skew-symmetric`

**非サポート（v0.1）:**
- Matrix Market array（密行列形式。疎行列ツールの v0.1 には不要）
- 複素数、パターン形式
- Harwell-Boeing, PETSc バイナリ

### 8.2 SPD 判定

```python
def diagnose_matrix(A: scipy.sparse.csr_matrix) -> MatrixInfo:
    """
    Determine SPD status.

    positive diagonal + symmetric → 'screen_passed'
    non-symmetric / non-square → 'failed'
    otherwise → 'unknown'

    NOTE: positive diagonal + symmetric is NOT sufficient for SPD.
    This is a screening heuristic only. Users must pass --assume-spd
    to run CG.
    """
    is_square = A.shape[0] == A.shape[1]
    if not is_square:
        return MatrixInfo(spd_status='failed', ...)

    ratio = _symmetry_ratio(A)
    symmetric = ratio > 0.9999999999  # 1e-10 tolerance

    diag = A.diagonal()
    all_positive = np.all(diag > 0)

    if symmetric and all_positive:
        spd_status = 'screen_passed'
    elif not symmetric:
        spd_status = 'failed'
    else:
        spd_status = 'unknown'

    return MatrixInfo(
        is_square=is_square,
        symmetry_ratio=ratio,
        diagonal_sign=_diag_sign(diag),
        spd_status=spd_status,
        ...
    )
```

### 8.3 `--assume-spd` フラグ

- `spd_status == 'failed'` → CG 実行を拒否（非対称/非正方）
- `spd_status == 'screen_passed'` → 自動で CG 実行
- `spd_status == 'unknown'` → `--assume-spd` がないと警告＋拒否。`--assume-spd` があればユーザ責任で実行

---

## 9. ベンチマーク方法論

### 9.1 計測モード

| モード | 含むもの | 含まないもの |
|--------|---------|------------|
| `end-to-end` | CSR 正規化 + Host→Device 転送 + 前処理 setup + solve + Device→Host 転送 | 行列ファイル読み込み |
| `steady-state` | solve のみ | 転送・前処理（デバイス上に保持済み前提） |

### 9.2 タイミングフロー（バックエンドネイティブ CG 使用）

```
[end-to-end]
  t0 = perf_counter()
  A_gpu = transfer_to_device(A_csr)    # 転送
  setup_preconditioner(A_gpu)          # 前処理セットアップ
  x_gpu, info = native_cg(A_gpu, b_gpu, tol, max_iter)
  sync_device()                        # GPU 完了待ち
  t1 = perf_counter()
  total_time = t1 - t0

[steady-state]
  # 行列・前処理はデバイス上に既に存在
  t0 = perf_counter()
  x_gpu, info = native_cg(A_gpu, b_gpu, tol, max_iter)
  sync_device()
  t1 = perf_counter()
  solve_time = t1 - t0
```

### 9.3 独立残差検証

全バックエンドの解に対し、**CPU 上で**残差を独立計算:

```python
def validate_solution(A_csr, x, b) -> float:
    """Compute ||Ax - b|| / ||b|| on CPU."""
    residual = A_csr @ x - b
    rel_res = np.linalg.norm(residual) / np.linalg.norm(b)
    return rel_res
```

### 9.4 CG 終了理由の判定

```python
def info_to_status(info: int, x: ndarray, residual: float, tol: float) -> str:
    """Map scipy/cupy CG info code to SolveStatus."""
    if info == 0:
        # Double-check: sometimes info=0 but residual is still high
        if residual <= tol:
            return "converged"
        else:
            return "max_iter"
    elif info > 0:
        return "max_iter"
    elif info < 0:
        # Illegal input or breakdown (non-positive curvature, etc.)
        return "breakdown"

    if not np.all(np.isfinite(x)):
        return "nan_inf"
    return "internal_error"
```

### 9.5 自動選択ロジック

```python
def recommend(results: list[SolverResult]) -> dict:
    """
    Return {'end_to_end': best_backend, 'steady_state': best_backend}

    Only considers results with status == 'converged'.
    Selects based on total_seconds (end-to-end) or solve_seconds (steady-state).
    """
    end_to_end = min(
        [r for r in results if r.status == 'converged'],
        key=lambda r: r.total_seconds,
        default=None
    )
    steady_state = min(
        [r for r in results if r.status == 'converged'],
        key=lambda r: r.solve_seconds,
        default=None
    )
    return {
        'end_to_end': end_to_end.backend if end_to_end else None,
        'steady_state': steady_state.backend if steady_state else None,
    }
```

### 9.6 複数試行

- デフォルト 5 回実行、**中央値**を採用
- 各試行前にバックエンドメモリを解放（サブプロセスなのでプロセス終了＝完全解放）
- サブプロセス単位なのでメモリ干渉は原理的に発生しない

---

## 10. プロファイルキャッシュ

### 10.1 設計思想

`backend="auto"` で毎回全バックエンドを試すと、常に単一バックエンドより遅い。
このオーバーヘッドを避けるため、tune/solve を分離:

```python
# 一度だけチューニング
profile = sparsetune.tune(A, dtype='float64')

# 以降はプロファイルから最適バックエンドで求解
result = sparsetune.solve(A, b, profile=profile)
# or
result = sparsetune.solve(A, b, backend='cupy:cuda:0')
```

### 10.2 プロファイルのキャッシュキー

```python
cache_key = hashlib.sha256(json.dumps({
    'schema_version': '1.0',
    'matrix_fingerprint': matrix_info.fingerprint,
    'dtype': 'float64',
    'rtol': 1e-6,
    'atol': 0.0,
    'max_iter': 10000,
    'gpu_model': env.gpu_model,
    'gpu_uuid': env.gpu_uuid,
    'cuda_driver': env.cuda_driver,
    'cuda_runtime': env.cuda_runtime,
    'cupy_version': env.cupy_version,
    'scipy_version': env.scipy_version,
    'cpu_model': env.cpu_model,
    'cpu_cores_physical': env.cpu_cores_physical,
    'blas_implementation': env.blas_implementation,
}).encode()).hexdigest()
```

### 10.3 プロファイル検証

`solve()` 時にプロファイルを検証し、不一致は `ProfileMismatchError`:

```python
class ProfileMismatchError(Exception):
    """Raised when cached profile is incompatible with current environment."""

def validate_profile(profile: Profile, matrix_info: MatrixInfo, config: dict):
    """Raise ProfileMismatchError on version, fingerprint, dtype, or device mismatch."""
    if profile['schema_version'] != '1.0':
        raise ProfileMismatchError(f"Schema v{profile['schema_version']} not supported")

    if profile['matrix']['fingerprint'] != matrix_info.fingerprint:
        raise ProfileMismatchError("Matrix fingerprint mismatch — re-tune required")

    if profile['matrix']['dtype'] != config['dtype']:
        raise ProfileMismatchError(
            f"Profile dtype {profile['matrix']['dtype']} != requested {config['dtype']}"
        )

    # Backend version check (warning, not error)
    if profile['environment']['cupy_version'] != env.cupy_version:
        warnings.warn(f"CuPy version changed: {profile['environment']['cupy_version']} → {env.cupy_version}")
```

### 10.4 CLI

```bash
# チューニング
sparsetune tune model.mtx --output model.profile.json

# プロファイルから求解
sparsetune solve model.mtx \
    --profile model.profile.json \
    --rhs rhs.mtx \
    --output x.mtx
```

---

## 11. インストール仕様

```bash
pip install sparsetune
```

**依存関係（必須）:**
```
scipy>=1.10.0
numpy>=1.24.0
```

**依存関係（オプション）:**

```bash
# 利用者の CUDA バージョンに合わせて選択
pip install sparsetune[cuda12]   # CUDA 12.x
pip install sparsetune[cuda13]   # CUDA 13.x
```

または手動で:

```bash
pip install sparsetune
pip install cupy-cuda12x
```

`[cupy]` 単一 extras は CUDA バージョン不明瞭なため提供しない。

`import sparsetune` は副作用を持たない（GPU 検出・警告は行わない）。
環境診断は `sparsetune doctor` で明示的に行う。

---

## 12. テスト戦略

### 12.1 テスト階層

```
tests/
├── unit/
│   ├── test_matrix.py       # load_matrix, mmread canonicalization
│   ├── test_inspect.py      # MatrixInfo, SPD diagnosis
│   ├── test_autoselect.py   # recommend() logic (mock results)
│   └── test_profile.py      # Profile serialization
├── integration/
│   ├── test_cli.py          # CLI end-to-end
│   ├── test_api.py          # Python API
│   └── test_subprocess.py   # subprocess runner
└── fixtures/
    ├── small_spd.mtx         # 10×10 SPD
    ├── small_symmetric.mtx   # 10×10 symmetric (mmread 展開確認用)
    ├── small_nonsquare.mtx   # 非正方（エラーケース）
    └── indefinite.mtx        # 対称・対角正だが非正定値（[[1,2],[2,1]]）
```

### 12.2 重要なテストケース

| テスト | 内容 |
|--------|------|
| `test_mmread_symmetric_already_expanded` | symmetric Matrix Market を mmread 後、二重展開されていないことを確認 |
| `test_mmread_returns_COO_not_CSR` | mmread の返り値が COO であることを確認、.tocsr() が必要 |
| `test_positive_diagonal_not_spd` | [[1,2],[2,1]] が spd_status='screen_passed' だが実際は非正定値 |
| `test_reject_nonsquare` | 非正方行列は spd_status='failed' で CG 拒否 |
| `test_assume_spd_required_for_unknown` | spd_status='unknown' は --assume-spd なしで拒否 |
| `test_autoselect_fastest_converged` | 収束結果から最速を選ぶ（実GPUテスト不要、モックで検証） |
| `test_subprocess_timeout` | タイムアウト時に worker が kill される |
| `test_subprocess_oom_graceful` | OOM 時にエラーステータスが返る |
| `test_profile_roundtrip` | tune → save → load → solve が一致 |

### 12.3 GPU テスト分離

- GPU テストは `pytest.mark.gpu` でマーク
- CI では CPU テストのみ実行
- GPU テストでは「CuPy が選ばれること」ではなく「有効な結果が返り、残差が tol 以下であること」を検証

---

## 13. プロジェクト構成

```
sparsetune/
├── pyproject.toml
├── README.md
├── LICENSE                 # Apache 2.0
├── .gitignore
├── src/
│   └── sparsetune/
│       ├── __init__.py     # __version__ = "0.1.0"
│       ├── _matrix.py      # load_matrix, canonicalize
│       ├── _inspect.py     # diagnose_matrix, MatrixInfo
│       ├── _backends.py    # SciPyBackend, CuPyBackend, probe
│       ├── _runner.py      # SubprocessRunner
│       ├── _worker.py      # Subprocess worker entry point
│       ├── _benchmark.py   # benchmark, recommend
│       ├── _profile.py     # tune, solve, Profile cache
│       ├── _cli.py         # CLI
│       └── _types.py       # Data classes, SolveStatus
├── tests/
│   ├── unit/
│   │   ├── test_matrix.py
│   │   ├── test_inspect.py
│   │   ├── test_autoselect.py
│   │   └── test_profile.py
│   ├── integration/
│   │   ├── test_cli.py
│   │   ├── test_api.py
│   │   └── test_subprocess.py
│   └── fixtures/
│       ├── small_spd.mtx
│       ├── small_symmetric.mtx
│       ├── small_nonsquare.mtx
│       └── indefinite.mtx
└── docs/
    └── SPEC.md
```

---

## 14. 非スコープ（v0.1）

- PyTorch バックエンド
- MPS / Apple Silicon
- 統一 CG 実装（normalized mode）
- GMRES / BiCGSTAB / 直接ソルバ
- out-of-core
- 自前 GPU カーネル
- AMGX バックエンド
- 複素数行列
- Matrix Market array 形式
- Harwell-Boeing / PETSc バイナリ
- マルチ GPU
- マルチ RHS
- 条件数推定
- プラグイン API
- メモリ使用量の厳密比較（v0.1 では参考値扱い）

---

## 15. マイルストーン

| バージョン | 内容 | 目標期間 |
|-----------|------|---------|
| v0.1.0 | SciPy + CuPy ネイティブCG、サブプロセス隔離、tune/solve 分離、CLI、JSON レポート | 2 週間 |
| v0.1.1 | PyPI 公開、README、CI/CD | +1 週間 |
| v0.2.0 | PyTorch CUDA 追加、GMRES/BiCGSTAB、メモリ測定の精緻化 | +4 週間 |
| v0.3.0 | MPS 実験的サポート、normalized mode（統一 CG による公平比較）、AMGX 検討 | +8 週間 |

---

## 16. リスク

| リスク | 深刻度 | 対策 |
|--------|--------|------|
| CuPy のインストール障壁 | 中 | extras_require、conda 手順を明記 |
| サブプロセス実装の複雑さ | 中 | Python 標準の subprocess で十分。JSON で通信 |
| 非 SPD 行列の誤判定 | 中 | screen_passed/unknown/failed の 3 段階 + --assume-spd |
| sparselab 名称競合 | 低 | sparsetune に変更済み |
| CI で GPU テスト不可 | 低 | GPU テストを pytest.mark.gpu で分離 |

---

## 17. レビュー履歴

| 日付 | レビューア | 結果 | 主な指摘・変更 |
|------|-----------|------|-------------|
| 2026-07-25 | サブエージェント (Hermes) | CONDITIONAL GO | CG統一化、BackendProtocolと計測フロー整合性、RHS生成仕様、条件数推定の非スコープ化、MPS実験的格下げ、Jacobiプリコンディショナ、OOMハンドリング、用語統一、アーキテクチャ図 |
| 2026-07-25 | ChatGPT 再レビュー | CONDITIONAL GO → GO | **v2.1 変更**: tol→rtol/atol（SciPy 1.14互換）、dtypeを実処理に伝播、end-to-end計測境界を統一（CSR正規化を親プロセスに移動）、steady-state worker単位を修正（1 worker = 全試行）、subprocessのNamedTemporaryFile→TemporaryDirectory+TimeoutExpired捕捉、iterationカウントcallback追加、status判定順序修正+ACCURACY_FAILED追加、ゼロRHS対応（閾値=max(rtol*b_norm,atol)）、RHSをMatrix Market形式に統一、メモリ値を参考値扱いに格下げ、プロファイル検証（ProfileMismatchError）+キャッシュキー拡充、CuPy extrasをcuda12/cuda13に分割 |
