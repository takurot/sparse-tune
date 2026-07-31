# sparsetune

> **開発状況:** v0.1.11
> **最終更新:** 2026-07-31

`sparsetune` は、ユーザーの疎行列に対して利用可能な CPU / GPU 線形ソルバを
隔離環境で比較し、再現可能な性能・精度レポートと推奨構成を生成する
クロスバックエンド・オートチューナーです。

ソルバそのものを再実装するのではなく、既存バックエンドを安全かつ公平に比較し、
「この行列を一度解く場合」と「同じ行列を繰り返し解く場合」のそれぞれに適した
バックエンドを選びます。

> [!IMPORTANT]
> v0.1.11 は PyPI から利用できます。実GPUでの機能検証には
> [Google Colab GPU validation notebook](notebooks/colab_gpu_validation.ipynb)を
> 使用できます。2026-07-31 の v0.1.11 Tesla T4 実行結果は
> [GPU機能検証レポート](docs/GPU_VALIDATION.md)にまとめています。

## 主な特徴

- **ユーザー行列中心** — Matrix Market 形式の疎行列を渡すだけで比較
- **CPU / GPU 横断比較** — SciPy（CPU）と CuPy（NVIDIA CUDA）に対応
- **サブプロセス隔離** — タイムアウト、OOM、プロセスクラッシュをバックエンド単位で捕捉
- **2 種類の性能評価** — 一回の求解向け `end-to-end` と反復利用向け `steady-state`
- **独立した精度検証** — 各バックエンドの解を CPU 上で再検証
- **再現可能なレポート** — ハードウェア、ドライバ、ライブラリのバージョンを記録
- **チューニング結果の再利用** — `tune` と `solve` を分離し、毎回の再計測を回避

## 対応環境

| 項目 | v0.1.11 |
| --- | --- |
| Python | 3.10–3.14 |
| CPU バックエンド | SciPy (`scipy:cpu`) |
| GPU バックエンド | CuPy (`cupy:cuda:N`) |
| GPU | NVIDIA CUDA 12.x / 13.x |
| ソルバ | 各バックエンドのネイティブ CG |
| 入力 | Matrix Market coordinate（real / integer） |

`inspect` は非正方行列も受け付け、形状と構造を診断します。CG を実行する
`bench`、`tune`、`solve` は正方行列だけを対象とします。複素数、密行列形式、
マルチ GPU、マルチ RHS、PyTorch、MPS、GMRES、BiCGSTAB、直接法は対象外です。

## インストール

PyPI から CPU 版をインストールできます。

```bash
pip install "sparsetune==0.1.11"
```

CUDA を利用する場合は、環境に合う追加依存関係を指定します。

```bash
# CUDA 12.x
pip install "sparsetune[cuda12]==0.1.11"

# CUDA 13.x
pip install "sparsetune[cuda13]==0.1.11"
```

Python 3.10〜3.14をサポートし、CIで各versionを継続検証します。
必須依存関係は NumPy 1.24 以上と SciPy 1.10 以上です。SciPy 1.10/1.11 の
`tol` と新しいSciPyの `rtol` の両CG呼び出し規約に対応します。CuPy は任意依存で、
`import sparsetune` の時点では GPU 検出などの副作用を発生させません。

### Google Colab でGPU検証

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/takurot/sparse-tune/blob/main/notebooks/colab_gpu_validation.ipynb)

このノートブックはproduction PyPIの `sparsetune==0.1.11` をインストールし、割り当てられた
NVIDIA GPU上でCPU/GPUの収束、相対残差、計測結果、推奨を検証してJSONへ保存します。
ColabのGPU、CPU割り当て、クロック、セッション寿命は変動するため、結果は機能検証であり、
安定した性能ベースラインではありません。

## クイックスタート

### 1. 環境を確認する

```bash
sparsetune doctor
```

### 2. 行列を調べる

```bash
sparsetune inspect model.mtx --format table
```

### 3. バックエンドを比較する

```bash
sparsetune bench model.mtx \
  --backends scipy,cupy \
  --runs 5 \
  --rtol 1e-6 \
  --format table
```

### 4. 推奨構成を保存する

```bash
sparsetune tune model.mtx --output model.profile.json
```

### 5. 保存した構成で解く

```bash
sparsetune solve model.mtx \
  --profile model.profile.json \
  --rhs rhs.mtx \
  --output x.mtx
```

右辺ベクトルを省略した場合は、`x = ones` を真の解として `b = A @ x` を生成します。
`--profile` の代わりに `--backend scipy:cpu` のようにバックエンドを直接指定することも
できます。

CLIのstdoutには、成功・失敗とも解ベクトルを含まない小さなJSON metrics
（status、残差、反復回数、計測時間など）を出力します。解ベクトルは `--output`
指定時だけMatrix Market fileへ保存され、stdoutへ重複出力されません。
`--quiet` はprogressとmetricsを抑制し、結果はexit codeで判定できます。
Python APIの `SolveResult.x` は引き続き完全な解ベクトルを保持します。

## CLI

| コマンド | 説明 |
| --- | --- |
| `sparsetune inspect <matrix>` | 形状、非ゼロ要素数、対称性などを表示 |
| `sparsetune bench <matrix>` | 利用可能なバックエンドを比較 |
| `sparsetune tune <matrix>` | 比較結果と推奨構成をプロファイルに保存 |
| `sparsetune solve <matrix>` | プロファイルまたは指定バックエンドで求解 |
| `sparsetune doctor` | Python、ライブラリ、CUDA、GPU の情報を表示 |
| `sparsetune --version` | バージョンを表示 |

代表的なベンチマークオプション:

| オプション | デフォルト | 説明 |
| --- | --- | --- |
| `--backends` | `scipy,cupy` | 比較するバックエンド |
| `--dtype` | `float64` | `float32` または `float64` |
| `--measure` | `end-to-end,steady-state` | 計測モード |
| `--runs` | `5` | 試行回数。中央値を採用 |
| `--rtol` | `1e-6` | 相対収束許容値 |
| `--atol` | `0.0` | 絶対収束許容値 |
| `--max-iter` | `10000` | 最大反復回数 |
| `--timeout` | `300.0` | バックエンドごとの制限時間（秒） |
| `--assume-spd` | 無効 | SPD と仮定して CG を実行 |
| `--format` | `json` | `json`、`table`、`csv` |

すべてのオプションと出力スキーマは
[CLI 仕様](docs/SPEC.md#4-cli-仕様)を参照してください。

## Python API

```python
import sparsetune
from sparsetune import SolveStatus

A = sparsetune.load_matrix("model.mtx")

profile = sparsetune.tune(
    A,
    backends=["scipy:cpu", "cupy:cuda:0"],
    dtype="float64",
    runs=5,
)

result = sparsetune.solve(A, profile=profile)

if result.status == SolveStatus.CONVERGED:
    x = result.x
    print(result.backend, result.total_seconds)
else:
    print(result.status, result.error)
```

主な公開 API は `list_backends()`、`load_matrix()`、`inspect()`、
`benchmark()`、`tune()`、`solve()` です。`solve()` は解ベクトルだけでなく、
バックエンド、反復回数、残差、所要時間、終了ステータスを含む `SolveResult` を返します。

## 入力行列と SPD 判定

対応する Matrix Market 入力:

- coordinate 形式
- real または integer
- `symmetric`、`general`、`skew-symmetric`
- `inspect` では非正方行列も可。`bench`、`tune`、`solve` では正方行列

CG は対称正定値（SPD）行列向けの反復法です。`sparsetune` の事前診断は、
対称性と正の対角成分を確認するスクリーニングであり、正定値性を証明しません。

- `screen_passed` — 対称かつ対角成分が正。CG を実行
- `unknown` — `--assume-spd` 指定時のみ、ユーザー責任で実行
- `failed` — 非対称または非正方。CG を拒否

## ベンチマーク方法

| モード | 評価する時間 | 想定用途 |
| --- | --- | --- |
| `end-to-end` | backend準備（GPUではHost→Device転送を含む）、求解、同期、解の取得 | 正規化済み行列を一度だけ解く |
| `steady-state` | デバイス上に行列を保持した状態での求解 | 同じ行列を繰り返し解く |

行列ファイルの読込、CSR正規化、RHS生成は親プロセスで先に行うため、どちらの
モードにも含みません。`setup_seconds` はbackend準備、`solve_seconds` は求解と同期、
`transfer_seconds` は解の取得、`total_seconds` はこれらを合わせたworker内の実時間です。
steady-stateの順位付けには `solve_seconds` を使い、解の取得時間は含めません。

各バックエンドは独立した worker プロセスで実行されます。モードごとに1回の
計測外warmupを行い、デフォルトではその後に5回計測して中央値を採用します。
あるモードの全sampleがCPU上の独立残差検証にも合格した `converged` の場合だけ、
そのbackendを同モードの推奨対象にします。一方のモードの失敗は他方の推奨を
無効にしません。

### 推奨理由と break-even

CPU が一回の求解で最速の場合、`end_to_end` は次の形になります。

```json
{
  "mode": "end-to-end",
  "backend": "scipy:cpu",
  "reason": "Fastest converged end-to-end result (0.0031 seconds)",
  "speedup": 1.7,
  "break_even_solves": null
}
```

GPU が反復利用で最速の場合、転送・セットアップの初期コストを回収する求解回数も
記録されます。

```json
{
  "mode": "steady-state",
  "backend": "cupy:cuda:0",
  "reason": "Fastest converged steady-state result (0.0012 seconds)",
  "speedup": 2.4,
  "break_even_solves": 8
}
```

この例では、同じ行列を 8 回以上解くと GPU の初期コストを回収できる計算です。
`break_even_solves` は最初の求解を1回目とする正の整数で、steady-stateの節約がない
場合や比較可能なCPU結果がない場合は `null` です。
時間、speedup、break-even は行列と実行環境ごとに変わるため、実際の JSON レポートを
判断に使用してください。収束した GPU 結果がない場合は CPU が選ばれ、
`break_even_solves` は `null` になります。

主な終了ステータス:

- `converged` — 収束し、独立残差検証にも合格
- `accuracy_failed` — ソルバは成功を返したが、残差が許容値を超過
- `max_iter` / `breakdown` — 反復上限到達または数値的破綻
- `nan_inf` — 解に NaN または Inf を検出
- `oom` / `timeout` / `process_crash` — 隔離 worker 内の実行エラー
- `unsupported` / `internal_error` — 非対応構成または内部エラー

## アーキテクチャ

```text
CLI / Python API
        |
        v
Matrix Inspector -> Experiment Planner
        |
        v
Subprocess Runner
  |- SciPy Worker (CPU)
  `- CuPy Worker  (CUDA)
        |
        v
Independent Validator (CPU)
        |
        v
Recommender -> Profile Cache / JSON
```

サブプロセス分離により、バックエンド単位のタイムアウト、CUDA コンテキストの破棄、
メモリアロケータの分離、セグメンテーションフォルトからの親プロセス保護を行います。
`bench`、`tune`、プロファイル付き `solve` が利用するGPU可用性と実行環境IDも
検証済みのプローブサブプロセスから取得します。明示的な環境診断である
`list_backends()` と `sparsetune doctor` だけは、呼び出し元プロセスでCUDAを検出します。

doctor、benchmark JSON、profile JSON は同じ実行環境fieldを使用します。CPUは
`cpu_model`、`cpu_cores_physical`、`blas_implementation`、SciPy versionを、
GPUは `gpu_uuid`、`gpu_model`、`cuda_driver`、`cuda_runtime`、`cupy_version` を
backend identityとして記録します。取得できないportable fieldは `null` です。
選択backendのidentity変更はprofile mismatch、Python/NumPy/SciPy/CPU/BLAS変更は
明示的なstale-profile opt-inが必要です。

## ロードマップ

次の項目は計画であり、上記の v0.1.11 の対応範囲には含まれません。

| バージョン | 予定内容 |
| --- | --- |
| v0.1.0 | SciPy / CuPy、ネイティブ CG、隔離実行、CLI、プロファイル（リリース済み） |
| v0.1.11 | 互換性・入力検証・再現性・CI/CD・GPU検証の強化（リリース済み） |
| v0.2.0 | PyTorch CUDA、GMRES / BiCGSTAB、メモリ測定改善 |
| v0.3.0 | MPS 実験対応、統一 CG、AMGX の検討 |

## ドキュメント

- [機能仕様書 SPEC v2.1](docs/SPEC.md)
- [開発ワークフロー](docs/WORKFLOW.md)
- [v0.1.11 リリース候補検証レポート](docs/VALIDATION.md)
- [v0.1.11 公開証跡](docs/RELEASE_EVIDENCE.md)
- [Tesla T4 GPU機能検証レポート](docs/GPU_VALIDATION.md)
- [v0.1.11 リリースノート](RELEASE_NOTES.md)
- [Google Colab GPU validation notebook](notebooks/colab_gpu_validation.ipynb)

## ライセンス

[MIT License](LICENSE)
