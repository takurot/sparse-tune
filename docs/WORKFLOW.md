# sparsetune 開発ワークフロー

**最終更新:** 2026-07-26
**対象:** `sparsetune` v0.1.0

このドキュメントは、`sparsetune` の機能追加・不具合修正・ドキュメント更新を、
仕様と実装のずれを抑えながら進めるための手順です。

## 1. ドキュメントと実装の優先順位

変更を始める前に、次の順番で確認します。

1. [機能仕様書](SPEC.md) — CLI、Python API、データ型、計測方法の契約
2. [README](../README.md) — 利用者向けの概要、導入例、対応範囲
3. 実装コード — `src/sparsetune/` 配下の実際の挙動
4. テスト — 実装された契約を保護する検証

仕様と実装が異なる場合、意図を推測して片方だけを変更しません。差分の理由を
確認し、仕様を変更するなら `docs/SPEC.md` と README も同じ変更に更新します。
実装前の v0.1.0 では、SPEC にない機能を追加しません。

## 2. 作業開始時の確認

### 2.1 リポジトリと作業ツリー

```bash
git checkout main
git pull --ff-only
git status --short --branch
```

既存の変更を削除したり、`git reset --hard` で上書きしたりしません。自分の変更と
無関係な変更が混在している場合は、対象ファイルを明示して作業します。

### 2.2 タスクのコンテキスト

GitHub Issue がある場合は、実装前に内容、コメント、受け入れ条件を確認します。

```bash
gh issue view <ISSUE> --json title,body,comments,labels
```

次に、対象機能に関係する SPEC の節と README の記載を読み、次を短く整理します。

- 変更対象ファイル
- 既存 API との互換性
- 追加・変更するテスト
- 完了条件と、確認できない条件

## 3. ブランチと計画

`main` に直接コミットせず、タスクごとにブランチを作成します。

```bash
git switch -c issue/<ISSUE>-<short-description>
# Issue がない場合
git switch -c feature/<short-description>
```

実装前に、変更を小さな順序付きステップへ分解します。各ステップには、完了を
確認するテストまたはコマンドを対応付けます。計画に含める項目は次のとおりです。

| 項目 | 内容 |
| --- | --- |
| 変更対象 | `src/sparsetune/`、`tests/`、`docs/` などの具体的なパス |
| API 契約 | CLI オプション、戻り値、例外、JSON スキーマ |
| テスト | unit / integration / GPU の区分とケース |
| リスク | 数値精度、バックエンド差、タイムアウト、リソース解放 |
| 完了条件 | SPEC のどの要件を、どのテストで確認するか |

## 4. 実装サイクル

Python の変更は、可能な限り Red → Green → Refactor の順で進めます。

### Red: 失敗するテストを先に作る

仕様に対応する最小の再現テストを追加し、意図した理由で失敗することを確認します。
v0.1.0 で優先する境界は次のとおりです。

- Matrix Market の `mmread` 結果を canonical CSR に変換できる
- symmetric 行列を二重展開しない
- 非正方行列や非対応 dtype を拒否する
- 正の対角成分だけでは SPD を証明しない
- `--assume-spd` が必要な行列を無断で CG に渡さない
- `info`、有限値、残差の順序に従って `SolveStatus` を分類する
- timeout / OOM / worker crash を親プロセスの異常終了にしない
- プロファイルの fingerprint、dtype、schema を検証する

### Green: 最小限の実装

テストを通すために必要な範囲だけを実装します。v0.1.0 の責務分割は SPEC の
構成に合わせます。

| モジュール | 主な責務 |
| --- | --- |
| `_matrix.py` | Matrix Market 読み込み、CSR 正規化、fingerprint |
| `_inspect.py` | 行列統計、対称性、SPD スクリーニング |
| `_backends.py` | SciPy / CuPy の能力確認とネイティブ CG |
| `_runner.py` | worker の起動、JSON 通信、timeout と異常終了 |
| `_worker.py` | 単一バックエンドの隔離実行 |
| `_benchmark.py` | 複数試行、中央値、推奨バックエンド |
| `_profile.py` | tune / solve、プロファイル保存と検証 |
| `_cli.py` | `inspect`、`bench`、`tune`、`solve`、`doctor` |
| `_types.py` | dataclass、`SolveStatus`、結果スキーマ |

バックエンド固有の処理を親プロセスへ漏らさず、結果を利用する前に shape と型を
検証します。成功結果に対する使用量計測、監査、キャッシュなどの副作用は、
プロトコル結果の検証後に行います。

### Refactor: 挙動を保った整理

Green 後に重複や命名を整理します。リファクタリング後は同じテストを再実行し、
JSON 出力、ステータス、計測境界、エラーメッセージの契約を維持します。

## 5. テストと品質ゲート

実装が存在する段階では、次の順で検証します。`pyproject.toml` が追加されるまでは、
利用可能な静的確認だけを行い、存在しないコマンドを成功扱いにしません。

```bash
pytest
python -m compileall src tests
ruff check .
ruff format --check .
mypy src
```

GPU 固有の検証は通常の CPU テストと分離します。

```bash
pytest -m "not gpu"
pytest -m gpu
```

GPU がない環境で GPU テストを無理に実行しません。GPU テストでは、特定の
バックエンドが選ばれることではなく、結果が有効で残差が許容値以下であることを
検証します。

ベンチマークの確認では、次の境界を守ります。

- `end-to-end`: 正規化後の転送、セットアップ、求解、結果転送を含める
- `steady-state`: デバイス上に保持済みの行列で求解のみを測る
- 各 worker は独立プロセスで実行し、試行間のメモリ干渉を避ける
- 推奨対象は独立 CPU 残差検証に合格した `converged` の結果だけにする

## 6. セキュリティと安全性

- 秘密情報、トークン、GPU 識別子以外の不要な環境情報をコミットしない
- 入力パス、設定 JSON、プロファイル JSON を境界で検証する
- worker の stderr をエラーに含める場合は長さを制限し、秘密情報を露出させない
- timeout 発生時は worker を確実に終了し、一時ファイルを後始末する
- `import sparsetune` で GPU 検出や警告などの副作用を発生させない
- 行列や RHS を外部から受け取る処理では、shape、dtype、有限値を確認する

## 7. ドキュメント同期

実装変更が次のいずれかに影響する場合は、コードと同じ変更で docs を更新します。

- CLI オプション、既定値、終了ステータス → `docs/SPEC.md` と README
- 公開 API、結果型、JSON スキーマ → `docs/SPEC.md` と README の API 例
- 対応環境、インストール、非スコープ → README と SPEC
- 実装手順、検証手順 → この `docs/WORKFLOW.md`

すべてのドキュメントに古いパスや、まだ存在しないコマンドを残しません。README の
予定仕様という注記は、実装が v0.1.0 として動作するまで維持します。

## 8. コミットとプルリクエスト

コミット前に、対象ファイルだけを確認します。

```bash
git diff --check
git diff
git status --short
git add <explicit-files>
git commit -m "<concise change description>"
git push -u origin HEAD
```

コミットには、変更と無関係な仮想環境、キャッシュ、一時ファイルを含めません。
`.gitignore` で `.venv/`、`.tokensave/`、`temp/` を除外します。

プルリクエストには次を記載します。

- 何を変更したか、なぜ変更したか
- SPEC のどの要件に対応するか
- 実行したテストと、環境上実行できなかったテスト
- GPU、性能、互換性への影響
- 関連 Issue があれば `Closes #<ISSUE>`

GitHub Actions が設定された後は、PR の全チェックを確認し、失敗した場合はログを
読んで最小修正と回帰テストを追加します。

## 9. リリース前チェック

v0.1.0 のリリース前には、次を確認します。

- [ ] `README.md`、`docs/SPEC.md`、このワークフローの記載が一致している
- [ ] `inspect`、`bench`、`tune`、`solve`、`doctor` が CLI から実行できる
- [ ] CPU unit / integration テストが通る
- [ ] GPU テストは利用可能な環境で通る、または明確にスキップされる
- [ ] timeout、OOM、process crash、非 SPD の負例を確認している
- [ ] プロファイルの保存・読み込み・不一致エラーを確認している
- [ ] `LICENSE` が MIT で、README からリンクされている
- [ ] GitHub の `main` とローカルのリリースコミットが一致している

## 10. 現在のプロジェクト構成

```text
.github/workflows/
├── publish.yml
├── test.yml
└── testpypi.yml
README.md
RELEASE_NOTES.md
LICENSE
pyproject.toml
docs/
├── SPEC.md
├── VALIDATION.md
└── WORKFLOW.md
src/sparsetune/
├── __init__.py
├── __main__.py
├── _backends.py
├── _benchmark.py
├── _cli.py
├── _inspect.py
├── _matrix.py
├── _probe_worker.py
├── _profile.py
├── _runner.py
├── _solve_worker.py
├── _types.py
└── _worker.py
tests/
├── fixtures/
├── integration/
└── unit/
```
