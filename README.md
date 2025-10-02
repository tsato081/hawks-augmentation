# Hawks データ拡張システム

カテゴリ別データを各100件に増やすためのデータ拡張スクリプト（クラウド環境対応版）

## 📋 概要

既存の5,130件のデータから、カテゴリ毎に100件まで拡張し、文体の多様性を確保しながら高品質な学習データを生成します。

### 主な特徴

✅ **カテゴリ整合性**: 96カテゴリの定義に厳密準拠
✅ **文体多様性**: 11スタイル（ニュース、ブログ、PR、議事録、行政告知、QA、SNS、コラム、FAQ、商品案内、社内連絡）
✅ **品質保証**: コサイン類似度<0.92、4-gram重複<0.35
✅ **安全性**: 実名・URL・日付の自動除外
✅ **スケーラビリティ**: バッチ処理、中間保存、リトライ機能

## 🚀 クイックスタート（クラウド環境）

### 1. ディレクトリ配置

```bash
# クラウド環境想定パス
/home/ubuntu/cross-encoder-experiment/hawks_augmentation/
```

### 2. セットアップ

```bash
cd /home/ubuntu/cross-encoder-experiment/hawks_augmentation

# 依存関係インストール
pip install -r requirements.txt

# 環境変数設定
cp .env.example .env
# エディタで.envを開き、OpenAI APIキーを設定
nano .env
```

### 3. 実行

```bash
# 実行スクリプトを使用（推奨）
bash run.sh

# または直接実行
python3 augmentation/augment_data.py

# ドライラン（分析のみ）
python3 augmentation/augment_data.py --dry-run
```

## 🧪 テスト実行

本番実行前に、動作確認を行うことを推奨します。

### 動作確認（1件だけ生成）

```bash
# APIキーと基本動作の確認（約30秒）
python3 augmentation/augment_data.py \
  --target-per-category 1 \
  --batch-size 1 \
  --concurrency 1
```

### 小規模テスト（5件生成）

```bash
# 品質フィルタの動作確認（約2-3分）
python3 augmentation/augment_data.py \
  --target-per-category 5 \
  --batch-size 5 \
  --concurrency 3
```

### 並列処理テスト（高速化）

```bash
# 3並列で5件生成（デフォルト、推奨）
python3 augmentation/augment_data.py \
  --target-per-category 5 \
  --batch-size 5 \
  --concurrency 3

# 5並列で高速化（メモリ5GB以上推奨）
python3 augmentation/augment_data.py \
  --target-per-category 5 \
  --batch-size 5 \
  --concurrency 5
```

### 分析のみ（生成なし）

```bash
# ドライラン: データ分析と必要件数の確認
python3 augmentation/augment_data.py --dry-run
```

## 📁 ディレクトリ構成

```
hawks_augmentation/
├── augmentation/              # モジュール群
│   ├── __init__.py
│   ├── data_analyzer.py      # データ分析
│   ├── fewshot_selector.py   # Few-shot選定
│   ├── style_config.py       # スタイル定義
│   ├── prompt_builder.py     # プロンプト生成
│   ├── llm_generator.py      # LLM生成
│   ├── quality_filter.py     # 品質フィルタ
│   ├── batch_controller.py   # バッチ制御
│   └── augment_data.py       # メインスクリプト
├── definitions/              # カテゴリ定義
│   └── category_defines.py
├── data/                     # 入力データ
│   └── merged_person_categories.csv
├── output/                   # 出力先
├── requirements.txt          # 依存関係
├── .env.example             # 環境変数テンプレート
├── run.sh                   # 実行スクリプト
└── README.md                # このファイル
```

## ⚙️ パラメータ設定

### run.sh の編集

```bash
python3 augmentation/augment_data.py \
  --input data/merged_person_categories.csv \
  --output output/augmented_data.csv \
  --target-per-category 100 \
  --model gpt-4o-mini \
  --temperature 0.8 \
  --batch-size 25 \
  --concurrency 3
```

### コマンドラインオプション

| オプション | デフォルト | 説明 |
|----------|----------|------|
| `--input` | `data/merged_person_categories.csv` | 入力CSVファイル |
| `--output` | `output/augmented_data.csv` | 出力CSVファイル |
| `--target-per-category` | `100` | カテゴリ毎の目標件数 |
| `--api-key` | 環境変数 | OpenAI APIキー |
| `--model` | `gpt-4o-mini` | 使用するLLMモデル |
| `--temperature` | `0.8` | 生成温度（0.7-1.0推奨） |
| `--batch-size` | `25` | バッチサイズ（推奨: 10-50） |
| `--concurrency` | `3` | 並列API呼び出し数（推奨: 3-5） |
| `--dry-run` | - | ドライラン（分析のみ） |

## ⚙️ バッチサイズ設定ガイド

バッチサイズは環境のメモリとレート制限に応じて調整してください：

| 環境 | 推奨バッチサイズ | メモリ使用量 | 処理時間（5,000件） |
|-----|----------------|-------------|-------------------|
| **標準環境** | **25** ✅ | 〜1GB | 2-3時間 |
| **低メモリ** | 10 | 〜500MB | 3-4時間 |
| **高性能** | 50 | 〜2GB | 1-2時間 |

### 選択基準

- **メモリ < 2GB**: `--batch-size 10`
- **メモリ 2-4GB**: `--batch-size 25` ✅ **推奨（デフォルト）**
- **メモリ > 4GB**: `--batch-size 50`

### 設定例

```bash
# 低メモリ環境（t2.small等）
python3 augmentation/augment_data.py --batch-size 10

# 標準環境（t2.medium等）
python3 augmentation/augment_data.py --batch-size 25

# 高性能環境（t2.large以上）
python3 augmentation/augment_data.py --batch-size 50
```

### 注意事項

- バッチサイズが大きいほどメモリ使用量が増加
- OpenAI APIのレート制限（3,500 RPM）に注意
- 中間保存は10カテゴリ毎（約250-500件）

## ⚙️ 並列数（Concurrency）設定ガイド

**非同期処理により、3並列で約3倍高速化！**

| 並列数 | メモリ使用量 | 処理時間（5,000件） | レート制限 | 推奨環境 |
|-------|-------------|-------------------|----------|---------|
| **3** ✅ | 3GB | **40-60分** | 安全 | t2.medium以上 |
| **5** | 5GB | 25-35分 | 注意 | t2.large以上 |
| **7** | 7GB | 20-25分 | リスク | t2.xlarge以上 |

### 選択基準

- **メモリ < 4GB**: `--concurrency 3` ✅ **デフォルト（推奨）**
- **メモリ 4-8GB**: `--concurrency 5`
- **メモリ > 8GB**: `--concurrency 7`（レート制限注意）

### 設定例

```bash
# 標準（3並列、安全）
python3 augmentation/augment_data.py --concurrency 3

# 高速（5並列、メモリ5GB以上）
python3 augmentation/augment_data.py --concurrency 5 --batch-size 25

# 最速（7並列、メモリ8GB以上、レート制限リスク）
python3 augmentation/augment_data.py --concurrency 7 --batch-size 50
```

### 並列処理の効果

| 項目 | 従来（直列） | 3並列 | 5並列 | 改善率 |
|-----|------------|------|------|-------|
| **処理時間** | 2-3時間 | 40-60分 | 25-35分 | **3-5倍高速** |
| **メモリ** | 1GB | 3GB | 5GB | 3-5倍 |
| **API効率** | 30 RPM | 90 RPM | 150 RPM | 3-5倍 |

### 注意事項

- **レート制限**: OpenAI APIは3,500 RPM（gpt-4o-mini）
- **並列数を上げすぎるとレート制限に達する可能性**
- **3並列が最も安全かつ効果的**

## 📊 処理フロー

1. **データ分析**: カテゴリ別件数、文字数分布を算出
2. **Few-shot選定**: 埋め込みベースのクラスタリングで代表例を抽出
3. **スタイル配分**: カテゴリ×スタイルの配分マトリクスを生成
4. **LLM生成**: プロンプト構築→API呼び出し→JSON抽出
5. **品質フィルタ**: 類似度、N-gram、言語品質、安全性をチェック
6. **出力**: 合格サンプルをCSVに保存

## 📈 出力データスキーマ

| カラム | 説明 |
|-------|------|
| `title` | タイトル |
| `body` | 本文 |
| `category` | カテゴリ |
| `style` | スタイル（生成時のプリセット） |
| `is_synth` | 合成フラグ（True） |
| `seed_ids` | 参照したfew-shot例のID |
| `prompt_style` | プロンプトで指定したスタイル |
| `gen_model` | 使用したモデル名 |
| `created_at` | 生成日時（ISO 8601） |
| `uuid` | 一意ID |

## 🔧 トラブルシューティング

### APIエラー

```bash
# レート制限に達した場合
python3 augmentation/augment_data.py --batch-size 25
```

### メモリエラー

```bash
# バッチサイズを減らす
python3 augmentation/augment_data.py --batch-size 10
```

### 生成品質が低い

```bash
# 温度を下げる
python3 augmentation/augment_data.py --temperature 0.7

# モデルをアップグレード
python3 augmentation/augment_data.py --model gpt-4o
```

### 途中で中断された場合

中間保存ファイルが `output/checkpoint_*.csv` に保存されています。これを結合して再開できます。

```bash
# チェックポイントファイルを確認
ls -lh output/checkpoint_*.csv

# 最新のチェックポイントから再開（手動で入力CSVを更新）
```

## 💰 コスト見積もり

### gpt-4o-mini使用時
- **入力**: 約5,000トークン/リクエスト
- **出力**: 約1,500トークン/リクエスト
- **総リクエスト数**: 約1,500回（5,000件生成時）
- **推定コスト**: $5-10 USD

### gpt-4o使用時
- **推定コスト**: $50-100 USD

## 🛡️ 品質保証

### 類似度チェック
- コサイン類似度 < 0.92
- 4-gram重複率 < 0.35

### 言語品質
- 日本語率 ≥ 60%
- 句点の存在確認
- 平均文長 10-200字

### 安全性
- URL、具体的日付、電話番号の除外
- 実名パターンの検出

## 📝 ログとモニタリング

実行中の出力例：

```
============================================================
データ拡張開始: 94カテゴリ
============================================================

=== カテゴリ: 労働争議 (必要数: 56) ===
  スタイル: ニュース (11件)
    ニュース: 100%|██████████| 11/11 [00:25<00:00,  2.3s/it]
  スタイル: ブログ (11件)
    ブログ: 100%|██████████| 11/11 [00:23<00:00,  2.1s/it]
  ...
  完了: 56/56件生成

  中間保存: output/checkpoint_10.csv (560件)
```

## 📧 サポート

問題が発生した場合は、以下の情報と共に報告してください：

1. エラーメッセージ全文
2. 使用した実行コマンド
3. `python3 --version` と `pip list` の出力
4. 入力CSVの最初の5行

---

**クラウド環境配置先**: `/home/ubuntu/cross-encoder-experiment/hawks_augmentation/`
