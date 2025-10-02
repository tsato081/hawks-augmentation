# Feature: Dynamic Style Classification (実験的実装)

## 概要

`feature/dynamic-style-classification` ブランチには、**埋め込みベースの動的スタイル分類**機能が実装されています。

### 現在のmainブランチ（固定11スタイル方式）
- 事前定義された11個のスタイル（ニュース、ブログ、プレスリリース、議事録、行政告知、QA、SNS投稿、コラム、FAQ、商品案内、社内連絡）
- 全カテゴリで同じスタイルセットを使用
- スタイル別のfew-shot例はなく、カテゴリ全体から選ばれた代表例を全スタイルで共有

### featureブランチ（動的スタイル分類方式）
- **カテゴリごとにデータから文体を自動学習**
- KMeansクラスタリングでカテゴリ内の文体を3-6個のスタイルに分類
- 各スタイルから独立したfew-shot例を抽出
- スタイル数はカテゴリのデータ量に応じて動的に決定: `K = min(6, max(3, ⌊√(n/3)⌋))`

---

## 変更内容の詳細

### 1. `fewshot_selector.py`
#### 追加メソッド
- `select_style_based_representatives(df, category, text_column, samples_per_style)`
  - カテゴリ内データを埋め込みベースでKMeansクラスタリング
  - 各クラスタ（=文体スタイル）から代表例を複数選定
  - 戻り値: `({style_id: [examples]}, num_styles)`

#### 変更メソッド
- `select_for_all_categories(df, text_column, samples_per_style)`
  - 全カテゴリでスタイル別代表例を選定
  - 戻り値: `{category: ({style_id: [examples]}, num_styles)}`

### 2. `augment_data.py`
#### ステップ2の変更
```python
# 旧: 固定スタイル用のfew-shot選定
fewshot_reps = selector.select_for_all_categories(valid_df, text_column='body')
# → {category: DataFrame}

# 新: スタイル別few-shot選定
fewshot_reps = selector.select_for_all_categories(valid_df, text_column='body', samples_per_style=3)
# → {category: ({style_id: [examples]}, num_styles)}
```

#### ステップ3の変更
```python
# 旧: 固定11スタイルの配分マトリクス生成
style_matrix = get_category_style_matrix(categories, analysis['needed_samples'])
# → {category: {style_name: count}}

# 新: 動的スタイル配分（均等配分）
for category in analysis['needed_samples'].keys():
    style_reps, num_styles = fewshot_reps[category]
    needed = analysis['needed_samples'][category]
    # スタイルごとに均等配分
    style_allocation = {style_id: base_count + (1 if style_id < remainder else 0)
                       for style_id in range(num_styles)}
# → {category: {style_id: count}}
```

### 3. `batch_controller.py`
#### 型定義の変更
```python
# 旧
fewshot_reps: Dict[str, pd.DataFrame]
style_matrix: Dict[str, Dict[str, int]]

# 新
fewshot_reps: Dict[str, tuple]  # {category: ({style_id: [examples]}, num_styles)}
style_matrix: Dict[str, Dict[int, int]]  # {category: {style_id: count}}
```

#### メソッドの変更
- `_get_fewshot_examples(category, style_id)` - スタイルIDパラメータ追加
- `_generate_style_instruction(few_shot_examples, style_id)` - **新規追加**
  - Few-shot例から動的にスタイル指示を生成
  - 平均文字数、口調（である/ます）を自動検出

#### 生成ループの変更
```python
# 旧: 固定スタイル名でループ
for style_name, needed_count in style_allocation.items():
    few_shot_examples = self._get_fewshot_examples(category)  # 全スタイル共通
    style_instruction = get_style_prompt(style_name)  # 固定定義

# 新: スタイルIDでループ
for style_id, needed_count in style_allocation.items():
    few_shot_examples = self._get_fewshot_examples(category, style_id)  # スタイル別
    style_instruction = self._generate_style_instruction(few_shot_examples, style_id)  # 動的生成
```

---

## メリット

### 1. データ駆動のスタイル学習
- カテゴリごとに実際のデータから文体を学習
- 事前定義不要で、自然な文体の多様性を確保

### 2. カテゴリ適応性
- カテゴリの特性に応じたスタイル数
- 例: データ量が少ないカテゴリは3スタイル、多いカテゴリは6スタイル

### 3. より自然な生成
- スタイル別のfew-shot例により、文体の一貫性が向上
- 固定ラベル（「ニュース調」など）に縛られない柔軟性

---

## デメリット・懸念点

### 1. クラスタリングの品質
- **簡易埋め込み使用**: 現在はダミー埋め込み（文字数ベース）
  - 本番では`sentence-transformers`が必要
- **クラスタが文体を表すとは限らない**: トピックや内容でクラスタリングされる可能性

### 2. 再現性
- 動的に決まるため、スタイルIDが何を意味するか不明瞭
- デバッグやモニタリングが難しい

### 3. プロンプトの質
- `_generate_style_instruction()`の特徴抽出が簡易的
- 固定スタイルの詳細な指示と比べて曖昧になる可能性

---

## 評価方法（提案）

### 比較実験
1. **mainブランチ（固定11スタイル）で100件生成**
2. **featureブランチ（動的スタイル）で100件生成**
3. **品質評価**:
   - 多様性: 文体のバリエーション
   - 一貫性: カテゴリ定義への適合度
   - 自然さ: 人間による読みやすさ評価

### 評価メトリクス
- カテゴリ分類精度（生成データがPickとして判定されるか）
- スタイル多様性（埋め込み空間での分散）
- テキスト品質スコア

---

## ブランチの切り替え

### featureブランチを試す
```bash
cd /home/ubuntu/cross-encoder-experiment/hawks_augmentation
git fetch origin
git checkout feature/dynamic-style-classification
git pull origin feature/dynamic-style-classification

# 100件生成（動的スタイル版）
python3 augmentation/augment_data.py \
  --input data/merged_person_categories.csv \
  --output output/augmented_dynamic_100.csv \
  --target-per-category 100 \
  --batch-size 5 \
  --concurrency 3
```

### mainブランチに戻る
```bash
git checkout main
git pull origin main

# 100件生成（固定11スタイル版）
python3 augmentation/augment_data.py \
  --input data/merged_person_categories.csv \
  --output output/augmented_fixed_100.csv \
  --target-per-category 100 \
  --batch-size 5 \
  --concurrency 3
```

---

## 今後の改善案

### 1. 埋め込みモデルの改善
```python
from sentence_transformers import SentenceTransformer

# augment_data.py で初期化
embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
selector = FewShotSelector(embedding_model=embedding_model)
```

### 2. スタイル特徴の高度化
- 文末表現の分析（である/だ/ます/です）
- 句読点の使用パターン
- 専門用語の頻度
- 文の長さの分布

### 3. スタイルの可視化
- 各スタイルの代表的な特徴をログ出力
- クラスタの特徴をt-SNEで可視化

### 4. ハイブリッドアプローチ
- 主要なスタイル（ニュース、ブログ、プレスリリース）は固定
- その他をデータから学習した動的スタイルで補完

---

## 問い合わせ先

このfeatureブランチは実験的実装です。問題があれば以下を確認してください：

1. **エラーが出る場合**: mainブランチに戻して動作確認
2. **品質が悪い場合**: `samples_per_style`を増やす（デフォルト3→5など）
3. **スタイル数を調整したい場合**: `fewshot_selector.py:172`のK値計算式を変更

```python
# 現在: K = min(6, max(3, ⌊√(n/3)⌋))
k = min(6, max(3, int(math.sqrt(n / 3))))

# 例: 常に5スタイルにする
k = min(5, n)
```

---

**作成日**: 2025-10-02
**ブランチ**: `feature/dynamic-style-classification`
**ベース**: `main` (commit: ec609e0)
