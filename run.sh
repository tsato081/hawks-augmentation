#!/bin/bash
# クラウド環境での実行想定: /home/ubuntu/cross-encoder-experiment/hawks_augmentation/

# スクリプトのディレクトリに移動
cd "$(dirname "$0")"

# Pythonパスを設定
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# .envファイルから環境変数を読み込み
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# データ拡張実行
python3 augmentation/augment_data.py \
  --input data/merged_person_categories.csv \
  --output output/augmented_data.csv \
  --target-per-category 100 \
  --model gpt-4o-mini \
  --temperature 0.8 \
  --batch-size 25 \
  --concurrency 3

echo ""
echo "処理完了！出力: output/augmented_data.csv"
