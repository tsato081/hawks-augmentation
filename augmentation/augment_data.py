#!/usr/bin/env python3
"""
データ拡張メインスクリプト
カテゴリ毎に100件にデータを増やす
"""

import argparse
import asyncio
import pandas as pd
from pathlib import Path
import sys
import os
from dotenv import load_dotenv

# .envファイルを読み込み
load_dotenv()

# モジュールのインポート
from data_analyzer import DataAnalyzer
from fewshot_selector import FewShotSelector
from style_config import get_category_style_matrix, STYLE_PRESETS
from prompt_builder import PromptBuilder
from llm_generator import LLMGenerator
from llm_generator_async import AsyncLLMGenerator
from quality_filter import QualityFilter
from batch_controller import BatchController


def main():
    parser = argparse.ArgumentParser(description="データ拡張スクリプト")
    parser.add_argument(
        "--input",
        default="data/merged_person_categories.csv",
        help="入力CSVファイル"
    )
    parser.add_argument(
        "--output",
        default="output/augmented_data.csv",
        help="出力CSVファイル"
    )
    parser.add_argument(
        "--target-per-category",
        type=int,
        default=100,
        help="カテゴリ毎の目標件数"
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="OpenAI APIキー（未指定時は環境変数OPENAI_API_KEYを使用）"
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="使用するLLMモデル"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="生成温度"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=25,
        help="バッチサイズ（推奨: 10-50、標準環境では25）"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=3,
        help="最大同時API呼び出し数（推奨: 3-5、メモリと相談）"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="ドライラン（分析のみ実行）"
    )

    args = parser.parse_args()

    # 入力ファイル確認
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"エラー: 入力ファイルが見つかりません: {input_path}")
        return 1

    # 出力ディレクトリ作成
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("データ拡張スクリプト")
    print("="*60)
    print(f"入力: {input_path}")
    print(f"出力: {output_path}")
    print(f"目標件数/カテゴリ: {args.target_per_category}")
    print(f"モデル: {args.model}")
    print(f"温度: {args.temperature}")
    print(f"並列数: {args.concurrency}")
    print("="*60)

    # ステップ1: データ読み込みと分析
    print("\n[ステップ1] データ分析中...")
    df = pd.read_csv(input_path)
    analyzer = DataAnalyzer(df, target_per_category=args.target_per_category)
    analysis = analyzer.analyze()
    analyzer.print_summary()

    # 有効サンプルのみ使用
    valid_df = analyzer.get_valid_samples()

    if args.dry_run:
        print("\nドライラン完了")
        return 0

    # ステップ2: スタイル別Few-shot代表例の選定（埋め込みベースのクラスタリング）
    print("\n[ステップ2] スタイル別Few-shot代表例を選定中（埋め込みクラスタリング）...")
    selector = FewShotSelector(embedding_model=None, min_similarity=0.90)
    fewshot_reps = selector.select_for_all_categories(valid_df, text_column='body', samples_per_style=3)

    total_styles = sum(num_styles for _, num_styles in fewshot_reps.values())
    print(f"選定完了: {len(fewshot_reps)}カテゴリ, 合計{total_styles}スタイル")

    # ステップ3: 動的スタイル配分マトリクス生成
    print("\n[ステップ3] 動的スタイル配分マトリクスを生成中...")
    style_matrix = {}
    for category in analysis['needed_samples'].keys():
        if category in fewshot_reps:
            style_reps, num_styles = fewshot_reps[category]
            needed = analysis['needed_samples'][category]

            # スタイルごとに均等配分
            base_count = needed // num_styles
            remainder = needed % num_styles

            style_allocation = {}
            for style_id in range(num_styles):
                count = base_count + (1 if style_id < remainder else 0)
                style_allocation[style_id] = count

            style_matrix[category] = style_allocation

    print(f"配分完了: {len(style_matrix)}カテゴリ（スタイル数は動的）")

    # ステップ4: モジュール初期化
    print("\n[ステップ4] LLM生成モジュールを初期化中...")
    prompt_builder = PromptBuilder()

    # 非同期LLMジェネレータを使用
    llm_generator = AsyncLLMGenerator(
        api_key=args.api_key,
        model=args.model,
        temperature=args.temperature,
        max_concurrent=args.concurrency,
        max_retries=3
    )

    quality_filter = QualityFilter(
        cosine_threshold=0.92,
        ngram_threshold=0.35,
        min_body_length=120,
        max_body_length=600
    )

    # ステップ5: バッチ生成（非同期）
    print("\n[ステップ5] データ生成開始（非同期処理）...")
    controller = BatchController(
        prompt_builder=prompt_builder,
        llm_generator=llm_generator,
        quality_filter=quality_filter,
        fewshot_reps=fewshot_reps,
        style_matrix=style_matrix,
        max_retry_per_sample=3
    )

    # asyncio.run() で非同期実行
    generated_df = asyncio.run(
        controller.generate_all_categories_async(
            original_df=valid_df,
            save_interval=10
        )
    )

    # ステップ6: 出力
    print("\n[ステップ6] 結果を保存中...")
    generated_df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"保存完了: {output_path}")

    # サマリ
    print("\n" + "="*60)
    print("データ拡張完了")
    print("="*60)
    print(f"元データ: {len(valid_df)}件")
    print(f"生成データ: {len(generated_df)}件")
    print(f"合計: {len(valid_df) + len(generated_df)}件")
    print("="*60)

    # スタイル分布
    if 'prompt_style' in generated_df.columns:
        print("\nスタイル分布:")
        style_counts = generated_df['prompt_style'].value_counts()
        for style, count in style_counts.items():
            pct = count / len(generated_df) * 100
            print(f"  {style}: {count}件 ({pct:.1f}%)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
