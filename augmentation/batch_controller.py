"""
バッチ制御モジュール
カテゴリ毎の生成・フィルタリングを管理
"""

import asyncio
import pandas as pd
from typing import Dict, List, Optional
from tqdm import tqdm
import uuid
from datetime import datetime
import json
from style_config import get_style_prompt, STYLE_PRESETS


class BatchController:
    def __init__(
        self,
        prompt_builder,
        llm_generator,
        quality_filter,
        fewshot_reps: Dict[str, tuple],  # {category: ({style_id: [examples]}, num_styles)}
        style_matrix: Dict[str, Dict[int, int]],  # {category: {style_id: count}}
        max_retry_per_sample: int = 3
    ):
        """
        Args:
            prompt_builder: PromptBuilderインスタンス
            llm_generator: LLMGeneratorインスタンス
            quality_filter: QualityFilterインスタンス
            fewshot_reps: カテゴリ毎のスタイル別few-shot代表例
            style_matrix: カテゴリ×スタイル配分マトリクス
            max_retry_per_sample: サンプル毎の最大再生成回数
        """
        self.prompt_builder = prompt_builder
        self.llm_generator = llm_generator
        self.quality_filter = quality_filter
        self.fewshot_reps = fewshot_reps
        self.style_matrix = style_matrix
        self.max_retry_per_sample = max_retry_per_sample

    def generate_for_category(
        self,
        category: str,
        original_df: pd.DataFrame,
        batch_size: int = 50
    ) -> List[Dict]:
        """
        カテゴリ単位で生成

        Args:
            category: 対象カテゴリ
            original_df: 元データ（参照用）
            batch_size: バッチサイズ

        Returns:
            生成されたサンプルリスト
        """
        if category not in self.style_matrix:
            print(f"カテゴリ {category} はスタイルマトリクスに存在しません")
            return []

        style_allocation = self.style_matrix[category]
        few_shot_examples = self._get_fewshot_examples(category)
        reference_bodies = original_df[original_df['category'] == category]['body'].astype(str).tolist()

        all_generated = []
        total_needed = sum(style_allocation.values())

        print(f"\n=== カテゴリ: {category} (必要数: {total_needed}) ===")

        for style, needed_count in style_allocation.items():
            if needed_count == 0:
                continue

            print(f"  スタイル: {style} ({needed_count}件)")
            generated_count = 0
            retry_count = 0

            with tqdm(total=needed_count, desc=f"    {style}", leave=False) as pbar:
                while generated_count < needed_count and retry_count < self.max_retry_per_sample * needed_count:
                    # プロンプト生成
                    from .style_config import get_style_prompt, STYLE_PRESETS

                    style_instruction = get_style_prompt(style)
                    messages = self.prompt_builder.build_messages(
                        category=category,
                        style=style,
                        style_instruction=style_instruction,
                        few_shot_examples=few_shot_examples,
                        num_samples=min(3, needed_count - generated_count)
                    )

                    # LLM生成
                    samples = self.llm_generator.generate(messages, num_samples=3)

                    if not samples:
                        retry_count += 1
                        continue

                    # 品質フィルタ
                    style_preset = STYLE_PRESETS.get(style, {})
                    title_range = style_preset.get('title_length', (10, 50))
                    body_range = style_preset.get('body_length', (120, 500))

                    passed_samples, rejected_info = self.quality_filter.filter_batch(
                        samples,
                        reference_bodies,
                        title_range=title_range,
                        body_range=body_range
                    )

                    # メタデータ追加
                    for sample in passed_samples:
                        sample['is_synth'] = True
                        sample['seed_ids'] = [str(ex.get('id', '')) for ex in few_shot_examples]
                        sample['prompt_style'] = style
                        sample['gen_model'] = self.llm_generator.model
                        sample['created_at'] = datetime.now().isoformat()
                        sample['uuid'] = str(uuid.uuid4())

                        all_generated.append(sample)
                        generated_count += 1
                        pbar.update(1)

                        if generated_count >= needed_count:
                            break

                    retry_count += 1

        print(f"  完了: {len(all_generated)}/{total_needed}件生成")
        return all_generated

    def generate_all_categories(
        self,
        original_df: pd.DataFrame,
        batch_size: int = 50,
        save_interval: int = 10
    ) -> pd.DataFrame:
        """
        全カテゴリで生成

        Args:
            original_df: 元データ
            batch_size: バッチサイズ
            save_interval: 中間保存の間隔（カテゴリ数）

        Returns:
            生成データのDataFrame
        """
        all_results = []
        categories = list(self.style_matrix.keys())

        print(f"\n{'='*60}")
        print(f"データ拡張開始: {len(categories)}カテゴリ")
        print(f"{'='*60}")

        for i, category in enumerate(categories):
            results = self.generate_for_category(category, original_df, batch_size)
            all_results.extend(results)

            # 中間保存
            if (i + 1) % save_interval == 0:
                self._save_intermediate(all_results, i + 1)

        print(f"\n{'='*60}")
        print(f"データ拡張完了: {len(all_results)}件生成")
        print(f"{'='*60}")

        return pd.DataFrame(all_results)

    def _get_fewshot_examples(self, category: str, style_id: int) -> List[Dict[str, str]]:
        """カテゴリとスタイルIDに対応するfew-shot例を取得"""
        if category not in self.fewshot_reps:
            return []

        style_reps, _ = self.fewshot_reps[category]
        if style_id not in style_reps:
            return []

        # style_repsは既に辞書のリストなのでそのまま返す
        return style_reps[style_id]

    def _generate_style_instruction(self, few_shot_examples: List[Dict], style_id: int) -> str:
        """Few-shot例から動的にスタイル指示を生成"""
        if not few_shot_examples:
            return f"文体スタイル{style_id}で生成してください。"

        # 簡易的な特徴抽出
        avg_title_len = sum(len(ex.get('title', '')) for ex in few_shot_examples) // len(few_shot_examples)
        avg_body_len = sum(len(ex.get('body', '')) for ex in few_shot_examples) // len(few_shot_examples)

        # である調・ます調の判定
        dearu_count = sum(ex.get('body', '').count('である') for ex in few_shot_examples)
        masu_count = sum(ex.get('body', '').count('ます') for ex in few_shot_examples)
        tone = "である調" if dearu_count > masu_count else "です・ます調"

        return f"""
文体スタイル: スタイル{style_id}（データから学習）
- 参考例の平均タイトル長: 約{avg_title_len}字
- 参考例の平均本文長: 約{avg_body_len}字
- 推定口調: {tone}
- 本文文字数: 120〜600字
- 参考例の文体に合わせつつ、内容は独自に生成してください
"""

    def _save_intermediate(self, results: List[Dict], checkpoint_num: int):
        """中間結果を保存"""
        if not results:
            return

        df = pd.DataFrame(results)
        output_path = f"output/checkpoint_{checkpoint_num}.csv"
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"\n  中間保存: {output_path} ({len(results)}件)")

    # ========== 非同期メソッド ==========

    async def generate_for_category_async(
        self,
        category: str,
        original_df: pd.DataFrame
    ) -> List[Dict]:
        """
        カテゴリ単位で非同期生成

        Args:
            category: 対象カテゴリ
            original_df: 元データ（参照用）

        Returns:
            生成されたサンプルリスト
        """
        if category not in self.style_matrix:
            print(f"カテゴリ {category} はスタイルマトリクスに存在しません")
            return []

        style_allocation = self.style_matrix[category]
        reference_bodies = original_df[original_df['category'] == category]['body'].astype(str).tolist()

        all_generated = []
        total_needed = sum(style_allocation.values())

        print(f"\n=== カテゴリ: {category} (必要数: {total_needed}) ===")

        for style_id, needed_count in style_allocation.items():
            if needed_count == 0:
                continue

            # スタイルIDに対応するfew-shot例を取得
            few_shot_examples = self._get_fewshot_examples(category, style_id)
            if not few_shot_examples:
                print(f"  スタイル{style_id}: few-shot例なし、スキップ")
                continue

            print(f"  スタイル{style_id} ({needed_count}件)")

            # 動的スタイル指示（few-shot例の特徴から生成）
            style_instruction = self._generate_style_instruction(few_shot_examples, style_id)

            prompts = []
            num_batches = (needed_count + 2) // 3  # 3件ずつ生成

            for _ in range(num_batches):
                messages = self.prompt_builder.build_messages(
                    category=category,
                    style=f"スタイル{style_id}",
                    style_instruction=style_instruction,
                    few_shot_examples=few_shot_examples,
                    num_samples=3
                )
                prompts.append(messages)

            # 非同期バッチ生成
            print(f"    非同期生成中... ({len(prompts)}バッチ)")
            batch_results = await self.llm_generator.generate_batch_async(prompts, num_samples_per_prompt=3)

            # フィルタリング（文字数範囲は120-600で固定）
            title_range = None  # タイトル制限なし
            body_range = (120, 600)

            generated_count = 0
            for batch_idx, samples in enumerate(batch_results):
                if not samples:
                    print(f"    [DEBUG] バッチ{batch_idx}: サンプルなし")
                    continue

                print(f"    [DEBUG] バッチ{batch_idx}: {len(samples)}件受信")
                passed_samples, filtered_out = self.quality_filter.filter_batch(
                    samples,
                    reference_bodies,
                    title_range=title_range,
                    body_range=body_range
                )
                print(f"    [DEBUG] フィルター結果: {len(passed_samples)}件合格 / {len(filtered_out)}件除外")
                if filtered_out:
                    for reject in filtered_out[:2]:  # 最初の2件の理由を表示
                        print(f"    [DEBUG] 除外理由: {', '.join(reject['reasons'])}")

                for sample in passed_samples:
                    sample['is_synth'] = True
                    sample['seed_ids'] = [str(ex.get('id', '')) for ex in few_shot_examples]
                    sample['prompt_style'] = f"スタイル{style_id}"
                    sample['style_cluster_id'] = style_id
                    sample['gen_model'] = self.llm_generator.model
                    sample['created_at'] = datetime.now().isoformat()
                    sample['uuid'] = str(uuid.uuid4())

                    all_generated.append(sample)
                    generated_count += 1

                    if generated_count >= needed_count:
                        break

                if generated_count >= needed_count:
                    break

            print(f"    完了: {generated_count}/{needed_count}件")

        print(f"  カテゴリ完了: {len(all_generated)}/{total_needed}件生成")
        return all_generated

    async def generate_all_categories_async(
        self,
        original_df: pd.DataFrame,
        save_interval: int = 10
    ) -> pd.DataFrame:
        """
        全カテゴリで非同期生成

        Args:
            original_df: 元データ
            save_interval: 中間保存の間隔（カテゴリ数）

        Returns:
            生成データのDataFrame
        """
        all_results = []
        categories = list(self.style_matrix.keys())

        print(f"\n{'='*60}")
        print(f"データ拡張開始（非同期処理）: {len(categories)}カテゴリ")
        print(f"{'='*60}")

        # tqdmで進捗表示
        from tqdm import tqdm
        for i, category in enumerate(tqdm(categories, desc="カテゴリ生成進捗", unit="カテゴリ")):
            results = await self.generate_for_category_async(category, original_df)
            all_results.extend(results)

            # 中間保存
            if (i + 1) % save_interval == 0:
                self._save_intermediate(all_results, i + 1)

        print(f"\n{'='*60}")
        print(f"データ拡張完了: {len(all_results)}件生成")
        print(f"{'='*60}")

        return pd.DataFrame(all_results)
