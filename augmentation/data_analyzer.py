"""
データ分析モジュール
カテゴリ別件数、文字数分布、生成必要数を算出
"""

import pandas as pd
from typing import Dict, Tuple
from collections import Counter


class DataAnalyzer:
    def __init__(self, df: pd.DataFrame, target_per_category: int = 100):
        """
        Args:
            df: 入力データフレーム (title, body, category)
            target_per_category: カテゴリ毎の目標件数
        """
        self.df = df
        self.target_per_category = target_per_category

    def analyze(self) -> Dict:
        """データ分析を実行"""
        # カテゴリ別件数
        category_counts = self.df['category'].value_counts().to_dict()

        # 生成必要数の計算
        needed_samples = {}
        for category, count in category_counts.items():
            if count < self.target_per_category:
                needed_samples[category] = self.target_per_category - count

        # 文字数統計
        self.df['title_len'] = self.df['title'].astype(str).str.len()
        self.df['body_len'] = self.df['body'].astype(str).str.len()

        length_stats = {
            'title': self.df['title_len'].describe().to_dict(),
            'body': self.df['body_len'].describe().to_dict()
        }

        # 文字数フィルタ（本文120字以上）
        valid_df = self.df[self.df['body_len'] >= 120]
        filtered_count = len(self.df) - len(valid_df)

        return {
            'total_samples': len(self.df),
            'valid_samples': len(valid_df),
            'filtered_count': filtered_count,
            'category_counts': category_counts,
            'needed_samples': needed_samples,
            'total_needed': sum(needed_samples.values()),
            'length_stats': length_stats,
            'num_categories': len(category_counts)
        }

    def get_valid_samples(self) -> pd.DataFrame:
        """120字以上のサンプルのみ返す"""
        self.df['body_len'] = self.df['body'].astype(str).str.len()
        return self.df[self.df['body_len'] >= 120].copy()

    def print_summary(self):
        """分析結果サマリを表示"""
        analysis = self.analyze()

        print("=" * 60)
        print("データ分析サマリ")
        print("=" * 60)
        print(f"総件数: {analysis['total_samples']}")
        print(f"有効件数 (本文120字以上): {analysis['valid_samples']}")
        print(f"除外件数: {analysis['filtered_count']}")
        print(f"カテゴリ数: {analysis['num_categories']}")
        print(f"\n生成必要総数: {analysis['total_needed']}")
        print(f"目標カテゴリ件数: {self.target_per_category}")

        print(f"\n--- 文字数統計 ---")
        print(f"タイトル平均: {analysis['length_stats']['title']['mean']:.1f}字")
        print(f"本文平均: {analysis['length_stats']['body']['mean']:.1f}字")
        print(f"本文中央値: {analysis['length_stats']['body']['50%']:.1f}字")

        print(f"\n--- 生成が必要なカテゴリ (上位10) ---")
        sorted_needed = sorted(
            analysis['needed_samples'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        for cat, needed in sorted_needed[:10]:
            current = analysis['category_counts'][cat]
            print(f"  {cat}: {current}件 → {self.target_per_category}件 (+ {needed}件)")

        print("=" * 60)
