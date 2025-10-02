"""
Few-shot選定モジュール
埋め込みベースのクラスタリングで文体を分類し、スタイル別にfew-shot例を選定
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import math


class FewShotSelector:
    def __init__(self, embedding_model=None, min_similarity: float = 0.90):
        """
        Args:
            embedding_model: 埋め込みモデル（未指定の場合はダミー）
            min_similarity: 相互類似度の下限
        """
        self.embedding_model = embedding_model
        self.min_similarity = min_similarity

    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """テキストから埋め込みベクトルを生成"""
        if self.embedding_model:
            # 実際の埋め込みモデルを使用
            return self.embedding_model.encode(texts)
        else:
            # ダミー埋め込み（文字数ベースの簡易実装）
            # 本番では sentence-transformers を使用
            embeddings = []
            for text in texts:
                # 簡易的な特徴量: 文字数、句点数、カンマ数、数字の有無など
                features = [
                    len(text),
                    text.count('。'),
                    text.count('、'),
                    sum(c.isdigit() for c in text),
                    text.count('は'),
                    text.count('が'),
                    text.count('を'),
                ]
                embeddings.append(features)
            return np.array(embeddings)

    def select_representatives(
        self,
        df: pd.DataFrame,
        category: str,
        text_column: str = 'body'
    ) -> pd.DataFrame:
        """
        カテゴリ内で代表的なサンプルを選定

        Args:
            df: データフレーム
            category: 対象カテゴリ
            text_column: テキストカラム名

        Returns:
            代表例のDataFrame
        """
        cat_df = df[df['category'] == category].copy()
        n = len(cat_df)

        if n == 0:
            return pd.DataFrame()

        # K値の計算: K = min(6, max(3, ⌊√(n/3)⌋))
        k = min(6, max(3, int(math.sqrt(n / 3))))
        k = min(k, n)  # サンプル数を超えないように

        # 埋め込み生成
        texts = cat_df[text_column].astype(str).tolist()
        embeddings = self._get_embeddings(texts)

        if k == n:
            # 全件選択
            return cat_df

        # KMeansクラスタリング
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cat_df['cluster'] = kmeans.fit_predict(embeddings)

        # 各クラスタから1件ずつ、中心に最も近いものを選択
        representatives = []
        for cluster_id in range(k):
            cluster_df = cat_df[cat_df['cluster'] == cluster_id]
            if len(cluster_df) == 0:
                continue

            cluster_indices = cluster_df.index.tolist()
            # cat_df内の位置を使ってembeddingsにアクセス
            cluster_positions = [cat_df.index.get_loc(idx) for idx in cluster_indices]
            cluster_embeddings = embeddings[cluster_positions]
            centroid = kmeans.cluster_centers_[cluster_id]

            # 中心に最も近いサンプルを選択
            distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
            closest_idx = cluster_indices[np.argmin(distances)]
            representatives.append(closest_idx)

        rep_df = cat_df.loc[representatives].copy()

        # 相互類似度チェック（0.90未満）
        rep_df = self._filter_by_mutual_similarity(rep_df, embeddings, cat_df)

        return rep_df.drop(columns=['cluster'], errors='ignore')

    def _filter_by_mutual_similarity(
        self,
        rep_df: pd.DataFrame,
        all_embeddings: np.ndarray,
        cat_df: pd.DataFrame
    ) -> pd.DataFrame:
        """相互類似度が高すぎる組を除外"""
        if len(rep_df) <= 1:
            return rep_df

        # 代表例の埋め込みを抽出（cat_df内の位置を使用）
        rep_indices = [cat_df.index.get_loc(idx) for idx in rep_df.index]
        rep_embeddings = all_embeddings[rep_indices]

        # コサイン類似度行列
        sim_matrix = cosine_similarity(rep_embeddings)
        np.fill_diagonal(sim_matrix, 0)  # 自分自身は除外

        # 類似度が閾値以上のペアを検出
        high_sim_pairs = np.argwhere(sim_matrix >= self.min_similarity)

        # 重複の多い方を除外
        to_remove = set()
        for i, j in high_sim_pairs:
            if i not in to_remove and j not in to_remove:
                # 類似度の合計が高い方を除外
                if sim_matrix[i].sum() > sim_matrix[j].sum():
                    to_remove.add(i)
                else:
                    to_remove.add(j)

        # 除外後のインデックス
        keep_indices = [idx for i, idx in enumerate(rep_df.index) if i not in to_remove]
        return rep_df.loc[keep_indices]

    def select_style_based_representatives(
        self,
        df: pd.DataFrame,
        category: str,
        text_column: str = 'body',
        samples_per_style: int = 3
    ) -> Tuple[Dict[int, List[Dict]], int]:
        """
        カテゴリ内で文体ベースのクラスタリングを行い、スタイル別に代表例を選定

        Args:
            df: データフレーム
            category: 対象カテゴリ
            text_column: テキストカラム名
            samples_per_style: スタイル（クラスタ）あたりのサンプル数

        Returns:
            ({style_id: [examples]}, num_styles)のタプル
        """
        cat_df = df[df['category'] == category].copy()
        n = len(cat_df)

        if n == 0:
            return {}, 0

        # K値の計算: K = min(6, max(3, ⌊√(n/3)⌋))
        k = min(6, max(3, int(math.sqrt(n / 3))))
        k = min(k, n)

        # 埋め込み生成
        texts = cat_df[text_column].astype(str).tolist()
        embeddings = self._get_embeddings(texts)

        if k == 1:
            # 1クラスタの場合
            selected = cat_df.head(min(samples_per_style, n))
            return {0: selected.to_dict('records')}, 1

        # KMeansクラスタリング（クラスタ = 文体スタイル）
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cat_df['style_cluster'] = kmeans.fit_predict(embeddings)

        # 各クラスタ（スタイル）から複数サンプルを選択
        style_representatives = {}
        for cluster_id in range(k):
            cluster_df = cat_df[cat_df['style_cluster'] == cluster_id]
            if len(cluster_df) == 0:
                continue

            cluster_indices = cluster_df.index.tolist()
            cluster_positions = [cat_df.index.get_loc(idx) for idx in cluster_indices]
            cluster_embeddings = embeddings[cluster_positions]
            centroid = kmeans.cluster_centers_[cluster_id]

            # 中心に近い順にsamples_per_style件を選択
            distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
            sorted_indices = np.argsort(distances)
            num_to_select = min(samples_per_style, len(cluster_indices))
            selected_indices = [cluster_indices[i] for i in sorted_indices[:num_to_select]]

            selected_samples = cat_df.loc[selected_indices].to_dict('records')
            style_representatives[cluster_id] = selected_samples

        return style_representatives, k

    def select_for_all_categories(
        self,
        df: pd.DataFrame,
        text_column: str = 'body',
        samples_per_style: int = 3
    ) -> Dict[str, Tuple[Dict[int, List[Dict]], int]]:
        """全カテゴリでスタイル別代表例を選定

        Returns:
            {category: ({style_id: [examples]}, num_styles)}
        """
        result = {}
        categories = df['category'].unique()

        for category in categories:
            style_reps, num_styles = self.select_style_based_representatives(
                df, category, text_column, samples_per_style
            )
            if style_reps:
                result[category] = (style_reps, num_styles)

        return result
