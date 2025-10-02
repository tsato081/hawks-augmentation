"""
Few-shot選定モジュール
埋め込みベースのクラスタリングでカテゴリ内多様性を確保
"""

import numpy as np
import pandas as pd
from typing import List, Dict
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

    def select_for_all_categories(
        self,
        df: pd.DataFrame,
        text_column: str = 'body'
    ) -> Dict[str, pd.DataFrame]:
        """全カテゴリで代表例を選定"""
        result = {}
        categories = df['category'].unique()

        for category in categories:
            rep_df = self.select_representatives(df, category, text_column)
            if len(rep_df) > 0:
                result[category] = rep_df

        return result
