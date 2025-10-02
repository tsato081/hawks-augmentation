"""
品質フィルタモジュール
類似度チェック、カテゴリ整合性、言語品質、安全性を検証
"""

import re
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


class QualityFilter:
    def __init__(
        self,
        cosine_threshold: float = 0.92,
        ngram_threshold: float = 0.35,
        min_body_length: int = 120,
        max_body_length: int = 600
    ):
        """
        Args:
            cosine_threshold: コサイン類似度の上限
            ngram_threshold: 4-gram重複率の上限
            min_body_length: 本文の最小文字数
            max_body_length: 本文の最大文字数
        """
        self.cosine_threshold = cosine_threshold
        self.ngram_threshold = ngram_threshold
        self.min_body_length = min_body_length
        self.max_body_length = max_body_length

        # TF-IDFベクトライザー（類似度計算用）
        self.vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 3))

        # 禁止ワードリスト（実名・固有名詞のパターン）
        self.forbidden_patterns = [
            r'https?://[^\s]+',  # URL
            r'\d{4}年\d{1,2}月\d{1,2}日',  # 具体的な日付
            r'〒\d{3}-\d{4}',  # 郵便番号
            r'電話.*\d{2,4}-\d{2,4}-\d{4}',  # 電話番号
        ]

    def check_similarity(
        self,
        generated_text: str,
        reference_texts: List[str]
    ) -> Tuple[bool, float]:
        """
        コサイン類似度チェック

        Returns:
            (合格判定, 最大類似度)
        """
        if not reference_texts:
            return True, 0.0

        try:
            # ベクトル化
            all_texts = [generated_text] + reference_texts
            vectors = self.vectorizer.fit_transform(all_texts)

            # 生成テキストと参照テキスト間の類似度
            gen_vec = vectors[0:1]
            ref_vecs = vectors[1:]
            similarities = cosine_similarity(gen_vec, ref_vecs)[0]

            max_sim = similarities.max()
            passed = max_sim < self.cosine_threshold

            return passed, float(max_sim)

        except Exception as e:
            print(f"類似度計算エラー: {e}")
            return True, 0.0

    def check_ngram_overlap(
        self,
        generated_text: str,
        reference_texts: List[str],
        n: int = 4
    ) -> Tuple[bool, float]:
        """
        N-gram重複率チェック

        Returns:
            (合格判定, 最大重複率)
        """
        if not reference_texts:
            return True, 0.0

        # 4-gramを抽出
        def get_ngrams(text: str, n: int) -> set:
            text = text.replace(' ', '').replace('\n', '')
            return set(text[i:i+n] for i in range(len(text) - n + 1))

        gen_ngrams = get_ngrams(generated_text, n)
        if not gen_ngrams:
            return True, 0.0

        max_overlap = 0.0
        for ref_text in reference_texts:
            ref_ngrams = get_ngrams(ref_text, n)
            if not ref_ngrams:
                continue

            overlap = len(gen_ngrams & ref_ngrams) / len(gen_ngrams)
            max_overlap = max(max_overlap, overlap)

        passed = max_overlap < self.ngram_threshold
        return passed, max_overlap

    def check_length(
        self,
        title: str,
        body: str,
        title_range: Optional[Tuple[int, int]] = None,
        body_range: Optional[Tuple[int, int]] = None
    ) -> Tuple[bool, str]:
        """文字数チェック"""
        title_len = len(title)
        body_len = len(body)

        # デフォルトレンジ
        if title_range is None:
            title_range = (10, 50)
        if body_range is None:
            body_range = (self.min_body_length, self.max_body_length)

        # タイトルチェック
        if not (title_range[0] <= title_len <= title_range[1]):
            return False, f"タイトル文字数が範囲外: {title_len}字 (範囲: {title_range[0]}-{title_range[1]})"

        # 本文チェック
        if not (body_range[0] <= body_len <= body_range[1]):
            return False, f"本文文字数が範囲外: {body_len}字 (範囲: {body_range[0]}-{body_range[1]})"

        return True, ""

    def check_language_quality(self, text: str) -> Tuple[bool, str]:
        """言語品質チェック"""
        # 日本語率
        japanese_chars = sum(1 for c in text if '\u3000' <= c <= '\u9fff')
        if len(text) > 0:
            jp_ratio = japanese_chars / len(text)
            if jp_ratio < 0.6:
                return False, f"日本語率が低い: {jp_ratio:.2f}"

        # 句点の存在（最低1つ）
        if '。' not in text and '.' not in text:
            return False, "句点が存在しない"

        # 平均文長チェック（極端に長い・短い文を検出）
        sentences = text.split('。')
        sentences = [s.strip() for s in sentences if s.strip()]
        if sentences:
            avg_len = sum(len(s) for s in sentences) / len(sentences)
            if avg_len < 10 or avg_len > 200:
                return False, f"平均文長が異常: {avg_len:.1f}字"

        return True, ""

    def check_safety(self, text: str) -> Tuple[bool, str]:
        """安全性チェック（禁止パターンの検出）"""
        for pattern in self.forbidden_patterns:
            if re.search(pattern, text):
                return False, f"禁止パターン検出: {pattern}"

        # 実名っぽいパターン（カタカナ+さん/氏/社長など）
        if re.search(r'[ァ-ヶー]{2,}(さん|氏|社長|代表)', text):
            return False, "実名の可能性があるパターン検出"

        return True, ""

    def validate_sample(
        self,
        sample: Dict[str, str],
        reference_bodies: List[str],
        title_range: Optional[Tuple[int, int]] = None,
        body_range: Optional[Tuple[int, int]] = None
    ) -> Tuple[bool, Dict[str, any]]:
        """
        サンプルを総合的に検証

        Returns:
            (合格判定, 検証結果の詳細)
        """
        title = sample.get('title', '')
        body = sample.get('body', '')
        category = sample.get('category', '')

        result = {
            'passed': True,
            'reasons': [],
            'metrics': {}
        }

        # 1. 文字数チェック
        length_ok, length_msg = self.check_length(title, body, title_range, body_range)
        if not length_ok:
            result['passed'] = False
            result['reasons'].append(length_msg)

        # 2. 類似度チェック
        sim_ok, max_sim = self.check_similarity(body, reference_bodies)
        result['metrics']['cosine_similarity'] = max_sim
        if not sim_ok:
            result['passed'] = False
            result['reasons'].append(f"類似度が高すぎる: {max_sim:.3f}")

        # 3. N-gram重複チェック
        ngram_ok, max_overlap = self.check_ngram_overlap(body, reference_bodies)
        result['metrics']['ngram_overlap'] = max_overlap
        if not ngram_ok:
            result['passed'] = False
            result['reasons'].append(f"4-gram重複が多すぎる: {max_overlap:.3f}")

        # 4. 言語品質チェック
        lang_ok, lang_msg = self.check_language_quality(body)
        if not lang_ok:
            result['passed'] = False
            result['reasons'].append(lang_msg)

        # 5. 安全性チェック
        safety_ok_title, safety_msg_title = self.check_safety(title)
        safety_ok_body, safety_msg_body = self.check_safety(body)
        if not safety_ok_title:
            result['passed'] = False
            result['reasons'].append(f"タイトル: {safety_msg_title}")
        if not safety_ok_body:
            result['passed'] = False
            result['reasons'].append(f"本文: {safety_msg_body}")

        return result['passed'], result

    def filter_batch(
        self,
        samples: List[Dict[str, str]],
        reference_bodies: List[str],
        title_range: Optional[Tuple[int, int]] = None,
        body_range: Optional[Tuple[int, int]] = None
    ) -> Tuple[List[Dict[str, str]], List[Dict]]:
        """
        バッチでフィルタリング

        Returns:
            (合格サンプル, 不合格情報)
        """
        passed_samples = []
        rejected_info = []

        for i, sample in enumerate(samples):
            is_valid, validation_result = self.validate_sample(
                sample,
                reference_bodies,
                title_range,
                body_range
            )

            if is_valid:
                passed_samples.append(sample)
            else:
                rejected_info.append({
                    'index': i,
                    'sample': sample,
                    'reasons': validation_result['reasons'],
                    'metrics': validation_result.get('metrics', {})
                })

        return passed_samples, rejected_info
