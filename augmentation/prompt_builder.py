"""
プロンプト生成モジュール
カテゴリ定義、スタイル指示、few-shot例を組み合わせてプロンプトを構築
"""

import pandas as pd
from typing import List, Dict, Optional
from pathlib import Path
import sys
import importlib.util

# カテゴリ定義の読み込み
def load_category_definitions() -> Dict[str, str]:
    """カテゴリ定義を読み込み"""
    # 相対パスでカテゴリ定義を読み込み（クラウド環境対応）
    base_dir = Path(__file__).parent.parent  # hawks_augmentation/
    target = base_dir / "definitions" / "category_defines.py"

    try:
        spec = importlib.util.spec_from_file_location("category_defines", str(target))
        mod = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(mod)
        return getattr(mod, "CATEGORY_DEFINE_LIST", {})
    except Exception:
        return {}


class PromptBuilder:
    def __init__(self, category_definitions: Optional[Dict[str, str]] = None):
        """
        Args:
            category_definitions: カテゴリ名: 定義のdict
        """
        if category_definitions is None:
            category_definitions = load_category_definitions()
        self.category_definitions = category_definitions

    def build_system_prompt(self) -> str:
        """システムプロンプト生成"""
        return """あなたは企業リスク監視用の学習データ生成の専門家です。以下の要件に従って、カテゴリ整合性を保ちつつ文体多様性のある学習データを生成してください。

【Pickデータの基本要件】
すべての生成データは、企業リスク監視システムで「Pick」と判定されるべき内容です：
1. **固有名詞が必須**: 本文に企業名・組織名・ブランド名・行政機関名・実在人物名（架空）が明確に含まれること
2. **事実ベースまたは重大な疑義**: 企業のリスク管理部門が注意すべき具体的な事実・問題・批判
3. **企業リスクと連動**: 企業・組織・経営陣に関する重大リスク情報であること
4. **カテゴリ定義に厳密に従う**: 指定されたカテゴリに該当する内容のみを生成

【生成してはいけない内容（Decline相当）】
- インタビュー記事、人物紹介、経営者の哲学紹介
- 業界分析・トレンドまとめ、教育的な解説記事
- 複数トピックを寄せ集めたニュースダイジェスト
- ポジティブな宣伝・広告・製品紹介のみの内容
- 企業リスクと無関係な雑談や一般論
- 政治・行政・選挙の話で企業・法人リスクと無関係なもの
- 個人への単純な批判・誹謗のみで企業リスクが示されないもの
- ※「批判的投稿」カテゴリの場合のみ、批判内容を生成すること

【技術的制約】
1. **架空の固有名詞を使用**: 実在の企業・個人は使用禁止
2. 具体的な日付・URL・詳細な住所は使用禁止（「○○年○月」「関東地方」程度はOK）
3. 元のサンプルの語順や決まり文句を模倣しない（内容の本質のみを参考）
4. 指定されたスタイルと文字数範囲を厳守
5. 出力はJSON形式のみ（説明文は不要）

【固有名詞の生成ルール】
- 企業名: 「アクロス株式会社」「サンライズ工業」「ミライテック」など具体的な架空名
- 人物名: 「田中太郎社長」「佐藤花子氏」「山田一郎代表」など日本人名+役職
- **禁止**: 「○○社」「△△氏」「A社」などの記号的な表現は使わない
- **必須**: フルネームまたは姓+役職の形式（例: 鈴木氏、高橋社長）

【生成の方針】
- **誰が何をしたのか**を明確に記述（主語となる企業名・人物名は必須）
- 具体的な問題・リスク・事件を記述（抽象的な一般論は避ける）
- ニュース調への過度な偏りを避け、多様な文体で表現
- 自然な日本語で、機械翻訳調や不自然な表現は避ける
"""

    def build_user_prompt(
        self,
        category: str,
        style: str,
        style_instruction: str,
        few_shot_examples: List[Dict[str, str]],
        num_samples: int = 3
    ) -> str:
        """ユーザープロンプト生成"""
        # カテゴリ定義
        cat_def = self.category_definitions.get(category, "")

        prompt = f"""【生成指示】

カテゴリ: {category}
カテゴリ定義: {cat_def}

{style_instruction}

【必須要件（Pickルール準拠）】
- **架空の企業名・団体名を本文に必ず含める**
  - 良い例: 「アクロス株式会社」「サンライズ工業」「ミライテックホールディングス」
  - 悪い例: 「○○社」「△△株式会社」「A社」
- **架空の人物名・役職を本文に必ず含める**
  - 良い例: 「田中太郎社長」「佐藤花子氏」「山田代表」「鈴木専務」
  - 悪い例: 「○○氏」「△△社長」「A氏」
- **誰が・何をした**という主語と述語を明確に記述
- **企業リスクに関する具体的な事実・問題・疑義**を含める
- インタビュー、人物紹介、業界分析、教育的解説、ニュースダイジェストにならないこと

【禁止事項】
- 実在の人名・企業名・団体名の使用
- 特定の日付・URL・詳細な住所の記載
- 元サンプルの文章構造の模倣
- ポジティブな宣伝・広告のみの内容
- 企業リスクと無関係な一般論や雑談

【Few-shot例】
以下は同カテゴリの参考例です（内容の本質のみを参考にし、表現は独自に）：

"""
        # Few-shot例を追加
        for i, example in enumerate(few_shot_examples[:3], 1):
            # NaN対応: floatの場合は空文字列に変換
            title_val = example.get('title', '')
            title = str(title_val).strip() if not isinstance(title_val, float) or not pd.isna(title_val) else ''
            title = title or '（タイトルなし）'

            body_val = example.get('body', '')
            body = str(body_val).strip() if not isinstance(body_val, float) or not pd.isna(body_val) else ''
            body = body[:200]  # 200文字まで

            prompt += f"{i}. タイトル: {title}\n   本文抜粋: {body}...\n\n"

        prompt += f"""
【出力形式】
{num_samples}件のサンプルをJSON配列で出力してください。

[
  {{
    "title": "タイトル",
    "body": "本文",
    "category": "{category}",
    "style": "{style}"
  }},
  ...
]

※必ず上記のJSON形式で出力してください（コードブロックや説明文は不要）
"""
        return prompt

    def build_messages(
        self,
        category: str,
        style: str,
        style_instruction: str,
        few_shot_examples: List[Dict[str, str]],
        num_samples: int = 3
    ) -> List[Dict[str, str]]:
        """OpenAI API用のメッセージ配列を生成"""
        return [
            {
                "role": "system",
                "content": self.build_system_prompt()
            },
            {
                "role": "user",
                "content": self.build_user_prompt(
                    category,
                    style,
                    style_instruction,
                    few_shot_examples,
                    num_samples
                )
            }
        ]
