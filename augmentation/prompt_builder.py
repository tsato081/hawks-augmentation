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
        return """あなたはデータ拡張の専門家です。以下の要件に従って、カテゴリ整合性を保ちつつ文体多様性のある学習データを生成してください。

【重要な制約】
1. カテゴリ定義に厳密に従い、指定されたカテゴリに該当する内容のみを生成する
2. **必ず具体的な企業名・団体名・人物名を含めること**（架空のものでOK、例: 株式会社○○、△△社長、など）
3. 実在の企業や個人は使用禁止（必ず架空の名前を作成すること）
4. 具体的な日付・URL・詳細な住所は使用禁止（「○○年○月」「関東地方」程度はOK）
5. 元のサンプルの語順や決まり文句を模倣しない（内容の本質のみを参考にする）
6. 指定されたスタイルと文字数範囲を厳守する
7. 出力はJSON形式のみ（説明文は不要）

【生成の方針】
- **誰が何をしたのか**を明確に記述する（主語となる企業名・人物名は必須）
- ニュース調への過度な偏りを避け、多様な文体で表現する
- 実在の事例は参考にしつつ、架空のシナリオとして再構成する
- 企業名は「○○株式会社」「△△ホールディングス」など具体的に
- 人物名は「○○氏」「△△社長」など役職付きで
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

【必須要件】
- **架空の企業名・団体名を必ず含める**（例: 株式会社○○、△△工業、など）
- **架空の人物名・役職を必ず含める**（例: ○○社長、△△氏、など）
- **誰が・何をした**という主語と述語を明確に

【禁止事項】
- 実在の人名・企業名・団体名の使用
- 特定の日付・URL・詳細な住所の記載
- 元サンプルの文章構造の模倣

【Few-shot例】
以下は同カテゴリの参考例です（内容の本質のみを参考にし、表現は独自に）：

"""
        # Few-shot例を追加
        for i, example in enumerate(few_shot_examples[:3], 1):
            title = example.get('title', '').strip() or '（タイトルなし）'
            body = example.get('body', '').strip()[:200]  # 200文字まで
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
