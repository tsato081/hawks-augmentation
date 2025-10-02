"""
LLM生成モジュール
OpenAI APIを使用してテキスト生成
"""

import json
import time
import os
from typing import List, Dict, Optional
import re


class LLMGenerator:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.8,
        max_retries: int = 3
    ):
        """
        Args:
            api_key: OpenAI APIキー（Noneの場合は環境変数から取得）
            model: 使用するモデル名
            temperature: 生成温度（0.7-1.0推奨）
            max_retries: 最大リトライ回数
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries

        # OpenAIクライアントの初期化
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("openai パッケージをインストールしてください: pip install openai")

    def generate(
        self,
        messages: List[Dict[str, str]],
        num_samples: int = 3
    ) -> List[Dict[str, str]]:
        """
        LLMでテキスト生成

        Args:
            messages: プロンプトメッセージ
            num_samples: 生成するサンプル数

        Returns:
            生成されたサンプルのリスト
        """
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=2000,
                    presence_penalty=0.3,
                    frequency_penalty=0.2
                )

                content = response.choices[0].message.content.strip()

                # JSONの抽出とパース
                samples = self._parse_json_response(content)

                if samples and len(samples) > 0:
                    return samples[:num_samples]

            except Exception as e:
                print(f"生成エラー (試行 {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # 指数バックオフ
                continue

        return []

    def _parse_json_response(self, content: str) -> List[Dict[str, str]]:
        """レスポンスからJSON配列を抽出"""
        # コードブロックを除去
        content = re.sub(r'```json\s*|\s*```', '', content)
        content = content.strip()

        try:
            # JSON配列として直接パース
            if content.startswith('['):
                return json.loads(content)

            # 最初のJSON配列を探索
            match = re.search(r'\[[\s\S]*\]', content)
            if match:
                return json.loads(match.group(0))

            # 単一のJSONオブジェクトの場合
            if content.startswith('{'):
                obj = json.loads(content)
                return [obj]

        except json.JSONDecodeError as e:
            print(f"JSON パースエラー: {e}")
            print(f"レスポンス: {content[:200]}...")

        return []

    def generate_batch(
        self,
        prompts: List[List[Dict[str, str]]],
        num_samples_per_prompt: int = 3,
        delay: float = 0.5
    ) -> List[List[Dict[str, str]]]:
        """
        バッチ生成

        Args:
            prompts: プロンプトメッセージのリスト
            num_samples_per_prompt: 1プロンプトあたりのサンプル数
            delay: リクエスト間の待機時間（秒）

        Returns:
            生成結果のリスト
        """
        results = []
        for i, messages in enumerate(prompts):
            print(f"バッチ生成中... ({i + 1}/{len(prompts)})")
            samples = self.generate(messages, num_samples_per_prompt)
            results.append(samples)

            # レート制限対策
            if i < len(prompts) - 1:
                time.sleep(delay)

        return results
