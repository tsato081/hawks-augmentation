"""
非同期LLM生成モジュール
AsyncOpenAI APIを使用した並列テキスト生成
"""

import asyncio
import json
import re
import os
from typing import List, Dict, Optional


class AsyncLLMGenerator:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.8,
        max_concurrent: int = 3,
        max_retries: int = 3
    ):
        """
        Args:
            api_key: OpenAI APIキー（Noneの場合は環境変数から取得）
            model: 使用するモデル名
            temperature: 生成温度（0.7-1.0推奨）
            max_concurrent: 最大同時実行数
            max_retries: 最大リトライ回数
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.temperature = temperature
        self.max_concurrent = max_concurrent
        self.max_retries = max_retries
        self.semaphore = asyncio.Semaphore(max_concurrent)

        # AsyncOpenAIクライアントの初期化
        try:
            from openai import AsyncOpenAI
            self.client = AsyncOpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("openai パッケージをインストールしてください: pip install openai")

    async def generate_async(
        self,
        messages: List[Dict[str, str]],
        num_samples: int = 3,
        attempt: int = 1
    ) -> List[Dict[str, str]]:
        """
        非同期でLLM生成

        Args:
            messages: プロンプトメッセージ
            num_samples: 生成するサンプル数
            attempt: 試行回数

        Returns:
            生成されたサンプルのリスト
        """
        async with self.semaphore:
            if attempt > 1:
                print(f"  リトライ中 (試行 {attempt}/{self.max_retries})")

            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=2000,
                    presence_penalty=0.3,
                    frequency_penalty=0.2
                )

                content = response.choices[0].message.content.strip()
                samples = self._parse_json_response(content)

                if samples and len(samples) > 0:
                    return samples[:num_samples]

            except Exception as e:
                print(f"  生成エラー (試行 {attempt}/{self.max_retries}): {e}")
                if attempt < self.max_retries:
                    await asyncio.sleep(2 ** attempt)  # 指数バックオフ
                    return await self.generate_async(messages, num_samples, attempt + 1)

        return []

    async def generate_batch_async(
        self,
        prompts: List[List[Dict[str, str]]],
        num_samples_per_prompt: int = 3
    ) -> List[List[Dict[str, str]]]:
        """
        複数プロンプトを並列生成

        Args:
            prompts: プロンプトメッセージのリスト
            num_samples_per_prompt: 1プロンプトあたりのサンプル数

        Returns:
            生成結果のリスト
        """
        tasks = [
            self.generate_async(messages, num_samples_per_prompt, 1)
            for messages in prompts
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # エラーハンドリング
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"  バッチエラー: プロンプト{i}: {result}")
                final_results.append([])
            else:
                final_results.append(result)

        return final_results

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
            print(f"  JSON パースエラー: {e}")
            print(f"  レスポンス: {content[:200]}...")

        return []
