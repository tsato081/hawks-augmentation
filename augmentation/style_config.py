"""
スタイル定義と配分設定モジュール
文体プリセット、文長ガイド、カテゴリ×スタイル配分を管理
"""

from typing import Dict, List, Tuple

# スタイルプリセット定義
STYLE_PRESETS = {
    "ニュース": {
        "description": "報道調の客観的な文体",
        "title_length": None,  # 制限なし
        "body_length": (120, 600),
        "tone": "である調、事実を淡々と伝える"
    },
    "ブログ": {
        "description": "個人的な視点を含む解説調",
        "title_length": None,  # 制限なし
        "body_length": (120, 600),
        "tone": "です・ます調、読者に語りかける"
    },
    "プレスリリース": {
        "description": "企業公式発表の形式",
        "title_length": None,  # 制限なし
        "body_length": (120, 600),
        "tone": "丁寧語、正式な表現"
    },
    "議事録": {
        "description": "会議や議論の記録",
        "title_length": None,  # 制限なし
        "body_length": (120, 600),
        "tone": "簡潔、箇条書き的"
    },
    "行政告知": {
        "description": "自治体や公的機関の通知",
        "title_length": None,  # 制限なし
        "body_length": (120, 600),
        "tone": "公的文書調、明確な指示"
    },
    "QA": {
        "description": "質問と回答の形式",
        "title_length": None,  # 制限なし
        "body_length": (120, 600),
        "tone": "Q:〜、A:〜の構造"
    },
    "SNS投稿": {
        "description": "SNS風の短文投稿",
        "title_length": None,  # 制限なし
        "body_length": (120, 300),
        "tone": "カジュアル、絵文字なし"
    },
    "コラム": {
        "description": "専門家の解説・意見",
        "title_length": None,  # 制限なし
        "body_length": (120, 600),
        "tone": "である調、分析的"
    },
    "FAQ": {
        "description": "よくある質問集",
        "title_length": None,  # 制限なし
        "body_length": (120, 600),
        "tone": "Q&A形式、明瞭"
    },
    "商品案内": {
        "description": "製品・サービス紹介",
        "title_length": None,  # 制限なし
        "body_length": (120, 600),
        "tone": "魅力を伝える、セールス調は避ける"
    },
    "社内連絡": {
        "description": "社内向けの通知文",
        "title_length": None,  # 制限なし
        "body_length": (120, 600),
        "tone": "簡潔、要点明確"
    }
}

# スタイル配分上限（全カテゴリ合算）
STYLE_DISTRIBUTION_LIMITS = {
    "ニュース": 0.25,          # 25%上限
    "ブログ": 0.30,            # 20-30%
    "コラム": 0.30,            # ブログと合わせて管理
    "プレスリリース": 0.15,    # 10-15%
    "行政告知": 0.30,          # 議事録・社内連絡と合計30%
    "議事録": 0.30,
    "社内連絡": 0.30,
    "QA": 0.15,               # FAQ・SNSと合計15%
    "FAQ": 0.15,
    "SNS投稿": 0.15,
    "商品案内": 0.10
}

# カテゴリ毎の最低スタイル数
MIN_STYLES_PER_CATEGORY = 5

# 各スタイルの最低件数
MIN_SAMPLES_PER_STYLE = 5

# カテゴリ毎の許可スタイル（指定がない場合は全スタイル許可）
CATEGORY_ALLOWED_STYLES = {
    "債権譲渡": ["ニュース"],  # ニュース調のみ
    # 他のカテゴリは全スタイル許可（追加する場合はここに記載）
}


def get_style_allocation(total_samples: int, category_count: int) -> Dict[str, int]:
    """
    総サンプル数とカテゴリ数から、各スタイルへの配分数を計算

    Args:
        total_samples: 生成する総サンプル数
        category_count: カテゴリ数

    Returns:
        スタイル名: 配分数のdict
    """
    allocation = {}
    remaining = total_samples

    # 上限に基づいて配分
    for style, limit in STYLE_DISTRIBUTION_LIMITS.items():
        max_count = int(total_samples * limit)
        # 最低件数を保証
        min_count = max(MIN_SAMPLES_PER_STYLE, category_count // 2)
        allocated = max(min_count, min(max_count, remaining // len(STYLE_PRESETS)))
        allocation[style] = allocated
        remaining -= allocated

    # 残りを均等配分
    if remaining > 0:
        styles = list(STYLE_PRESETS.keys())
        per_style = remaining // len(styles)
        for style in styles[:remaining % len(styles)]:
            allocation[style] = allocation.get(style, 0) + per_style + 1
        for style in styles[remaining % len(styles):]:
            allocation[style] = allocation.get(style, 0) + per_style

    return allocation


def get_style_prompt(style_name: str) -> str:
    """スタイルに応じたプロンプト指示を生成"""
    if style_name not in STYLE_PRESETS:
        style_name = "ニュース"  # デフォルト

    style = STYLE_PRESETS[style_name]
    title_length = style["title_length"]
    body_min, body_max = style["body_length"]

    # タイトル文字数指示（Noneの場合は省略）
    title_instruction = "" if title_length is None else f"- タイトル文字数: 適度な長さ（目安: 15-50字程度）\n"

    return f"""
文体スタイル: {style_name}
- {style['description']}
- 口調: {style['tone']}
{title_instruction}- 本文文字数: {body_min}〜{body_max}字
"""


def validate_length(text: str, length_range: Tuple[int, int]) -> bool:
    """文字数が範囲内かチェック"""
    min_len, max_len = length_range
    text_len = len(text)
    return min_len <= text_len <= max_len


def get_category_style_matrix(categories: List[str], samples_per_category: Dict[str, int]) -> Dict[str, Dict[str, int]]:
    """
    カテゴリ×スタイルの配分マトリクスを生成

    Args:
        categories: カテゴリリスト
        samples_per_category: カテゴリ毎の生成数

    Returns:
        {category: {style: count}}の2次元dict
    """
    matrix = {}

    for category in categories:
        needed = samples_per_category.get(category, 0)
        if needed == 0:
            continue

        # カテゴリ毎の許可スタイルを取得（指定がない場合は全スタイル）
        if category in CATEGORY_ALLOWED_STYLES:
            allowed_styles = CATEGORY_ALLOWED_STYLES[category]
        else:
            allowed_styles = list(STYLE_PRESETS.keys())

        # 最低スタイル数を保証（ただし許可スタイル数を超えない）
        num_styles = max(1, min(MIN_STYLES_PER_CATEGORY, len(allowed_styles), needed // MIN_SAMPLES_PER_STYLE))
        selected_styles = allowed_styles[:num_styles]

        # 配分計算
        base_count = needed // num_styles
        remainder = needed % num_styles

        style_counts = {}
        for i, style in enumerate(selected_styles):
            count = base_count + (1 if i < remainder else 0)
            style_counts[style] = count

        matrix[category] = style_counts

    return matrix
