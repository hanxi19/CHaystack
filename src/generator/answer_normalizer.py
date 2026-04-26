from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Sequence


@dataclass
class NormalizeConfig:
    """Configuration for answer normalization."""

    max_length: int = 256
    strip_whitespace: bool = True
    normalize_unicode: bool = True
    normalize_punctuation: bool = True
    normalize_case: bool = True
    remove_redundant_spaces: bool = True


class AnswerNormalizer:
    """Normalize generated answers for DocumentVQA.

    Handles common post-processing needs:
    - Unicode normalization (full-width/half-width)
    - Punctuation standardization
    - Case normalization
    - Redundant whitespace removal
    - Length truncation
    """

    # 全角转半角映射
    FULLWIDTH_TO_HALFWIDTH = str.maketrans(
        "！＂＃＄％＆＇（）＊＋，－．／０１２３４５６７８９：；＜＝＞？＠ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ［＼］＾＿｀ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ｛｜｝～",
        "!\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~",
    )

    # 中文标点映射（弯引号用 \\u 写出，避免与 Python 字符串定界符混淆）
    CN_PUNCTUATION_MAP = str.maketrans(
        "，。！？；：\u201c\u201d\u2018\u2019（）【】《》",
        ",.!?;:\"\"''()[]<>",
    )

    def __init__(self, config: NormalizeConfig | None = None) -> None:
        self.config = config or NormalizeConfig()

    def normalize(self, text: str) -> str:
        """Normalize answer text."""
        if not text:
            return ""

        # 基础清理
        text = text.strip()

        # Unicode 标准化
        if self.config.normalize_unicode:
            text = unicodedata.normalize("NFKC", text)

        # 全角转半角（英文数字标点）
        text = text.translate(self.FULLWIDTH_TO_HALFWIDTH)

        # 中文标点可选转换
        if self.config.normalize_punctuation:
            text = text.translate(self.CN_PUNCTUATION_MAP)

        # 大小写统一（英文字母）
        if self.config.normalize_case:
            text = text.lower()

        # 去除多余空白
        if self.config.remove_redundant_spaces:
            text = re.sub(r"\s+", " ", text)

        # 首尾清理
        if self.config.strip_whitespace:
            text = text.strip()

        # 截断过长答案
        if len(text) > self.config.max_length:
            text = text[: self.config.max_length].rsplit(" ", 1)[0]

        return text

    def is_uncertain_answer(self, text: str) -> bool:
        """Check if the answer indicates uncertainty."""
        uncertain_patterns = [
            r"无法确定",
            r"无法回答",
            r"不知道",
            r"不确定",
            r"没有答案",
            r"未提及",
            r"无法找到",
            r"not\s+(?:sure|found|available)",
            r"cannot\s+(?:determine|answer|find)",
            r"unable\s+to",
            r"no\s+(?:answer|information)",
        ]
        text_lower = text.lower().strip()
        for pattern in uncertain_patterns:
            if re.search(pattern, text_lower):
                return True
        return False

    def clean_extracted_answer(
        self,
        text: str,
        remove_prefixes: Sequence[str] | None = None,
    ) -> str:
        """Clean answer by removing common prefixes/suffixes."""
        text = text.strip()

        # 去除常见前缀
        prefixes = remove_prefixes or [
            "答案：",
            "答案是",
            "回答：",
            "答案是：",
            "answer:",
            "the answer is",
            "答案是",
        ]
        for prefix in prefixes:
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix) :].strip()
                break

        # 去除解释性后缀（以"因为"、"由于"等开头的句子）
        explanation_markers = ["因为", "由于", "原因是", "based on", "because", "since"]
        for marker in explanation_markers:
            idx = text.lower().find(marker.lower())
            if idx > 0:
                text = text[:idx].strip()
                break

        return self.normalize(text)


# 默认实例，方便直接使用
default_normalizer = AnswerNormalizer()


def normalize_answer(text: str) -> str:
    """Convenience function using default normalizer."""
    return default_normalizer.normalize(text)
