"""Generator 模块。

提供文档问答的答案生成功能：
- infer: 推理脚本
- answer_normalizer: 答案规范化
"""

from .answer_normalizer import AnswerNormalizer, NormalizeConfig

__all__ = [
    "AnswerNormalizer",
    "NormalizeConfig",
]
