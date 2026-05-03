"""统一的模型管理模块。

提供三类模型的统一接口：
- Embedding: 图文检索的向量编码（CLIP/SigLIP/Qwen3-VL-Embedding）
- Reranker: 检索结果的重排序（自动检测：Embedding 或 Qwen3-VL-Reranker）
- Generator: 文档问答的答案生成（Qwen-VL / LLaVA / InternVL）
"""

from .base import BaseEmbeddingModel, BaseGenerator, BaseReranker
from .embedding import MultimodalEmbeddingModel
from .generator import (
    GeneratorConfig,
    GeneratorFactory,
    InternVLGenerator,
    LLaVAGenerator,
    QwenVLGenerator,
)
from .reranker import Reranker

__all__ = [
    # 基类
    "BaseEmbeddingModel",
    "BaseReranker",
    "BaseGenerator",
    # 实现类
    "MultimodalEmbeddingModel",
    "Reranker",
    "QwenVLGenerator",
    "LLaVAGenerator",
    "InternVLGenerator",
    "GeneratorFactory",
    "GeneratorConfig",
]
