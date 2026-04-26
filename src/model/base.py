"""模型基类定义。

定义三类模型的统一接口，确保所有实现类遵循相同的协议。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image


@dataclass
class EmbeddingOutput:
    """向量编码输出。"""

    embeddings: np.ndarray
    model_name: str
    image_paths: list[str] | None = None


class BaseEmbeddingModel(ABC):
    """图文检索的向量编码模型基类。

    支持的模型类型：
    - CLIP 系列（AltCLIP 等）
    - SigLIP 系列
    - Qwen3-VL-Embedding
    """

    @abstractmethod
    def encode_texts(
        self,
        texts: Sequence[str],
        batch_size: int = 32,
        **kwargs,
    ) -> EmbeddingOutput:
        """编码文本为向量。"""
        pass

    @abstractmethod
    def encode_images(
        self,
        image_paths: Sequence[str],
        batch_size: int = 16,
        skip_bad_images: bool = False,
        **kwargs,
    ) -> EmbeddingOutput:
        """编码图片为向量。"""
        pass

    @abstractmethod
    def embedding_dim(self) -> int:
        """返回向量维度。"""
        pass


class BaseReranker(ABC):
    """检索结果重排序模型基类。

    支持的模型类型：
    - Qwen3-VL-Reranker
    """

    @abstractmethod
    def process(self, inputs: dict[str, Any]) -> list[float]:
        """对检索结果进行重排序。

        Args:
            inputs: 包含 query 和 documents 的字典

        Returns:
            每个文档的相关性分数列表
        """
        pass


class BaseGenerator(ABC):
    """文档问答的答案生成模型基类。

    支持的模型类型：
    - Qwen2-VL
    - Qwen2.5-VL
    - Qwen3-VL
    """

    @abstractmethod
    def generate(
        self,
        question: str,
        image_paths: Sequence[str | Path],
    ) -> dict[str, Any]:
        """根据问题和图片生成答案。

        Args:
            question: 问题文本
            image_paths: 文档图片路径列表

        Returns:
            包含 answer, raw_answer, is_uncertain, num_images 的字典
        """
        pass

    @abstractmethod
    def batch_generate(
        self,
        questions: Sequence[str],
        image_paths_list: Sequence[Sequence[str | Path]],
    ) -> list[dict[str, Any]]:
        """批量生成答案。"""
        pass
