"""检索结果重排序模型实现。

支持：
- Embedding Reranker: 使用 embedding 模型重新编码并计算相似度
- Qwen3-VL-Reranker: 专门的视觉语言重排序模型
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import BaseReranker


def is_qwen3_vl_reranker(model_name: str) -> bool:
    """判断是否为 Qwen3-VL-Reranker 模型。"""
    return "reranker" in model_name.lower() and "qwen" in model_name.lower()


class Reranker(BaseReranker):
    """统一的 Reranker 接口。

    自动检测模型类型：
    - Qwen3-VL-Reranker: 专门的 VL reranker
    - 其他: 作为 embedding reranker 使用
    """

    def __init__(
        self,
        model_name: str,
        device: str | None = None,
        **kwargs,
    ) -> None:
        """初始化 Reranker。

        Args:
            model_name: 模型名称或路径
            device: 设备（cuda/cpu），默认自动检测
            **kwargs: 传递给底层实现的其他参数
        """
        self.model_name = model_name
        self.device = device
        self._backend: str
        self._impl: Any = None

        # 检测模型类型
        if is_qwen3_vl_reranker(model_name):
            self._backend = "qwen3_vl_reranker"
            self._init_qwen3_vl_reranker(**kwargs)
        else:
            self._backend = "embedding"
            self._init_embedding_reranker(**kwargs)

    def _init_qwen3_vl_reranker(self, **kwargs) -> None:
        """初始化 Qwen3-VL-Reranker。"""
        from .qwen3_vl_reranker import Qwen3VLReranker as _Qwen3VLRerankerImpl

        self._impl = _Qwen3VLRerankerImpl(
            model_name_or_path=self.model_name,
            device=self.device,
            **kwargs,
        )

    def _init_embedding_reranker(self, **kwargs) -> None:
        """初始化 Embedding Reranker。"""
        from .embedding import MultimodalEmbeddingModel

        self._impl = MultimodalEmbeddingModel(
            model_name=self.model_name,
            device=self.device,
            **kwargs,
        )

    def process(self, inputs: dict[str, Any]) -> list[float]:
        """对检索结果进行重排序。

        Args:
            inputs: 包含以下字段的字典：
                - query: dict，包含 text/image/video
                - documents: list[dict]，每个包含 text/image/video
                - instruction: str（可选，仅 VL reranker 使用）

        Returns:
            每个文档的相关性分数列表
        """
        if self._backend == "qwen3_vl_reranker":
            return self._impl.process(inputs)
        else:
            # Embedding reranker: 计算 query 和 documents 的相似度
            return self._process_embedding(inputs)

    def _process_embedding(self, inputs: dict[str, Any]) -> list[float]:
        """使用 embedding 模型进行重排序。"""
        query = inputs.get("query", {})
        documents = inputs.get("documents", [])

        if not query or not documents:
            return []

        # 编码 query
        query_text = query.get("text")
        if not query_text:
            raise ValueError("Embedding reranker requires query text")

        query_emb = self._impl.encode_texts([query_text]).embeddings[0]

        # 编码 documents（假设是图片路径）
        doc_paths = []
        for doc in documents:
            if "image" in doc:
                doc_paths.append(doc["image"])
            else:
                raise ValueError("Embedding reranker requires document images")

        doc_embs = self._impl.encode_images(doc_paths).embeddings

        # 计算相似度
        scores = np.dot(doc_embs, query_emb).tolist()
        return scores

    @property
    def backend(self) -> str:
        """返回当前使用的后端类型。"""
        return self._backend

