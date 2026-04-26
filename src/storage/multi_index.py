"""多索引管理器，支持类别感知的检索。

自动管理多个类别索引，检索时根据问题类别路由到对应索引。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .vector_store import NumpyVectorStore, SearchResult


VALID_CATEGORIES = ["paper", "camera", "webpage", "advertise"]


@dataclass
class MultiIndexConfig:
    """多索引配置。"""

    index_root: str | Path
    categories: list[str]
    index_pattern: str = "benchmark_{category}"


class CategoryAwareIndexManager:
    """类别感知的多索引管理器。

    功能：
    1. 自动加载多个类别索引
    2. 根据问题类别路由到对应索引
    3. 支持跨类别检索（可选）
    """

    def __init__(
        self,
        index_root: str | Path,
        categories: list[str] | None = None,
        index_pattern: str = "benchmark_{category}",
        lazy_load: bool = True,
    ) -> None:
        """初始化多索引管理器。

        Args:
            index_root: 索引根目录
            categories: 类别列表，None 表示使用所有有效类别
            index_pattern: 索引目录命名模式，{category} 会被替换为类别名
            lazy_load: 是否延迟加载索引（首次使用时才加载）
        """
        self.index_root = Path(index_root)
        self.categories = categories or VALID_CATEGORIES
        self.index_pattern = index_pattern
        self.lazy_load = lazy_load

        # 索引缓存
        self._stores: dict[str, NumpyVectorStore] = {}

        # 验证类别
        for cat in self.categories:
            if cat not in VALID_CATEGORIES:
                raise ValueError(f"无效类别: {cat}，可选: {VALID_CATEGORIES}")

        # 非延迟加载模式：立即加载所有索引
        if not lazy_load:
            self._load_all_indexes()

    def _get_index_dir(self, category: str) -> Path:
        """获取类别对应的索引目录。"""
        return self.index_root / self.index_pattern.format(category=category)

    def _load_index(self, category: str) -> NumpyVectorStore:
        """加载指定类别的索引。"""
        if category in self._stores:
            return self._stores[category]

        index_dir = self._get_index_dir(category)
        if not index_dir.exists():
            raise FileNotFoundError(
                f"类别 '{category}' 的索引不存在: {index_dir}\n"
                f"请先运行: bash scripts/build_index.sh"
            )

        store = NumpyVectorStore.load(index_dir)
        self._stores[category] = store
        return store

    def _load_all_indexes(self) -> None:
        """加载所有类别的索引。"""
        for category in self.categories:
            try:
                self._load_index(category)
            except FileNotFoundError as e:
                print(f"[warning] {e}")

    def get_store(self, category: str) -> NumpyVectorStore:
        """获取指定类别的向量存储。"""
        if category not in self.categories:
            raise ValueError(f"未配置类别: {category}，可用: {self.categories}")
        return self._load_index(category)

    def search(
        self,
        query_embedding: Any,
        category: str,
        top_k: int = 5,
    ) -> list[SearchResult]:
        """在指定类别的索引中检索。

        Args:
            query_embedding: 查询向量
            category: 目标类别
            top_k: 返回结果数

        Returns:
            检索结果列表
        """
        store = self.get_store(category)
        return store.search(query_embedding, top_k)

    def search_all(
        self,
        query_embedding: Any,
        top_k: int = 5,
        per_category_k: int | None = None,
    ) -> dict[str, list[SearchResult]]:
        """在所有类别的索引中检索。

        Args:
            query_embedding: 查询向量
            top_k: 每个类别返回的结果数（当 per_category_k 为 None 时）
            per_category_k: 每个类别的 top-k，覆盖 top_k

        Returns:
            字典，key 为类别名，value 为检索结果列表
        """
        k = per_category_k if per_category_k is not None else top_k
        results = {}
        for category in self.categories:
            try:
                store = self.get_store(category)
                results[category] = store.search(query_embedding, k)
            except FileNotFoundError:
                results[category] = []
        return results

    def get_available_categories(self) -> list[str]:
        """获取可用的类别列表（索引已存在）。"""
        available = []
        for category in self.categories:
            index_dir = self._get_index_dir(category)
            if index_dir.exists():
                available.append(category)
        return available

    def get_stats(self) -> dict[str, Any]:
        """获取所有索引的统计信息。"""
        stats = {}
        for category in self.categories:
            try:
                store = self.get_store(category)
                stats[category] = {
                    "num_images": len(store.metadata),
                    "embedding_dim": store.config.get("embedding_dim"),
                    "model_name": store.config.get("model_name"),
                    "index_dir": str(self._get_index_dir(category)),
                }
            except FileNotFoundError:
                stats[category] = {"status": "not_found"}
        return stats

    @classmethod
    def from_config_file(cls, config_path: str | Path) -> "CategoryAwareIndexManager":
        """从配置文件加载。

        配置文件格式（JSON）：
        {
            "index_root": "/path/to/indexes",
            "categories": ["paper", "camera", "webpage", "advertise"],
            "index_pattern": "benchmark_{category}",
            "lazy_load": true
        }
        """
        config_path = Path(config_path)
        with config_path.open("r", encoding="utf-8") as f:
            config = json.load(f)

        return cls(
            index_root=config["index_root"],
            categories=config.get("categories"),
            index_pattern=config.get("index_pattern", "benchmark_{category}"),
            lazy_load=config.get("lazy_load", True),
        )

    def save_config(self, config_path: str | Path) -> None:
        """保存配置到文件。"""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        config = {
            "index_root": str(self.index_root),
            "categories": self.categories,
            "index_pattern": self.index_pattern,
            "lazy_load": self.lazy_load,
        }

        with config_path.open("w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
