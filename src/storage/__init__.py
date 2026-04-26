"""存储模块。

提供向量索引的存储和管理：
- vector_store: 单个向量索引的存储和检索
- multi_index: 多索引管理（类别感知）
"""

from .multi_index import CategoryAwareIndexManager
from .vector_store import NumpyVectorStore, SearchResult

__all__ = [
    "NumpyVectorStore",
    "SearchResult",
    "CategoryAwareIndexManager",
]
