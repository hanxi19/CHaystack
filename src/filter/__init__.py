"""Filter 模块。

使用 VLM generator 对检索结果进行相关性判断，过滤掉无法回答问题的图片。
支持持久化缓存，多个 generator 可共享同一过滤结果。
"""

from .relevance_filter import FilterCache, FilterConfig, FilterResult, RelevanceFilter

__all__ = [
    "RelevanceFilter",
    "FilterCache",
    "FilterConfig",
    "FilterResult",
]
