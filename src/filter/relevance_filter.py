"""Relevance filter: judge whether a document image can answer a given question.

Uses a VLM generator to produce YES/NO judgments on each candidate image,
then returns only the relevant ones.  Supports persistent caching so that
filter results can be reused across different generator runs.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch

from ..model import BaseGenerator, GeneratorConfig, GeneratorFactory


DEFAULT_FILTER_SYSTEM_PROMPT = """你是一个文档相关性判断助手。你的任务是判断一张文档图片是否包含回答特定问题所需的信息。

规则：
1. 仔细观察图片中的所有文字、图表和布局信息
2. 判断这些信息是否足以回答给定的问题
3. 如果图片包含回答问题的关键证据或线索，回答 YES
4. 如果图片不包含相关信息、信息不完整、或无法从中推断答案，回答 NO

请只回答 YES 或 NO，不要解释。"""


# ---------------------------------------------------------------------------
# Filter cache
# ---------------------------------------------------------------------------


class FilterCache:
    """Persistent cache for filter judgments.

    Keyed by ``(question_hash, image_path)`` so the same filter results can be
    shared across multiple generator runs.
    """

    def __init__(self, cache_path: str | Path | None = None) -> None:
        self._data: dict[str, dict[str, Any]] = {}
        self._path = Path(cache_path) if cache_path else None
        self._dirty = False
        if self._path and self._path.exists():
            self._load()

    def _load(self) -> None:
        if not self._path:
            return
        try:
            with self._path.open("r", encoding="utf-8") as f:
                self._data = json.load(f)
        except (json.JSONDecodeError, OSError):
            self._data = {}

    def save(self) -> None:
        if not self._path:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False)

    @staticmethod
    def _make_key(question: str, image_path: str) -> str:
        qhash = hashlib.sha256(question.encode()).hexdigest()[:16]
        phash = hashlib.sha256(image_path.encode()).hexdigest()[:16]
        return f"{qhash}_{phash}"

    def get(self, question: str, image_path: str) -> dict[str, Any] | None:
        key = self._make_key(question, image_path)
        return self._data.get(key)

    def put(
        self, question: str, image_path: str,
        is_relevant: bool, raw_response: str = "", error: str | None = None,
    ) -> None:
        key = self._make_key(question, image_path)
        self._data[key] = {
            "is_relevant": is_relevant,
            "raw_response": raw_response,
            "error": error,
        }
        self._dirty = True

    def __len__(self) -> int:
        return len(self._data)


# ---------------------------------------------------------------------------
# Config / Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class FilterResult:
    """Single filter judgment result."""

    image_path: str
    is_relevant: bool
    raw_response: str = ""
    error: str | None = None
    cached: bool = False


@dataclass
class FilterConfig:
    """Configuration for the relevance filter."""

    model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    device: str | None = None
    max_new_tokens: int = 8
    system_prompt: str = DEFAULT_FILTER_SYSTEM_PROMPT
    trust_remote_code: bool = True
    cache_path: str | None = None
    concurrency: int = 1  # reserved for future batched filtering
    skip_on_error: bool = True


# ---------------------------------------------------------------------------
# RelevanceFilter
# ---------------------------------------------------------------------------


class RelevanceFilter:
    """Filter candidates by asking a VLM whether each image answers the question.

    Typical usage::

        config = FilterConfig(
            model_name="Qwen/Qwen2.5-VL-3B-Instruct",
            cache_path="/path/to/filter_cache.json",
        )
        filt = RelevanceFilter(config)

        candidates = ["/path/to/img1.png", "/path/to/img2.png"]
        results = filt.filter("发票号码是多少？", candidates)
        relevant = [r.image_path for r in results if r.is_relevant]
        filt.save_cache()  # persist for later reuse
    """

    def __init__(self, config: FilterConfig | None = None) -> None:
        self.config = config or FilterConfig()

        self._cache = FilterCache(self.config.cache_path)

        gen_config = GeneratorConfig(
            model_name=self.config.model_name,
            device=self.config.device,
            max_new_tokens=self.config.max_new_tokens,
            temperature=0.0,
            do_sample=False,
            system_prompt=self.config.system_prompt,
            trust_remote_code=self.config.trust_remote_code,
        )
        self._generator: BaseGenerator = GeneratorFactory.create(gen_config)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def filter(
        self,
        question: str,
        image_paths: Sequence[str | Path],
        progress_callback=None,
    ) -> list[FilterResult]:
        """Judge each image and return filtered results.

        Checks cache first; only calls the VLM for uncached (question, image) pairs.
        """
        results: list[FilterResult] = []
        total = len(image_paths)

        for idx, path in enumerate(image_paths):
            path_str = str(path)

            if progress_callback:
                progress_callback(idx + 1, total)

            # --- cache hit ---
            cached = self._cache.get(question, path_str)
            if cached is not None:
                results.append(FilterResult(
                    image_path=path_str,
                    is_relevant=cached["is_relevant"],
                    raw_response=cached.get("raw_response", ""),
                    error=cached.get("error"),
                    cached=True,
                ))
                continue

            # --- cache miss: call VLM ---
            try:
                answer = self._generator.generate(question, [path_str])
                raw = answer["raw_answer"].strip()
                is_rel = self._parse_yes_no(raw)
                self._cache.put(question, path_str, is_relevant=is_rel, raw_response=raw)
                results.append(FilterResult(
                    image_path=path_str,
                    is_relevant=is_rel,
                    raw_response=raw,
                    cached=False,
                ))
            except Exception as exc:
                if self.config.skip_on_error:
                    self._cache.put(question, path_str, is_relevant=False, error=str(exc))
                    results.append(FilterResult(
                        image_path=path_str,
                        is_relevant=False,
                        error=str(exc),
                        cached=False,
                    ))
                else:
                    raise

        return results

    def filter_inplace(
        self,
        question: str,
        candidates: list[dict[str, Any]],
        image_key: str = "image_path",
        progress_callback=None,
    ) -> list[dict[str, Any]]:
        """Filter a list of candidate dicts, keeping only relevant ones.

        Each candidate dict is augmented with ``filter_relevant`` (bool) and
        ``filter_raw`` (str) fields.  Only relevant candidates are returned.
        """
        image_paths = [c[image_key] for c in candidates]
        judgments = self.filter(question, image_paths, progress_callback)

        filtered: list[dict[str, Any]] = []
        for cand, judge in zip(candidates, judgments):
            cand = dict(cand)
            cand["filter_relevant"] = judge.is_relevant
            cand["filter_raw"] = judge.raw_response
            if judge.error:
                cand["filter_error"] = judge.error
            if judge.is_relevant:
                filtered.append(cand)

        return filtered

    def save_cache(self) -> None:
        """Persist the filter cache to disk."""
        self._cache.save()

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_yes_no(text: str) -> bool:
        """Parse a YES/NO response from the generator."""
        cleaned = text.strip().upper()
        cleaned = re.sub(r"[.。,，！!？?]+$", "", cleaned)
        if cleaned in ("YES", "Y", "是"):
            return True
        if cleaned in ("NO", "N", "否", "不是"):
            return False
        if re.search(r"\bYES\b", cleaned):
            return True
        if re.search(r"\bNO\b", cleaned):
            return False
        if "是" in cleaned and "否" not in cleaned and "不是" not in cleaned:
            return True
        return False
