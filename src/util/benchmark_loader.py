"""Load CHaystack evaluation JSONL under ``<benchmark_root>/data/eval/``."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

ALL_CATEGORIES: tuple[str, ...] = ("paper", "camera", "webpage", "advertise")


class BenchmarkLoader:
    def __init__(self, benchmark_root: str) -> None:
        self.root = Path(benchmark_root).expanduser().resolve()

    def _eval_path(self, category: str) -> Path:
        if category not in ALL_CATEGORIES:
            raise ValueError(
                f"未知类别 {category!r}，可选: {', '.join(ALL_CATEGORIES)}"
            )
        return self.root / "data" / "eval" / f"{category}.jsonl"

    def load_eval_data(self, category: str) -> Iterator[dict[str, Any]]:
        path = self._eval_path(category)
        if not path.is_file():
            raise FileNotFoundError(f"评测集不存在: {path}")

        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                image_id = rec.get("image_id")
                question = rec.get("question") or rec.get("query")
                answer = rec.get("answer") or rec.get("gold_answer")
                sample_id = rec.get("sample_id") or f"{category}_{image_id}"

                row: dict[str, Any] = {
                    "sample_id": sample_id,
                    "image_id": image_id,
                    "question": question,
                    "answer": answer,
                    "gold_answer": answer,
                }
                if "answer_type" in rec:
                    row["answer_type"] = rec["answer_type"]
                yield row
