from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np


FAISS_INDEX_FILE = "faiss.index"
METADATA_FILE = "metadata.jsonl"
CONFIG_FILE = "config.json"


@dataclass
class SearchResult:
    rank: int
    score: float
    image_id: str
    image_path: str
    metadata: dict[str, Any]


class NumpyVectorStore:
    """A lightweight FAISS-backed vector store."""

    def __init__(
        self,
        index: faiss.Index,
        metadata: list[dict[str, Any]],
        config: dict[str, Any],
    ) -> None:
        self.index = index
        self.metadata = metadata
        self.config = config

    @classmethod
    def load(cls, index_dir: str | Path, mmap: bool = True) -> "NumpyVectorStore":
        del mmap
        index_dir = Path(index_dir)
        index = faiss.read_index(str(index_dir / FAISS_INDEX_FILE))
        with (index_dir / CONFIG_FILE).open("r", encoding="utf-8") as fp:
            config = json.load(fp)

        ef_search = config.get("hnsw_ef_search")
        if ef_search is not None and hasattr(index, "hnsw"):
            index.hnsw.efSearch = int(ef_search)

        metadata: list[dict[str, Any]] = []
        with (index_dir / METADATA_FILE).open("r", encoding="utf-8") as fp:
            for line in fp:
                metadata.append(json.loads(line))
        return cls(index=index, metadata=metadata, config=config)

    @staticmethod
    def build_index(
        embeddings: np.ndarray,
        factory: str | None = None,
        train_size: int = 20000,
        hnsw_m: int = 32,
        hnsw_ef_construction: int = 200,
        hnsw_ef_search: int | None = None,
    ) -> faiss.Index:
        embeddings = np.asarray(embeddings, dtype=np.float32)
        dim = embeddings.shape[1]

        if not factory or factory == "flat":
            index = faiss.IndexFlatIP(dim)
            index.add(embeddings)
            return index

        factory_norm = factory.strip()
        if factory_norm.lower() == "hnsw":
            index = faiss.IndexHNSWFlat(
                dim,
                int(hnsw_m),
                faiss.METRIC_INNER_PRODUCT,
            )
            index.hnsw.efConstruction = int(hnsw_ef_construction)
            if hnsw_ef_search is not None:
                index.hnsw.efSearch = int(hnsw_ef_search)
            index.add(embeddings)
            return index

        match = re.fullmatch(r"(?i)HNSW(\d+)", factory_norm)
        if match:
            m = int(match.group(1))
            index = faiss.IndexHNSWFlat(dim, m, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = int(hnsw_ef_construction)
            if hnsw_ef_search is not None:
                index.hnsw.efSearch = int(hnsw_ef_search)
            index.add(embeddings)
            return index

        index = faiss.index_factory(dim, factory_norm, faiss.METRIC_INNER_PRODUCT)
        if not index.is_trained:
            train_vectors = embeddings[: min(train_size, embeddings.shape[0])]
            index.train(train_vectors)
        index.add(embeddings)
        return index

    @staticmethod
    def save_index(index_dir: str | Path, index: faiss.Index) -> None:
        index_dir = Path(index_dir)
        faiss.write_index(index, str(index_dir / FAISS_INDEX_FILE))

    @staticmethod
    def save_metadata(index_dir: str | Path, metadata: list[dict[str, Any]]) -> None:
        index_dir = Path(index_dir)
        with (index_dir / METADATA_FILE).open("w", encoding="utf-8") as fp:
            for item in metadata:
                fp.write(json.dumps(item, ensure_ascii=False) + "\n")

    @staticmethod
    def save_config(index_dir: str | Path, config: dict[str, Any]) -> None:
        index_dir = Path(index_dir)
        with (index_dir / CONFIG_FILE).open("w", encoding="utf-8") as fp:
            json.dump(config, fp, ensure_ascii=False, indent=2)

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[SearchResult]:
        query_embedding = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        top_k = min(top_k, len(self.metadata))
        if top_k == 0:
            return []

        scores, indices = self.index.search(query_embedding, top_k)
        scores = scores[0]
        indices = indices[0]

        results: list[SearchResult] = []
        for rank, (score, idx) in enumerate(zip(scores.tolist(), indices.tolist()), start=1):
            if idx < 0:
                continue
            meta = self.metadata[int(idx)]
            results.append(
                SearchResult(
                    rank=rank,
                    score=float(score),
                    image_id=meta["image_id"],
                    image_path=meta["image_path"],
                    metadata=meta,
                )
            )
        return results
