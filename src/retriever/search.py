from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..model import MultimodalEmbeddingModel
from ..storage import NumpyVectorStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按文本问题检索相关文档图片")
    parser.add_argument(
        "--index_dir",
        type=str,
        required=True,
        help="index_images.py 生成的向量库目录",
    )
    parser.add_argument("--query", type=str, required=True, help="输入查询问题")
    parser.add_argument("--top_k", type=int, default=5, help="最终返回结果数")
    parser.add_argument(
        "--candidate_k",
        type=int,
        default=20,
        help="初召回候选数，重排后再截断到 top_k",
    )
    parser.add_argument(
        "--query_model_name",
        type=str,
        default=None,
        help="第一阶段召回模型；默认读取 index 内记录的模型",
    )
    parser.add_argument(
        "--reranker",
        type=str,
        default=None,
        help="重排序模型（自动检测类型）：embedding 模型或 Qwen3-VL-Reranker 路径",
    )
    parser.add_argument(
        "--reranker_instruction",
        type=str,
        default=None,
        help="Reranker instruction（仅 Qwen3-VL-Reranker 使用）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="cuda / cpu，默认自动判断",
    )
    parser.add_argument(
        "--qwen_query_instruction",
        type=str,
        default=None,
        help="仅 Qwen3-VL-Embedding 查询侧 instruction",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="可选，将检索结果写入 JSON",
    )
    return parser.parse_args()


def rerank_results(
    query: str,
    results: list[dict],
    reranker_model: str,
    reranker_instruction: str | None,
    device: str | None,
) -> tuple[list[dict], str]:
    """使用统一的 Reranker 接口重排序。"""
    from ..model import Reranker

    print(f"rerank with {reranker_model}")
    reranker = Reranker(reranker_model, device=device)

    documents = [{"image": item["image_path"]} for item in results]
    inputs = {
        "query": {"text": query},
        "documents": documents,
    }
    if reranker_instruction:
        inputs["instruction"] = reranker_instruction

    scores = reranker.process(inputs)

    reranked = []
    for item, score in zip(results, scores):
        merged = dict(item)
        merged["rerank_score"] = float(score)
        reranked.append(merged)
    reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
    for rank, item in enumerate(reranked, start=1):
        item["rank"] = rank
    return reranked, reranker.backend


def main() -> None:
    args = parse_args()

    store = NumpyVectorStore.load(args.index_dir, mmap=True)

    query_model_name = args.query_model_name or store.config["model_name"]
    query_model = MultimodalEmbeddingModel(
        query_model_name,
        device=args.device,
        qwen_query_instruction=args.qwen_query_instruction,
    )
    query_embedding = query_model.encode_texts([args.query]).embeddings[0]

    retrieved = store.search(query_embedding=query_embedding, top_k=args.candidate_k)
    results = [
        {
            "rank": item.rank,
            "score": item.score,
            "image_id": item.image_id,
            "image_path": item.image_path,
        }
        for item in retrieved
    ]

    rerank_backend = None
    if args.reranker:
        results, rerank_backend = rerank_results(
            query=args.query,
            results=results,
            reranker_model=args.reranker,
            reranker_instruction=args.reranker_instruction,
            device=args.device,
        )

    results = results[: args.top_k]
    payload = {
        "query": args.query,
        "index_dir": str(Path(args.index_dir).resolve()),
        "query_model_name": query_model_name,
        "reranker": args.reranker,
        "rerank_backend": rerank_backend,
        "results": results,
    }

    if args.output_path:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
