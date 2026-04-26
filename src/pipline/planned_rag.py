"""
Anchor-aware CDocRAG pipeline: planner + multi-query retrieval + reranking + generator.

Example:
    python -m src.pipeline.planned_rag \
        --benchmark_root /volume/zlhuang/lhx/Chinese_documentVQA/benchmark \
        --index_dir /volume/zlhuang/lhx/Chinese_documentVQA/data/indexes/benchmark_all \
        --planner_model Qwen/Qwen2.5-0.5B-Instruct \
        --generator_model Qwen/Qwen2.5-VL-3B-Instruct \
        --top_k 5 \
        --candidate_k 30 \
        --output_path /volume/zlhuang/lhx/Chinese_documentVQA/data/output/planned_rag.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model import GeneratorConfig, MultimodalEmbeddingModel, Reranker, QwenVLGenerator
from src.planner import AnchorQueryPlan, AnchorQueryPlanner
from src.storage import NumpyVectorStore, SearchResult, CategoryAwareIndexManager
from src.util.benchmark_loader import ALL_CATEGORIES, BenchmarkLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Anchor-aware CDocRAG inference")
    parser.add_argument("--benchmark_root", type=str, required=True)
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        choices=ALL_CATEGORIES,
        help="评测类别；不指定则评测全部类别",
    )
    parser.add_argument(
        "--index_root",
        type=str,
        default=None,
        help="多索引根目录（推荐），会自动加载 benchmark_{category} 子目录",
    )
    parser.add_argument(
        "--index_dir",
        type=str,
        default=None,
        help="单一索引目录（兼容旧版），与 --index_root 二选一",
    )
    parser.add_argument(
        "--planner_model",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="HuggingFace causal LM used for anchor-aware query planning",
    )
    parser.add_argument(
        "--retriever_model",
        type=str,
        default=None,
        help="检索模型；默认使用索引 config 中记录的模型",
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
        "--generator_model",
        type=str,
        default="Qwen/Qwen2.5-VL-3B-Instruct",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument(
        "--candidate_k",
        type=int,
        default=30,
        help="融合/重排前保留的候选文档数量",
    )
    parser.add_argument(
        "--per_query_k",
        type=int,
        default=10,
        help="每个规划子查询召回的候选数量",
    )
    parser.add_argument("--max_planned_queries", type=int, default=4)
    parser.add_argument("--planner_max_new_tokens", type=int, default=256)
    parser.add_argument("--generator_max_new_tokens", type=int, default=128)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--system_prompt", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--print_every", type=int, default=10)
    return parser.parse_args()


def load_samples(
    benchmark_root: str,
    *,
    category: str | None,
    limit: int | None,
) -> list[dict[str, Any]]:
    loader = BenchmarkLoader(benchmark_root)
    categories = [category] if category else list(ALL_CATEGORIES)
    samples: list[dict[str, Any]] = []

    for cat in categories:
        for sample in loader.load_eval_data(category=cat):
            samples.append(
                {
                    "sample_id": sample.get("sample_id") or f"{cat}_{sample.get('image_id')}",
                    "qid": sample.get("sample_id") or f"{cat}_{sample.get('image_id')}",
                    "question": sample["question"],
                    "gold_answer": sample.get("answer") or sample.get("gold_answer"),
                    "gold_image_id": sample.get("image_id"),
                    "category": cat,
                    "answer_type": sample.get("answer_type"),
                }
            )

    return samples[:limit] if limit is not None else samples


def search_with_planned_queries(
    plan: AnchorQueryPlan,
    retriever: MultimodalEmbeddingModel,
    store: NumpyVectorStore,
    *,
    per_query_k: int,
    candidate_k: int,
) -> list[dict[str, Any]]:
    candidate_map: dict[str, dict[str, Any]] = {}
    rrf_k = 60.0

    for query_idx, query in enumerate(plan.retrieval_queries):
        if not query:
            continue
        query_embedding = retriever.encode_texts([query]).embeddings[0]
        results = store.search(query_embedding, top_k=per_query_k)

        for result in results:
            key = result.image_path
            item = candidate_map.setdefault(key, _candidate_from_result(result))
            rrf_score = 1.0 / (rrf_k + result.rank)
            item["fusion_score"] += rrf_score
            item["best_score"] = max(item["best_score"], result.score)
            item["source_queries"].append(
                {
                    "query_index": query_idx,
                    "query": query,
                    "rank": result.rank,
                    "score": result.score,
                }
            )

    candidates = list(candidate_map.values())
    candidates.sort(
        key=lambda item: (item["fusion_score"], item["best_score"]),
        reverse=True,
    )
    for rank, item in enumerate(candidates[:candidate_k], start=1):
        item["rank"] = rank
    return candidates[:candidate_k]


def _candidate_from_result(result: SearchResult) -> dict[str, Any]:
    return {
        "rank": result.rank,
        "score": result.score,
        "best_score": result.score,
        "fusion_score": 0.0,
        "image_id": result.image_id,
        "image_path": result.image_path,
        "metadata": result.metadata,
        "source_queries": [],
    }


def rerank_candidates(
    question: str,
    candidates: list[dict[str, Any]],
    *,
    reranker: Reranker | None,
    reranker_instruction: str | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """使用统一的 Reranker 接口重排序候选结果。

    注意：Reranker 应在主流程中只加载一次并传入，勿在每次样本中重复 ``Reranker(...)``，
    否则会反复从磁盘加载大模型（表现为终端不断出现 Loading weights）。"""
    if not candidates:
        return [], "none"

    if reranker is None:
        return candidates, "fusion"

    # 准备输入
    documents = [{"image": item["image_path"]} for item in candidates]
    inputs = {
        "query": {"text": question},
        "documents": documents,
    }
    if reranker_instruction:
        inputs["instruction"] = reranker_instruction

    # 重排序
    scores = reranker.process(inputs)

    reranked: list[dict[str, Any]] = []
    for item, score in zip(candidates, scores):
        updated = dict(item)
        updated["rerank_score"] = float(score)
        reranked.append(updated)

    reranked.sort(key=lambda item: item["rerank_score"], reverse=True)
    for rank, item in enumerate(reranked, start=1):
        item["rank"] = rank

    return reranked, reranker.backend


def compact_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for item in candidates:
        compact.append(
            {
                "rank": item.get("rank"),
                "image_id": item.get("image_id"),
                "image_path": item.get("image_path"),
                "category": (item.get("metadata") or {}).get("category"),
                "score": item.get("score"),
                "fusion_score": item.get("fusion_score"),
                "rerank_score": item.get("rerank_score"),
            }
        )
    return compact


def main() -> None:
    args = parse_args()

    # 验证索引参数
    if not args.index_root and not args.index_dir:
        raise ValueError("必须指定 --index_root 或 --index_dir")
    if args.index_root and args.index_dir:
        raise ValueError("--index_root 和 --index_dir 不能同时指定")

    samples = load_samples(
        args.benchmark_root,
        category=args.category,
        limit=args.limit,
    )
    category_counts: dict[str, int] = defaultdict(int)
    for sample in samples:
        category_counts[str(sample.get("category") or "unknown")] += 1
    print(f"[data] loaded {len(samples)} samples")
    print(f"[data] category distribution: {dict(category_counts)}")

    print(f"[init] loading planner: {args.planner_model}")
    planner = AnchorQueryPlanner(
        args.planner_model,
        device=args.device,
        max_new_tokens=args.planner_max_new_tokens,
    )

    # 加载索引：多索引模式或单一索引模式
    use_multi_index = args.index_root is not None
    index_manager = None
    store = None

    if use_multi_index:
        print(f"[init] loading multi-index from: {args.index_root}")
        index_manager = CategoryAwareIndexManager(
            index_root=args.index_root,
            categories=ALL_CATEGORIES,
            lazy_load=True,
        )
        print(f"[init] available categories: {index_manager.get_available_categories()}")
        # 从第一个可用索引获取模型名
        first_cat = index_manager.get_available_categories()[0]
        first_store = index_manager.get_store(first_cat)
        retriever_model_name = args.retriever_model or first_store.config["model_name"]
    else:
        print(f"[init] loading single index: {args.index_dir}")
        store = NumpyVectorStore.load(args.index_dir, mmap=True)
        retriever_model_name = args.retriever_model or store.config["model_name"]

    print(f"[init] loading retriever: {retriever_model_name}")
    retriever = MultimodalEmbeddingModel(retriever_model_name, device=args.device)

    print(f"[init] loading generator: {args.generator_model}")
    generator = QwenVLGenerator(
        GeneratorConfig(
            model_name=args.generator_model,
            device=args.device,
            max_new_tokens=args.generator_max_new_tokens,
            do_sample=args.do_sample,
            system_prompt=args.system_prompt,
        )
    )

    shared_reranker: Reranker | None = None
    if args.reranker:
        print(f"[init] loading reranker (once): {args.reranker}")
        shared_reranker = Reranker(args.reranker, device=args.device)

    output_fp = None
    if args.output_path:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_fp = output_path.open("w", encoding="utf-8")
        print(f"[output] writing results to {output_path}")

    try:
        for idx, sample in enumerate(samples, start=1):
            question = sample["question"]
            sample_category = sample.get("category")

            # 根据索引模式选择对应的 store
            if use_multi_index:
                if not sample_category:
                    print(f"[warning] sample {idx} missing category, skipping")
                    continue
                current_store = index_manager.get_store(sample_category)
            else:
                current_store = store

            plan = planner.plan(question, max_queries=args.max_planned_queries)
            candidates = search_with_planned_queries(
                plan,
                retriever,
                current_store,
                per_query_k=args.per_query_k,
                candidate_k=args.candidate_k,
            )
            reranked, rerank_backend = rerank_candidates(
                question,
                candidates,
                reranker=shared_reranker,
                reranker_instruction=args.reranker_instruction,
            )
            top_results = reranked[: args.top_k]
            image_paths = [item["image_path"] for item in top_results]
            answer = generator.generate(question, image_paths)

            output = {
                "sample_id": sample.get("sample_id"),
                "qid": sample.get("qid"),
                "question": question,
                "category": sample.get("category"),
                "answer_type": sample.get("answer_type"),
                "gold_image_id": sample.get("gold_image_id"),
                "gold_answer": sample.get("gold_answer"),
                "plan": plan.to_dict(),
                "rerank_backend": rerank_backend,
                "retrieved_image_ids": [item["image_id"] for item in top_results],
                "retrieved_categories": [
                    (item.get("metadata") or {}).get("category", "unknown")
                    for item in top_results
                ],
                "retrieval_results": compact_candidates(top_results),
                "answer": answer["answer"],
                "raw_answer": answer["raw_answer"],
                "is_uncertain": answer["is_uncertain"],
            }

            line = json.dumps(output, ensure_ascii=False)
            if output_fp:
                output_fp.write(line + "\n")
                output_fp.flush()
            else:
                print(line)

            if args.print_every > 0 and idx % args.print_every == 0:
                q_short = question[:40] + "..." if len(question) > 40 else question
                print(f"[{idx}/{len(samples)}] {q_short}", file=sys.stderr)
    finally:
        if output_fp:
            output_fp.close()

    print(f"[done] processed {len(samples)} samples")


if __name__ == "__main__":
    main()
