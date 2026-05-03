"""
Retriever → Filter → Generator pipeline for CDocRAG.

Supports phased execution so only one model resides in GPU VRAM at a time:

    Phase 1 (retrieve):  retriever only       → writes retrieval candidates
    Phase 2 (filter):    filter VLM only      → writes filtered candidates + cache
    Phase 3 (generate):  generator VLM only   → writes final answers

    python -m src.pipline.filter_rag --phase retrieve ... --retrieval_output /tmp/r.jsonl
    python -m src.pipline.filter_rag --phase filter   ... --retrieval_cache /tmp/r.jsonl --candidate_output /tmp/f.jsonl
    python -m src.pipline.filter_rag --phase generate ... --candidate_cache /tmp/f.jsonl --output_path /tmp/o.jsonl

    # Or all phases sequentially (each loads/unloads its own model):
    python -m src.pipline.filter_rag --phase all ... --output_path /tmp/o.jsonl
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.util.benchmark_loader import ALL_CATEGORIES, BenchmarkLoader


# ======================================================================
# CLI
# ======================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retriever → Filter → Generator pipeline for CDocRAG (phased)"
    )

    # ── Phase selection ──
    parser.add_argument(
        "--phase",
        type=str,
        default="all",
        choices=["retrieve", "filter", "generate", "all"],
        help="执行阶段: retrieve / filter / generate / all（默认 all，顺序执行三段）",
    )

    # ── Data ──
    parser.add_argument("--benchmark_root", type=str, required=True)
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        choices=ALL_CATEGORIES,
        help="评测类别；不指定则评测全部类别",
    )

    # ── Index ──
    parser.add_argument(
        "--index_root", type=str, default=None,
        help="多索引根目录，自动加载 benchmark_{category} 子目录",
    )
    parser.add_argument(
        "--index_dir", type=str, default=None,
        help="单一索引目录（与 --index_root 二选一）",
    )

    # ── Models ──
    parser.add_argument(
        "--retriever_model", type=str, default=None,
        help="检索模型；默认使用索引 config 中记录的模型",
    )
    parser.add_argument(
        "--filter_model", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct",
        help="Filter 使用的 VLM 模型",
    )
    parser.add_argument(
        "--generator_model", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct",
        help="最终答案生成的 VLM 模型",
    )
    parser.add_argument(
        "--reranker", type=str, default=None,
        help="重排序模型（可选）",
    )
    parser.add_argument(
        "--reranker_instruction", type=str, default=None,
        help="Reranker instruction（仅 Qwen3-VL-Reranker 使用）",
    )
    parser.add_argument("--device", type=str, default=None)

    # ── Retrieval knobs ──
    parser.add_argument("--candidate_k", type=int, default=30)
    parser.add_argument("--top_k", type=int, default=5,
                        help="最终保留给 Generator 的图片数量")

    # ── Filter knobs ──
    parser.add_argument("--no_filter", action="store_true",
                        help="跳过过滤器（消融实验用）")
    parser.add_argument(
        "--filter_cache_path", type=str, default=None,
        help="Filter 判断结果缓存文件（JSON），跨 generator 共享",
    )

    # ── Generation knobs ──
    parser.add_argument("--generator_max_new_tokens", type=int, default=128)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--generator_system_prompt", type=str, default=None)

    # ── Data exchange between phases ──
    parser.add_argument(
        "--retrieval_cache", type=str, default=None,
        help="Phase 1 写出的检索结果（JSONL）；phase 2 从此读取",
    )
    parser.add_argument(
        "--candidate_cache", type=str, default=None,
        help="Phase 2 写出的过滤后结果（JSONL）；phase 3 从此读取",
    )

    # ── Output ──
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--print_every", type=int, default=10)

    return parser.parse_args()


# ======================================================================
# Shared helpers
# ======================================================================


def _resolve(path: str | None, default: str) -> str:
    if path:
        return path
    return str(Path(default).resolve())


def _free_gpu() -> None:
    """Best-effort GPU memory release between phases."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


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
            samples.append({
                "sample_id": sample.get("sample_id") or f"{cat}_{sample.get('image_id')}",
                "qid": sample.get("sample_id") or f"{cat}_{sample.get('image_id')}",
                "question": sample["question"],
                "gold_answer": sample.get("answer") or sample.get("gold_answer"),
                "gold_image_id": sample.get("image_id"),
                "category": cat,
                "answer_type": sample.get("answer_type"),
            })

    return samples[:limit] if limit is not None else samples


def _load_index(args: argparse.Namespace):
    """Load index (single or multi) and return (store_or_manager, retriever_model_name)."""
    from src.storage import NumpyVectorStore, CategoryAwareIndexManager

    use_multi = args.index_root is not None
    store = None
    index_manager = None

    if use_multi:
        index_manager = CategoryAwareIndexManager(
            index_root=args.index_root,
            categories=ALL_CATEGORIES,
            lazy_load=True,
        )
        available = index_manager.get_available_categories()
        first_cat = available[0]
        first_store = index_manager.get_store(first_cat)
        model_name = args.retriever_model or first_store.config["model_name"]
        return index_manager, model_name

    store = NumpyVectorStore.load(args.index_dir, mmap=True)
    model_name = args.retriever_model or store.config["model_name"]
    return store, model_name


# ======================================================================
# Phase 1 — Retrieve
# ======================================================================


def phase_retrieve(args: argparse.Namespace) -> str:
    """Encode questions and search the FAISS index.

    Writes one JSONL line per sample with ``candidates`` embedded.
    Returns the output path.
    """
    from src.model import MultimodalEmbeddingModel

    output_path = _resolve(args.retrieval_cache, "data/output/phase1_retrieval.jsonl")

    # ---- load data ----
    samples = load_samples(args.benchmark_root, category=args.category, limit=args.limit)
    print(f"[phase1] {len(samples)} samples")

    # ---- load index ----
    index_src, retriever_model_name = _load_index(args)
    use_multi = args.index_root is not None

    # ---- load retriever ----
    print(f"[phase1] loading retriever: {retriever_model_name}")
    retriever = MultimodalEmbeddingModel(retriever_model_name, device=args.device)

    # ---- retrieve ----
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fp:
        for idx, sample in enumerate(samples, start=1):
            question = sample["question"]
            cat = sample.get("category")

            if use_multi:
                if not cat:
                    print(f"[phase1] skip {idx}: missing category", file=sys.stderr)
                    continue
                store = index_src.get_store(cat)
            else:
                store = index_src

            query_emb = retriever.encode_texts([question]).embeddings[0]
            results = store.search(query_emb, top_k=args.candidate_k)

            candidates = [
                {
                    "rank": r.rank,
                    "score": r.score,
                    "image_id": r.image_id,
                    "image_path": r.image_path,
                    "metadata": r.metadata,
                }
                for r in results
            ]

            line = json.dumps({**sample, "candidates": candidates}, ensure_ascii=False)
            fp.write(line + "\n")

            if args.print_every > 0 and idx % args.print_every == 0:
                print(f"[phase1] {idx}/{len(samples)}", file=sys.stderr)

    del retriever
    _free_gpu()
    print(f"[phase1] done → {output_path}")
    return output_path


# ======================================================================
# Phase 2 — Filter
# ======================================================================


def phase_filter(args: argparse.Namespace) -> str:
    """Judge each retrieved image with a VLM, keep only relevant ones.

    Reads from phase 1 output, writes filtered candidates.
    """
    from src.filter import FilterConfig, RelevanceFilter

    input_path = _resolve(args.retrieval_cache, "data/output/phase1_retrieval.jsonl")
    output_path = _resolve(args.candidate_cache, "data/output/phase2_filtered.jsonl")

    if not Path(input_path).exists():
        raise FileNotFoundError(f"Phase 1 output not found: {input_path}")

    # ---- load data ----
    samples: list[dict[str, Any]] = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    print(f"[phase2] {len(samples)} samples from {input_path}")

    # Handle --no_filter: just copy candidates through
    if args.no_filter:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fp:
            for sample in samples:
                cands = sample.get("candidates", [])
                for c in cands:
                    c["filter_relevant"] = None
                    c["filter_raw"] = ""
                line = json.dumps({**sample, "candidates": cands}, ensure_ascii=False)
                fp.write(line + "\n")
        print(f"[phase2] no_filter → {output_path}")
        return output_path

    # ---- load filter VLM (the ONLY VLM in this phase) ----
    print(f"[phase2] loading filter: {args.filter_model}")
    filter_config = FilterConfig(
        model_name=args.filter_model,
        device=args.device,
        cache_path=args.filter_cache_path,
    )
    relevance_filter = RelevanceFilter(filter_config)

    # ---- filter ----
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fp:
        for idx, sample in enumerate(samples, start=1):
            question = sample["question"]
            candidates = sample.get("candidates", [])

            filtered = relevance_filter.filter_inplace(
                question, candidates, image_key="image_path",
            )

            pre_n = len(candidates)
            post_n = len(filtered)

            # Fallback: if filter rejects every image, keep all originals
            if post_n == 0 and pre_n > 0:
                for c in candidates:
                    c["filter_relevant"] = False
                    c["filter_raw"] = "[fallback: all rejected]"
                filtered = candidates
                post_n = len(filtered)

            line = json.dumps(
                {
                    **{k: v for k, v in sample.items() if k != "candidates"},
                    "candidates": filtered,
                    "filter_stats": {"pre": pre_n, "post": post_n},
                },
                ensure_ascii=False,
            )
            fp.write(line + "\n")

            if args.print_every > 0 and idx % args.print_every == 0:
                print(
                    f"[phase2] {idx}/{len(samples)} "
                    f"filter: {pre_n}→{post_n}",
                    file=sys.stderr,
                )

    relevance_filter.save_cache()
    del relevance_filter
    _free_gpu()

    print(f"[phase2] done → {output_path}")
    return output_path


# ======================================================================
# Phase 3 — Generate
# ======================================================================


def phase_generate(args: argparse.Namespace) -> str:
    """Generate final answers using filtered candidates.

    Reads from phase 2 output.  Only ONE VLM (the generator) is loaded.
    """
    from src.model import GeneratorConfig, GeneratorFactory

    input_path = _resolve(args.candidate_cache, "data/output/phase2_filtered.jsonl")
    output_path = _resolve(args.output_path, "data/output/filter_rag.jsonl")

    if not Path(input_path).exists():
        raise FileNotFoundError(f"Phase 2 output not found: {input_path}")

    # ---- load data ----
    samples: list[dict[str, Any]] = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    print(f"[phase3] {len(samples)} samples from {input_path}")

    # ---- load generator (the ONLY VLM in this phase) ----
    print(f"[phase3] loading generator: {args.generator_model}")
    generator = GeneratorFactory.create(
        GeneratorConfig(
            model_name=args.generator_model,
            device=args.device,
            max_new_tokens=args.generator_max_new_tokens,
            do_sample=args.do_sample,
            system_prompt=args.generator_system_prompt,
        )
    )

    # ---- generate ----
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fp:
        for idx, sample in enumerate(samples, start=1):
            question = sample["question"]
            candidates = sample.get("candidates", [])
            top_cands = candidates[: args.top_k]
            image_paths = [c["image_path"] for c in top_cands]

            answer = generator.generate(question, image_paths)

            output = {
                "sample_id": sample.get("sample_id"),
                "qid": sample.get("qid"),
                "question": question,
                "category": sample.get("category"),
                "answer_type": sample.get("answer_type"),
                "gold_image_id": sample.get("gold_image_id"),
                "gold_answer": sample.get("gold_answer"),
                "filter_enabled": not args.no_filter,
                "filter_stats": sample.get("filter_stats", {}),
                "retrieved_image_ids": [c["image_id"] for c in top_cands],
                "retrieved_categories": [
                    (c.get("metadata") or {}).get("category", "unknown")
                    for c in top_cands
                ],
                "retrieval_results": top_cands,
                "answer": answer["answer"],
                "raw_answer": answer["raw_answer"],
                "is_uncertain": answer["is_uncertain"],
            }

            line = json.dumps(output, ensure_ascii=False)
            fp.write(line + "\n")
            fp.flush()

            if args.print_every > 0 and idx % args.print_every == 0:
                q_short = question[:40] + "..." if len(question) > 40 else question
                print(
                    f"[phase3] {idx}/{len(samples)} {q_short} | "
                    f"A: {answer['answer'][:40]}...",
                    file=sys.stderr,
                )

    del generator
    _free_gpu()

    print(f"[phase3] done → {output_path}")
    return output_path


# ======================================================================
# Phase "all" — sequential orchestration
# ======================================================================


def phase_all(args: argparse.Namespace) -> None:
    """Run all three phases in one process.

    Between phases we ``del`` models and call ``torch.cuda.empty_cache``
    so each phase owns the GPU exclusively.
    """
    # Ensure we have paths for intermediate files
    retrieval_out = _resolve(args.retrieval_cache, "data/output/phase1_retrieval.jsonl")
    candidate_out = _resolve(args.candidate_cache, "data/output/phase2_filtered.jsonl")

    args.retrieval_cache = retrieval_out
    args.candidate_cache = candidate_out

    print("=" * 50)
    print("  Phase 1/3: Retrieve")
    print("=" * 50)
    phase_retrieve(args)

    print("=" * 50)
    print("  Phase 2/3: Filter")
    print("=" * 50)
    phase_filter(args)

    print("=" * 50)
    print("  Phase 3/3: Generate")
    print("=" * 50)
    phase_generate(args)


# ======================================================================
# Entry point
# ======================================================================


def main() -> None:
    args = parse_args()

    # Validate
    if not args.index_root and not args.index_dir:
        raise ValueError("必须指定 --index_root 或 --index_dir")
    if args.index_root and args.index_dir:
        raise ValueError("--index_root 和 --index_dir 不能同时指定")

    if args.phase == "retrieve":
        phase_retrieve(args)
    elif args.phase == "filter":
        phase_filter(args)
    elif args.phase == "generate":
        phase_generate(args)
    elif args.phase == "all":
        phase_all(args)


if __name__ == "__main__":
    main()
