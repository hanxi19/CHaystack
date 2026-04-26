"""
Inference script for DocumentVQA generator.

Example usage:
    # Single query with explicit images
    python -m src.generator.infer \
        --model_name Qwen/Qwen2.5-VL-3B-Instruct \
        --question "发票号码是什么？" \
        --image_paths /path/to/doc1.png /path/to/doc2.png

    # Batch inference from retriever output
    python -m src.generator.infer \
        --model_name Qwen/Qwen2.5-VL-3B-Instruct \
        --retrieval_results /path/to/retriever_output.jsonl \
        --output_path /path/to/generator_output.jsonl

    # With custom system prompt
    python -m src.generator.infer \
        --model_name Qwen/Qwen2.5-VL-3B-Instruct \
        --retrieval_results /path/to/retriever_output.jsonl \
        --system_prompt "你的自定义prompt"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from ..model import GeneratorConfig, QwenVLGenerator
from .answer_normalizer import NormalizeConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用 Qwen-VL 模型回答文档视觉问答问题"
    )

    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        help="Qwen-VL 模型名称或本地路径",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="计算设备: cuda / cpu / auto，默认自动选择",
    )

    # Generation arguments
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="最大生成token数",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="采样温度（do_sample=True时有效）",
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        help="使用采样而非贪心解码",
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default=None,
        help="自定义系统提示词，覆盖默认prompt",
    )

    # Single query mode
    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="单个问题（与 --retrieval_results 二选一）",
    )
    parser.add_argument(
        "--image_paths",
        type=str,
        nargs="+",
        default=None,
        help="文档图片路径列表（单个问题模式使用）",
    )

    # Batch mode from retriever output
    parser.add_argument(
        "--retrieval_results",
        type=str,
        default=None,
        help="Retriever 输出文件（JSONL格式），包含 question 和 results 字段",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="使用的 top-k 图片数量",
    )
    parser.add_argument(
        "--result_image_key",
        type=str,
        default="image_path",
        help="retriever 结果中图片路径的字段名",
    )

    # Output arguments
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="输出文件路径（JSONL格式），默认输出到 stdout",
    )
    parser.add_argument(
        "--print_every",
        type=int,
        default=10,
        help="每处理 N 条打印进度，0 表示不打印",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="仅处理前 N 条，用于调试",
    )

    # Normalizer arguments
    parser.add_argument(
        "--max_answer_length",
        type=int,
        default=256,
        help="答案最大长度（规范化后）",
    )

    return parser.parse_args()


def build_generator_config(args: argparse.Namespace) -> GeneratorConfig:
    """Build generator config from CLI args."""
    system_prompt = args.system_prompt

    return GeneratorConfig(
        model_name=args.model_name,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        do_sample=args.do_sample,
        system_prompt=system_prompt,
    )


def build_normalizer_config(args: argparse.Namespace) -> NormalizeConfig:
    """Build normalizer config from CLI args."""
    return NormalizeConfig(max_length=args.max_answer_length)


def load_retrieval_results(path: str) -> list[dict[str, Any]]:
    """Load retriever output from JSONL file."""
    results = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            results.append(json.loads(line))
    return results


def extract_image_paths(
    retrieval_item: dict[str, Any],
    top_k: int,
    image_key: str,
) -> list[str]:
    """Extract top-k image paths from retriever result."""
    results = retrieval_item.get("results", [])
    paths = []
    for item in results[:top_k]:
        if isinstance(item, dict) and image_key in item:
            paths.append(item[image_key])
        elif isinstance(item, str):
            paths.append(item)
    return paths


def process_single_query(
    generator: QwenVLGenerator,
    question: str,
    image_paths: list[str],
) -> dict[str, Any]:
    """Process a single query and return result."""
    result = generator.generate(question, image_paths)
    return {
        "question": question,
        "image_paths": image_paths,
        "answer": result["answer"],
        "raw_answer": result["raw_answer"],
        "is_uncertain": result["is_uncertain"],
        "num_images": result["num_images"],
    }


def process_batch(
    generator: QwenVLGenerator,
    retrieval_results: list[dict[str, Any]],
    top_k: int,
    image_key: str,
    output_path: str | None,
    print_every: int,
    limit: int | None,
) -> None:
    """Process batch queries from retriever output."""
    output_fp = None
    if output_path:
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        output_fp = open(output_path, "w", encoding="utf-8")

    total = len(retrieval_results) if limit is None else min(limit, len(retrieval_results))
    processed = 0

    try:
        for idx, item in enumerate(retrieval_results[:total]):
            question = item.get("question") or item.get("query")
            if not question:
                print(f"[warning] 跳过第 {idx+1} 条：缺少 question 字段", file=sys.stderr)
                continue

            image_paths = extract_image_paths(item, top_k, image_key)

            result = process_single_query(generator, question, image_paths)

            # 合并原始数据
            output_item = {
                **item,
                "generator_answer": result["answer"],
                "generator_raw": result["raw_answer"],
                "generator_is_uncertain": result["is_uncertain"],
                "generator_num_images": result["num_images"],
            }

            # 输出
            json_line = json.dumps(output_item, ensure_ascii=False)
            if output_fp:
                output_fp.write(json_line + "\n")
            else:
                print(json_line)

            processed += 1

            if print_every > 0 and processed % print_every == 0:
                print(
                    f"[{processed}/{total}] Q: {question[:40]}... A: {result['answer'][:40]}...",
                    file=sys.stderr,
                )

    finally:
        if output_fp:
            output_fp.close()

    print(f"[done] 共处理 {processed} 条查询", file=sys.stderr)


def main() -> None:
    args = parse_args()

    # 验证参数
    if args.question and args.retrieval_results:
        print(
            "错误：--question 和 --retrieval_results 不能同时使用",
            file=sys.stderr,
        )
        sys.exit(1)

    if not args.question and not args.retrieval_results:
        print(
            "错误：必须指定 --question 或 --retrieval_results",
            file=sys.stderr,
        )
        sys.exit(1)

    # 构建配置
    generator_config = build_generator_config(args)
    normalizer_config = build_normalizer_config(args)

    # 初始化 generator
    print(f"[init] 加载模型: {args.model_name}", file=sys.stderr)
    from .answer_normalizer import AnswerNormalizer

    generator = QwenVLGenerator(
        config=generator_config,
        normalizer=AnswerNormalizer(normalizer_config),
    )
    print(f"[init] 模型加载完成，设备: {generator.device}", file=sys.stderr)

    # 处理模式
    if args.question:
        # 单条查询模式
        image_paths = args.image_paths or []
        result = process_single_query(generator, args.question, image_paths)

        output = {
            "question": result["question"],
            "answer": result["answer"],
            "raw_answer": result["raw_answer"],
            "is_uncertain": result["is_uncertain"],
            "num_images": result["num_images"],
        }

        if args.output_path:
            Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(args.output_path, "w", encoding="utf-8") as f:
                json.dump(output, f, ensure_ascii=False, indent=2)
            print(f"[done] 结果已保存: {args.output_path}", file=sys.stderr)
        else:
            print(json.dumps(output, ensure_ascii=False, indent=2))

    else:
        # 批量模式
        retrieval_results = load_retrieval_results(args.retrieval_results)
        print(
            f"[batch] 加载 {len(retrieval_results)} 条 retriever 结果",
            file=sys.stderr,
        )

        process_batch(
            generator=generator,
            retrieval_results=retrieval_results,
            top_k=args.top_k,
            image_key=args.result_image_key,
            output_path=args.output_path,
            print_every=args.print_every,
            limit=args.limit,
        )


if __name__ == "__main__":
    main()
