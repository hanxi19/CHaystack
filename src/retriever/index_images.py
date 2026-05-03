"""
为 Benchmark 图片建立向量索引

支持四类图片：paper, camera, webpage, advertise
可以为每个类别单独建索引，也可以建立统一索引

用法:
    # 为所有类别建立统一索引
    python -m src.retriever.index_images \
        --image_root /volume/zlhuang/lhx/Chinese_documentVQA/benchmark/data/image \
        --index_dir /volume/zlhuang/lhx/Chinese_documentVQA/data/indexes/benchmark_all \
        --model_name qwen3-vl-embedding-2b

    # 为单个类别建索引
    python -m src.retriever.index_images \
        --image_root /volume/zlhuang/lhx/Chinese_documentVQA/benchmark/data/image \
        --categories paper \
        --index_dir /volume/zlhuang/lhx/Chinese_documentVQA/data/indexes/benchmark_paper \
        --model_name qwen3-vl-embedding-2b

    # 为多个类别建索引
    python -m src.retriever.index_images \
        --image_root /volume/zlhuang/lhx/Chinese_documentVQA/benchmark/data/image \
        --categories paper,webpage \
        --index_dir /volume/zlhuang/lhx/Chinese_documentVQA/data/indexes/benchmark_paper_webpage
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..model import MultimodalEmbeddingModel
from ..storage import NumpyVectorStore


VALID_CATEGORIES = ["paper", "camera", "webpage", "advertise"]
DEFAULT_EXTENSIONS = [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]


def scan_image_paths(
    image_root: str | Path,
    categories: list[str] | None = None,
    extensions: list[str] | None = None,
) -> list[Path]:
    """
    扫描图片路径

    支持两种目录结构：
    1. 分类目录结构（新 benchmark）:
       image_root/
         ├── paper/*.jpg
         ├── camera/*.jpg
         ├── webpage/*.png
         └── advertise/*.jpg

    2. 扁平目录结构（旧 DuReader）:
       image_root/*.png
       或
       image_root/dureader_vis_images_part_*/*.png

    Args:
        image_root: 图片根目录
        categories: 类别列表，None 表示所有类别或扁平结构
        extensions: 图片扩展名列表

    Returns:
        图片路径列表
    """
    image_root = Path(image_root)
    if not image_root.exists():
        raise FileNotFoundError(f"图片根目录不存在: {image_root}")

    if extensions is None:
        extensions = DEFAULT_EXTENSIONS

    image_paths = []

    # 如果指定了类别，使用分类目录结构
    if categories is not None:
        for cat in categories:
            if cat not in VALID_CATEGORIES:
                raise ValueError(f"无效类别: {cat}，可选: {VALID_CATEGORIES}")

            cat_dir = image_root / cat
            if not cat_dir.exists():
                print(f"警告: 类别目录不存在: {cat_dir}")
                continue

            for ext in extensions:
                cat_images = sorted(cat_dir.glob(f"*{ext}"))
                image_paths.extend(cat_images)
                if cat_images:
                    print(f"[{cat}] 找到 {len(cat_images)} 张图片 (*{ext})")

    else:
        # 尝试分类目录结构
        has_category_dirs = any((image_root / cat).is_dir() for cat in VALID_CATEGORIES)

        if has_category_dirs:
            # 使用分类目录结构，加载所有类别
            print("检测到分类目录结构，加载所有类别...")
            for cat in VALID_CATEGORIES:
                cat_dir = image_root / cat
                if not cat_dir.exists():
                    continue

                for ext in extensions:
                    cat_images = sorted(cat_dir.glob(f"*{ext}"))
                    image_paths.extend(cat_images)
                    if cat_images:
                        print(f"[{cat}] 找到 {len(cat_images)} 张图片 (*{ext})")

        else:
            # 尝试 DuReader 风格的目录结构
            print("检测到扁平目录结构，尝试 DuReader 格式...")

            # 尝试 dureader_vis_images_part_*/*.png
            for ext in extensions:
                part_images = sorted(image_root.glob(f"dureader_vis_images_part_*/*{ext}"))
                if part_images:
                    image_paths.extend(part_images)
                    print(f"找到 {len(part_images)} 张图片 (dureader_vis_images_part_*/*{ext})")

            # 尝试根目录下的图片
            if not image_paths:
                for ext in extensions:
                    root_images = sorted(image_root.glob(f"*{ext}"))
                    if root_images:
                        image_paths.extend(root_images)
                        print(f"找到 {len(root_images)} 张图片 (*{ext})")

    if not image_paths:
        raise FileNotFoundError(
            f"在 {image_root} 下没有找到图片。\n"
            f"支持的目录结构：\n"
            f"  1. 分类目录: {image_root}/{{paper,camera,webpage,advertise}}/*.{{jpg,png}}\n"
            f"  2. DuReader: {image_root}/dureader_vis_images_part_*/*.png\n"
            f"  3. 扁平目录: {image_root}/*.png"
        )

    return image_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="为 Benchmark 图片建立向量索引")
    parser.add_argument(
        "--image_root",
        type=str,
        required=True,
        help="图片根目录（支持分类目录或扁平目录）",
    )
    parser.add_argument(
        "--index_dir",
        type=str,
        required=True,
        help="向量库输出目录",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="qwen3-vl-embedding-2b",
        help="多模态检索模型名或预设名",
    )
    parser.add_argument(
        "--categories",
        type=str,
        default=None,
        help="类别列表（逗号分隔），如 paper,webpage。不指定则加载所有",
    )
    parser.add_argument(
        "--extensions",
        type=str,
        default=".png,.jpg,.jpeg,.PNG,.JPG,.JPEG",
        help="图片扩展名（逗号分隔）",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help=(
            "每批送入 encode_images 的图片数；Qwen3-VL 显存占用大，24GB 上 OOM 请改为 1–8（可先 1–2）。"
            "CLIP/SigLIP 可保持较大 batch。"
        ),
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--faiss_factory",
        type=str,
        default="hnsw",
        help=(
            "FAISS 索引类型。支持："
            "flat（IndexFlatIP）;"
            "hnsw 或 HNSW32/HNSW64（IndexHNSWFlat + 内积）;"
            "以及任意 faiss.index_factory 字符串（例如 IVF4096,Flat）"
        ),
    )
    parser.add_argument(
        "--hnsw_m",
        type=int,
        default=32,
        help="当 --faiss_factory=hnsw 时使用的 M 参数",
    )
    parser.add_argument(
        "--hnsw_ef_construction",
        type=int,
        default=200,
        help="HNSW 建图阶段的 efConstruction",
    )
    parser.add_argument(
        "--hnsw_ef_search",
        type=int,
        default=128,
        help="HNSW 查询阶段的 efSearch",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="仅处理前 N 张图片（用于调试）",
    )
    parser.add_argument(
        "--no_skip_bad_images",
        action="store_true",
        help="遇到坏图直接报错退出（默认会跳过）",
    )
    parser.add_argument(
        "--bad_image_log",
        type=str,
        default=None,
        help="将跳过的图片路径写入该文件",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    skip_bad_images = not args.no_skip_bad_images

    # 解析类别
    categories = None
    if args.categories:
        categories = [c.strip() for c in args.categories.split(",") if c.strip()]

    # 解析扩展名
    extensions = [e.strip() for e in args.extensions.split(",") if e.strip()]

    # 扫描图片
    print(f"扫描图片目录: {args.image_root}")
    if categories:
        print(f"指定类别: {', '.join(categories)}")
    else:
        print("加载所有类别")

    image_paths = scan_image_paths(args.image_root, categories, extensions)
    print(f"总共找到 {len(image_paths)} 张图片")

    if args.limit is not None:
        image_paths = image_paths[: args.limit]
        print(f"限制处理前 {args.limit} 张")

    index_dir = Path(args.index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    # 加载模型
    print(f"加载模型: {args.model_name}")
    model = MultimodalEmbeddingModel(
        model_name=args.model_name,
        device=args.device,
    )

    # 编码图片
    all_embeddings: list[np.ndarray] = []
    metadata: list[dict[str, str]] = []
    embedding_dim = 0
    skipped = 0
    bad_log_fp = None

    if args.bad_image_log:
        bad_log_path = Path(args.bad_image_log)
        bad_log_path.parent.mkdir(parents=True, exist_ok=True)
        bad_log_fp = bad_log_path.open("w", encoding="utf-8")

    print("开始编码图片...")
    for start in range(0, len(image_paths), args.batch_size):
        batch_paths = [str(path) for path in image_paths[start : start + args.batch_size]]
        batch_output = model.encode_images(
            batch_paths,
            batch_size=args.batch_size,
            skip_bad_images=skip_bad_images,
        )

        if batch_output.embeddings.shape[0] > 0:
            if embedding_dim == 0:
                embedding_dim = int(batch_output.embeddings.shape[1])
            all_embeddings.append(batch_output.embeddings.astype(np.float32, copy=False))

        valid_paths = batch_output.image_paths or []
        valid_set = set(valid_paths)

        for path_str in valid_paths:
            path = Path(path_str)

            # 提取类别信息
            category = "unknown"
            if path.parent.name in VALID_CATEGORIES:
                category = path.parent.name
            elif "dureader" in path.parent.name.lower():
                category = "dureader"

            metadata.append(
                {
                    "image_id": path.stem,
                    "image_path": str(path.resolve()),
                    "category": category,
                    "part_dir": path.parent.name,
                    "filename": path.name,
                }
            )

        skipped += len(batch_paths) - len(valid_paths)

        if bad_log_fp is not None:
            for path_str in batch_paths:
                if path_str not in valid_set:
                    bad_log_fp.write(path_str + "\n")

        end = start + len(batch_paths)
        print(f"[{end}/{len(image_paths)}] 已编码")

    if not all_embeddings:
        raise RuntimeError(
            "没有任何图片被成功编码；请检查 image_root 或关闭 --no_skip_bad_images"
        )

    # 构建索引
    print("构建 FAISS 索引...")
    embeddings = np.concatenate(all_embeddings, axis=0)
    index = NumpyVectorStore.build_index(
        embeddings=embeddings,
        factory=args.faiss_factory,
        hnsw_m=args.hnsw_m,
        hnsw_ef_construction=args.hnsw_ef_construction,
        hnsw_ef_search=args.hnsw_ef_search,
    )

    # 保存索引
    print(f"保存索引到: {index_dir}")
    NumpyVectorStore.save_index(index_dir, index)
    NumpyVectorStore.save_metadata(index_dir, metadata)

    # 统计类别分布
    category_counts = {}
    for meta in metadata:
        cat = meta.get("category", "unknown")
        category_counts[cat] = category_counts.get(cat, 0) + 1

    NumpyVectorStore.save_config(
        index_dir,
        {
            "model_name": model.model_name,
            "embedding_dim": embedding_dim,
            "num_images": len(metadata),
            "num_images_scanned": len(image_paths),
            "num_images_skipped": skipped,
            "image_root": str(Path(args.image_root).resolve()),
            "categories": categories or "all",
            "category_counts": category_counts,
            "faiss_factory": args.faiss_factory,
            "hnsw_m": args.hnsw_m,
            "hnsw_ef_construction": args.hnsw_ef_construction,
            "hnsw_ef_search": args.hnsw_ef_search,
            "skip_bad_images": skip_bad_images,
        },
    )

    print("\n" + "=" * 60)
    print("索引构建完成")
    print("=" * 60)
    print(f"输出目录: {index_dir}")
    print(f"扫描图片: {len(image_paths)}")
    print(f"入库向量: {len(metadata)}")
    if skipped:
        print(f"跳过损坏: {skipped}")
    print(f"向量维度: {embedding_dim}")
    print(f"\n类别分布:")
    for cat, count in sorted(category_counts.items()):
        print(f"  {cat:12s}: {count:6d}")
    print("=" * 60)

    if bad_log_fp is not None:
        bad_log_fp.close()


if __name__ == "__main__":
    main()
