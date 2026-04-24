#!/usr/bin/env python3
"""使用 Hugging Face `datasets` 下载 wulipc/CC-OCR 的 multi_scene_ocr 子集到 ./data/cc_ocr/。

数据集主页: https://huggingface.co/datasets/wulipc/CC-OCR
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

DATASET_ID = "wulipc/CC-OCR"
CONFIG_NAME = "multi_scene_ocr"


def ensure_dependencies() -> None:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "datasets", "huggingface_hub"],
        stdout=sys.stdout,
        stderr=sys.stderr,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"下载 {DATASET_ID} 子集 {CONFIG_NAME} 到 ./data/cc_ocr/",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="datasets 缓存根目录，默认为本脚本所在目录下的 data/cc_ocr/",
    )
    parser.add_argument(
        "--skip-pip",
        action="store_true",
        help="跳过 pip install（已安装 datasets 时使用）",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Hugging Face 访问令牌（可选，用于私有或限流场景）",
    )
    args = parser.parse_args()

    if not args.skip_pip:
        ensure_dependencies()

    from datasets import DownloadMode, load_dataset

    script_dir = Path(__file__).resolve().parent
    cache_dir = (args.data_dir or (script_dir / "data" / "cc_ocr")).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"数据集: {DATASET_ID}")
    print(f"配置: {CONFIG_NAME}")
    print(f"缓存目录: {cache_dir}")

    ds = load_dataset(
        DATASET_ID,
        CONFIG_NAME,
        cache_dir=str(cache_dir),
        token=args.token,
        download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS,
    )

    print("已就绪。划分与规模:")
    if isinstance(ds, dict):
        for split_name, split_ds in ds.items():
            n = len(split_ds) if hasattr(split_ds, "__len__") else "?"
            print(f"  - {split_name}: {n} 条")
    else:
        n = len(ds) if hasattr(ds, "__len__") else "?"
        print(f"  - default: {n} 条")
    print(f"数据与元数据位于: {cache_dir}")


if __name__ == "__main__":
    main()
