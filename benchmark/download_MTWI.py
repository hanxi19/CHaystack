#!/usr/bin/env python3
"""使用 ModelScope SDK 下载 iic/MTWI 数据集到本目录下的 data/。"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def ensure_dependencies() -> None:
    """按需在运行前安装 pip 依赖（与文档一致：先装 modelscope；OSS 拉取需 oss2）。"""
    pkgs = ["modelscope", "oss2"]
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", *pkgs],
        stdout=sys.stdout,
        stderr=sys.stderr,
    )


def load_mtwi(subset_name: str, split: str, cache_dir: str):
    """依次尝试用户指定的 subset_name，失败则回退到 Hub 当前使用的 MTWI。"""
    from modelscope.msdatasets import MsDataset

    candidates: list[str] = []
    if subset_name:
        candidates.append(subset_name)
    if "MTWI" not in candidates:
        candidates.append("MTWI")

    last_err: ValueError | None = None
    for sub in candidates:
        try:
            return MsDataset.load(
                "iic/MTWI",
                subset_name=sub,
                split=split,
                cache_dir=cache_dir,
            ), sub
        except ValueError as e:
            msg = str(e)
            if "not found" in msg and "Available" in msg:
                last_err = e
                continue
            raise
    raise RuntimeError(f"无法加载 MTWI 子集，已尝试 {candidates}: {last_err}") from last_err


def main() -> None:
    parser = argparse.ArgumentParser(description="下载 ModelScope iic/MTWI 到 ./data")
    parser.add_argument(
        "--subset-name",
        default="images",
        help="子集名；若 Hub 中不存在则自动尝试 MTWI（默认: images，与官方示例一致）",
    )
    parser.add_argument("--split", default="test", help="划分，默认 test")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="保存目录，默认为本脚本所在目录下的 data/",
    )
    parser.add_argument(
        "--skip-pip",
        action="store_true",
        help="跳过 pip install（已手动安装依赖时使用）",
    )
    args = parser.parse_args()

    if not args.skip_pip:
        ensure_dependencies()

    script_dir = Path(__file__).resolve().parent
    data_dir = (args.data_dir or (script_dir / "data")).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    ds, subset = load_mtwi(args.subset_name, args.split, str(data_dir))
    if subset != args.subset_name:
        print(f"提示: subset_name 已从 {args.subset_name!r} 自动切换为 {subset!r}（当前 Hub 可用子集）")
    n = len(ds) if hasattr(ds, "__len__") else "未知"
    print(f"已加载 iic/MTWI ({subset=}, {args.split=})，样本数: {n}")
    print(f"缓存与数据文件目录: {data_dir}")


if __name__ == "__main__":
    main()
