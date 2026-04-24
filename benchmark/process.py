#!/usr/bin/env python3
"""
将 benchmark/data 下的数据集图像整理到 benchmark/data/image/<类别>/。

四个类别：
  - paper      : CDLA 数据集的论文图片
  - camera     : CC-OCR 和 XFUND 的实拍文档图片
  - webpage    : DuReader 的网页截图
  - advertise  : MTWI 的广告/招牌图片

用法（建议在仓库 Chinese_documentVQA 根目录执行）:
    python benchmark/process.py
    python benchmark/process.py --data-root ./benchmark/data --verbose
"""

from __future__ import annotations

import argparse
import base64
import json
import shutil
import sys
import tarfile
from pathlib import Path
from typing import Iterable

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable, *args, **kwargs):
        return iterable


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

# CC-OCR：仅保留 document_text / zh_doc（doc_zh）子集
CC_OCR_ZH_DOC_SUBSETS: dict[tuple[str, str], str] = {
    ("document_text", "zh_doc"): "doc_zh",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="整理 benchmark/data 下各数据集图像到 data/image/<类别>/")
    p.add_argument(
        "--data-root",
        type=Path,
        default=SCRIPT_DIR / "data",
        help="数据根目录（默认：benchmark/data）",
    )
    p.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="输出根目录（默认：<data-root>/image）",
    )
    p.add_argument("--force", action="store_true", help="已存在同名文件时仍覆盖")
    p.add_argument("--verbose", action="store_true", help="显示详细进度")
    return p.parse_args()


def _archive_signature(archive: Path) -> str:
    """生成压缩包签名，用于判断是否需要重新解压"""
    st = archive.stat()
    return f"{archive.resolve()}\n{st.st_size}\n{int(st.st_mtime_ns)}"


def _marker_path(archive: Path, dest: Path) -> Path:
    """生成解压标记文件路径"""
    safe = archive.name.replace(".", "_")
    return dest / f".extracted_{safe}"


def _extract_tar_gz(archive: Path, dest: Path, verbose: bool = False) -> None:
    """解压 .tar.gz 文件"""
    dest.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"解压: {archive.name} -> {dest}")

    extract_kw: dict = {}
    if sys.version_info >= (3, 12):
        extract_kw["filter"] = "data"

    with tarfile.open(archive, "r:gz") as tf:
        tf.extractall(path=dest, **extract_kw)


def extract_cdla_to_paper(data_root: Path, out_root: Path, force: bool, verbose: bool) -> int:
    """将 CDLA 数据集的图片复制到 image/paper/"""
    cdla_dir = data_root / "cdla" / "CDLA_DATASET"
    if not cdla_dir.is_dir():
        print(f"[paper/cdla] 跳过：未找到 {cdla_dir}", file=sys.stderr)
        return 0

    dest_dir = out_root / "paper"
    dest_dir.mkdir(parents=True, exist_ok=True)

    # 递归查找所有图片文件
    image_files = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
        image_files.extend(cdla_dir.rglob(ext))

    n = 0
    iterator = tqdm(image_files, desc="[paper/cdla]", unit="图") if verbose else image_files
    for src in iterator:
        dst = dest_dir / src.name
        if dst.is_file() and not force and dst.stat().st_size == src.stat().st_size:
            n += 1
            continue
        shutil.copy2(src, dst)
        n += 1

    print(f"[paper/cdla] 复制 {n} 张 -> {dest_dir}")
    return n


def extract_cc_ocr_to_camera(data_root: Path, out_root: Path, force: bool, verbose: bool) -> int:
    """将 CC-OCR 数据集的图片复制到 image/camera/"""
    cc_root = data_root / "cc_ocr"
    arrows = sorted(cc_root.glob("**/cc-ocr-test-*-of-*.arrow"))
    if not arrows:
        print(f"[camera/cc_ocr] 跳过：未在 {cc_root} 下找到 cc-ocr-test-*.arrow", file=sys.stderr)
        return 0

    try:
        from datasets import Dataset, concatenate_datasets
    except ImportError as e:
        raise SystemExit(
            "提取 CC-OCR 需要安装 datasets：pip install datasets pyarrow\n" + str(e)
        ) from e

    if verbose:
        print(f"[camera/cc_ocr] 加载 {len(arrows)} 个 Arrow 文件...")

    parts = [Dataset.from_file(str(p)) for p in arrows]
    ds = parts[0] if len(parts) == 1 else concatenate_datasets(parts)

    dest_dir = out_root / "camera"
    dest_dir.mkdir(parents=True, exist_ok=True)

    n = 0
    iterator = tqdm(ds, desc="[camera/cc_ocr]", unit="图") if verbose else ds
    for row in iterator:
        l2 = row.get("l2-category") or row.get("l2_category") or ""
        sp = row.get("split") or ""
        if not isinstance(l2, str):
            l2 = str(l2)
        if not isinstance(sp, str):
            sp = str(sp)

        key = (l2, sp)
        if key not in CC_OCR_ZH_DOC_SUBSETS:
            continue

        b64 = row.get("image")
        name = row.get("image_name") or f"{row.get('index', n)}.png"
        idx = row.get("index", n)

        if not isinstance(b64, str):
            continue

        raw = b64.strip()
        if raw.startswith("data:"):
            raw = raw.split(",", 1)[-1]
        pad = (-len(raw)) % 4
        if pad:
            raw += "=" * pad

        try:
            blob = base64.b64decode(raw)
        except (ValueError, base64.binascii.Error):
            continue

        safe_name = Path(str(name)).name
        # 与评测 jsonl 中 image_id 一致：{index:06d}_{image_name}，勿加 cc_ocr_ 前缀（否则检索 Recall 对不上）
        out_name = f"{int(idx):06d}_{safe_name}"
        dst = dest_dir / out_name

        if dst.is_file() and not force and dst.stat().st_size == len(blob):
            n += 1
            continue

        dst.write_bytes(blob)
        n += 1

    print(f"[camera/cc_ocr] 写出 {n} 张 -> {dest_dir}")
    return n


def extract_xfund_to_camera(data_root: Path, out_root: Path, force: bool, verbose: bool) -> int:
    """将 XFUND 数据集的图片复制到 image/camera/"""
    xfund = data_root / "xfund"
    patterns = [xfund / "zh.train" / "*.jpg", xfund / "zh.val" / "*.jpg"]

    dest_dir = out_root / "camera"
    dest_dir.mkdir(parents=True, exist_ok=True)

    image_files = []
    for pat in patterns:
        if pat.parent.is_dir():
            image_files.extend(sorted(pat.parent.glob(pat.name)))

    if not image_files:
        print(f"[camera/xfund] 跳过：未找到 {xfund}/zh.train 或 zh.val 下的图片", file=sys.stderr)
        return 0

    n = 0
    iterator = tqdm(image_files, desc="[camera/xfund]", unit="图") if verbose else image_files
    for src in iterator:
        # 与评测 jsonl 中 image_id 一致，勿加 xfund_ 前缀
        dst = dest_dir / src.name
        if dst.is_file() and not force and dst.stat().st_size == src.stat().st_size:
            n += 1
            continue
        shutil.copy2(src, dst)
        n += 1

    print(f"[camera/xfund] 复制 {n} 张 -> {dest_dir}")
    return n


def extract_dureader_to_webpage(data_root: Path, out_root: Path, force: bool, verbose: bool) -> int:
    """解压 DuReader 并将图片复制到 image/webpage/"""
    dureader_dir = data_root / "dureader_vis_docvqa"
    tar_file = dureader_dir / "docvqa_dev_images.tar.gz"

    if not tar_file.is_file():
        print(f"[webpage/dureader] 跳过：未找到 {tar_file}", file=sys.stderr)
        return 0

    # 检查是否需要解压
    marker = _marker_path(tar_file, dureader_dir)
    sig = _archive_signature(tar_file)
    need_extract = True

    if marker.is_file():
        try:
            if marker.read_text(encoding="utf-8") == sig:
                need_extract = False
                if verbose:
                    print(f"[webpage/dureader] 跳过解压（已是最新）: {tar_file.name}")
        except OSError:
            pass

    if need_extract:
        _extract_tar_gz(tar_file, dureader_dir, verbose)
        try:
            marker.write_text(sig, encoding="utf-8")
        except OSError as e:
            print(f"[webpage/dureader] 警告：无法写入解压标记: {e}", file=sys.stderr)

    # 查找解压后的图片目录
    image_dir = dureader_dir / "dureader_images_dev"
    if not image_dir.is_dir():
        print(f"[webpage/dureader] 跳过：解压后未找到 {image_dir}", file=sys.stderr)
        return 0

    dest_dir = out_root / "webpage"
    dest_dir.mkdir(parents=True, exist_ok=True)

    image_files = list(image_dir.glob("*.png"))
    n = 0
    iterator = tqdm(image_files, desc="[webpage/dureader]", unit="图") if verbose else image_files
    for src in iterator:
        dst = dest_dir / src.name
        if dst.is_file() and not force and dst.stat().st_size == src.stat().st_size:
            n += 1
            continue
        shutil.copy2(src, dst)
        n += 1

    print(f"[webpage/dureader] 复制 {n} 张 -> {dest_dir}")
    return n


def resolve_mtwi_test_image_dir(data_root: Path) -> Path | None:
    """定位 MTWI test 图片目录。ModelScope 缓存下 content hash 目录名会变，故动态解析。"""
    extracted = data_root / "iic" / "MTWI" / "master" / "data_files" / "extracted"
    if not extracted.is_dir():
        return None

    def test_dir_has_jpgs(td: Path) -> bool:
        return td.is_dir() and any(td.glob("*.jpg"))

    stable = extracted / "current" / "test"
    if test_dir_has_jpgs(stable):
        return stable

    candidates: list[Path] = []
    for p in extracted.iterdir():
        if p.name == "current" or not p.is_dir():
            continue
        td = p / "test"
        if test_dir_has_jpgs(td):
            candidates.append(td)
    if not candidates:
        return None
    return max(candidates, key=lambda td: sum(1 for _ in td.glob("*.jpg")))


def extract_mtwi_to_advertise(data_root: Path, out_root: Path, force: bool, verbose: bool) -> int:
    """将 MTWI 数据集的图片复制到 image/advertise/"""
    mtwi_test_dir = resolve_mtwi_test_image_dir(data_root)

    if mtwi_test_dir is None or not mtwi_test_dir.is_dir():
        hint = data_root / "iic" / "MTWI" / "master" / "data_files" / "extracted"
        print(f"[advertise/mtwi] 跳过：未找到含 test/*.jpg 的 MTWI 目录（已查 {hint} 下各 hash 子目录）", file=sys.stderr)
        return 0

    dest_dir = out_root / "advertise"
    dest_dir.mkdir(parents=True, exist_ok=True)

    image_files = list(mtwi_test_dir.glob("*.jpg"))
    if not image_files:
        print(f"[advertise/mtwi] 跳过：{mtwi_test_dir} 下无 jpg 文件", file=sys.stderr)
        return 0

    n = 0
    iterator = tqdm(image_files, desc="[advertise/mtwi]", unit="图") if verbose else image_files
    for src in iterator:
        dst = dest_dir / src.name
        if dst.is_file() and not force and dst.stat().st_size == src.stat().st_size:
            n += 1
            continue
        shutil.copy2(src, dst)
        n += 1

    print(f"[advertise/mtwi] 复制 {n} 张 -> {dest_dir}")
    return n


def check_completeness(out_root: Path) -> None:
    """检查各类别图片是否完整"""
    print("\n" + "=" * 60)
    print("图片语料库完整性检查")
    print("=" * 60)

    categories = {
        "paper": "论文类图片（CDLA）",
        "camera": "实拍文档类图片（CC-OCR + XFUND）",
        "webpage": "网页类图片（DuReader）",
        "advertise": "广告类图片（MTWI）",
    }

    all_complete = True
    for category, desc in categories.items():
        cat_dir = out_root / category
        if not cat_dir.is_dir():
            print(f"✗ {category:12s} - 目录不存在: {cat_dir}")
            all_complete = False
            continue

        count = len(list(cat_dir.glob("*")))
        if count == 0:
            print(f"✗ {category:12s} - 无图片文件")
            all_complete = False
        else:
            print(f"✓ {category:12s} - {count:6d} 张图片 ({desc})")

    print("=" * 60)
    if all_complete:
        print("✓ 所有类别图片已准备完成")
    else:
        print("✗ 部分类别图片缺失，请检查数据源")
    print("=" * 60 + "\n")


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    out_root = (args.out_root or (data_root / "image")).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"数据根目录: {data_root}")
    print(f"输出根目录: {out_root}\n")

    # 1. CDLA -> paper
    extract_cdla_to_paper(data_root, out_root, args.force, args.verbose)

    # 2. CC-OCR -> camera
    extract_cc_ocr_to_camera(data_root, out_root, args.force, args.verbose)

    # 3. XFUND -> camera
    extract_xfund_to_camera(data_root, out_root, args.force, args.verbose)

    # 4. DuReader -> webpage
    extract_dureader_to_webpage(data_root, out_root, args.force, args.verbose)

    # 5. MTWI -> advertise
    extract_mtwi_to_advertise(data_root, out_root, args.force, args.verbose)

    # 检查完整性
    check_completeness(out_root)

    print(f"[done] 输出目录: {out_root}")


if __name__ == "__main__":
    main()
