#!/usr/bin/env python3
"""从 Google Drive 下载 CDLA 中文文档版面数据集到 ./data/cdla/。

CDLA（Chinese Document Layout Analysis）为中文学术文献场景版面数据，
官方发布见: https://github.com/buptlihang/CDLA
Google Drive 直链文件 ID 与 README 中 Google Drive 下载一致。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

# buptlihang/CDLA README 中的 Google Drive 文件
GOOGLE_DRIVE_FILE_ID = "14SUsp_TG8OPdK0VthRXBcAbYzIBjSNLm"
ZIP_NAME = "CDLA_DATASET.zip"


def ensure_gdown() -> None:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "gdown"],
        stdout=sys.stdout,
        stderr=sys.stderr,
    )


def download_zip(dest: Path, *, force: bool) -> None:
    import gdown

    url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and not force:
        print(f"已存在文件，跳过下载: {dest}（使用 --force 可重新下载）")
        return
    print(f"正在下载: {url}")
    print(f"保存到: {dest}")
    gdown.download(url, str(dest), quiet=False)


def unzip_if_needed(zip_path: Path, extract_to: Path) -> None:
    if not zip_path.is_file():
        raise FileNotFoundError(f"未找到压缩包: {zip_path}")
    print(f"正在解压到: {extract_to}")
    extract_to.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(path=extract_to)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="下载 CDLA 数据集（Google Drive）到 ./data/cdla/",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="输出目录，默认为本脚本所在目录下的 data/cdla/",
    )
    parser.add_argument(
        "--skip-pip",
        action="store_true",
        help="跳过 pip install（已安装 gdown 时使用）",
    )
    parser.add_argument(
        "--no-unzip",
        action="store_true",
        help="只下载 zip，不解压",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="即使已存在 zip 也重新下载",
    )
    args = parser.parse_args()

    if not args.skip_pip:
        ensure_gdown()

    script_dir = Path(__file__).resolve().parent
    data_dir = (args.data_dir or (script_dir / "data" / "cdla")).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    zip_path = data_dir / ZIP_NAME
    download_zip(zip_path, force=args.force)

    if not args.no_unzip:
        unzip_if_needed(zip_path, data_dir)

    print("完成。")


if __name__ == "__main__":
    main()
