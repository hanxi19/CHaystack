#!/usr/bin/env python3
"""从百度 CDN 下载 DuReader-vis 的 DocVQA 数据包并解压到本目录下的 data/。

官方说明与文件列表见：
https://github.com/baidu/DuReader/tree/master/DuReader-vis
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import tarfile
import urllib.request
from pathlib import Path

# README 中给出的文件名、下载地址与 MD5
ARCHIVE_NAME = "dureader_vis_docvqa.tar.gz"
DOWNLOAD_URL = (
    "https://dataset-bj.cdn.bcebos.com/qianyan/dureader%5Fvis%5Fdocvqa.tar.gz"
)
EXPECTED_MD5 = "03559a8d01b3939020c71d4fec250926"


def _human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024.0 or unit == "TB":
            return f"{n:.2f} {unit}" if unit != "B" else f"{int(n)} {unit}"
        n /= 1024.0
    return f"{n:.2f} PB"


def md5_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path, chunk_size: int = 2 * 1024 * 1024) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=600) as resp, dest.open("wb") as out:
        cl = resp.headers.get("Content-Length")
        total: int | None = int(cl) if cl and cl.isdigit() else None
        done = 0
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            if total is not None:
                pct = 100.0 * done / total
                msg = f"\r下载进度: {_human_bytes(done)} / {_human_bytes(total)} ({pct:.1f}%)"
            else:
                msg = f"\r下载进度: {_human_bytes(done)}（总大小未知）"
            print(msg, end="", flush=True)
        print()


def extract_tar_gz(archive: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    extract_kw: dict = {}
    if sys.version_info >= (3, 12):
        extract_kw["filter"] = "data"
    with tarfile.open(archive, "r:gz") as tf:
        tf.extractall(path=dest_dir, **extract_kw)


def main() -> None:
    parser = argparse.ArgumentParser(description="下载 DuReader-vis docvqa 包并解压到 ./data")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="输出目录，默认为本脚本所在目录下的 data/",
    )
    parser.add_argument(
        "--no-verify-md5",
        action="store_true",
        help="下载后跳过 MD5 校验",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="即使已存在完整压缩包也重新下载",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    data_dir = (args.data_dir or (script_dir / "data")).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    archive_path = data_dir / ARCHIVE_NAME

    need_download = args.force or not archive_path.is_file()
    if need_download:
        print(f"正在下载: {DOWNLOAD_URL}")
        print(f"保存到: {archive_path}")
        download(DOWNLOAD_URL, archive_path)
    else:
        print(f"已存在压缩包，跳过下载: {archive_path}")

    if not args.no_verify_md5:
        got = md5_file(archive_path)
        if got.lower() != EXPECTED_MD5.lower():
            raise SystemExit(
                f"MD5 不匹配: 期望 {EXPECTED_MD5}, 实际 {got}。"
                "可删除该文件后重试，或使用 --no-verify-md5（不推荐）。"
            )
        print("MD5 校验通过。")

    print(f"正在解压到: {data_dir}")
    extract_tar_gz(archive_path, data_dir)
    print("完成。")


if __name__ == "__main__":
    main()
