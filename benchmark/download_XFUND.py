#!/usr/bin/env python3
"""从 GitHub Release v1.0 下载 XFUND 中以 zh 开头的四个文件并解压 zip。

文件默认保存到本脚本所在目录下的 data/xfund/。
发布页: https://github.com/doc-analysis/XFUND/releases/tag/v1.0
"""

from __future__ import annotations

import argparse
import json
import urllib.request
import zipfile
from pathlib import Path

RELEASE_API = "https://api.github.com/repos/doc-analysis/XFUND/releases/tags/v1.0"
RELEASE_PAGE = "https://github.com/doc-analysis/XFUND/releases/tag/v1.0"


def _human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024.0 or unit == "TB":
            return f"{n:.2f} {unit}" if unit != "B" else f"{int(n)} {unit}"
        n /= 1024.0
    return f"{n:.2f} PB"


def _github_request(url: str) -> urllib.request.Request:
    return urllib.request.Request(
        url,
        headers={
            "User-Agent": "Chinese_documentVQA-benchmark-download",
            "Accept": "application/vnd.github+json",
        },
    )


def list_zh_assets() -> list[tuple[str, str]]:
    """返回 (文件名, browser_download_url)，仅 name 以 zh 开头的资源。"""
    req = _github_request(RELEASE_API)
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.load(resp)
    out: list[tuple[str, str]] = []
    for asset in data.get("assets", []):
        name = asset.get("name") or ""
        url = asset.get("browser_download_url") or ""
        if name.startswith("zh") and url:
            out.append((name, url))
    out.sort(key=lambda x: x[0])
    return out


def download(url: str, dest: Path, label: str, chunk_size: int = 2 * 1024 * 1024) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = _github_request(url)
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
                msg = (
                    f"\r[{label}] {_human_bytes(done)} / {_human_bytes(total)} ({pct:.1f}%)"
                )
            else:
                msg = f"\r[{label}] {_human_bytes(done)}（总大小未知）"
            print(msg, end="", flush=True)
        print()


def extract_zip(archive: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "r") as zf:
        zf.extractall(path=dest_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="下载 XFUND v1.0 中 zh 开头的 4 个文件并解压 zip 到 ./data/xfund/",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="输出目录，默认为本脚本所在目录下的 data/xfund/",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="已存在同名文件时仍重新下载",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    data_dir = (args.data_dir or (script_dir / "data" / "xfund")).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"查询资源列表: {RELEASE_API}")
    assets = list_zh_assets()
    if len(assets) != 4:
        raise SystemExit(
            f"期望 4 个以 zh 开头的资源，实际得到 {len(assets)} 个: {[a[0] for a in assets]}。"
            f"请核对 {RELEASE_PAGE}"
        )

    for name, url in assets:
        dest = data_dir / name
        if dest.is_file() and not args.force:
            print(f"已存在，跳过下载: {dest}")
        else:
            print(f"下载: {name}")
            print(f"  URL: {url}")
            download(url, dest, label=name)

        if name.endswith(".zip"):
            out_sub = data_dir / Path(name).stem
            print(f"解压: {dest} -> {out_sub}/")
            extract_zip(dest, out_sub)

    print(f"完成。文件目录: {data_dir}")


if __name__ == "__main__":
    main()
