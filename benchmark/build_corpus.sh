#!/usr/bin/env bash
# 依次下载各数据源到 benchmark/data/，再运行 process.py 将图像整理到 data/image/<类别>/。
#
# 用法（可在任意目录执行）:
#   bash benchmark/build_corpus.sh
#   PYTHON=python3.11 bash benchmark/build_corpus.sh
#
# 说明: 下载耗时长且依赖网络；若某步失败，修复环境后单独重跑对应 download_*.py 再执行 process.py 即可。

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"

run_step() {
  local title="$1"
  shift
  echo "========== ${title} =========="
  "$@"
  echo
}

run_step "下载 CDLA" "$PYTHON" download_CDLA.py
run_step "下载 CC-OCR" "$PYTHON" download_cc_ocr.py
run_step "下载 XFUND" "$PYTHON" download_XFUND.py
run_step "下载 DuReader-vis" "$PYTHON" download_dureader.py
run_step "下载 MTWI (ModelScope)" "$PYTHON" download_MTWI.py
run_step "整理图像到 data/image/" "$PYTHON" process.py

echo "全部完成。"
