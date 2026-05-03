#!/bin/bash
#
# Filter RAG 消融实验（分阶段执行，每阶段只驻留一个模型）
#
#   Phase 1 (retrieve):  加载 retriever → 检索全部样本 → 释放
#   Phase 2 (filter):    加载 filter    → 逐图判断 YES/NO → 缓存 + 释放
#   Phase 3 (generate):  加载 generator → 逐样本生成答案 → 释放
#
#   三种 generator × 两种条件 = 6 组，filter 缓存跨组共享
#
# ============================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# ── 可配置参数 ───────────────────────────────────────────────

BENCHMARK_ROOT="benchmark"
INDEX_ROOT="data/indexes"
OUTPUT_ROOT="data/output/ablation"
PHASE1_OUT="$OUTPUT_ROOT/phase1_retrieval.jsonl"
PHASE2_FILTERED="$OUTPUT_ROOT/phase2_filtered.jsonl"
PHASE2_UNFILTERED="$OUTPUT_ROOT/phase2_unfiltered.jsonl"
FILTER_CACHE="$OUTPUT_ROOT/filter_cache.json"

RETRIEVER_MODEL="qwen3-vl-embedding-2b"
FILTER_MODEL="Qwen/Qwen2.5-VL-3B-Instruct"

GENERATOR_MODELS=(
    "Qwen/Qwen2.5-VL-3B-Instruct"
    # "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
    # "OpenGVLab/InternVL2_5-4B"
)
GENERATOR_LABELS=(
    "qwen25_vl"
    # "llava_onevision"
    # "internvl25"
)

DEVICE="cuda"
CANDIDATE_K=15
TOP_K=10
GENERATOR_MAX_NEW_TOKENS=128
DO_SAMPLE="false"
GENERATOR_SYSTEM_PROMPT=""
CATEGORY=""
LIMIT=""
PRINT_EVERY=1

# ── 路径处理 ─────────────────────────────────────────────────

_resolve() {
    local val="$1"
    if [[ "$val" = /* ]]; then echo "$val"; else echo "$PROJECT_ROOT/$val"; fi
}

_BENCHMARK_ROOT=$(_resolve "$BENCHMARK_ROOT")
_INDEX_ROOT=$(_resolve "$INDEX_ROOT")
_OUTPUT_ROOT=$(_resolve "$OUTPUT_ROOT")
_FILTER_CACHE=$(_resolve "$FILTER_CACHE")
_PHASE1_OUT=$(_resolve "$PHASE1_OUT")
_PHASE2_FILTERED=$(_resolve "$PHASE2_FILTERED")
_PHASE2_UNFILTERED=$(_resolve "$PHASE2_UNFILTERED")

export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# ── 前置检查 ─────────────────────────────────────────────────

if [ ! -d "$_BENCHMARK_ROOT" ]; then
    echo "错误: benchmark 目录不存在: $_BENCHMARK_ROOT" >&2; exit 1
fi
if [ ! -d "$_INDEX_ROOT" ]; then
    echo "错误: 索引根目录不存在: $_INDEX_ROOT" >&2; exit 1
fi

mkdir -p "$_OUTPUT_ROOT/with_filter"
mkdir -p "$_OUTPUT_ROOT/no_filter"

# ── 公共参数 ─────────────────────────────────────────────────

COMMON_ARGS=(
    --benchmark_root "$_BENCHMARK_ROOT"
    --index_root "$_INDEX_ROOT"
    --retriever_model "$RETRIEVER_MODEL"
    --candidate_k "$CANDIDATE_K"
    --top_k "$TOP_K"
    --generator_max_new_tokens "$GENERATOR_MAX_NEW_TOKENS"
    --print_every "$PRINT_EVERY"
    --device "$DEVICE"
)
if [ -n "$CATEGORY" ]; then COMMON_ARGS+=(--category "$CATEGORY"); fi
if [ -n "$LIMIT" ]; then COMMON_ARGS+=(--limit "$LIMIT"); fi
if [ -n "$GENERATOR_SYSTEM_PROMPT" ]; then COMMON_ARGS+=(--generator_system_prompt "$GENERATOR_SYSTEM_PROMPT"); fi
if [ "$DO_SAMPLE" = "true" ] || [ "$DO_SAMPLE" = "1" ]; then COMMON_ARGS+=(--do_sample); fi

# ── 打印实验矩阵 ─────────────────────────────────────────────

echo "============================================"
echo "  Filter RAG 消融实验 (分阶段)"
echo "============================================"
echo ""
echo "retriever:  $RETRIEVER_MODEL"
echo "filter:     $FILTER_MODEL"
echo "candidate_k: $CANDIDATE_K   top_k: $TOP_K"
echo "limit:      ${LIMIT:-(全量)}"
echo ""
echo "实验矩阵 (共 $(( ${#GENERATOR_MODELS[@]} * 2 )) 组):"
echo "  condition     | generator"
echo "  --------------|------------"
for label in "${GENERATOR_LABELS[@]}"; do
    printf "  with_filter   | %s\n" "$label"
done
for label in "${GENERATOR_LABELS[@]}"; do
    printf "  no_filter     | %s\n" "$label"
done
echo ""

# ==================================================================
# Phase 1: Retrieve (只加载 retriever，输出 → phase1_retrieval.jsonl)
# ==================================================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Phase 1: Retrieve"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

python -m src.pipline.filter_rag \
    --phase retrieve \
    "${COMMON_ARGS[@]}" \
    --retrieval_cache "$_PHASE1_OUT"

echo ""
echo "✓ Phase 1 完成 → $_PHASE1_OUT"
echo ""

# ==================================================================
# Phase 2a: Filter (只加载 filter VLM，读 phase1 → 写 phase2_filtered)
# ==================================================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Phase 2a: Filter (with filter)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

python -m src.pipline.filter_rag \
    --phase filter \
    "${COMMON_ARGS[@]}" \
    --filter_model "$FILTER_MODEL" \
    --filter_cache_path "$_FILTER_CACHE" \
    --retrieval_cache "$_PHASE1_OUT" \
    --candidate_cache "$_PHASE2_FILTERED"

echo ""
echo "✓ Phase 2a 完成 → $_PHASE2_FILTERED"
echo ""

# ==================================================================
# Phase 2b: No-filter pass (直接从 phase1 复制，不加载 VLM)
# ==================================================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Phase 2b: No-filter pass"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

python -m src.pipline.filter_rag \
    --phase filter \
    "${COMMON_ARGS[@]}" \
    --no_filter \
    --retrieval_cache "$_PHASE1_OUT" \
    --candidate_cache "$_PHASE2_UNFILTERED"

echo ""
echo "✓ Phase 2b 完成 → $_PHASE2_UNFILTERED"
echo ""

# ==================================================================
# Phase 3: Generate (6 runs, 每种 generator 只加载一次)
# ==================================================================

# total_gen=$(( ${#GENERATOR_MODELS[@]} * 2 ))
# gen_idx=0

# for condition in "with_filter" "no_filter"; do
#     if [ "$condition" = "with_filter" ]; then
#         candidate_src="$_PHASE2_FILTERED"
#     else
#         candidate_src="$_PHASE2_UNFILTERED"
#     fi

#     for i in "${!GENERATOR_MODELS[@]}"; do
#         gen_idx=$((gen_idx + 1))
#         gen_model="${GENERATOR_MODELS[$i]}"
#         label="${GENERATOR_LABELS[$i]}"
#         output_path="$_OUTPUT_ROOT/${condition}/${label}.jsonl"

#         echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
#         echo "  Phase 3 [$gen_idx/$total_gen]: $condition / $label"
#         echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
#         echo ""

#         python -m src.pipline.filter_rag \
#             --phase generate \
#             "${COMMON_ARGS[@]}" \
#             --generator_model "$gen_model" \
#             --candidate_cache "$candidate_src" \
#             --output_path "$output_path"

#         echo ""
#         echo "✓ [$gen_idx/$total_gen] $condition / $label → $output_path"
#         echo ""
#     done
# done

# ==================================================================
# 汇总
# ==================================================================

# echo "============================================"
# echo "  消融实验完成"
# echo "============================================"
# echo ""
# echo "中间产物:"
# echo "  phase1:  $_PHASE1_OUT  ($(wc -l < "$_PHASE1_OUT" 2>/dev/null || echo 0) lines)"
# echo "  phase2a: $_PHASE2_FILTERED  ($(wc -l < "$_PHASE2_FILTERED" 2>/dev/null || echo 0) lines)"
# echo "  phase2b: $_PHASE2_UNFILTERED  ($(wc -l < "$_PHASE2_UNFILTERED" 2>/dev/null || echo 0) lines)"
# if [ -f "$_FILTER_CACHE" ]; then
#     entries=$(python -c "import json; print(len(json.load(open('$_FILTER_CACHE'))))" 2>/dev/null || echo "?")
#     echo "  filter_cache: $_FILTER_CACHE ($entries entries)"
# fi
# echo ""
# echo "输出文件:"
# echo "  with_filter:"
# for label in "${GENERATOR_LABELS[@]}"; do
#     f="$_OUTPUT_ROOT/with_filter/${label}.jsonl"
#     if [ -f "$f" ]; then
#         echo "    $label  ($(wc -l < "$f") lines)"
#     else
#         echo "    $label  (MISSING)"
#     fi
# done
# echo "  no_filter:"
# for label in "${GENERATOR_LABELS[@]}"; do
#     f="$_OUTPUT_ROOT/no_filter/${label}.jsonl"
#     if [ -f "$f" ]; then
#         echo "    $label  ($(wc -l < "$f") lines)"
#     else
#         echo "    $label  (MISSING)"
#     fi
# done
# echo ""
# echo "快速对比 (前 3 条样本):"
# echo "─────────────────────────────────────────────"
# for label in "${GENERATOR_LABELS[@]}"; do
#     echo "--- $label ---"
#     for condition in "with_filter" "no_filter"; do
#         f="$_OUTPUT_ROOT/${condition}/${label}.jsonl"
#         if [ -f "$f" ]; then
#             echo "  $condition:"
#             head -3 "$f" | python -c "
# import sys, json
# for line in sys.stdin:
#     d = json.loads(line.strip())
#     q = d.get('question','')[:50]
#     a = d.get('answer','')[:40]
#     fs = d.get('filter_stats',{})
#     pre = fs.get('pre','?') if isinstance(fs,dict) else '?'
#     post = fs.get('post','?') if isinstance(fs,dict) else '?'
#     print(f'    Q: {q}')
#     print(f'    A: {a}')
#     if isinstance(fs,dict) and 'pre' in fs:
#         print(f'    filter: {pre}→{post}')
#     print()" 2>/dev/null || true
#         fi
#     done
#     echo ""
# done
