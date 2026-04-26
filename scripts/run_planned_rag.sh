#!/bin/bash
#
# 运行 CDocRAG：Anchor-Aware Query Planner + 多子查询检索 + 可选重排 + 生成
# 对应: python -m src.pipeline.planned_rag
#
# 所有参数请在本文件内直接修改
# ============================================

# 数据与路径（相对项目根目录）
BENCHMARK_ROOT="benchmark"
# 多索引模式：指向索引根目录，会自动加载 benchmark_{category} 子目录
INDEX_ROOT="data/indexes"
OUTPUT_PATH="data/output/planned_rag.jsonl"

# 评测子集：空字符串表示评测全部四类；否则填 paper / webpage / camera / advertise 之一
CATEGORY=""

# 模型
PLANNER_MODEL="Qwen/Qwen2.5-3B-Instruct"
# 检索器：与建索引时一致；留空则使用索引 config 中的 model_name
RETRIEVER_MODEL="qwen3-vl-embedding-2b"
# 重排序模型（自动检测类型）：embedding 模型或 Qwen3-VL-Reranker 路径
RERANKER="qwen3-vl-embedding-2b"
RERANKER_INSTRUCTION=""
GENERATOR_MODEL="Qwen/Qwen2.5-VL-3B-Instruct"
DEVICE="cuda"

# 检索与规划
TOP_K=10
CANDIDATE_K=30
PER_QUERY_K=10
MAX_PLANNED_QUERIES=4
PLANNER_MAX_NEW_TOKENS=256
GENERATOR_MAX_NEW_TOKENS=128

# 留空不限制；填数字可调试
LIMIT="50"

# 是否采样（true 则加 --do_sample；否则不加）
DO_SAMPLE="false"
SYSTEM_PROMPT=""

PRINT_EVERY=10

# ============================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# 将相对路径转为绝对路径
_BENCHMARK_ROOT="$BENCHMARK_ROOT"
if [[ ! "$_BENCHMARK_ROOT" = /* ]]; then
    _BENCHMARK_ROOT="$PROJECT_ROOT/$_BENCHMARK_ROOT"
fi
_INDEX_ROOT="$INDEX_ROOT"
if [[ ! "$_INDEX_ROOT" = /* ]]; then
    _INDEX_ROOT="$PROJECT_ROOT/$_INDEX_ROOT"
fi
_OUTPUT_PATH="$OUTPUT_PATH"
if [[ ! "$_OUTPUT_PATH" = /* ]]; then
    _OUTPUT_PATH="$PROJECT_ROOT/$_OUTPUT_PATH"
fi

export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

if [ ! -d "$_INDEX_ROOT" ]; then
    echo "错误: 索引根目录不存在: $_INDEX_ROOT" >&2
    exit 1
fi
mkdir -p "$(dirname "$_OUTPUT_PATH")"

echo "项目根: $PROJECT_ROOT"
echo "benchmark: $_BENCHMARK_ROOT"
echo "index_root: $_INDEX_ROOT"
echo "output:  $_OUTPUT_PATH"
if [ -n "$CATEGORY" ]; then
    echo "category: $CATEGORY"
else
    echo "category: (all)"
fi
echo ""

CMD=(
    python -m src.pipeline.planned_rag
    --benchmark_root "$_BENCHMARK_ROOT"
    --index_root "$_INDEX_ROOT"
    --planner_model "$PLANNER_MODEL"
    --generator_model "$GENERATOR_MODEL"
    --output_path "$_OUTPUT_PATH"
    --top_k "$TOP_K"
    --candidate_k "$CANDIDATE_K"
    --per_query_k "$PER_QUERY_K"
    --max_planned_queries "$MAX_PLANNED_QUERIES"
    --planner_max_new_tokens "$PLANNER_MAX_NEW_TOKENS"
    --generator_max_new_tokens "$GENERATOR_MAX_NEW_TOKENS"
    --print_every "$PRINT_EVERY"
)

if [ -n "$CATEGORY" ]; then
    CMD+=(--category "$CATEGORY")
fi
if [ -n "$RETRIEVER_MODEL" ]; then
    CMD+=(--retriever_model "$RETRIEVER_MODEL")
fi
if [ -n "$RERANKER" ]; then
    CMD+=(--reranker "$RERANKER")
fi
if [ -n "$RERANKER_INSTRUCTION" ]; then
    CMD+=(--reranker_instruction "$RERANKER_INSTRUCTION")
fi
if [ -n "$DEVICE" ]; then
    CMD+=(--device "$DEVICE")
fi
if [ -n "$LIMIT" ]; then
    CMD+=(--limit "$LIMIT")
fi
if [ -n "$SYSTEM_PROMPT" ]; then
    CMD+=(--system_prompt "$SYSTEM_PROMPT")
fi
if [ "$DO_SAMPLE" = "true" ] || [ "$DO_SAMPLE" = "1" ]; then
    CMD+=(--do_sample)
fi

exec "${CMD[@]}"
