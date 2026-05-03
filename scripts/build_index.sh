#!/bin/bash
#
# 为 Benchmark 图片建立向量索引
#
# 配置参数（修改这里）
# ============================================

# 模型配置
MODEL_NAME="qwen3-vl-embedding-2b"  # 或使用自定义路径: /path/to/checkpoint
DEVICE="cuda"                        # cuda, cuda:0, cpu

# 索引配置
BUILD_UNIFIED_INDEX=false             # 是否构建统一索引（所有类别）
BUILD_CATEGORY_INDEXES=true         # 是否为每个类别单独建索引

# 类别选择（仅当 BUILD_CATEGORY_INDEXES=true 时生效）
BUILD_PAPER=true
BUILD_CAMERA=true
BUILD_WEBPAGE=true
BUILD_ADVERTISE=true

# 其他参数
BATCH_SIZE=16
LIMIT=""                             # 限制图片数量（调试用），留空表示处理全部

# ============================================

set -e

# 项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 路径配置
BENCHMARK_ROOT="$PROJECT_ROOT/benchmark"
IMAGE_ROOT="$BENCHMARK_ROOT/data/image"
INDEX_ROOT="$PROJECT_ROOT/data/indexes"

echo "========================================"
echo "  Benchmark 索引构建"
echo "========================================"
echo "模型: $MODEL_NAME"
echo "设备: $DEVICE"
echo "批次大小: $BATCH_SIZE"
if [ -n "$LIMIT" ]; then
    echo "限制数量: $LIMIT 张图片"
fi
echo "图片根目录: $IMAGE_ROOT"
echo "索引根目录: $INDEX_ROOT"
echo ""

# 检查图片目录
if [ ! -d "$IMAGE_ROOT" ]; then
    echo "错误: 图片目录不存在: $IMAGE_ROOT"
    echo "请先运行: python benchmark/process.py"
    exit 1
fi

# 创建索引目录
mkdir -p "$INDEX_ROOT"

# 构建统一索引
if [ "$BUILD_UNIFIED_INDEX" = true ]; then
    echo "========================================"
    echo "  构建统一索引（所有类别）"
    echo "========================================"
    INDEX_DIR="$INDEX_ROOT/benchmark_all"
    echo "索引目录: $INDEX_DIR"
    echo ""

    CMD="python -m src.retriever.index_images \
        --image_root \"$IMAGE_ROOT\" \
        --index_dir \"$INDEX_DIR\" \
        --model_name \"$MODEL_NAME\" \
        --device \"$DEVICE\" \
        --batch_size $BATCH_SIZE"

    if [ -n "$LIMIT" ]; then
        CMD="$CMD --limit $LIMIT"
    fi

    eval $CMD
    echo ""
    echo "✓ 统一索引构建完成: $INDEX_DIR"
    echo ""
fi

# 构建分类索引
if [ "$BUILD_CATEGORY_INDEXES" = true ]; then
    echo "========================================"
    echo "  构建分类索引"
    echo "========================================"

    # Paper
    if [ "$BUILD_PAPER" = true ]; then
        echo ""
        echo "构建 paper 索引..."
        INDEX_DIR="$INDEX_ROOT/benchmark_paper"

        CMD="python -m src.retriever.index_images \
            --image_root \"$IMAGE_ROOT\" \
            --categories paper \
            --index_dir \"$INDEX_DIR\" \
            --model_name \"$MODEL_NAME\" \
            --device \"$DEVICE\" \
            --batch_size $BATCH_SIZE"

        if [ -n "$LIMIT" ]; then
            CMD="$CMD --limit $LIMIT"
        fi

        eval $CMD
        echo "✓ paper 索引构建完成: $INDEX_DIR"
    fi

    # Camera
    if [ "$BUILD_CAMERA" = true ]; then
        echo ""
        echo "构建 camera 索引..."
        INDEX_DIR="$INDEX_ROOT/benchmark_camera"

        CMD="python -m src.retriever.index_images \
            --image_root \"$IMAGE_ROOT\" \
            --categories camera \
            --index_dir \"$INDEX_DIR\" \
            --model_name \"$MODEL_NAME\" \
            --device \"$DEVICE\" \
            --batch_size $BATCH_SIZE"

        if [ -n "$LIMIT" ]; then
            CMD="$CMD --limit $LIMIT"
        fi

        eval $CMD
        echo "✓ camera 索引构建完成: $INDEX_DIR"
    fi

    # Webpage
    if [ "$BUILD_WEBPAGE" = true ]; then
        echo ""
        echo "构建 webpage 索引..."
        INDEX_DIR="$INDEX_ROOT/benchmark_webpage"

        CMD="python -m src.retriever.index_images \
            --image_root \"$IMAGE_ROOT\" \
            --categories webpage \
            --index_dir \"$INDEX_DIR\" \
            --model_name \"$MODEL_NAME\" \
            --device \"$DEVICE\" \
            --batch_size $BATCH_SIZE"

        if [ -n "$LIMIT" ]; then
            CMD="$CMD --limit $LIMIT"
        fi

        eval $CMD
        echo "✓ webpage 索引构建完成: $INDEX_DIR"
    fi

    # Advertise
    if [ "$BUILD_ADVERTISE" = true ]; then
        echo ""
        echo "构建 advertise 索引..."
        INDEX_DIR="$INDEX_ROOT/benchmark_advertise"

        CMD="python -m src.retriever.index_images \
            --image_root \"$IMAGE_ROOT\" \
            --categories advertise \
            --index_dir \"$INDEX_DIR\" \
            --model_name \"$MODEL_NAME\" \
            --device \"$DEVICE\" \
            --batch_size $BATCH_SIZE"

        if [ -n "$LIMIT" ]; then
            CMD="$CMD --limit $LIMIT"
        fi

        eval $CMD
        echo "✓ advertise 索引构建完成: $INDEX_DIR"
    fi
fi

echo ""
echo "========================================"
echo "  索引构建完成"
echo "========================================"
echo "索引目录: $INDEX_ROOT"
