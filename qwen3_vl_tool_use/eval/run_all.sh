#!/bin/bash

# 要执行的 Python 命令通用部分
MODEL_NAME="REVPT_models"
DATASET="vsibench"
PORT_POOL=10010

# 定义路径列表
PATH_LIST=(
    "/vepfs_c/gaolei/REVPT/data/vsibench/train4.parquet"
    "/vepfs_c/gaolei/REVPT/data/vsibench/train5.parquet"
    "/vepfs_c/gaolei/REVPT/data/vsibench/train6.parquet"
    "/vepfs_c/gaolei/REVPT/data/vsibench/train7.parquet"
)

# 错误时立即退出
set -e

# 遍历路径
for DATA_PATH in "${PATH_LIST[@]}"; do
    echo "=============================================="
    echo "🚀 Starting evaluation for: $DATA_PATH"
    echo "=============================================="

    # 执行命令并捕获退出状态
    python agent_eval.py \
        --model-name "$MODEL_NAME" \
        --dataset "$DATASET" \
        --port-pool "$PORT_POOL" \
        --path "$DATA_PATH"

    STATUS=$?

    # 检查是否执行成功
    if [ $STATUS -ne 0 ]; then
        echo "❌ Error occurred while processing $DATA_PATH"
        echo "Script stopped."
        exit 1
    else
        echo "✅ Finished processing $DATA_PATH"
    fi

    echo "----------------------------------------------"
    echo "Waiting 5 seconds before next run..."
    echo "----------------------------------------------"
    sleep 10
done

echo "🎉 All tasks completed successfully!"
