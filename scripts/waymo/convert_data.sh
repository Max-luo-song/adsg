#!/bin/bash
###############################################################################
# Waymo 基础数据转换脚本
# 用法: ./convert_data.sh <scene_id> [input_dir] [output_dir]
###############################################################################

# 参数处理
if [ $# -lt 1 ]; then
    echo "ERROR: 用法: $0 <scene_id> [input_dir] [output_dir]"
    echo "ERROR: 示例: $0 621 /path/to/raw /path/to/processed"
    exit 1
fi

scene_id="$1"

# 获取输入和输出目录（如果提供了参数）
# 如果提供了参数，使用提供的路径；否则使用相对路径
if [ $# -ge 2 ]; then
    input_dir="$2"
else
    input_dir="data/waymo/raw"
fi

if [ $# -ge 3 ]; then
    output_dir="$3"
else
    output_dir="data/waymo/processed"
fi

# 规范化路径（移除末尾斜杠，避免双斜杠）
input_dir="${input_dir%/}"
output_dir="${output_dir%/}"

echo "[INFO] Scene ID: $scene_id"
echo "[INFO] Input Dir: $input_dir"
echo "[INFO] Output Dir: $output_dir"

# ============================================================================
# STAGE 1: 基础数据处理
# ============================================================================
echo "[INFO] ========== STAGE 1: Convert Data =========="

python datasets/preprocess.py \
    --data_root "$input_dir" \
    --target_dir "$output_dir" \
    --dataset waymo \
    --split training \
    --scene_ids "$scene_id" \
    --workers 8 \
    --process_keys images lidar calib pose dynamic_masks objects

if [ $? -ne 0 ]; then
    echo "[ERROR] 数据转换失败"
    exit 1
fi

echo "[SUCCESS] 数据转换完成"
