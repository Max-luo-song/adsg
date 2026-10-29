#!/bin/bash
###############################################################################
# Waymo 语义分割脚本
# 用法: ./semantic_segment.sh <scene_id> <output_dir> [gpu_id]
###############################################################################

# 参数处理
if [ $# -lt 2 ]; then
    echo "ERROR: 用法: $0 <scene_id> <output_dir> [gpu_id]"
    echo "ERROR: 示例: $0 621 /path/to/processed 0"
    exit 1
fi

scene_id="$1"
output_dir="${2%/}"  # 移除末尾斜杠
gpu="${3:--1}"       # 如果未指定，使用-1（自动选择）

echo "[INFO] Scene ID: $scene_id"
echo "[INFO] Output Dir: $output_dir"
echo "[INFO] GPU ID: $gpu"

# ============================================================================
# STAGE 2: 语义分割
# ============================================================================
echo "[INFO] ========== STAGE 2: Semantic Segmentation =========="

segformer_path=third_party/SegFormer

# Pick an available gpu
source scripts/utils.sh
if [ "${gpu}" = "-1" ]; then
    gpu=$(pick_gpu)
    if [ -z "${gpu}" ]; then
        echo "[ERROR] 没有可用的 GPU"
        exit 1
    fi
fi
echo "[INFO] 使用 GPU: ${gpu}"

# 激活 Conda 环境
CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate segformer


CUDA_VISIBLE_DEVICES=${gpu} python datasets/tools/extract_masks.py \
    --data_root "$output_dir/training" \
    --segformer_path=$segformer_path \
    --checkpoint=$segformer_path/pretrained/segformer.b5.1024x1024.city.160k.pth \
    --scene_ids="$scene_id" \
    --process_dynamic_mask

if [ $? -ne 0 ]; then
    echo "[ERROR] 语义分割失败"
    exit 1
fi

echo "[SUCCESS] 语义分割完成"
