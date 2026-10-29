#!/bin/bash
###############################################################################
# Waymo 数据预处理脚本
# 用法: ./preprocess.sh <scene_id> [input_dir] [output_dir]
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

# ============================================================================
# STAGE 2: 语义分割
# ============================================================================
echo "[INFO] ========== STAGE 2: Semantic Segmentation =========="

gpu=-1

segformer_path=third_party/SegFormer

# Pick an avaliable gpu
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

# ============================================================================
# STAGE 3: SMPL 人体姿态提取
# ============================================================================
echo "[INFO] ========== STAGE 3: SMPL Pose Extraction =========="

conda activate 4D-humans
export PYOPENGL_PLATFORM=osmesa

CUDA_VISIBLE_DEVICES=${gpu} python datasets/tools/humanpose_process.py \
    --data_root "$output_dir/training" \
    --dataset waymo \
    --scene_id "$scene_id"

if [ $? -ne 0 ]; then
    echo "[ERROR] SMPL 提取失败"
    exit 1
fi

echo "[SUCCESS] SMPL 提取完成"
echo "[SUCCESS] ========== 预处理完成 =========="

