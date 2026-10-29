#!/bin/bash
###############################################################################
# 提取交通灯脚本
# 用法: ./extract_traffic_lights.sh <scene_id> [processed_dir]
###############################################################################

# 参数处理
if [ $# -lt 1 ]; then
    echo "ERROR: 用法: $0 <scene_id> [processed_dir]"
    echo "ERROR: 示例: $0 172 /path/to/processed"
    exit 1
fi

scene_id="$1"

if [ $# -ge 2 ]; then
    processed_dir="$2"
else
    processed_dir="data/waymo/processed"
fi

# 规范化路径（移除末尾斜杠）
processed_dir="${processed_dir%/}"

echo "[INFO] Scene ID: $scene_id"
echo "[INFO] Processed Dir: $processed_dir"

# 智能查找场景目录
scene_dir="$processed_dir/training/$scene_id"

# 尝试 zero-padded (023)
padded_id=$(printf "%03d" "$scene_id")
if [ ! -d "$scene_dir" ] && [ -d "$processed_dir/training/$padded_id" ]; then
    scene_dir="$processed_dir/training/$padded_id"
    echo "[INFO] Found zero-padded directory: $scene_dir"
fi

# 尝试模糊匹配
if [ ! -d "$scene_dir" ]; then
    found=$(find "$processed_dir/training" -maxdepth 1 -type d -name "*$scene_id" 2>/dev/null | head -n 1)
    if [ -n "$found" ]; then
        scene_dir="$found"
        echo "[INFO] Found matching directory: $scene_dir"
    fi
fi

if [ ! -d "$scene_dir" ]; then
    echo "[ERROR] Cannot find scene directory for ID: $scene_id in $processed_dir/training"
    exit 1
fi

# 激活 Conda 环境
CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate grounddino

python adsutido_tools/extract_traffic_lights.py \
    --scene_dir "$scene_dir" \
    --gdino_config third_party/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
    --gdino_checkpoint third_party/GroundingDINO/ckpt/grounding_dino/groundingdino_swint_ogc.pth \
    --cam_ids 0 1 2 --save_masks --device cuda --merge