#!/bin/bash
###############################################################################
# 提取交通灯脚本 v3 —— 综合版（2D跟踪 + LiDAR深度 + 三角化 + DBSCAN + 反投影补全）
# 用法: ./extract_traffic_lights_v3.sh <scene_id> [processed_dir]
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

echo ""
echo "=========================================="
echo "  红绿灯检测与跟踪 v3（综合版）"
echo "=========================================="
echo "  策略:"
echo "    1. 2D检测（Grounding DINO）+ bbox扩展"
echo "    2. 2D跟踪（IoU关联）"
echo "    3. 深度估计（LiDAR优先 + 三角化）"
echo "    4. 3D聚类合并（DBSCAN）"
echo "    4.5 反投影补全（已知3D→所有帧所有相机）"
echo "  场景: $scene_dir"
echo "=========================================="
echo ""

TL_EXPAND_RATIO="${TL_EXPAND_RATIO:-0.45}"
echo "[INFO] Traffic-light bbox expand ratio: ${TL_EXPAND_RATIO}"

python adsutido_tools/extract_traffic_lights_v3.py \
    --scene_dir "$scene_dir" \
    --gdino_config third_party/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
    --gdino_checkpoint third_party/GroundingDINO/ckpt/grounding_dino/groundingdino_swint_ogc.pth \
    --box_threshold 0.35 \
    --text_threshold 0.30 \
    --cam_ids 0 1 2 \
    --lidar_margin 20 \
    --lidar_top_fraction 0.6 \
    --lidar_depth_percentile 15.0 \
    --bbox_head_fraction 0.3 \
    --iou_threshold 0.3 \
    --max_gap 10 \
    --motion_thresh 0.05 \
    --min_motion_dets 10 \
    --triag_cond_thresh 1e-4 \
    --merge_dist 5 \
    --expand_ratio "${TL_EXPAND_RATIO}" \
    --reproject \
    --reproject_min_depth 5.0 \
    --reproject_max_depth 80.0 \
    --reproject_margin 50 \
    --min_frames 5 \
    --min_height 1.5 \
    --save_masks \
    --merge \
    --device cuda

echo ""
echo "=========================================="
echo "  完成！"
echo "=========================================="
