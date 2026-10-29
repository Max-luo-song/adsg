#!/bin/bash
###############################################################################
# NuScenes Traffic Light Extraction Script v3 (comprehensive)
# Usage: ./scripts/nuscene/extract_traffic_lights_v3.sh <scene_name> [processed_dir] [gpu_id] [cam_ids_csv]
# Example: ./scripts/nuscene/extract_traffic_lights_v3.sh scene-0061 /path/to/processed/mini 0 0,1,2
###############################################################################

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "ERROR: Usage: $0 <scene_name> [processed_dir] [gpu_id] [cam_ids_csv]"
    echo "ERROR: Example: $0 scene-0061 /path/to/processed/mini 0 0,1,2"
    exit 1
fi

scene_name="$1"
gpu_id="${3:-0}"
cam_ids_csv="${4:-0,1,2}"

if [ $# -ge 2 ]; then
    processed_dir="$2"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
    PARENT_ROOT="$(dirname "$PROJECT_ROOT")"
    processed_dir="${PARENT_ROOT}/data/nuscenes/processedads/mini"
fi

processed_dir="${processed_dir%/}"

if [[ "$scene_name" != scene-* ]] && ! [[ "$scene_name" =~ ^[0-9]+$ ]]; then
    echo "ERROR: Invalid scene_name: $scene_name (expected scene-XXXX or numeric index)"
    exit 1
fi

if [[ "$scene_name" == scene-* ]]; then
    scene_digits="${scene_name#scene-}"
else
    scene_digits="$scene_name"
fi
scene_dir_name=$(printf "%03d" "$((10#$scene_digits))")
scene_dir="${processed_dir}/${scene_dir_name}"

if [ ! -d "$scene_dir" ]; then
    echo "[ERROR] Cannot find scene directory: $scene_dir"
    echo "[ERROR] Looking for scene ${scene_name} -> ${scene_dir_name} under ${processed_dir}"
    exit 1
fi

IFS=',' read -r -a cam_ids <<< "$cam_ids_csv"
if [ "${#cam_ids[@]}" -eq 0 ]; then
    echo "[ERROR] cam_ids_csv is empty: $cam_ids_csv"
    exit 1
fi

echo "[INFO] Scene Name: $scene_name"
echo "[INFO] Scene Dir: $scene_dir"
echo "[INFO] Processed Dir: $processed_dir"
echo "[INFO] GPU ID: $gpu_id"
echo "[INFO] Cam IDs: $cam_ids_csv"

CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate grounddino

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$PROJECT_ROOT" || exit 1
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

echo ""
echo "=========================================="
echo "  Traffic Light Detection & Tracking v3"
echo "=========================================="
echo "  Scene: $scene_dir"
echo "=========================================="
echo ""

CUDA_VISIBLE_DEVICES="${gpu_id}" python adsutido_tools/extract_traffic_lights_v3.py \
    --scene_dir "$scene_dir" \
    --dataset_type nuscenes \
    --gdino_config third_party/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
    --gdino_checkpoint third_party/GroundingDINO/ckpt/grounding_dino/groundingdino_swint_ogc.pth \
    --box_threshold 0.35 \
    --text_threshold 0.30 \
    --cam_ids "${cam_ids[@]}" \
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
    --expand_ratio 0.0 \
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
echo "  Done"
echo "=========================================="
