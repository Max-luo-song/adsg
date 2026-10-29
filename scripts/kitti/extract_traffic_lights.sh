#!/bin/bash
###############################################################################
# KITTI Traffic Light Extraction Script
# Usage: ./scripts/kitti/extract_traffic_lights.sh <scene_id> [processed_dir]
###############################################################################

# Check args
if [ $# -lt 1 ]; then
    echo "ERROR: Usage: $0 <scene_id> [processed_dir]"
    exit 1
fi

scene_id="$1"

if [ $# -ge 2 ]; then
    processed_dir="$2"
else
    # Default to project/data/kitti/processed
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
    PARENT_ROOT="$(dirname "$PROJECT_ROOT")"
    processed_dir="${PARENT_ROOT}/data/kitti/processed"
fi

processed_dir="${processed_dir%/}"
echo "[INFO] Scene ID: $scene_id"
echo "[INFO] Processed Dir: $processed_dir"

# Find scene directory
# KITTI preprocess typically outputs to processed_dir/scene_id or processed_dir/date/scene_id
scene_dir="$processed_dir/$scene_id"

if [ ! -d "$scene_dir" ]; then
    # Try date-based structure
    date_part=$(echo "$scene_id" | cut -d'_' -f1-3)
    if [ -d "$processed_dir/$date_part/$scene_id" ]; then
        scene_dir="$processed_dir/$date_part/$scene_id"
    elif [ -d "$processed_dir/$date_part" ]; then
         # Sometimes the scene directory IS the date directory if structure is flat?
         # Unlikely given scene_id includes drive.
         pass
    fi
fi

if [ ! -d "$scene_dir" ]; then
    echo "[ERROR] Cannot find scene directory for ID: $scene_id in $processed_dir"
    exit 1
fi

echo "[INFO] Scene Directory: $scene_dir"

# Activate environment if needed
# Assuming 'grounddino' env exists as per Waymo scripts, or use current if compatible.
# We try to source conda if available.
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    if conda env list | grep -q "grounddino"; then
        conda activate grounddino
    fi
fi

# Ensure PYTHONPATH
export PYTHONPATH="$(pwd):$PYTHONPATH"

# Run extraction
# KITTI has 2 cameras: 0 (Left), 1 (Right)
# Use --dataset_type kitti to handle coordinate system differences
python adsutido_tools/extract_traffic_lights.py \
    --scene_dir "$scene_dir" \
    --gdino_config third_party/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
    --gdino_checkpoint third_party/GroundingDINO/ckpt/grounding_dino/groundingdino_swint_ogc.pth \
    --cam_ids 0 1 \
    --save_masks --device cuda --merge \
    --dataset_type kitti
