#!/bin/bash
###############################################################################
# NuScenes Traffic Light Extraction Script for AdStudio
# Usage: ./scripts/nuscene/extract_traffic_lights.sh <scene_name> [processed_dir]
# Example: ./scripts/nuscene/extract_traffic_lights.sh scene-0061 /path/to/processed/mini
###############################################################################

# Check args
if [ $# -lt 1 ]; then
    echo "ERROR: Usage: $0 <scene_name> [processed_dir]"
    echo "ERROR: Example: $0 scene-0061 /path/to/processed/mini"
    exit 1
fi

scene_name="$1"

if [ $# -ge 2 ]; then
    processed_dir="$2"
else
    # Default to project/data/nuscenes/processedads/mini
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
    PARENT_ROOT="$(dirname "$PROJECT_ROOT")"
    processed_dir="${PARENT_ROOT}/data/nuscenes/processedads/mini"
fi

processed_dir="${processed_dir%/}"
echo "[INFO] Scene Name: $scene_name"
echo "[INFO] Processed Dir: $processed_dir"

# Convert scene name to directory name (scene-0061 -> 061)
scene_idx=$(echo "$scene_name" | sed 's/scene-0*//')
scene_dir_name=$(printf "%03d" "$scene_idx")

# Find scene directory
scene_dir="$processed_dir/$scene_dir_name"

if [ ! -d "$scene_dir" ]; then
    echo "[ERROR] Cannot find scene directory: $scene_dir"
    echo "[ERROR] Looking for: $scene_dir_name (converted from $scene_name)"
    exit 1
fi

echo "[INFO] Scene Directory: $scene_dir"

# Activate environment if needed
# Assuming 'grounddino' env exists, or use current if compatible.
if command -v conda &> /dev/null; then
    CONDA_BASE=$(conda info --base)
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    if conda env list | grep -q "grounddino"; then
        conda activate grounddino
    fi
fi

# Get project root and set PYTHONPATH
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$PROJECT_ROOT" || exit 1
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Run extraction
# NuScenes typically has 6 cameras: 
# 0=CAM_FRONT, 1=CAM_FRONT_LEFT, 2=CAM_FRONT_RIGHT, 
# 3=CAM_BACK_LEFT, 4=CAM_BACK_RIGHT, 5=CAM_BACK
# For traffic lights, we primarily use front cameras
python adsutido_tools/extract_traffic_lights.py \
    --scene_dir "$scene_dir" \
    --gdino_config third_party/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
    --gdino_checkpoint third_party/GroundingDINO/ckpt/grounding_dino/groundingdino_swint_ogc.pth \
    --cam_ids 0 1 2 \
    --save_masks --device cuda --merge \
    --dataset_type nuscenes
