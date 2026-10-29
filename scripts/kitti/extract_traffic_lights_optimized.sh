#!/bin/bash
###############################################################################
# KITTI Traffic Light Extraction Script (Optimized)
# 优化参数以提高红绿灯检测率
# Usage: ./scripts/kitti/extract_traffic_lights_optimized.sh <scene_id> [processed_dir]
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
scene_dir="$processed_dir/$scene_id"

if [ ! -d "$scene_dir" ]; then
    # Try date-based structure
    date_part=$(echo "$scene_id" | cut -d'_' -f1-3)
    if [ -d "$processed_dir/$date_part/$scene_id" ]; then
        scene_dir="$processed_dir/$date_part/$scene_id"
    elif [ -d "$processed_dir/$date_part" ]; then
         pass
    fi
fi

if [ ! -d "$scene_dir" ]; then
    echo "[ERROR] Cannot find scene directory for ID: $scene_id in $processed_dir"
    exit 1
fi

echo "[INFO] Scene Directory: $scene_dir"

# Activate environment
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    if conda env list | grep -q "grounddino"; then
        conda activate grounddino
    fi
fi

# Ensure PYTHONPATH
export PYTHONPATH="$(pwd):$PYTHONPATH"

# Run extraction with optimized parameters for KITTI
echo "[INFO] Running optimized traffic light extraction..."
echo "[INFO] Optimizations:"
echo "  - Lower detection thresholds (box: 0.25, text: 0.20)"
echo "  - Relaxed clustering (min_samples: 1, eps: 3.0)"
echo "  - Lower frame threshold (min_tl_frames: 2)"

python adsutido_tools/extract_traffic_lights.py \
    --scene_dir "$scene_dir" \
    --gdino_config third_party/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
    --gdino_checkpoint third_party/GroundingDINO/ckpt/grounding_dino/groundingdino_swint_ogc.pth \
    --cam_ids 0 1 \
    --box_threshold 0.25 \
    --text_threshold 0.20 \
    --cluster_eps 3.0 \
    --cluster_min_samples 1 \
    --cluster_merge_eps 2.0 \
    --min_tl_frames 2 \
    --save_masks --device cuda --merge \
    --dataset_type kitti

if [ $? -eq 0 ]; then
    echo "[SUCCESS] Traffic light extraction completed"
    
    # Show statistics
    echo ""
    echo "[INFO] Extraction statistics:"
    python3 << EOF
import json
import os

inst_file = os.path.join("$scene_dir", "instances/instances_info.json")
if os.path.exists(inst_file):
    with open(inst_file) as f:
        data = json.load(f)
    
    tl_count = 0
    tl_frames = []
    for k, v in data.items():
        if v['class_name'] == 'TrafficLight':
            tl_count += 1
            tl_frames.append(len(v['frame_annotations']['frame_idx']))
    
    if tl_count > 0:
        print(f"  Total TrafficLights detected: {tl_count}")
        print(f"  Total observations: {sum(tl_frames)}")
        print(f"  Avg frames per TL: {sum(tl_frames)/len(tl_frames):.1f}")
        print(f"  Min/Max frames: {min(tl_frames)}/{max(tl_frames)}")
    else:
        print("  WARNING: No traffic lights detected!")
else:
    print("  ERROR: instances_info.json not found")
EOF
else
    echo "[ERROR] Traffic light extraction failed"
    exit 1
fi
