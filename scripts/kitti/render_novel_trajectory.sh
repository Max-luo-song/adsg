#!/bin/bash
###############################################################################
# KITTI Render Novel Trajectory Script
# Usage: ./scripts/kitti/render_novel_trajectory.sh <scene_id> [processed_dir]
###############################################################################

# Check args
if [ $# -lt 1 ]; then
    echo "ERROR: Usage: $0 <scene_id> [processed_dir]"
    echo "ERROR: Example: $0 2011_09_26_drive_0001_sync /path/to/processed"
    exit 1
fi
CONDA_ENV_NAME="drivestudio"
source /opt/conda/etc/profile.d/conda.sh
conda activate "$CONDA_ENV_NAME"
scene_id="$1"

if [ $# -ge 2 ]; then
    processed_dir="$2"
else
    # Default to project/data/kitti/processed
    # Get project root from script location
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
    PARENT_ROOT="$(dirname "$PROJECT_ROOT")"
    processed_dir="${PARENT_ROOT}/data/kitti/processed"
fi

# Normalize path
processed_dir="${processed_dir%/}"

echo "[INFO] Scene ID: $scene_id"
echo "[INFO] Processed Dir: $processed_dir"

# Parameters
################################################################################
# Auto-detect checkpoint (use the most recent one)
found_ckpt=$(find output -path "output/kitti_${scene_id}/*/checkpoints/checkpoint_final.pth" -printf '%T@ %p\n' | sort -rn | head -n 1 | cut -d' ' -f2)

if [ -z "$found_ckpt" ]; then
    found_ckpt=$(find output -path "output/kitti_${scene_id}/*/checkpoint_final.pth" -printf '%T@ %p\n' | sort -rn | head -n 1 | cut -d' ' -f2)
fi

if [ -n "$found_ckpt" ]; then
    ckpt_path="$found_ckpt"
    echo "[INFO] Auto-detected checkpoint: $ckpt_path"
else
    echo "[ERROR] Could not find checkpoint for scene ${scene_id}. Please ensure training finished successfully."
    exit 1
fi

traj_types=(
    # original_traj
    left_shift_1m
    left_shift_3m
    # left_shift_5m
    right_shift_1m
    right_shift_3m
    # right_shift_5m
    # front_shift_1m
    # front_shift_3m
    # front_shift_5m
    # back_shift_1m
    # back_shift_3m
    # back_shift_5m
    # change_lane_1m
    # change_lane_2m
    # change_lane_3.5m
)

cam_ids=(0 1)
downscales=(1 1)
fps=10

render_rgb=true
render_depth=false
save_images=true
generate_lidar_pc=false
################################################################################

# Pick GPU
# Check if scripts/utils.sh exists relative to current dir or script dir
if [ -f "scripts/utils.sh" ]; then
    source scripts/utils.sh
elif [ -f "${PROJECT_ROOT}/scripts/utils.sh" ]; then
    source "${PROJECT_ROOT}/scripts/utils.sh"
else
    echo "Warning: scripts/utils.sh not found. GPU selection might fail if not specified."
fi

if command -v pick_gpu &> /dev/null; then
    gpu=$(pick_gpu)
else
    gpu=0
fi

if [ -z "${gpu}" ]; then
    echo "No GPU found"
    exit 1
fi

export PYTHONPATH=$(pwd)
echo "Using checkpoint: $ckpt_path"

bool_args=""
if [ "$render_rgb" = true ]; then
    bool_args="$bool_args --render_rgb"
fi
if [ "$render_depth" = true ]; then
    bool_args="$bool_args --render_depth"
fi
if [ "$save_images" = true ]; then
    bool_args="$bool_args --save_images"
fi
if [ "$generate_lidar_pc" = true ]; then
    bool_args="$bool_args --generate_lidar_pc"
fi

CUDA_VISIBLE_DEVICES=${gpu} python sim_render/cam/render_novel_trajectory.py \
    --resume_from $ckpt_path \
    --traj_types "${traj_types[@]}" \
    --cam_ids "${cam_ids[@]}" \
    --downscales "${downscales[@]}" \
    --fps $fps \
    $bool_args

# Capture exit code
RET_CODE=$?

if [ $RET_CODE -eq 0 ]; then
    echo "Rendering finished successfully. Starting video stitching..."

    LOG_DIR=$(dirname "$ckpt_path")
    
    # Iterate over trajectory types
    for traj in "${traj_types[@]}"; do
        if ls -d "$LOG_DIR/novel_traj/${traj}_step"* 1> /dev/null 2>&1; then
            LATEST_TRAJ_DIR=$(ls -td "$LOG_DIR/novel_traj/${traj}_step"* | head -n 1)
            
            VIDEO_DIR="${LATEST_TRAJ_DIR}/videos"
            
            if [ -d "$VIDEO_DIR" ]; then
                echo "[INFO] Stitching videos in: $VIDEO_DIR"
                if [ -f "scripts/stitch_videos.sh" ]; then
                     bash scripts/stitch_videos.sh "$VIDEO_DIR"
                elif [ -f "${PROJECT_ROOT}/scripts/stitch_videos.sh" ]; then
                     bash "${PROJECT_ROOT}/scripts/stitch_videos.sh" "$VIDEO_DIR"
                fi
                
                # FID Calculation
                CAM0_VIDEO=$(find "$VIDEO_DIR" -name "*cam0*.mp4" | head -n 1)

                if [ -n "$CAM0_VIDEO" ] && [ -f "$CAM0_VIDEO" ]; then
                    echo "[INFO] Calculating FID for $(basename "$CAM0_VIDEO")"
                    
                    FID_GT_BASE="${processed_dir}"
                    # Assuming scene_id is the folder name in processed dir
                    GT_FOLDER="${FID_GT_BASE}/${scene_id}/images"
                    
                    if [ -d "$GT_FOLDER" ]; then
                        FID_OUTPUT_FILE="$VIDEO_DIR/fid_result.txt"
                        
                        python adsutido_tools/cal_video_fid.py \
                            --video "$CAM0_VIDEO" \
                            --gt_folder "$GT_FOLDER" > "$FID_OUTPUT_FILE" 2>&1
                        
                        if [ $? -eq 0 ]; then
                            echo "[INFO] FID calculation successful. Results saved to $FID_OUTPUT_FILE"
                        else
                            echo "[ERROR] FID calculation failed. Check $FID_OUTPUT_FILE for details."
                        fi
                    else
                        echo "[WARN] GT folder not found at $GT_FOLDER. Skipping FID calculation."
                    fi
                else
                    echo "[WARN] cam0 video not found, skipping FID calculation."
                fi
            else
                echo "[WARN] 'videos' directory not found in $LATEST_TRAJ_DIR"
            fi
        else
            echo "[WARN] No output directory found for trajectory type: $traj in $LOG_DIR/novel_traj/"
        fi
    done
else
    echo "[ERROR] Rendering failed with exit code $RET_CODE"
    exit $RET_CODE
fi
