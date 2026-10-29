#!/bin/bash
###############################################################################
# NuScenes Render Novel Trajectory Script for AdStudio
# Usage: ./scripts/nuscene/render_novel_trajectory.sh <scene_name> [processed_dir] [gpu_id] [output_root] [cam_ids_csv]
# Example: ./scripts/nuscene/render_novel_trajectory.sh scene-0061 /path/to/processed/mini 0 output6cams 0,1,2,3,4,5
###############################################################################

# Check args
if [ $# -lt 1 ]; then
    echo "ERROR: Usage: $0 <scene_name> [processed_dir] [gpu_id] [output_root] [cam_ids_csv]"
    echo "ERROR: Example: $0 scene-0061 /path/to/processed/mini 0 output6cams 0,1,2,3,4,5"
    exit 1
fi

CONDA_ENV_NAME="drivestudio"
CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME"

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

# Normalize path
processed_dir="${processed_dir%/}"
gpu_arg="${3:--1}"
output_root="${4:-output6cams}"
cam_ids_csv="${5:-0,1,2}"

IFS=',' read -r -a cam_ids <<< "$cam_ids_csv"
if [ "${#cam_ids[@]}" -eq 0 ]; then
    echo "[ERROR] cam_ids_csv is empty: $cam_ids_csv"
    exit 1
fi
downscales=()
for _ in "${cam_ids[@]}"; do
    downscales+=(1)
done

echo "[INFO] Scene Name: $scene_name"
echo "[INFO] Processed Dir: $processed_dir"
echo "[INFO] GPU ID: $gpu_arg"
echo "[INFO] Output Root: $output_root"
echo "[INFO] Cam IDs: $cam_ids_csv"

# Parameters
################################################################################
project_name="nuscene"
run_root="${output_root}/${project_name}/${scene_name}"

# Auto-detect checkpoint (use the most recent one)
found_ckpt=$(find "$run_root" -path "${run_root}/checkpoints/checkpoint_final.pth" -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -n 1 | cut -d' ' -f2)

if [ -z "$found_ckpt" ]; then
    found_ckpt=$(find "$run_root" -path "${run_root}/checkpoint_final.pth" -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -n 1 | cut -d' ' -f2)
fi

# Backward-compatible fallback for legacy output layout
if [ -z "$found_ckpt" ]; then
    found_ckpt=$(find output -path "output/nuscenes_${scene_name}/*/checkpoints/checkpoint_final.pth" -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -n 1 | cut -d' ' -f2)
fi

if [ -z "$found_ckpt" ]; then
    found_ckpt=$(find output -path "output/nuscenes_${scene_name}/*/checkpoint_final.pth" -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -n 1 | cut -d' ' -f2)
fi

if [ -n "$found_ckpt" ]; then
    ckpt_path="$found_ckpt"
    echo "[INFO] Auto-detected checkpoint: $ckpt_path"
else
    echo "[ERROR] Could not find checkpoint for scene ${scene_name}. Please ensure training finished successfully."
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

fps=10

render_rgb=true
render_depth=false
save_images=true
generate_lidar_pc=false
################################################################################

# Get project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Pick GPU
if [ -f "${PROJECT_ROOT}/scripts/utils.sh" ]; then
    source "${PROJECT_ROOT}/scripts/utils.sh"
fi

if [ "$gpu_arg" = "-1" ] || [ "$gpu_arg" = "auto" ]; then
    if command -v pick_gpu &> /dev/null; then
        gpu=$(pick_gpu)
    else
        gpu=0
    fi
else
    gpu="$gpu_arg"
fi

if [ -z "${gpu}" ]; then
    echo "No GPU found"
    exit 1
fi

cd "$PROJECT_ROOT" || exit 1
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
                    
                    # Convert scene name to directory name (scene-0061 -> 061)
                    scene_idx=$(echo "$scene_name" | sed 's/scene-0*//')
                    scene_dir_name=$(printf "%03d" "$scene_idx")
                    
                    FID_GT_BASE="${processed_dir}"
                    GT_FOLDER="${FID_GT_BASE}/${scene_dir_name}/images"
                    
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
