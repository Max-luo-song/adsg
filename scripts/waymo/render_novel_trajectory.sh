#!/bin/bash
###############################################################################
# 渲染新轨迹脚本
# 用法: ./render_novel_trajectory.sh <scene_id> [processed_dir]
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

# 参数设置
################################################################################
# 尝试自动查找当前场景的 checkpoint
# 默认路径结构: output/waymo_${scene_id}/<timestamp>_<config>/checkpoints/checkpoint_final.pth
# 实际发现结构: output/waymo_23/20260317_lidar+cam0_1_2adstudiov1/checkpoint_final.pth
# 也就是说 checkpoints 子目录可能不存在，或者是在根目录下

# 先尝试标准结构
found_ckpt=$(find output -path "output/waymo_${scene_id}/*/checkpoints/checkpoint_final.pth" | head -n 1)

# 如果没找到，尝试在实验目录下直接查找
if [ -z "$found_ckpt" ]; then
    found_ckpt=$(find output -path "output/waymo_${scene_id}/*/checkpoint_final.pth" | head -n 1)
fi

if [ -n "$found_ckpt" ]; then
    ckpt_path="$found_ckpt"
    echo "[INFO] Auto-detected checkpoint: $ckpt_path"
else
    # Fallback to hardcoded path (warning user)
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

# cam_ids=(0 1 2 3 5 6 7 9 10 11 12)
# downscales=(1 1 1 1 1 1 1 1 1 1 1)

cam_ids=(0 1 2)
downscales=(1 1 1)
fps=10

render_rgb=true
render_depth=false
save_images=true
generate_lidar_pc=false
################################################################################
source scripts/utils.sh

export PYTHONPATH=$(pwd)

gpu=$(pick_gpu)
if [ -z "${gpu}" ]; then
    echo "no gpu found"
    exit 1
fi


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

# Capture the exit code
RET_CODE=$?

if [ $RET_CODE -eq 0 ]; then
    echo "Rendering finished successfully. Starting video stitching..."

    # Determine the log directory (directory containing the checkpoint)
    LOG_DIR=$(dirname "$ckpt_path")
    
    # Iterate over each requested trajectory type
    for traj in "${traj_types[@]}"; do
        # Search for the output directory for this trajectory
        # Pattern: ${LOG_DIR}/novel_traj/${traj}_step*
        # We sort by modification time (newest first) to get the one we just generated
        
        # Check if any matching directory exists first
        if ls -d "$LOG_DIR/novel_traj/${traj}_step"* 1> /dev/null 2>&1; then
            LATEST_TRAJ_DIR=$(ls -td "$LOG_DIR/novel_traj/${traj}_step"* | head -n 1)
            
            VIDEO_DIR="${LATEST_TRAJ_DIR}/videos"
            
            if [ -d "$VIDEO_DIR" ]; then
                echo "[INFO] Stitching videos in: $VIDEO_DIR"
                bash scripts/stitch_videos.sh "$VIDEO_DIR"
                
                # FID Calculation
                # Find cam0 video
                CAM0_VIDEO=$(find "$VIDEO_DIR" -name "*cam0*.mp4" | head -n 1)

                if [ -n "$CAM0_VIDEO" ] && [ -f "$CAM0_VIDEO" ]; then
                    echo "[INFO] Calculating FID for $(basename "$CAM0_VIDEO")"
                    
                    # Define GT folder path base
                    FID_GT_BASE="/inspire/hdd/global_user/guoluosong-253108120129/project/data/waymo/processed/training"
                    # Handle scene_id padding if necessary (assuming scene_id variable is available)
                    # Simple check: if scene_id < 100 and gt folder not found, try padding
                    GT_FOLDER="${FID_GT_BASE}/${scene_id}/images"
                    
                    if [ ! -d "$GT_FOLDER" ]; then
                         # Try padding with 3 zeros
                         PADDED_ID=$(printf "%03d" "$scene_id")
                         GT_FOLDER="${FID_GT_BASE}/${PADDED_ID}/images"
                    fi
                    
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
                        echo "[WARN] GT folder not found at $GT_FOLDER (or padded version). Skipping FID calculation."
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

