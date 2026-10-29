#!/bin/bash
###############################################################################
# KITTI Drivestudio Render Novel Views
# Runs Rendering using the Drivestudio codebase for KITTI dataset
# Usage: ./drivestudio_render_novel.sh <scene_id> [gpu_id]
###############################################################################

# ============================================================================
# Configuration
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
# Drivestudio Paths
DRIVESTUDIO_ROOT="/inspire/hdd/global_user/guoluosong-253108120129/project/drivestudio"
# Environment used in train.sh
CONDA_ENV_NAME="drivestudio"

# Data Paths (KITTI)
PROCESSED_DATA_ROOT="/inspire/hdd/global_user/guoluosong-253108120129/project/data/kitti/processeddrivestudio"

# Output Paths
OUTPUT_ROOT="${DRIVESTUDIO_ROOT}/output"
PROJECT_NAME="kitti"

# Logs
LOG_DIR="${PROJECT_ROOT}/logs/drivestudio_render_novel_kitti"
mkdir -p "$LOG_DIR"

SCENE_ID="${1:-}"
GPU_ID="${2:-0}"
SCENE_LOG="${LOG_DIR}/scene_${SCENE_ID}.log"

# ============================================================================
# Helper Functions
# ============================================================================

log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] $1" | tee -a "$SCENE_LOG"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] $1" | tee -a "$SCENE_LOG"
}

log_success() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [SUCCESS] $1" | tee -a "$SCENE_LOG"
}

check_gpu() {
    if ! command -v nvidia-smi &> /dev/null; then
        log_error "nvidia-smi not found"
        return 1
    fi
    return 0
}

get_training_scene_dir() {
    local scene_dir="$PROCESSED_DATA_ROOT/$SCENE_ID"
    
    # Check if scene directory exists directly
    if [[ -d "$scene_dir" ]]; then
        echo "$scene_dir"
        return 0
    fi
    
    return 1
}

# ============================================================================
# Main Render Logic
# ============================================================================

render_novel() {
    log_info "========== STAGE: Render Novel Trajectory (Drivestudio - KITTI) =========="
    
    local run_name="kittitest_${SCENE_ID}"
    local exp_dir="${OUTPUT_ROOT}/${PROJECT_NAME}/${run_name}"
    local ckpt_path="${exp_dir}/checkpoint_final.pth"
    
    if [[ ! -f "$ckpt_path" ]]; then
        log_error "Checkpoint not found for rendering: $ckpt_path"
        return 1
    fi
    
    log_info "Rendering using checkpoint: $ckpt_path"
    
    # Activate Environment (if not already)
    source /opt/conda/etc/profile.d/conda.sh
    conda activate "$CONDA_ENV_NAME"
    
    cd "$DRIVESTUDIO_ROOT" || exit 1
    export PYTHONPATH="$DRIVESTUDIO_ROOT"
    
    # Render Config
    local fps=10
    # KITTI usually has 2 cameras (dataset="kitti/2cams"), so we render 0 and 1
    local cam_ids=(0 1) 
    local downscales=(1 1)
    local traj_types=(
        "left_shift_1m"
        "left_shift_3m"
        "right_shift_1m"
        "right_shift_3m"
    )
    
    CUDA_VISIBLE_DEVICES="$GPU_ID" python sim_render/cam/render_novel_trajectory.py \
        --resume_from "$ckpt_path" \
        --traj_types "${traj_types[@]}" \
        --cam_ids "${cam_ids[@]}" \
        --downscales "${downscales[@]}" \
        --fps "$fps" \
        --render_rgb \
        --save_images \
        2>&1 | tee -a "$SCENE_LOG"
        
    local status=${PIPESTATUS[0]}
    
    if [[ $status -eq 0 ]]; then
        log_success "Rendering completed. Starting post-processing (Stitch & FID)..."

        # 1. Identify Log Dir
        local log_dir="${OUTPUT_ROOT}/${PROJECT_NAME}/${run_name}"
        
        # 2. Get GT images folder
        local scene_data_dir=$(get_training_scene_dir)
        local gt_folder="${scene_data_dir}/images"
        
        # 3. Iterate traj types
        for traj in "${traj_types[@]}"; do
             # Find latest output dir for this traj
             local traj_dirs=$(ls -td "${log_dir}/novel_traj/${traj}_step"* 2>/dev/null)
             
             if [[ -z "$traj_dirs" ]]; then
                 log_error "No output directory found for trajectory: $traj"
                 continue
             fi
             
             local latest_traj_dir=$(echo "$traj_dirs" | head -n 1)
             local video_dir="${latest_traj_dir}/videos"
             
             if [[ -d "$video_dir" ]]; then
                 log_info "Processing videos in: $video_dir"
                 
                 # Stitching
                 # Assuming stitch_videos.sh works for 2 cams too (it likely stitches whatever valid videos are there)
                 if [[ -f "${PROJECT_ROOT}/scripts/stitch_videos.sh" ]]; then
                     bash "${PROJECT_ROOT}/scripts/stitch_videos.sh" "$video_dir" >> "$SCENE_LOG" 2>&1
                     local combined_video="${video_dir}/combined_layout.mp4"
                     if [[ -f "$combined_video" ]]; then
                         log_info "Stitching successful: $combined_video"
                     fi
                 else
                     log_error "Stitching script not found at ${PROJECT_ROOT}/scripts/stitch_videos.sh"
                 fi

                 # FID Calculation (Use cam0 video)
                 local cam0_video=$(find "$video_dir" -name "*cam0*.mp4" | head -n 1)
                 
                 if [[ -n "$cam0_video" && -f "$cam0_video" ]]; then
                     if [[ -d "$gt_folder" ]]; then
                         local fid_output="${video_dir}/fid_result.txt"
                         log_info "Calculating FID for cam0 video: $(basename "$cam0_video")..."
                         
                         if [[ -f "${PROJECT_ROOT}/adsutido_tools/cal_video_fid.py" ]]; then
                             python "${PROJECT_ROOT}/adsutido_tools/cal_video_fid.py" \
                                 --video "$cam0_video" \
                                 --gt_folder "$gt_folder" > "$fid_output" 2>&1
                             
                             log_info "FID Result saved to $fid_output"
                         else
                             log_error "FID script not found at ${PROJECT_ROOT}/adsutido_tools/cal_video_fid.py"
                         fi
                     else
                         log_error "GT images not found at $gt_folder. Skipping FID."
                     fi
                 else
                     log_error "cam0 video not found in $video_dir. Skipping FID."
                 fi
             else
                 log_error "Video directory not found in $latest_traj_dir"
             fi
        done
        
        return 0
    else
        log_error "Rendering failed with exit code $status"
        return 1
    fi
}

# ============================================================================
# Main
# ============================================================================

if [[ -z "$SCENE_ID" ]]; then
    echo "Usage: $0 <scene_id> [gpu_id]"
    exit 1
fi

log_info "Initializing Drivestudio Render Novel for KITTI Scene $SCENE_ID"
check_gpu || exit 1

if ! render_novel; then
    log_error "Rendering failed. Aborting."
    exit 1
fi

log_success "Rendering completed successfully for Scene $SCENE_ID"
