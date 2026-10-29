#!/bin/bash
###############################################################################
# Waymo Drivestudio Pipeline
# Runs Training and Rendering using the Drivestudio codebase
# Usage: ./drivestudio_pipeline.sh <scene_id> [gpu_id]
###############################################################################

# ============================================================================
# Configuration
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
PARENT_ROOT="$(dirname "$PROJECT_ROOT")"

# Drivestudio Paths
DRIVESTUDIO_ROOT="/inspire/hdd/global_user/guoluosong-253108120129/project/drivestudio"
DRIVESTUDIO_ENV="/opt/conda/envs/drivestudio"

# Data Paths (Same as Adstudio)
PROCESSED_DATA_ROOT="/inspire/hdd/global_user/guoluosong-253108120129/project/data/waymo/processed"

# Output Paths (Store in Adstudio output or Drivestudio output? 
# User wanted "adstudio to run according to pipeline.sh flow", implies running IN adstudio context
# but using drivestudio algo. Let's store output in Drivestudio output to keep it with the code,
# or user might want it in adstudio. 
# "project/drivestudio also run according to pipeline.sh flow" -> likely implies running ON drivestudio.
# I'll use drivestudio output root to avoid permission/path mixing issues if they differ.)
OUTPUT_ROOT="${DRIVESTUDIO_ROOT}/output"

# Logs
LOG_DIR="${PROJECT_ROOT}/logs/drivestudio_pipeline"
mkdir -p "$LOG_DIR"

SCENE_ID="${1:-}"
GPU_ID="${2:-0}"
SCENE_LOG="${LOG_DIR}/scene_${SCENE_ID}.log"

# Configs
TRAIN_CONFIG="/inspire/hdd/global_user/guoluosong-253108120129/project/drivestudio/configs/omnire_extended_cam.yaml"
# Render config args are passed via command line in drivestudio

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

# ============================================================================
# Validation Functions (Adapted from pipeline.sh)
# ============================================================================

get_training_scene_dir() {
    local scene_dir="$PROCESSED_DATA_ROOT/training1/$SCENE_ID"
    
    # 1. Direct match
    if [[ -d "$scene_dir" ]]; then
        echo "$scene_dir"
        return 0
    fi
    
    # 2. Zero-padded (023 vs 23)
    local padded_id=$(printf "%03d" "$SCENE_ID" 2>/dev/null || echo "$SCENE_ID")
    if [[ -d "$PROCESSED_DATA_ROOT/training1/$padded_id" ]]; then
        echo "$PROCESSED_DATA_ROOT/training1/$padded_id"
        return 0
    fi
    
    return 1
}

verify_train_data() {
    if [[ ! -d "$PROCESSED_DATA_ROOT/training1" ]]; then
        log_error "Training data directory not found: $PROCESSED_DATA_ROOT/training1"
        return 1
    fi
    
    local scene_dir=$(get_training_scene_dir)
    if [[ -z "$scene_dir" ]]; then
        log_error "Scene data not found for ID: $SCENE_ID in $PROCESSED_DATA_ROOT/training1"
        return 1
    fi
    
    log_info "Found scene data at: $scene_dir"
    
    # Check for minimal required files (images)
    if [[ ! -d "$scene_dir/images" ]]; then
        log_error "Missing images directory in $scene_dir"
        return 1
    fi
    
    return 0
}

# ============================================================================
# Stage: Train
# ============================================================================

stage_train() {
    log_info "========== STAGE: Train (Drivestudio) =========="
    
    # Define experiment name and output path
    # Drivestudio uses output_root/project/run_name
    local project="waymo"
    local run_name="origintest_${SCENE_ID}"
    local exp_dir="${OUTPUT_ROOT}/${project}/${run_name}"
    local ckpt_path="${exp_dir}/checkpoint_final.pth"
    
    # Check if already trained
    if [[ -f "$ckpt_path" ]]; then
        log_info "Checkpoint found at $ckpt_path. Skipping training."
        return 0
    fi

    log_info "Starting training for scene $SCENE_ID..."
    log_info "Output Dir: $exp_dir"
    
    # Activate Environment
    source /opt/conda/etc/profile.d/conda.sh
    conda activate drivestudio
    
    # Run Training
    # Note: Using python from env
    cd "$DRIVESTUDIO_ROOT" || exit 1ß
    export PYTHONPATH="$DRIVESTUDIO_ROOT"
    
    CUDA_VISIBLE_DEVICES="$GPU_ID" python tools/train.py \
        --config_file "$TRAIN_CONFIG" \
        --output_root "$OUTPUT_ROOT" \
        --project "$project" \
        --run_name "$run_name" \
        dataset="waymo/3cams" \
        data.scene_idx="$SCENE_ID" \
        data.start_timestep=0 \
        data.end_timestep=-1 \
        2>&1 | tee -a "$SCENE_LOG"
        
    local status=${PIPESTATUS[0]}
    
    if [[ $status -eq 0 ]]; then
        if [[ -f "$ckpt_path" ]]; then
            log_success "Training completed. Checkpoint: $ckpt_path"
            return 0
        else
            log_error "Training script exited successfully but checkpoint not found at $ckpt_path"
            return 1
        fi
    else
        log_error "Training failed with exit code $status"
        return 1
    fi
}

# ============================================================================
# Stage: Render Novel Trajectory
# ============================================================================

stage_render() {
    log_info "========== STAGE: Render Novel Trajectory (Drivestudio) =========="
    
    local project="waymo"
    local run_name="origin_${SCENE_ID}"
    local exp_dir="${OUTPUT_ROOT}/${project}/${run_name}"
    local ckpt_path="${exp_dir}/checkpoint_final.pth"
    
    if [[ ! -f "$ckpt_path" ]]; then
        log_error "Checkpoint not found for rendering: $ckpt_path"
        return 1
    fi
    
    log_info "Rendering using checkpoint: $ckpt_path"
    
    # Activate Environment (if not already)
    source /opt/conda/etc/profile.d/conda.sh
    conda activate drivestudio
    
    cd "$DRIVESTUDIO_ROOT" || exit 1
    export PYTHONPATH="$DRIVESTUDIO_ROOT"
    
    # Render Config
    local fps=10
    local cam_ids=(0 1 2)
    local downscales=(1 1 1)
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
        local log_dir="${OUTPUT_ROOT}/${project}/${run_name}"
        
        # 2. Get GT images folder (using helper function)
        local scene_data_dir=$(get_training_scene_dir)
        local gt_folder="${scene_data_dir}/images"
        
        # 3. Iterate traj types
        for traj in "${traj_types[@]}"; do
             # Find latest output dir for this traj
             # Pattern: ${log_dir}/novel_traj/${traj}_step*
             # Use ls -td to sort by time
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
                 bash "${PROJECT_ROOT}/scripts/stitch_videos.sh" "$video_dir" >> "$SCENE_LOG" 2>&1
                 
                 local combined_video="${video_dir}/combined_layout.mp4"
                 if [[ -f "$combined_video" ]]; then
                     log_info "Stitching successful: $combined_video"
                 else
                     log_error "Stitching failed, combined video not found."
                 fi

                 # FID Calculation (Use cam0 video, NOT combined video)
                 # Find cam0 video
                 local cam0_video=$(find "$video_dir" -name "*cam0*.mp4" | head -n 1)
                 
                 if [[ -n "$cam0_video" && -f "$cam0_video" ]]; then
                     if [[ -d "$gt_folder" ]]; then
                         local fid_output="${video_dir}/fid_result.txt"
                         log_info "Calculating FID for cam0 video: $(basename "$cam0_video")..."
                         python "${PROJECT_ROOT}/adsutido_tools/cal_video_fid.py" \
                             --video "$cam0_video" \
                             --gt_folder "$gt_folder" > "$fid_output" 2>&1
                         
                         log_info "FID Result saved to $fid_output"
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

log_info "Initializing Drivestudio Pipeline for Scene $SCENE_ID"
check_gpu || exit 1

# 1. Verify Data
if ! verify_train_data; then
    log_error "Data verification failed. Aborting."
    exit 1
fi

# 2. Train
if ! stage_train; then
    log_error "Training failed. Aborting."
    exit 1
fi

# 3. Render
if ! stage_render; then
    log_error "Rendering failed. Aborting."
    exit 1
fi

log_success "All stages completed successfully for Scene $SCENE_ID"
