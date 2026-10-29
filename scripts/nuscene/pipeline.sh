#!/bin/bash
###############################################################################
# NuScenes Single Scene Processing Pipeline for AdStudio
# Usage: ./scripts/nuscene/pipeline.sh <scene_name> [gpu_id] [split]
# Example: ./scripts/nuscene/pipeline.sh scene-0000 0 v1.0-mini
# Example: ./scripts/nuscene/pipeline.sh scene-0061 0 v1.0-trainval
###############################################################################

# ============================================================================
# Configuration
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
PARENT_ROOT="$(dirname "$PROJECT_ROOT")"
LOG_DIR="${PROJECT_ROOT}/logs/pipeline_nuscene"

# Data Paths
RAW_DATA_ROOT_DEFAULT="${PARENT_ROOT}/data/sets/nuscenes"
RAW_DATA_ROOT_FULL="/inspire/hdd/global_user/guoluosong-253108120129/project/data/nuscenesdownloadzip"
RAW_DATA_ROOT="$RAW_DATA_ROOT_DEFAULT"
PROCESSED_DATA_ROOT="${PARENT_ROOT}/data/nuscenes/processedads"

# Pipeline-driven training/render config
TRAIN_OUTPUT_ROOT="output6cams"
TRAIN_PROJECT_NAME="nuscene"
TRAIN_DATASET_CONFIG="nuscenes/6cams"
TRAIN_CAMERA_IDS="0,1,2,3,4,5"
RENDER_CAMERA_IDS="0,1,2,3,4,5"
TL_EXTRACT_SCRIPT="${SCRIPT_DIR}/extract_traffic_lights_v3.sh"

# Default split
SPLIT_INPUT="${3:-v1.0-mini}"
case "$SPLIT_INPUT" in
    full|trainval|v1.0-trainval)
        SPLIT="v1.0-trainval"
        RAW_DATA_ROOT="$RAW_DATA_ROOT_FULL"
        ;;
    mini|v1.0-mini)
        SPLIT="v1.0-mini"
        ;;
    *)
        SPLIT="$SPLIT_INPUT"
        ;;
esac
SPLIT_SHORT="${SPLIT#v1.0-}"

# Keep this aligned with preprocess.sh invocation below.
PREPROCESS_INTERPOLATE_N=4
PREPROCESS_WORKERS=32
if [[ "$PREPROCESS_INTERPOLATE_N" -gt 0 ]]; then
    PREPROCESS_HZ=$(( (PREPROCESS_INTERPOLATE_N + 1) * 2 ))
    PROCESSED_WORK_ROOT="${PROCESSED_DATA_ROOT/processed/processed_${PREPROCESS_HZ}Hz}"
else
    PROCESSED_WORK_ROOT="$PROCESSED_DATA_ROOT"
fi
PROCESSED_SPLIT_ROOT="${PROCESSED_WORK_ROOT}/${SPLIT_SHORT}"

# Default GPU ID
GPU_ID="${2:-0}"

# Init log file (will be updated after scene_name is validated)
SCENE_LOG="/tmp/pipeline_nuscene_init.log"

set -euo pipefail

# ============================================================================
# Markers
# ============================================================================
get_scene_index() {
    # Extract scene index from scene name (e.g., scene-0061 -> 61)
    echo "$SCENE_NAME" | sed 's/scene-0*//'
}

get_scene_dir_name() {
    # Convert scene name to directory name (e.g., scene-0061 -> 061)
    local idx=$(get_scene_index)
    printf "%03d" "$idx"
}

get_scene_marker_dir() {
    local scene_dir=$(get_scene_dir_name)
    echo "${PROCESSED_SPLIT_ROOT}/${scene_dir}"
}

init_scene_markers() {
    # Define markers for incremental processing
    PREPROCESS_MARKER="$(get_scene_marker_dir)/.preprocess_complete"
    TRAIN_MARKER="$(get_scene_marker_dir)/.train_complete"
    RENDER_MARKER="$(get_scene_marker_dir)/.render_complete"
    EXTRACT_TL_MARKER="$(get_scene_marker_dir)/.extract_traffic_lights_complete"

    # Define locks
    PREPROCESS_LOCK="$(get_scene_marker_dir)/.preprocess.lock"
    TRAIN_LOCK="$(get_scene_marker_dir)/.train.lock"
    EXTRACT_TL_LOCK="$(get_scene_marker_dir)/.extract_traffic_lights.lock"
}

# ============================================================================
# Logging Functions
# ============================================================================
log_info() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[${timestamp}] [INFO] $*" | tee -a "$SCENE_LOG"
}

log_error() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[${timestamp}] [ERROR] $*" | tee -a "$SCENE_LOG" >&2
}

log_success() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[${timestamp}] [SUCCESS] $*" | tee -a "$SCENE_LOG"
}

log_warning() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[${timestamp}] [WARN] $*" | tee -a "$SCENE_LOG"
}

# ============================================================================
# Verification Functions
# ============================================================================

check_preprocess_done() {
    [[ -f "$PREPROCESS_MARKER" ]] && return 0
    return 1
}

verify_preprocessed_data() {
    local scene_dir_name=$(get_scene_dir_name)
    local scene_dir="${PROCESSED_SPLIT_ROOT}/${scene_dir_name}"
    
    if [[ ! -d "$scene_dir" ]]; then
        log_error "Validation failed: Scene directory not found at $scene_dir"
        return 1
    fi

    # Check for images
    if [[ ! -d "$scene_dir/images" ]]; then
        log_error "Validation failed: images directory missing"
        return 1
    fi
    
    # Check for semantic masks (road/sky)
    if [[ ! -d "$scene_dir/road_masks" ]] || [[ ! -d "$scene_dir/sky_masks" ]]; then
        log_error "Validation failed: Semantic masks missing"
        return 1
    fi
    
    # Check for human pose (smpl.pkl)
    if [[ ! -d "$scene_dir/humanpose" ]]; then
        log_error "Validation failed: humanpose directory missing"
        return 1
    fi
    
    return 0
}

verify_traffic_lights_data() {
    local scene_dir_name=$(get_scene_dir_name)
    local scene_dir="${PROCESSED_SPLIT_ROOT}/${scene_dir_name}"
    
    # Check for traffic light instances json
    if [[ ! -f "$scene_dir/instances/traffic_light_instances_info.json" ]] && \
       [[ ! -f "$scene_dir/instances/instances_info.json" ]]; then
        log_error "Validation failed: Traffic light instances info missing"
        return 1
    fi
    return 0
}

verify_training_complete() {
    local train_output_dir="$PROJECT_ROOT/${TRAIN_OUTPUT_ROOT}/${TRAIN_PROJECT_NAME}/${SCENE_NAME}"
    
    if [[ ! -d "$train_output_dir" ]]; then
        log_error "Validation failed: Training output directory not found at $train_output_dir"
        return 1
    fi
    
    local checkpoints=$(find "$train_output_dir" -name "*.pth" | head -n 1)
    if [[ -z "$checkpoints" ]]; then
        log_error "Validation failed: No checkpoints found in $train_output_dir"
        return 1
    fi
    
    return 0
}

verify_rendered_output() {
    local train_output_dir="$PROJECT_ROOT/${TRAIN_OUTPUT_ROOT}/${TRAIN_PROJECT_NAME}/${SCENE_NAME}"
    
    if [[ ! -d "$train_output_dir" ]]; then
        log_error "Validation failed: Training output directory not found"
        return 1
    fi
    
    local novel_traj_dirs=$(find "$train_output_dir" -type d -name "novel_traj")
    if [[ -z "$novel_traj_dirs" ]]; then
        log_error "Validation failed: novel_traj directory not found"
        return 1
    fi
    
    local render_files=$(find "$novel_traj_dirs" -name "*.mp4" -o -name "*.jpg" -o -name "*.png" | head -n 1)
    if [[ -z "$render_files" ]]; then
        log_error "Validation failed: No rendered files found"
        return 1
    fi
    
    return 0
}

validate_inputs() {
    if [[ $# -lt 1 ]]; then
        echo "ERROR: Usage: $0 <scene_name> [gpu_id] [split]"
        echo "ERROR: Example: $0 scene-0061 0 v1.0-mini"
        exit 1
    fi

    SCENE_NAME="$1"
    
    if ! [[ "$GPU_ID" =~ ^[0-9]+$ ]]; then
        echo "ERROR: gpu_id must be a number, got: $GPU_ID"
        exit 1
    fi

    log_info "========================================="
    log_info "NuScenes Pipeline - Scene $SCENE_NAME"
    log_info "========================================="
    log_info "Split: $SPLIT ($SPLIT_SHORT)"
    log_info "GPU ID: $GPU_ID"
    log_info "Project Root: $PROJECT_ROOT"
    log_info "Raw Data Root: $RAW_DATA_ROOT"
    log_info "Processed Data Root: $PROCESSED_DATA_ROOT"
    log_info "Processed Work Root: $PROCESSED_WORK_ROOT"
    log_info "Processed Split Root: $PROCESSED_SPLIT_ROOT"
    log_info "Train Output Root: ${TRAIN_OUTPUT_ROOT}/${TRAIN_PROJECT_NAME}"
    log_info "Train Dataset Config: $TRAIN_DATASET_CONFIG"
    log_info "Train Camera IDs: $TRAIN_CAMERA_IDS"
    log_info "Render Camera IDs: $RENDER_CAMERA_IDS"
    log_info "Traffic Light Script: $TL_EXTRACT_SCRIPT"
    log_info "Log File: $SCENE_LOG"
}

# ============================================================================
# Stages
# ============================================================================

stage_preprocess() {
    log_info "========== STAGE 1-3: Preprocessing (Convert, Segment, Pose) =========="
    
    mkdir -p "$(dirname "$PREPROCESS_MARKER")"
    
    # Check if already done
    if [[ -f "$PREPROCESS_MARKER" ]]; then
        if verify_preprocessed_data; then
            log_warning "Preprocessing already completed and valid (Marker: $PREPROCESS_MARKER)"
            log_info "Skipping preprocessing stage"
            return 0
        else
            log_warning "Marker exists but data invalid, re-running"
            rm -f "$PREPROCESS_MARKER"
        fi
    fi

    mkdir -p "$(dirname "$PREPROCESS_LOCK")"
    {
        flock 200 || { log_error "Cannot acquire preprocess lock"; return 1; }
        
        # Double check marker inside lock
        if [[ -f "$PREPROCESS_MARKER" ]]; then
            log_info "Preprocessing completed by another process. Skipping."
            return 0
        fi

        log_info "Starting Preprocessing for Scene $SCENE_NAME..."
        
        # Get scene index for start_idx
        scene_idx=$(get_scene_index)
        
        if [[ -f "${SCRIPT_DIR}/preprocess.sh" ]]; then
            # Call preprocess.sh
            # Usage: ./preprocess.sh [start_idx] [num_scenes] [input_dir] [output_dir] [split] [interpolate_N] [workers] [gpu_id]
            bash "${SCRIPT_DIR}/preprocess.sh" "$scene_idx" 1 "$RAW_DATA_ROOT" "$PROCESSED_DATA_ROOT" "$SPLIT" "$PREPROCESS_INTERPOLATE_N" "$PREPROCESS_WORKERS" "$GPU_ID" 2>&1 | tee -a "$SCENE_LOG"
            
            if [[ ${PIPESTATUS[0]} -eq 0 ]]; then
                if verify_preprocessed_data; then
                    log_success "Preprocessing completed successfully"
                    mkdir -p "$(dirname "$PREPROCESS_MARKER")"
                    touch "$PREPROCESS_MARKER"
                else
                    log_error "Preprocessing succeeded but validation failed"
                    return 1
                fi
            else
                log_error "Preprocessing failed"
                return 1
            fi
        else
            log_error "preprocess.sh not found in ${SCRIPT_DIR}"
            return 1
        fi
    } 200>"$PREPROCESS_LOCK"
}

stage_extract_traffic_lights() {
    log_info "========== STAGE 4: Extract Traffic Lights =========="
    
    mkdir -p "$(dirname "$EXTRACT_TL_MARKER")"
    
    if [[ -f "$EXTRACT_TL_MARKER" ]]; then
        if verify_traffic_lights_data; then
            log_warning "Traffic light extraction already completed and valid"
            log_info "Skipping traffic light extraction stage"
            return 0
        else
            log_warning "Marker exists but data invalid, re-running"
            rm -f "$EXTRACT_TL_MARKER"
        fi
    fi
    
    if [[ ! -f "$PREPROCESS_MARKER" ]]; then
        log_error "Preprocessing not complete, cannot extract traffic lights"
        return 1
    fi
    
    mkdir -p "$(dirname "$EXTRACT_TL_LOCK")"
    {
        flock 200 || { log_error "Cannot acquire traffic light extraction lock"; return 1; }
        
        if [[ -f "$EXTRACT_TL_MARKER" ]]; then
            log_info "Traffic light extraction completed by another process. Skipping."
            return 0
        fi

        log_info "Starting Traffic Light Extraction for Scene $SCENE_NAME..."
        
        if [[ -f "$TL_EXTRACT_SCRIPT" ]]; then
            bash "$TL_EXTRACT_SCRIPT" "$SCENE_NAME" "$PROCESSED_SPLIT_ROOT" "$GPU_ID" 2>&1 | tee -a "$SCENE_LOG"
            
            if [[ ${PIPESTATUS[0]} -eq 0 ]]; then
                if verify_traffic_lights_data; then
                    log_success "Traffic light extraction completed successfully"
                    mkdir -p "$(dirname "$EXTRACT_TL_MARKER")"
                    touch "$EXTRACT_TL_MARKER"
                else
                    log_error "Traffic light extraction succeeded but validation failed"
                    return 1
                fi
            else
                log_error "Traffic light extraction failed"
                return 1
            fi
        else
            log_warning "Traffic light script not found: $TL_EXTRACT_SCRIPT, skipping"
            # Mark as complete since this is optional
            touch "$EXTRACT_TL_MARKER"
            return 0
        fi
    } 200>"$EXTRACT_TL_LOCK"
}

stage_train() {
    log_info "========== STAGE 5: Training =========="
    
    mkdir -p "$(dirname "$TRAIN_MARKER")"
    
    if [[ -f "$TRAIN_MARKER" ]]; then
        if verify_training_complete; then
            log_warning "Training already completed and valid"
            log_info "Skipping training stage"
            return 0
        else
            log_warning "Marker exists but model invalid, re-training"
            rm -f "$TRAIN_MARKER"
        fi
    fi
    
    if [[ ! -f "$PREPROCESS_MARKER" ]]; then
        log_error "Preprocessing not complete, cannot train"
        return 1
    fi
    
    log_info "Starting Training for Scene $SCENE_NAME (GPU: $GPU_ID)..."
    
    if [[ -f "${SCRIPT_DIR}/train.sh" ]]; then
        bash "${SCRIPT_DIR}/train.sh" \
            "$SCENE_NAME" \
            "$PROCESSED_SPLIT_ROOT" \
            0 \
            -1 \
            "$GPU_ID" \
            "$TRAIN_OUTPUT_ROOT" \
            "$TRAIN_DATASET_CONFIG" \
            "$TRAIN_CAMERA_IDS" 2>&1 | tee -a "$SCENE_LOG"
        
        if [[ ${PIPESTATUS[0]} -eq 0 ]]; then
            if verify_training_complete; then
                log_success "Training completed successfully"
                mkdir -p "$(dirname "$TRAIN_MARKER")"
                touch "$TRAIN_MARKER"
            else
                log_error "Training succeeded but validation failed"
                return 1
            fi
        else
            log_error "Training failed"
            return 1
        fi
    else
        log_error "train.sh not found"
        return 1
    fi
}

stage_render() {
    log_info "========== STAGE 6: Render Novel Trajectory =========="
    
    mkdir -p "$(dirname "$RENDER_MARKER")"
    
    if [[ -f "$RENDER_MARKER" ]]; then
        if verify_rendered_output; then
            log_warning "Rendering already completed and valid"
            log_info "Skipping rendering stage"
            return 0
        else
            log_warning "Marker exists but output invalid, re-rendering"
            rm -f "$RENDER_MARKER"
        fi
    fi
    
    if [[ ! -f "$TRAIN_MARKER" ]]; then
        log_error "Training not complete, cannot render"
        return 1
    fi
    
    log_info "Starting Rendering for Scene $SCENE_NAME (GPU: $GPU_ID)..."
    
    if [[ -f "${SCRIPT_DIR}/render_novel_trajectory.sh" ]]; then
        bash "${SCRIPT_DIR}/render_novel_trajectory.sh" \
            "$SCENE_NAME" \
            "$PROCESSED_SPLIT_ROOT" \
            "$GPU_ID" \
            "$TRAIN_OUTPUT_ROOT" \
            "$RENDER_CAMERA_IDS" 2>&1 | tee -a "$SCENE_LOG"
        
        if [[ ${PIPESTATUS[0]} -eq 0 ]]; then
            if verify_rendered_output; then
                log_success "Rendering completed successfully"
                mkdir -p "$(dirname "$RENDER_MARKER")"
                touch "$RENDER_MARKER"
            else
                log_error "Rendering succeeded but validation failed"
                return 1
            fi
        else
            log_warning "Rendering failed (this is optional)"
            # Mark as complete anyway since rendering is optional
            touch "$RENDER_MARKER"
        fi
    else
        log_warning "render_novel_trajectory.sh not found, skipping"
        # Mark as complete since this is optional
        touch "$RENDER_MARKER"
        return 0
    fi
}

# ============================================================================
# Main
# ============================================================================
main() {
    if [[ $# -lt 1 ]]; then
        echo "ERROR: Usage: $0 <scene_name> [gpu_id] [split]"
        echo "ERROR: Example: $0 scene-0061 0 v1.0-mini"
        exit 1
    fi
    
    local scene_name_arg="$1"
    SCENE_NAME="$scene_name_arg"
    
    init_scene_markers
    
    mkdir -p "$LOG_DIR"
    export SCENE_LOG="${LOG_DIR}/scene_${SCENE_NAME}.log"
    
    validate_inputs "$@"
    
    local start_time=$(date +%s)
    log_info "Start processing Scene $SCENE_NAME"
    
    if ! stage_preprocess; then
        log_error "Pipeline aborted: Preprocessing failed"
        exit 1
    fi
    
    if ! stage_extract_traffic_lights; then
        log_error "Pipeline aborted: Traffic Light Extraction failed"
        exit 1
    fi
    
    if ! stage_train; then
        log_error "Pipeline aborted: Training failed"
        exit 1
    fi
    
    if ! stage_render; then
        log_error "Pipeline aborted: Rendering failed"
        exit 1
    fi
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local hours=$((duration / 3600))
    local minutes=$(((duration % 3600) / 60))
    local seconds=$((duration % 60))
    
    log_success "========================================="
    log_success "Pipeline Completed! Scene $SCENE_NAME"
    log_success "Total Duration: ${hours}h ${minutes}m ${seconds}s"
    log_success "========================================="
}

main "$@"
