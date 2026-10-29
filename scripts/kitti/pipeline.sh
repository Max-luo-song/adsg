#!/bin/bash
###############################################################################
# KITTI Single Scene Processing Pipeline
# Adapts the Waymo processing pipeline for KITTI dataset
# Usage: ./scripts/kitti/pipeline.sh <scene_id> [gpu_id]
# Example: ./scripts/kitti/pipeline.sh 2011_09_26_drive_0059_sync 0
###############################################################################

# ============================================================================
# Configuration
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
PARENT_ROOT="$(dirname "$PROJECT_ROOT")"
LOG_DIR="${PROJECT_ROOT}/logs/pipeline_kitti"

# Data Paths
# We assume data is located in project/data/kitti/
RAW_DATA_ROOT="${PARENT_ROOT}/data/kitti/raw"
PROCESSED_DATA_ROOT="${PARENT_ROOT}/data/kitti/processedads"

# Default GPU ID
GPU_ID="${2:-0}"

# Init log file (will be updated after scene_id is validated)
SCENE_LOG="/tmp/pipeline_kitti_init.log"

set -euo pipefail

# ============================================================================
# Markers
# ============================================================================
get_scene_marker_dir() {
    echo "${PROCESSED_DATA_ROOT}/scene_${SCENE_ID}"
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
    # Check if processed data exists
    local scene_dir="${PROCESSED_DATA_ROOT}/${SCENE_ID}"
    
    # Preprocess creates scene_id folder directly in output_dir
    # Or sometimes it might be date/scene_id depending on script
    # Looking at preprocess.sh:
    # python datasets/preprocess.py ... --target_dir "$output_dir"
    # Drivestudio preprocess usually creates "kitti/processed/<scene_id>" or "kitti/processed/date/scene_id"
    # But let's assume flat structure based on Waymo pipeline or date-based.
    # KITTI preprocess usually organizes by date.
    
    # Check if scene dir exists directly or nested
    if [[ -d "${PROCESSED_DATA_ROOT}/${SCENE_ID}" ]]; then
        scene_dir="${PROCESSED_DATA_ROOT}/${SCENE_ID}"
    else
        # Try date based
        local date_part=$(echo "$SCENE_ID" | cut -d'_' -f1-3)
        if [[ -d "${PROCESSED_DATA_ROOT}/${date_part}/${SCENE_ID}" ]]; then
             scene_dir="${PROCESSED_DATA_ROOT}/${date_part}/${SCENE_ID}"
        elif [[ -d "${PROCESSED_DATA_ROOT}/${date_part}" ]]; then
             # Maybe just date folder
             scene_dir="${PROCESSED_DATA_ROOT}/${date_part}" 
        fi
    fi

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
    local scene_dir="${PROCESSED_DATA_ROOT}/${SCENE_ID}"
    
    # Check if scene dir exists directly or nested (copy logic from verify_preprocessed_data)
    if [[ -d "${PROCESSED_DATA_ROOT}/${SCENE_ID}" ]]; then
        scene_dir="${PROCESSED_DATA_ROOT}/${SCENE_ID}"
    else
        local date_part=$(echo "$SCENE_ID" | cut -d'_' -f1-3)
        if [[ -d "${PROCESSED_DATA_ROOT}/${date_part}/${SCENE_ID}" ]]; then
             scene_dir="${PROCESSED_DATA_ROOT}/${date_part}/${SCENE_ID}"
        fi
    fi

    # Check for traffic light instances json
    if [[ ! -f "$scene_dir/instances/traffic_light_instances_info.json" ]] && \
       [[ ! -f "$scene_dir/instances/instances_info.json" ]]; then
        log_error "Validation failed: Traffic light instances info missing"
        return 1
    fi
    return 0
}

verify_training_complete() {
    local train_output_dir="$PROJECT_ROOT/output/kitti_${SCENE_ID}"
    
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
    local train_output_dir="$PROJECT_ROOT/output/kitti_${SCENE_ID}"
    
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

# ============================================================================
# Input Validation
# ============================================================================
validate_inputs() {
    if [[ $# -lt 1 ]]; then
        echo "ERROR: Usage: $0 <scene_id> [gpu_id]"
        exit 1
    fi

    SCENE_ID="$1"
    # KITTI scene ID is a string (e.g. 2011_09_26_drive_0001_sync), so we relax the number check
    
    if ! [[ "$GPU_ID" =~ ^[0-9]+$ ]]; then
        echo "ERROR: gpu_id must be a number, got: $GPU_ID"
        exit 1
    fi

    log_info "========================================="
    log_info "KITTI Pipeline - Scene $SCENE_ID"
    log_info "========================================="
    log_info "GPU ID: $GPU_ID"
    log_info "Project Root: $PROJECT_ROOT"
    log_info "Raw Data Root: $RAW_DATA_ROOT"
    log_info "Processed Data Root: $PROCESSED_DATA_ROOT"
    log_info "Log Dir: $SCENE_LOG"
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

        log_info "Starting Preprocessing for Scene $SCENE_ID..."
        
        if [[ -f "${SCRIPT_DIR}/preprocess.sh" ]]; then
            # Call preprocess.sh
            # Usage: ./preprocess.sh <scene_id> [input_dir] [output_dir]
            bash "${SCRIPT_DIR}/preprocess.sh" "$SCENE_ID" "$RAW_DATA_ROOT" "$PROCESSED_DATA_ROOT" 2>&1 | tee -a "$SCENE_LOG"
            
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

        log_info "Starting Traffic Light Extraction for Scene $SCENE_ID..."
        
        if [[ -f "${SCRIPT_DIR}/extract_traffic_lights.sh" ]]; then
            bash "${SCRIPT_DIR}/extract_traffic_lights.sh" "$SCENE_ID" "$PROCESSED_DATA_ROOT" 2>&1 | tee -a "$SCENE_LOG"
            
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
             log_error "extract_traffic_lights.sh not found in ${SCRIPT_DIR}"
             return 1
        fi
    } 200>"$EXTRACT_TL_LOCK"
}

stage_train() {
    log_info "========== STAGE 4: Training =========="
    
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
    
    log_info "Starting Training for Scene $SCENE_ID (GPU: $GPU_ID)..."
    
    if [[ -f "${SCRIPT_DIR}/train.sh" ]]; then
        CUDA_VISIBLE_DEVICES="$GPU_ID" bash "${SCRIPT_DIR}/train.sh" "$SCENE_ID" "$PROCESSED_DATA_ROOT" 2>&1 | tee -a "$SCENE_LOG"
        
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
    log_info "========== STAGE 5: Render Novel Trajectory =========="
    
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
    
    log_info "Starting Rendering for Scene $SCENE_ID (GPU: $GPU_ID)..."
    
    if [[ -f "${SCRIPT_DIR}/render_novel_trajectory.sh" ]]; then
        CUDA_VISIBLE_DEVICES="$GPU_ID" bash "${SCRIPT_DIR}/render_novel_trajectory.sh" "$SCENE_ID" "$PROCESSED_DATA_ROOT" 2>&1 | tee -a "$SCENE_LOG"
        
        if [[ ${PIPESTATUS[0]} -eq 0 ]]; then
            if verify_rendered_output; then
                log_success "Rendering completed successfully"
                mkdir -p "$(dirname "$RENDER_MARKER")"
                touch "$RENDER_MARKER"
            else
                log_error "Rendering succeeded but validation failed"
                return 1
            fi
        fi
    else
        log_error "render_novel_trajectory.sh not found"
        return 1
    fi
}

# ============================================================================
# Main
# ============================================================================
main() {
    if [[ $# -lt 1 ]]; then
        echo "ERROR: Usage: $0 <scene_id> [gpu_id]"
        exit 1
    fi
    
    local scene_id_arg="$1"
    SCENE_ID="$scene_id_arg"
    
    init_scene_markers
    
    mkdir -p "$LOG_DIR"
    export SCENE_LOG="${LOG_DIR}/scene_${SCENE_ID}.log"
    
    validate_inputs "$@"
    
    local start_time=$(date +%s)
    log_info "Start processing Scene $SCENE_ID"
    
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
    log_success "Pipeline Completed! Scene $SCENE_ID"
    log_success "Total Duration: ${hours}h ${minutes}m ${seconds}s"
    log_success "========================================="
}

main "$@"
