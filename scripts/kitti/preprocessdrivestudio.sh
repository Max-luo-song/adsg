#!/bin/bash
###############################################################################
# KITTI Preprocessing Script
# Adapts the Waymo preprocessing pipeline for KITTI dataset
# Usage: ./scripts/kitti/preprocess.sh <scene_id> [input_dir] [output_dir]
# Example: ./scripts/kitti/preprocessdrivestudio.sh 2011_09_26_drive_0001_sync
###############################################################################

# Ensure script execution directory context
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")" # This is ADSTUDIO_ROOT
ADSTUDIO_ROOT="$PROJECT_ROOT"
DRIVESTUDIO_ROOT="/inspire/hdd/global_user/guoluosong-253108120129/project/drivestudio"

# Check args
if [ $# -lt 1 ]; then
    echo "ERROR: Usage: $0 <scene_id> [input_dir] [output_dir]"
    echo "ERROR: Example: $0 2011_09_26_drive_0001_sync ${DRIVESTUDIO_ROOT}/data/kitti/raw ${DRIVESTUDIO_ROOT}/data/kitti/processeddrivestudio"
    exit 1
fi

scene_id="$1"
# Extract date from scene_id (e.g., 2011_09_26_drive_0001_sync -> 2011_09_26)
date_part=$(echo "$scene_id" | cut -d'_' -f1-3)

if [ $# -ge 2 ]; then
    input_dir="$2"
else
    input_dir="/inspire/hdd/global_user/guoluosong-253108120129/project/data/kitti/raw"
fi

if [ $# -ge 3 ]; then
    output_dir="$3"
else
    output_dir="/inspire/hdd/global_user/guoluosong-253108120129/project/data/kitti/processeddrivestudio"
fi

# Normalize paths
input_dir="${input_dir%/}"
output_dir="${output_dir%/}"

echo "[INFO] Scene ID: $scene_id"
echo "[INFO] Date Split: $date_part"
echo "[INFO] Input Dir: $input_dir"
echo "[INFO] Output Dir: $output_dir"
echo "[INFO] DriveStudio Root: $DRIVESTUDIO_ROOT"
echo "[INFO] AdStudio Root: $ADSTUDIO_ROOT"

# Create temporary split files
# Stage 1 & 2 require: scene_id,start_timestep,end_timestep
# Note: Scripts skip the first line (header), so we must provide one.
temp_split_file_stage1="${DRIVESTUDIO_ROOT}/data/kitti_temp_split_stage1_${scene_id}.txt"
echo "scene_id,start,end" > "$temp_split_file_stage1"
echo "${scene_id},0,-1" >> "$temp_split_file_stage1"

# Stage 3 requires: scene_id (no timestamps)
# Note: Scripts skip the first line (header), so we must provide one.
temp_split_file_stage3="${DRIVESTUDIO_ROOT}/data/kitti_temp_split_stage3_${scene_id}.txt"
echo "scene_id" > "$temp_split_file_stage3"
echo "$scene_id" >> "$temp_split_file_stage3"

# Cleanup trap to remove temp files on exit
trap "rm -f $temp_split_file_stage1 $temp_split_file_stage3" EXIT

# Switch to Drivestudio root for execution as per docs/KITTI.md
cd "$DRIVESTUDIO_ROOT" || { echo "ERROR: Cannot cd to $DRIVESTUDIO_ROOT"; exit 1; }

# IMPORTANT: Add current directory to PYTHONPATH so that 'datasets' package is found
export PYTHONPATH="$DRIVESTUDIO_ROOT:$PYTHONPATH"

# ============================================================================
# STAGE 1: Basic Data Processing
# ============================================================================
echo "[INFO] ========== STAGE 1: Convert Data =========="

# Activate env if needed
CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate drivestudio

# Reference: drivestudio/docs/KITTI.md
# python datasets/preprocess.py --split 2011_09_26 --split_file ...
python datasets/preprocess.py \
    --data_root "$input_dir" \
    --target_dir "$output_dir" \
    --dataset kitti \
    --split "$date_part" \
    --split_file "$temp_split_file_stage1" \
    --workers 8 \
    --process_keys images lidar pose calib dynamic_masks objects

if [ $? -ne 0 ]; then
    echo "[ERROR] Data conversion failed"
    exit 1
fi

echo "[SUCCESS] Data conversion completed"

# ============================================================================
# STAGE 2: Semantic Segmentation (SegFormer)
# ============================================================================
echo "[INFO] ========== STAGE 2: Semantic Segmentation =========="

# SegFormer relies on adstudio installation
segformer_path="${ADSTUDIO_ROOT}/third_party/SegFormer"
checkpoint_path="$segformer_path/pretrained/segformer.b5.1024x1024.city.160k.pth"

# Pick GPU using adstudio utils.sh
if [ -f "${ADSTUDIO_ROOT}/scripts/utils.sh" ]; then
    source "${ADSTUDIO_ROOT}/scripts/utils.sh"
    gpu=$(pick_gpu)
else
    echo "[WARN] scripts/utils.sh not found, defaulting to GPU 0"
    gpu=0
fi

if [ -z "${gpu}" ]; then
    echo "[ERROR] No available GPU"
    exit 1
fi
echo "[INFO] Using GPU: ${gpu}"

# Activate SegFormer Env
conda activate segformer
# Keep PYTHONPATH for drivestudio's datasets tools
export PYTHONPATH="$DRIVESTUDIO_ROOT:$PYTHONPATH"

# Reference: drivestudio/docs/KITTI.md
CUDA_VISIBLE_DEVICES=${gpu} python datasets/tools/extract_masks.py \
    --data_root "$output_dir" \
    --segformer_path "$segformer_path" \
    --checkpoint "$checkpoint_path" \
    --split_file "$temp_split_file_stage1" \
    --process_dynamic_mask

if [ $? -ne 0 ]; then
    echo "[ERROR] Semantic segmentation failed"
    exit 1
fi

echo "[SUCCESS] Semantic segmentation completed"

# ============================================================================
# STAGE 3: SMPL Pose Extraction
# ============================================================================
echo "[INFO] ========== STAGE 3: SMPL Pose Extraction =========="

conda activate 4D-humans
export PYOPENGL_PLATFORM=osmesa
export PYTHONPATH="$ADSTUDIO_ROOT:$PYTHONPATH"

# Reference: drivestudio/scripts/kitti/extract_human_pose.sh
CUDA_VISIBLE_DEVICES=${gpu} python datasets/tools/humanpose_process.py \
    --dataset kitti \
    --data_root "$output_dir" \
    --split_file "$temp_split_file_stage3"

if [ $? -ne 0 ]; then
    echo "[ERROR] SMPL extraction failed"
    exit 1
fi

echo "[SUCCESS] SMPL extraction completed"
echo "[SUCCESS] ========== Preprocessing completed =========="
